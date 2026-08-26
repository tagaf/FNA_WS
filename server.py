#!/usr/bin/env python3
"""Live AD9643 scope/spectrum server: repeated block capture -> CUDA -> browser.

NOTE: the FPGA design is one-shot block capture, not a streaming pipeline.
This loop re-arms as fast as it can; `duty_cycle` and `coverage` in the metrics
report honestly what fraction of real time is actually being digitised.

DMA READBACK: this does NOT use gpu.FastC2H (raw os.readv() on the XDMA
character device). bisect_dma.py proved that path wedges the SoC even when
reading into ordinary malloc'd memory (it never got past that stage) — the
earlier theory that only CUDA-pinned destinations were unsafe was never
actually confirmed and was wrong. Readback instead goes through
ad9643.ddr_read_samples(), the vendor dma_from_device CLI via subprocess,
which is the one path bisect_dma.py confirmed completes without freezing.
See NOTES.md and bisect_marker.txt.
"""
import json, os, struct, sys, threading, time, argparse
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad9643 as A
import gpu

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# The full-resolution spectrum (nfft/2+1 bins, up to ~67M at max FFT size) is
# always computed on the GPU -- that part is fast (see NOTES.md). What was
# slow was shipping all of it over HTTP every frame. DISP_BINS caps the
# always-sent overview; ZOOM_MAX_BINS caps the opt-in full-resolution slice
# requested for a cropped frequency range (still far more resolution than an
# unzoomed view, bounded only so an extreme zoom request can't reintroduce
# the same multi-hundred-MB payload).
DISP_BINS = 4096
ZOOM_MAX_BINS = 1 << 20


def decimate_max(arr, out_n):
    """Max-decimate arr (1D) down to out_n points. No-op if already <= out_n.

    Uses np.maximum.reduceat over out_n (nearly) equal-width chunks -- a
    peak in any chunk still shows up, unlike mean/stride decimation, which
    matters for a spectrum (a narrow tone must not get averaged away)."""
    n = len(arr)
    if n <= out_n:
        return arr.astype(np.float32, copy=True)
    starts = np.linspace(0, n, out_n + 1).astype(np.int64)[:-1]
    return np.maximum.reduceat(arr, starts).astype(np.float32)


# ----------------------------------------------------------------- system
class SysMon:
    def __init__(self):
        self._prev = self._cpu()
        self.zones = []
        base = "/sys/class/thermal"
        if os.path.isdir(base):
            for z in sorted(os.listdir(base)):
                if z.startswith("thermal_zone"):
                    try:
                        t = open(f"{base}/{z}/type").read().strip()
                        self.zones.append((t, f"{base}/{z}/temp"))
                    except OSError:
                        pass

    @staticmethod
    def _cpu():
        with open("/proc/stat") as f:
            p = f.readline().split()[1:]
        v = [int(x) for x in p]
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0)

    def sample(self):
        tot, idle = self._cpu()
        dt, di = tot - self._prev[0], idle - self._prev[1]
        self._prev = (tot, idle)
        cpu = 100.0 * (1 - di / dt) if dt > 0 else 0.0
        temps = {}
        for name, path in self.zones:
            try:
                v = int(open(path).read()) / 1000.0
                if -50 < v < 200:
                    temps[name] = round(v, 1)
            except OSError:
                pass
        mem = {}
        try:
            for line in open("/proc/meminfo"):
                k, v = line.split(":")
                if k in ("MemTotal", "MemAvailable"):
                    mem[k] = int(v.split()[0]) * 1024
        except OSError:
            pass
        return {"cpu_pct": round(cpu, 1), "temps": temps,
                "mem_used": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0),
                "mem_total": mem.get("MemTotal", 0)}


# ------------------------------------------------------------ acquisition
class Engine:
    def __init__(self, nsamples=1 << 20, channel=1, speed=0, nfft=8192,
                 max_frames=64, trace_width=1024, avg=4, min_period=0.005):
        self.cfg = dict(nsamples=nsamples, channel=channel, speed=speed,
                        nfft=nfft, max_frames=max_frames, avg=avg)
        self.trace_width = trace_width
        self.min_period = min_period      # floor on loop period; leaves the
                                          # scheduler room for networking
        self.running = True
        self.lock = threading.Lock()
        self.frame = None
        self.err = None
        self.sysmon = SysMon()
        self._stop = threading.Event()
        self._dirty = threading.Event()
        self._single = threading.Event()   # set by trigger(): arm exactly one capture
        self.zoom = None                   # (lo_hz, hi_hz) or None: full-res crop request
        self.tot_frames = 0
        self.tot_samples = 0
        self.tot_bytes = 0
        self.timeouts = 0
        self.t_start = time.monotonic()
        self._acc = None
        self._acc_n = 0

    # ---- lifecycle
    def start(self):
        self.adc = A.Adc()
        self.sp = gpu.Spectrum(self.cfg["nfft"],
                               max_samples=max(1 << 22, self.cfg["nsamples"]),
                               trace_width=self.trace_width)
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def stop(self):
        self._stop.set()
        # One in-flight cycle at max samples/FFT takes ~2.3s (1.05s capture +
        # ~1s DMA + ~0.3s GPU); 3s left too little margin -- a single Ctrl-C
        # could still be waiting on join() when the terminal least expects it,
        # inviting an impatient second Ctrl-C mid-shutdown. 20s comfortably
        # covers worst case plus Spectrum teardown of the largest buffers.
        self.th.join(timeout=20)
        self.sp.close(); self.adc.close()

    def configure(self, **kw):
        trig = bool(kw.pop("trigger", False))
        UNSET = object()
        zoom = kw.pop("zoom", UNSET)
        with self.lock:
            for k, v in kw.items():
                if k == "running":
                    self.running = bool(v)
                elif k in self.cfg:
                    self.cfg[k] = int(v)
            self._acc = None; self._acc_n = 0
            if trig:
                self.running = False        # a trigger always arms exactly one
            if zoom is not UNSET:
                self.zoom = None if not zoom else (float(zoom[0]), float(zoom[1]))
        self._dirty.set()
        if trig:
            self._single.set()

    def trigger(self):
        """Arm exactly one capture, whatever the current run state."""
        self.configure(trigger=True)

    # ---- the loop
    def _loop(self):
        while not self._stop.is_set():
            if self.running:
                pass
            elif self._single.is_set():
                self._single.clear()
            else:
                time.sleep(0.02); continue
            with self.lock:
                cfg = dict(self.cfg)
            t0 = time.monotonic()
            try:
                self._one(cfg)
            except (A.CaptureTimeout, A.DmaTimeout) as e:
                self.timeouts += 1
                self.err = str(e)
                time.sleep(0.2)
            except Exception as e:
                self.err = f"{type(e).__name__}: {e}"
                time.sleep(0.3)
            if self.running:               # no artificial delay after a single shot
                slack = self.min_period - (time.monotonic() - t0)
                if slack > 0:
                    time.sleep(slack)

    def _one(self, cfg):
        t_loop0 = time.monotonic()
        N, ch, sp_, nfft = cfg["nsamples"], cfg["channel"], cfg["speed"], cfg["nfft"]

        # rebuild the GPU context if FFT size or capacity changed
        if nfft != self.sp.nfft or N > self.sp.max_samples:
            self.sp.close()
            self.sp = gpu.Spectrum(nfft, max_samples=max(1 << 22, N),
                                   trace_width=self.trace_width)
            self._acc = None; self._acc_n = 0

        A.Adc._validate(N, ch, sp_)

        # arm clears Adc_Finish (it otherwise idles high from the previous run)
        self.adc.wr(A.REG_SPEED, sp_)
        self.adc.wr(A.REG_CHANNEL, ch)
        self.adc.wr(A.REG_NSAMPLES, N)
        expect = N * (sp_ + 1) / A.BASE_CLOCK_HZ
        timeout = max(0.5, expect * 4 + 0.5)
        t0 = time.monotonic()
        self.adc.wr(A.REG_START, 0)
        self.adc.wr(A.REG_START, 1)
        while not self.adc.finished:
            if time.monotonic() - t0 > timeout:
                st = self.adc.regs(); self.adc.recover()
                raise A.CaptureTimeout(f"Adc_Finish low after {timeout:.3f}s regs={st}")
            time.sleep(1e-4)          # never busy-spin on MMIO reads
        t_cap = time.monotonic() - t0

        # Readback via the vendor dma_from_device CLI (subprocess + tempfile) —
        # NOT gpu.FastC2H's raw os.readv(), which bisect_dma.py showed wedges
        # the SoC regardless of destination buffer type. See module docstring.
        t1 = time.monotonic()
        nbytes = N * 2
        d = A.ddr_read_samples(N)
        n = len(d)
        np.copyto(self.sp.stage[:n], d)
        self.sp.load(n * 2)
        got = n * 2
        t_dma = time.monotonic() - t1

        # CUDA
        t2 = time.monotonic()
        spec = self.sp.process(N, max_frames=cfg["max_frames"])
        t_gpu_wall = time.monotonic() - t2

        # exponential spectrum averaging
        navg = max(1, cfg["avg"])
        if self._acc is None or self._acc.shape != spec.shape:
            self._acc = spec.astype(np.float64).copy(); self._acc_n = 1
        else:
            a = 1.0 / navg
            self._acc = (1 - a) * self._acc + a * spec
            self._acc_n = min(self._acc_n + 1, navg)
        shown = self._acc.astype(np.float32)

        t_loop = time.monotonic() - t_loop0
        self.tot_frames += 1
        self.tot_samples += N
        self.tot_bytes += nbytes

        fs = A.sample_rate(sp_)
        st = self.sp.stats
        bin_hz = fs / nfft
        # Peak/noise stay full-resolution (cheap: an argmax/median over the
        # real array) even though only a downsampled copy goes over HTTP.
        k = int(np.argmax(shown[1:]) + 1)
        peak_f = k * fs / nfft
        peak_db = float(shown[k])
        noise = float(np.median(shown))
        gt = self.sp.times

        disp = decimate_max(shown, DISP_BINS)
        disp_bin_hz = (len(shown) - 1) * bin_hz / max(1, len(disp) - 1) \
            if len(disp) > 1 else bin_hz

        zoom_meta = {"active": False, "bins": 0, "lo_hz": 0.0, "hi_hz": 0.0, "bin_hz": 0.0}
        zoom_bytes = b""
        if self.zoom is not None:
            lo_hz, hi_hz = self.zoom
            i0 = max(0, min(len(shown) - 1, int(lo_hz / bin_hz)))
            i1 = max(i0 + 1, min(len(shown), int(np.ceil(hi_hz / bin_hz))))
            zslice = shown[i0:i1]
            zbin_hz = bin_hz
            if len(zslice) > ZOOM_MAX_BINS:
                zbin_hz = (i1 - i0) * bin_hz / ZOOM_MAX_BINS
                zslice = decimate_max(zslice, ZOOM_MAX_BINS)
            zoom_bytes = zslice.astype(np.float32).tobytes()
            zoom_meta = {"active": True, "bins": len(zslice),
                         "lo_hz": i0 * bin_hz, "hi_hz": i1 * bin_hz, "bin_hz": zbin_hz}

        m = {
            "ts": time.time(),
            "cfg": cfg,
            "engine": {"running": self.running},
            "fs_hz": fs,
            "nbins": int(self.sp.nbins),          # true FFT resolution
            "disp_bins": int(len(disp)),          # length of the array actually sent
            "disp_bin_hz": disp_bin_hz,
            "zoom": zoom_meta,
            "trace_width": int(self.trace_width),
            "nframes": int(self.sp.nframes),
            "bin_hz": bin_hz,
            "acq": {
                "capture_ms": t_cap * 1e3,
                "capture_theory_ms": expect * 1e3,
                "dma_ms": t_dma * 1e3,
                "dma_gbps": (nbytes / t_dma) / 1e9 if t_dma > 0 else 0,
                "gpu_wall_ms": t_gpu_wall * 1e3,
                "loop_ms": t_loop * 1e3,
                "fps": 1.0 / t_loop if t_loop > 0 else 0,
                "duty_pct": 100.0 * t_cap / t_loop if t_loop > 0 else 0,
                "coverage_pct": 100.0 * (N / fs) / t_loop if t_loop > 0 else 0,
                "eff_msps": (N / t_loop) / 1e6 if t_loop > 0 else 0,
                "bytes_ok": got == nbytes,
                "dma_path": "vendor_subprocess",
            },
            "gpu": {
                "h2d_ms": float(gt[0]), "kern_ms": float(gt[1]),
                "fft_ms": float(gt[2]), "post_ms": float(gt[3]),
                "total_ms": float(gt[4]),
                "fft_gsps": (nfft * self.sp.nframes / (gt[2] * 1e-3)) / 1e9
                            if gt[2] > 0 else 0,
                "mem_free": gpu_mem_cached()[0], "mem_total": gpu_mem_cached()[1],
            },
            "sig": {
                "min": float(st[0]), "max": float(st[1]), "mean": float(st[2]),
                "std": float(st[3]),
                "vpp_codes": float(st[1] - st[0]),
                "peak_hz": peak_f, "peak_db": peak_db,
                "noise_db": noise, "snr_db": peak_db - noise,
                "clipping": bool(st[1] >= 16380 or st[0] <= 2),
                "is_test_ramp": ch == 0,
            },
            "totals": {
                "frames": self.tot_frames, "samples": self.tot_samples,
                "bytes": self.tot_bytes, "timeouts": self.timeouts,
                "uptime_s": time.monotonic() - self.t_start,
                "avg_depth": self._acc_n,
            },
            "sys": self.sysmon.sample(),
            "err": self.err,
        }
        hdr = json.dumps(m).encode()
        blob = (struct.pack("<I", len(hdr)) + hdr +
                disp.tobytes() + zoom_bytes +
                self.sp.tmin.tobytes() + self.sp.tmax.tobytes())
        with self.lock:
            self.frame = blob


_memcache = [0, 0, 0.0]


def gpu_mem_cached():
    if time.monotonic() - _memcache[2] > 1.0:
        try:
            f, t = gpu.gpu_mem(); _memcache[0], _memcache[1] = f, t
        except Exception:
            pass
        _memcache[2] = time.monotonic()
    return _memcache[0], _memcache[1]


# ----------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            try:
                b = open(os.path.join(WEB, "index.html"), "rb").read()
            except OSError:
                b = b"<h1>web/index.html missing</h1>"
            return self._send(200, b, "text/html; charset=utf-8")
        if p == "/frame":
            with self.engine.lock:
                f = self.engine.frame
            if f is None:
                return self._send(503, b"no frame yet", "text/plain")
            return self._send(200, f, "application/octet-stream")
        if p == "/limits":
            return self._send(200, json.dumps({
                "granularity": A.SAMPLE_GRANULARITY,
                "max_samples": A.MAX_SAMPLES,
                "base_clock": A.BASE_CLOCK_HZ,
            }).encode(), "application/json")
        if p == "/status":
            # Cheap, authoritative, polled independently of /frame: engine.err
            # is set the instant a capture fails, but a failed capture never
            # reaches the code that rebuilds the (possibly large) frame blob,
            # so a client relying on /frame alone would show a stale error
            # state (or none) until the next successful capture. This lets
            # the UI show "capturing" / "done" / "timed out" immediately.
            e = self.engine
            return self._send(200, json.dumps({
                "running": e.running,
                "frames": e.tot_frames,
                "timeouts": e.timeouts,
                "err": e.err,
                "has_frame": e.frame is not None,
            }).encode(), "application/json")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        p = self.path.split("?")[0]
        if p == "/restart":
            # Ack first, THEN release hardware and exit -- os._exit() so no
            # atexit/signal handling delays it further. Relies on a
            # supervisor (systemd, Restart=always) to actually bring the
            # process back; without one this just stops the server for good.
            self._send(200, b'{"ok":true,"restarting":true}', "application/json")
            def _restart():
                try:
                    self.engine.stop()
                finally:
                    os._exit(0)
            threading.Thread(target=_restart, daemon=True).start()
            return
        if p != "/control":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            cfg = json.loads(self.rfile.read(n) or b"{}")
            self.engine.configure(**cfg)
            self.engine.err = None
            return self._send(200, b'{"ok":true}', "application/json")
        except Exception as e:
            return self._send(400, json.dumps({"error": str(e)}).encode(),
                              "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", type=int, default=8090)  # 8080 is taken
                                                              # by openshell-gateway
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("-n", "--nsamples", type=int, default=1 << 20)
    ap.add_argument("-c", "--channel", type=int, default=1)
    ap.add_argument("--nfft", type=int, default=8192)
    ap.add_argument("--min-period", type=float, default=0.005,
                    help="floor on loop period in seconds (default 5 ms)")
    a = ap.parse_args()
    eng = Engine(nsamples=a.nsamples, channel=a.channel, nfft=a.nfft,
                 min_period=a.min_period)
    eng.start()
    Handler.engine = eng
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f"serving on http://{a.bind}:{a.port}/  (ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()


if __name__ == "__main__":
    main()
