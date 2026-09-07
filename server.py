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
import errno, fcntl, signal, socket, subprocess
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad9643 as A
import gpu
import noise

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")

# The full-resolution spectrum (nfft/2+1 bins, up to ~67M at max FFT size) is
# always computed on the GPU -- that part is fast (see NOTES.md). What was
# slow was shipping all of it over HTTP every frame. DISP_BINS caps the
# always-sent overview; ZOOM_MAX_BINS caps the opt-in full-resolution slice
# requested for a cropped frequency range (still far more resolution than an
# unzoomed view, bounded only so an extreme zoom request can't reintroduce
# the same multi-hundred-MB payload).
DISP_BINS = 4096
ZOOM_MAX_BINS = 1 << 20


def log_display(shown, bin_hz, out_n):
    """Max-pool the spectrum onto out_n points spaced uniformly in log(f),
    from the first bin (f0 = bin_hz) to Nyquist. A linear-in-frequency
    overview capped at out_n points pins its first point to fs/2/out_n
    (~31 kHz at 250 Msps) regardless of FFT length -- the low decades that a
    long record buys were invisible. Log spacing gives them true per-bin
    detail; the top decades pool many bins per point (max-pooled, so peaks
    survive). Display point k covers log-fraction [k/out_n,(k+1)/out_n] of
    [log f0, log f1] -- the client maps it linearly onto its log axis."""
    nb = len(shown)
    if nb < 4:
        d = shown[1:].astype(np.float32, copy=True)
        return d, bin_hz, max(1, nb - 1) * bin_hz
    idx = np.power(10.0, np.linspace(0.0, np.log10(nb - 1), out_n + 1))
    starts = np.floor(idx[:-1]).astype(np.int64)
    starts = np.maximum.accumulate(np.clip(starts, 1, nb - 2)) - 1  # into shown[1:]
    d = np.maximum.reduceat(shown[1:], starts).astype(np.float32)
    return d, bin_hz, (nb - 1) * bin_hz


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
        self.wifi_if = None
        try:
            for line in open("/proc/net/wireless"):
                if ":" in line:
                    self.wifi_if = line.split(":")[0].strip()
        except OSError:
            pass
        self._wifi_rate = [0.0, 0.0]     # [tx_mbps, last-checked]
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
        net = None
        if self.wifi_if:
            try:
                for line in open("/proc/net/wireless"):
                    if line.strip().startswith(self.wifi_if + ":"):
                        p = line.split()
                        net = {"iface": self.wifi_if,
                               "rssi_dbm": float(p[3].rstrip("."))}
            except (OSError, ValueError, IndexError):
                pass
            # tx bitrate needs `iw`; refresh at most every 5 s
            if net and time.monotonic() - self._wifi_rate[1] > 5.0:
                self._wifi_rate[1] = time.monotonic()
                try:
                    out = subprocess.run(
                        ["iw", "dev", self.wifi_if, "link"],
                        capture_output=True, text=True, timeout=1.5).stdout
                    for ln in out.splitlines():
                        if "tx bitrate:" in ln:
                            self._wifi_rate[0] = float(ln.split()[2])
                except Exception:
                    pass
            if net:
                net["tx_mbps"] = self._wifi_rate[0]
        return {"cpu_pct": round(cpu, 1), "temps": temps, "net": net,
                "mem_used": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0),
                "mem_total": mem.get("MemTotal", 0)}


# ---------------------------------------------------------- DMA readback
class DmaReader:
    """Readback with two paths and automatic fallback:

    helper_resident  native/xdma_shm_reader stays alive across frames --
                     vendor-exact device access (same open flags, aligned
                     bounce buffer, same chunked read loop) minus the
                     per-frame process spawn and file round-trip.
                     Opt-in via --fast-dma; NOT yet validated on hardware.
    vendor_subprocess  dma_from_device CLI per frame (the proven path,
                     default). Any helper failure falls back here for the
                     rest of the session and is reported in metrics.
    """

    def __init__(self, use_helper, dev=A.C2H_DEV):
        self.path = "vendor_subprocess"
        self.p = None
        self.view = None
        self._shm_path = None
        if use_helper:
            try:
                self._start_helper(dev)
            except Exception as e:
                self.path = f"vendor_subprocess (helper: {e})"
                self._kill_helper()

    def _start_helper(self, dev):
        import select
        exe = os.path.join(HERE, "native", "xdma_shm_reader")
        if not os.path.exists(exe):
            raise RuntimeError("helper binary missing (make -C native)")
        self._shm_path = f"/dev/shm/adc_helper_{os.getpid()}.buf"
        # tmpfs + malloc are both lazy: sizing for the full 500 MB window
        # costs nothing until pages are actually touched
        self.max_bytes = A.WR_WINDOW_BYTES
        self.p = subprocess.Popen([exe, dev, self._shm_path,
                                   str(self.max_bytes)],
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)
        r, _, _ = select.select([self.p.stdout], [], [], 3.0)
        banner = self.p.stdout.readline() if r else ""
        if not banner.startswith("READY"):
            raise RuntimeError(f"helper did not come up ({banner.strip()!r})")
        import mmap as _mmap
        self._f = open(self._shm_path, "r+b")
        self._mm = _mmap.mmap(self._f.fileno(), self.max_bytes)
        self.view = np.frombuffer(self._mm, dtype=np.uint16)
        self.path = "helper_resident"

    def _kill_helper(self):
        if self.p is not None:
            try:
                self.p.kill(); self.p.wait(timeout=2)
            except Exception:
                pass
            self.p = None
        self.view = None
        for attr in ("_mm", "_f"):
            try:
                getattr(self, attr).close()
            except Exception:
                pass
        if self._shm_path:
            try:
                os.unlink(self._shm_path)
            except OSError:
                pass

    def read_into(self, out, nsamples, addr=0):
        nbytes = nsamples * 2
        if self.p is not None:
            import select
            deadline = max(1.0, nbytes / 50e6 + 1.5)
            try:
                self.p.stdin.write(f"R {addr} {nbytes}\n")
                self.p.stdin.flush()
                r, _, _ = select.select([self.p.stdout], [], [], deadline)
                reply = self.p.stdout.readline().strip() if r else ""
            except (BrokenPipeError, OSError):
                reply = ""
            if reply.startswith("OK"):
                got = int(reply.split()[1]) // 2
                np.copyto(out[:got], self.view[:got])
                return got
            # helper failed or stalled: kill it, permanent vendor fallback
            self._kill_helper()
            self.path = f"vendor_subprocess (helper failed: {reply or 'timeout'})"
            raise A.DmaTimeout(
                f"resident DMA helper failed ({reply or 'no reply within '
                f'{deadline:.1f}s'}); fell back to vendor CLI for the rest "
                f"of this session")
        return A.ddr_read_into(out, nsamples, addr)

    def close(self):
        if self.p is not None:
            try:
                self.p.stdin.write("Q\n"); self.p.stdin.flush()
                self.p.wait(timeout=2)
            except Exception:
                pass
        self._kill_helper()


class MockAdc:
    """Synthetic stand-in with correct timing semantics (finish goes low on
    arm, comes back after N*(speed+1)/fs). NEVER opens /dev/*. For UI and
    server development when the FPGA is off."""

    def __init__(self):
        self._r = {A.REG_START: 0, A.REG_SPEED: 0, A.REG_CHANNEL: 1,
                   A.REG_NSAMPLES: 0, A.REG_FINISH: 1}
        self._t_done = 0.0

    def wr(self, off, val):
        prev = self._r.get(A.REG_START, 0)
        self._r[off] = int(val)
        if off == A.REG_NSAMPLES:
            self._r[A.REG_FINISH] = 0
        if off == A.REG_START and val == 1 and prev == 0:
            expect = (self._r[A.REG_NSAMPLES] *
                      (self._r[A.REG_SPEED] + 1) / A.BASE_CLOCK_HZ)
            self._t_done = time.monotonic() + expect
            self._r[A.REG_FINISH] = 0

    def rd(self, off):
        if off == A.REG_FINISH and time.monotonic() >= self._t_done:
            self._r[A.REG_FINISH] = 1
        return self._r.get(off, 0)

    @property
    def finished(self):
        return bool(self.rd(A.REG_FINISH) & 1)

    def regs(self):
        return {n: self.rd(o) for n, o in (
            ("start", A.REG_START), ("speed", A.REG_SPEED),
            ("channel", A.REG_CHANNEL), ("nsamples", A.REG_NSAMPLES),
            ("finish", A.REG_FINISH))}

    def recover(self):
        self._r[A.REG_FINISH] = 1
        return True

    def close(self):
        pass


class MockDma:
    """Phase-continuous 25 MHz tone at -14 dBFS + noise, mimicking real data.

    Reads the speed register from the paired MockAdc and drops samples like
    the FPGA's decimator, so a divided rate shows the tone at the physically
    correct (possibly aliased) frequency instead of a mock artifact."""
    path = "mock_synth"
    F0 = 25e6

    def __init__(self, adc=None):
        self._adc = adc
        self._phase = 0
        self._rng = np.random.default_rng(0)

    def read_into(self, out, nsamples, addr=0):
        step = (self._adc._r.get(A.REG_SPEED, 0) + 1) if self._adc else 1
        t = self._phase + step * np.arange(nsamples, dtype=np.float64)
        self._phase += step * nsamples
        sig = (8192.0 + 1638.0 * np.sin(2 * np.pi * self.F0 / A.BASE_CLOCK_HZ * t)
               + self._rng.normal(0, 6, nsamples))
        out[:nsamples] = np.clip(sig, 0, 16383).astype(np.uint16)
        return nsamples

    def close(self):
        pass


# ------------------------------------------------------------ acquisition
class Engine:
    def __init__(self, nsamples=1 << 20, channel=1, speed=0, nfft=8192,
                 max_frames=64, trace_width=1024, avg=4, min_period=0.005,
                 fast_dma=False, mock=False):
        self.fast_dma = fast_dma
        self.mock = mock
        self.cfg = dict(nsamples=nsamples, channel=channel, speed=speed,
                        nfft=nfft, max_frames=max_frames, avg=avg,
                        trace_samples=-1,     # -1: trace shows the whole record
                                              # (explicit count still accepted)
                        classify=0,           # 1: run peak/noise classification
                        readback=-1)   # -1 auto, 0 full N, >0 explicit
        self.trace_width = trace_width
        self.min_period = min_period      # floor on loop period; leaves the
                                          # scheduler room for networking
        self.running = True
        self.lock = threading.Lock()
        self.new_frame = threading.Condition(self.lock)
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
        self._acc_tag = None   # (fs, nfft, channel) the accumulator belongs to

        # Classification runs in its own thread: analyse() costs 100-560 ms on
        # a full-resolution spectrum (measured 2^20..2^24 bins), which would
        # throttle a ~50 fps capture loop to a crawl. EMI sources drift slowly,
        # so a ~1 Hz classification against a live-rate display is the right
        # trade. The worker takes a snapshot only when idle, so it can never
        # queue up work faster than it retires it.
        self.analysis = None
        self.classify_period = 1.0
        self.pfa = 1e-6
        self._an_lock = threading.Lock()
        self._an_cv = threading.Condition(self._an_lock)
        self._an_req = None
        self._an_last = 0.0

    # ---- lifecycle
    def start(self):
        if self.mock:
            self.adc = MockAdc()
            self.dma = MockDma(self.adc)
        else:
            self.adc = A.Adc()
            self.dma = DmaReader(self.fast_dma)
        self.sp = gpu.Spectrum(self.cfg["nfft"],
                               max_samples=max(1 << 22, self.cfg["nsamples"]),
                               trace_width=self.trace_width)
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        self.an_th = threading.Thread(target=self._analysis_loop, daemon=True)
        self.an_th.start()

    def stop(self):
        self._stop.set()
        # One in-flight cycle at max samples/FFT takes ~2.3s (1.05s capture +
        # ~1s DMA + ~0.3s GPU); 3s left too little margin -- a single Ctrl-C
        # could still be waiting on join() when the terminal least expects it,
        # inviting an impatient second Ctrl-C mid-shutdown. 20s comfortably
        # covers worst case plus Spectrum teardown of the largest buffers.
        with self._an_lock:
            self._an_cv.notify_all()
        self.th.join(timeout=20)
        if getattr(self, "an_th", None):
            self.an_th.join(timeout=5)
        self.sp.close(); self.dma.close(); self.adc.close()

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

    MAX_WIRE_FAMILIES = 8
    MAX_WIRE_MEMBERS = 96
    MAX_WIRE_SPURS = 32

    def _analysis_loop(self):
        while not self._stop.is_set():
            with self._an_lock:
                while self._an_req is None and not self._stop.is_set():
                    self._an_cv.wait(timeout=0.5)
                req = self._an_req
            if req is None:
                continue
            try:
                spec, bin_hz, fs, K, nfft, ch = req
                r = noise.analyse(spec, bin_hz, fs, nframes=K, pfa=self.pfa)
                self.analysis = self._shape_analysis(r, fs, nfft, ch)
            except Exception as e:
                self.analysis = {"error": f"{type(e).__name__}: {e}"}
            finally:
                with self._an_lock:
                    self._an_req = None

    def _live_analysis(self, shown, bin_hz, fs, nfft, ch):
        """Re-sample the stored family/spur frequencies against the CURRENT
        spectrum, every frame.

        The structural result (which frequencies belong to which family) comes
        from the ~1 Hz worker, but marker *levels* must track the trace or the
        dots visibly lag it. Frequencies are physical, so they stay valid when
        nfft changes -- only the bin mapping moves, and re-deriving it is a few
        hundred lookups. fs or channel changing does invalidate the structure
        (aliasing differs; a different source entirely), so that still clears.
        """
        an = self.analysis
        if not an or an.get("error"):
            return an
        if an.get("channel") != ch or abs(an.get("fs_hz", 0.0) - fs) > 1.0:
            return {"stale": True, "reason": "fs/channel changed",
                    "families": [], "spurs": []}
        nb = len(shown)
        if nb < 2 or bin_hz <= 0:
            return an

        def levels(freqs):
            if not freqs:
                return []
            idx = np.clip(np.rint(np.asarray(freqs, dtype=np.float64) / bin_hz)
                          .astype(np.int64), 0, nb - 1)
            return [float(v) for v in shown[idx]]

        out = dict(an)
        out["families"] = [dict(f, dbs=levels(f.get("freqs", [])))
                           for f in an.get("families", [])]
        out["spurs"] = [dict(sp, db=(levels([sp["freq_hz"]]) or [sp["db"]])[0])
                        for sp in an.get("spurs", [])]
        out["structure_nfft"] = an.get("nfft")
        out["nfft"] = nfft          # levels are current: markers are in sync
        out["live"] = True
        return out

    def _shape_analysis(self, r, fs, nfft, ch):
        """Trim to what the plot needs -- a full peak list can be thousands
        of entries and would dwarf the spectrum payload itself."""
        fams = []
        for i, f in enumerate(r["families"][:self.MAX_WIRE_FAMILIES]):
            mem = sorted(f["members"], key=lambda m: -m["db"])[:self.MAX_WIRE_MEMBERS]
            fams.append({
                "id": i,
                "f0_hz": f["f0_hz"],
                "label": f.get("label", "?"),
                "why": f.get("why", ""),
                "n_members": f["n_members"],
                "density": f["density"],
                "significance": f.get("significance", 0.0),
                "peak_db": f["peak_db"],
                "freqs": [m["freq_hz"] for m in mem],
                "dbs": [m["db"] for m in mem],
            })
        spurs = [p for p in r["peaks"] if p.get("label") not in (None, "family_member")]
        spurs.sort(key=lambda p: -p["db"])
        return {
            "ts": time.time(), "fs_hz": fs, "nfft": nfft, "channel": ch,
            "floor_db": r["floor_db"],
            "thr_db": r["threshold_db_over_floor"],
            "n_peaks": r["n_peaks"],
            "families": fams,
            "spurs": [{"freq_hz": p["freq_hz"], "db": p["db"],
                       "label": p["label"], "why": p.get("why", "")}
                      for p in spurs[:self.MAX_WIRE_SPURS]],
        }

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

        # Physics check. The FSM cannot digitise N samples faster than N/fs,
        # so a completion far short of that means Adc_Finish was still high
        # from the previous run (it idles high — see NOTES.md) and we are
        # about to read a buffer that was never filled. Flag it rather than
        # publishing a plausible-looking spectrum built from stale DDR.
        suspect = t_cap < 0.5 * expect

        # Read back only what is actually consumed. The spectrum uses exactly
        # max_frames*nfft samples and the trace is decimated to trace_width
        # columns, so at large N the full transfer is almost entirely wasted:
        # at N=262,144,000 the FFT consumes 1 MB of the 500 MB moved.
        #   readback = -1  auto: read what the FFT needs (default)
        #            =  0  full N
        #            = >0  explicit sample count
        # frames the record can actually supply caps the need: at nfft close
        # to N, max_frames*nfft would otherwise demand (and DMA) far more
        # than the FFT can consume
        need = min(cfg["max_frames"], max(1, N // nfft)) * nfft
        tr_req = cfg.get("trace_samples", -1)
        trace_n = N if tr_req < 0 else min(max(256, tr_req), N)
        rb = cfg.get("readback", -1)
        want = max(need, trace_n, rb) if rb > 0 else (N if rb == 0
                                                      else max(need, trace_n))
        read_n = min(N, want)
        g = A.SAMPLE_GRANULARITY
        read_n = max(g, -(-read_n // g) * g)     # round UP to granularity
        read_n = min(read_n, N)
        trace_n = min(trace_n, read_n)

        # Readback via the vendor dma_from_device CLI (subprocess + tempfile) —
        # NOT gpu.FastC2H's raw os.readv(), which bisect_dma.py showed wedges
        # the SoC regardless of destination buffer type. See module docstring.
        t1 = time.monotonic()
        nbytes = read_n * 2
        n = self.dma.read_into(self.sp.stage, read_n)
        self.sp.load(n * 2)
        got = n * 2
        t_dma = time.monotonic() - t1

        # CUDA
        t2 = time.monotonic()
        spec = self.sp.process(read_n, max_frames=cfg["max_frames"],
                               trace_n=trace_n)
        t_gpu_wall = time.monotonic() - t2

        # exponential spectrum averaging
        navg = max(1, cfg["avg"])
        # configure() resets the accumulator, but a frame produced under the
        # OLD cfg can still land afterwards and re-seed it; spectra from
        # different fs/nfft/channel share bin indices, so blending them shows
        # stale peaks at wrong frequencies (seen: a 6.25 MHz peak from
        # speed=7 displayed as 50 MHz after switching to speed=0). Tag the
        # accumulator with the producing config and reseed on any mismatch.
        tag = (fs := A.sample_rate(sp_), nfft, ch)
        if (self._acc is None or self._acc.shape != spec.shape
                or self._acc_tag != tag):
            self._acc = spec.copy(); self._acc_n = 1
            self._acc_tag = tag
        elif navg == 1:
            np.copyto(self._acc, spec); self._acc_n = 1
        else:
            a = np.float32(1.0 / navg)
            self._acc *= (1 - a); self._acc += a * spec
            self._acc_n = min(self._acc_n + 1, navg)
        shown = self._acc

        t_loop = time.monotonic() - t_loop0
        self.tot_frames += 1
        self.tot_samples += N
        self.tot_bytes += nbytes

        st = self.sp.stats
        bin_hz = fs / nfft
        # Peak/noise stay full-resolution (cheap: an argmax/median over the
        # real array) even though only a downsampled copy goes over HTTP.
        k = int(np.argmax(shown[1:]) + 1)
        peak_f = k * fs / nfft
        peak_db = float(shown[k])
        step = max(1, (len(shown) - 1) // 16384)
        noise = float(np.median(shown[1::step]))
        gt = self.sp.times

        if cfg.get("classify"):
            now = time.monotonic()
            if now - self._an_last >= self.classify_period:
                with self._an_lock:
                    if self._an_req is None:          # only when idle
                        self._an_req = (shown.copy(), bin_hz, fs,
                                        max(1, self.sp.nframes * max(1, self._acc_n)),
                                        nfft, ch)
                        self._an_last = now
                        self._an_cv.notify()
        elif self.analysis is not None:
            self.analysis = None

        disp, disp_f0, disp_f1 = log_display(shown, bin_hz, DISP_BINS)

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
            "disp_log": True,                     # log-spaced from disp_f0 to disp_f1
            "disp_f0": disp_f0,
            "disp_f1": disp_f1,
            "zoom": zoom_meta,
            "trace_width": int(self.trace_width),
            "nframes": int(self.sp.nframes),
            "bin_hz": bin_hz,
            "acq": {
                "capture_ms": t_cap * 1e3,
                "capture_theory_ms": expect * 1e3,
                "capture_suspect": bool(suspect),
                "capture_samples": N,
                "read_samples": read_n,
                "read_pct": 100.0 * read_n / N if N else 0.0,
                "trace_samples": trace_n,
                "fft_samples": min(need, read_n),
                "dma_ms": t_dma * 1e3,
                "dma_gbps": (nbytes / t_dma) / 1e9 if t_dma > 0 else 0,
                "gpu_wall_ms": t_gpu_wall * 1e3,
                "loop_ms": t_loop * 1e3,
                "fps": 1.0 / t_loop if t_loop > 0 else 0,
                "duty_pct": 100.0 * t_cap / t_loop if t_loop > 0 else 0,
                "coverage_pct": 100.0 * (N / fs) / t_loop if t_loop > 0 else 0,
                "eff_msps": (N / t_loop) / 1e6 if t_loop > 0 else 0,
                "bytes_ok": got == nbytes,
                "dma_path": self.dma.path,
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
            "analysis": self._live_analysis(shown, bin_hz, fs, nfft, ch)
                        if cfg.get("classify") else None,
            "err": self.err,
        }
        hdr = json.dumps(m).encode()
        blob = (struct.pack("<I", len(hdr)) + hdr +
                disp.tobytes() + zoom_bytes +
                self.sp.tmin.tobytes() + self.sp.tmax.tobytes())
        with self.new_frame:
            self.frame = blob
            self.new_frame.notify_all()   # wake every long-poll waiter
        self.err = None       # a completed capture clears any stale error


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
            # Long-poll: /frame?wait=1&since=<frames-counter> parks the
            # request until a frame newer than <since> exists (25 s cap,
            # then 304). One round-trip per frame instead of a stream of
            # /status polls -- far kinder to a lossy WiFi link.
            q = {}
            if "?" in self.path:
                for kv in self.path.split("?", 1)[1].split("&"):
                    k, _, v = kv.partition("=")
                    q[k] = v
            e = self.engine
            if q.get("wait") == "1":
                since = int(q.get("since", "0") or 0)
                deadline = time.monotonic() + 25.0
                with e.new_frame:
                    while (e.tot_frames <= since or e.frame is None):
                        left = deadline - time.monotonic()
                        if left <= 0 or not e.new_frame.wait(timeout=left):
                            return self._send(204, b"", "text/plain")
                    f = e.frame
                return self._send(200, f, "application/octet-stream")
            with e.lock:
                f = e.frame
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
                    A._xfer_cleanup()      # os._exit() skips atexit
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


LOCK_PATH = os.path.expanduser("~/.adc_capture.lock")


class ReuseServer(ThreadingHTTPServer):
    # SO_REUSEADDR: a stale TIME_WAIT socket from the previous run must never
    # stop us rebinding. (It does NOT let two live servers share a port.)
    allow_reuse_address = True
    daemon_threads = True
    # Nagle + delayed-ACK adds up to ~40 ms to each small reply; the UI polls
    # /status every animation frame, so leaving it on made the whole page
    # feel like it stuttered even when the engine was healthy.
    disable_nagle_algorithm = True


def acquire_single_instance(replace=False):
    """Only one process may drive the AXI_CMD registers: two engines
    interleaving triggers would corrupt each other's captures.

    Returns (lockfile, None) on success, or (None, "pid port") if another
    instance holds it."""
    f = open(LOCK_PATH, "a+")
    for attempt in range(2):
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.seek(0); f.truncate()
            return f, None
        except OSError:
            f.seek(0)
            info = f.read().strip() or "?"
            if not replace or attempt:
                return None, info
            pid = int(info.split()[0]) if info.split()[0].isdigit() else None
            if pid:
                print(f"--replace: stopping existing instance (pid {pid}) …")
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                for _ in range(50):
                    try:
                        os.kill(pid, 0); time.sleep(0.1)
                    except ProcessLookupError:
                        break
    return None, "?"


def bind_server(bind, port, tries=20):
    """Bind without ever dying on EADDRINUSE: walk forward from the requested
    port, then fall back to a kernel-assigned one."""
    if port == 0:
        srv = ReuseServer((bind, 0), Handler)
        return srv, srv.server_address[1]
    first_err = None
    for p in range(port, port + tries):
        try:
            return ReuseServer((bind, p), Handler), p
        except OSError as e:
            if e.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
            first_err = first_err or e
            if p == port:
                who = port_holder(p)
                print(f"port {p} busy{f' ({who})' if who else ''} — trying "
                      f"{p + 1}–{port + tries - 1}", file=sys.stderr)
    srv = ReuseServer((bind, 0), Handler)          # kernel picks a free port
    return srv, srv.server_address[1]


def port_holder(port):
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True,
                             timeout=2).stdout
        for line in out.splitlines():
            if f":{port} " in line and "users:" in line:
                return line.split("users:")[1].strip().strip('()')
    except Exception:
        pass
    return None


def local_urls(port):
    urls = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope",
                              "global"], capture_output=True, text=True,
                             timeout=2).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) > 3:
                urls.append(f"http://{parts[3].split('/')[0]}:{port}/")
    except Exception:
        pass
    return urls or [f"http://localhost:{port}/"]


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
    ap.add_argument("--replace", action="store_true",
                    help="stop an already-running instance and take over")
    ap.add_argument("--fast-dma", action="store_true",
                    help="use the resident DMA helper (native/xdma_shm_reader)"
                         " instead of spawning dma_from_device per frame."
                         " Validate on hardware with validate_fast_dma.py"
                         " before trusting it")
    ap.add_argument("--mock", action="store_true",
                    help="no hardware: synthetic 25 MHz tone, never opens"
                         " /dev/*. For UI/server development")
    a = ap.parse_args()

    lock = None
    if a.mock:
        print("MOCK MODE - synthetic data, hardware untouched")
    else:
        lock, holder = acquire_single_instance(a.replace)
    if lock is None and not a.mock:
        pid, _, oport = holder.partition(" ")
        print(f"\nAlready running (pid {pid})"
              + (f" on port {oport}" if oport else "") + ".", file=sys.stderr)
        if oport.isdigit():
            for u in local_urls(int(oport)):
                print(f"  {u}", file=sys.stderr)
        print("\nThat instance already owns the FPGA. Use it, or restart with:"
              "\n  python3 server.py --replace\n", file=sys.stderr)
        return 1

    A._xfer_reap_stale()      # clear tmpfs files stranded by earlier crashes
    eng = Engine(nsamples=a.nsamples, channel=a.channel, nfft=a.nfft,
                 min_period=a.min_period, fast_dma=a.fast_dma, mock=a.mock)
    eng.start()
    Handler.engine = eng
    srv, port = bind_server(a.bind, a.port)
    if lock:
        lock.write(f"{os.getpid()} {port}\n"); lock.flush()

    # systemd's `stop` sends SIGTERM, whose default action kills the process
    # without unwinding -- so the finally block below (release ADC/GPU, drop
    # the lock, delete the tmpfs transfer file) never ran. Turn it into the
    # same clean unwind as Ctrl-C.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(
        KeyboardInterrupt()))

    print(f"serving on port {port}  (ctrl-C to stop)")
    for u in local_urls(port):
        print(f"  {u}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        eng.stop()
        if lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
                os.unlink(LOCK_PATH)
            except OSError:
                pass


if __name__ == "__main__":
    main()
