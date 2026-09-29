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
import stream as ST
import diag as DG
import gpu
import noise
import phasenoise as PN

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


def lin_display(shown, bin_hz, out_n):
    """Max-pool the spectrum onto out_n points spaced uniformly in FREQUENCY,
    from the first bin to Nyquist -- the linear counterpart to log_display().

    Same max-pooling, so a narrow tone still survives the decimation; only the
    spacing differs. The client maps display point k linearly onto its axis
    either way, so the two only agree if the axis type matches the spacing --
    which is why the choice is reported back in the frame metadata rather than
    assumed."""
    nb = len(shown)
    if nb < 4:
        d = shown[1:].astype(np.float32, copy=True)
        return d, bin_hz, max(1, nb - 1) * bin_hz
    body = shown[1:]
    starts = np.linspace(0, len(body), out_n + 1).astype(np.int64)[:-1]
    d = np.maximum.reduceat(body, starts).astype(np.float32)
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


class DiagSession:
    """Owns the board while the diagnostics tab is open.

    Three things have to be true at once and none of them survive being left
    to the client:

    1. Acquisition is STOPPED. Diagnostics change ChannelSel, DataNum and the
       ADC test pattern; a capture loop running underneath would fight them.
    2. Only ONE test runs at a time, so two sweeps cannot both be driving
       reg13.
    3. The ADC is ALWAYS restored -- normal output plus the stored data delay
       -- when a test ends, fails, is cancelled, or the client disappears.

    (3) is why this is a LEASE rather than a flag. The transport is polling,
    not a websocket, so there is no disconnect to catch: the client renews
    every ~2 s and the watchdog releases the board if renewals stop. Closing
    the tab, navigating away and pulling the network cable all look the same,
    which is what we want.
    """

    # Browsers throttle setInterval in hidden/background tabs (often to once a
    # minute), so a short lease WILL lapse while the tab is still open. That is
    # fine -- the board is released and the ADC restored, which is the point --
    # but the client must be able to take it straight back. keepalive()
    # re-acquires rather than failing, so a lapse costs a stopped acquisition,
    # not a stuck tab.
    LEASE_S = 15.0         # tolerate a few missed 2 s keepalives
    WATCH_S = 1.0

    def __init__(self, engine):
        self.e = engine
        self.lock = threading.Lock()
        self.owner = None              # opaque client token
        self.expires = 0.0
        self.prev_running = False
        self.test = None               # name of the running test
        self.frac = 0.0
        self.msg = ""
        self.points = []               # streamed progress points for live plots
        self.result = None             # DiagResult
        self.error = None
        self._cancel = False
        self._th = None
        self._watch = threading.Thread(target=self._watchdog, daemon=True)
        self._watch.start()

    # ---------------------------------------------------------- lease
    @property
    def active(self):
        return self.owner is not None and time.monotonic() < self.expires

    def enter(self, token):
        with self.lock:
            if self.active and self.owner != token:
                raise RuntimeError("diagnostics already in use by another client")
            first = not self.active
            self.owner = token
            self.expires = time.monotonic() + self.LEASE_S
            if first:
                self.prev_running = self.e.running
                self._stop_acquisition()
        return self.state()

    def keepalive(self, token):
        with self.lock:
            if self.owner == token:
                self.expires = time.monotonic() + self.LEASE_S
                return self.state()
            if self.owner is not None and time.monotonic() < self.expires:
                raise RuntimeError("another client holds the diagnostics board")
            # Free, or our own lapsed lease: take it back. Re-entering redoes
            # the stop-acquisition sequence, which is exactly what is needed
            # after a lapse handed the board back to the capture loop.
            first = not self.active
            self.owner = token
            self.expires = time.monotonic() + self.LEASE_S
            if first:
                self.prev_running = self.e.running
                self._stop_acquisition()
            return self.state()

    def leave(self, token=None):
        with self.lock:
            if self.owner is None:
                return self.state()
            if token is not None and token != self.owner:
                raise RuntimeError("not the diagnostics owner")
            self._release()
        return self.state()

    def _release(self):
        """Caller holds self.lock."""
        self._cancel = True
        th = self._th
        if th is not None and th.is_alive():
            th.join(timeout=15.0)
        try:
            if not self.e.mock:
                DG.restore_adc(self.e.adc)
        except Exception as ex:
            self.error = f"restore failed: {type(ex).__name__}: {ex}"
        self.owner = None
        self.expires = 0.0
        self.test = None
        self.e.running = self.prev_running
        self.e._dirty.set()

    def _watchdog(self):
        while True:
            time.sleep(self.WATCH_S)
            with self.lock:
                if self.owner is not None and time.monotonic() >= self.expires:
                    # client vanished: same path as an explicit leave
                    self._release()

    def _stop_acquisition(self):
        """reg0 = 0 and wait for reg5 to settle, per interface section 6.5."""
        self.e.running = False
        time.sleep(0.3)
        if self.e.mock:
            return
        try:
            self.e._teardown_stream()
            self.e.adc.wr(A.REG_START, 0)
            last = -1
            for _ in range(12):            # up to ~1.2 s; a lap is <= 65.5 ms
                time.sleep(0.1)
                now = self.e.adc.rd(A.REG_SEGCNT)
                if now == last:
                    break
                last = now
        except Exception:
            pass

    # ---------------------------------------------------------- running tests
    def start(self, token, name, params):
        fn = DG.ALL_TESTS.get(name)
        if fn is None:
            raise ValueError(f"unknown test {name!r}")
        with self.lock:
            if self.owner != token:
                raise RuntimeError(
                    "diagnostics lease lapsed or held by another client"
                    " -- re-enter the tab")
            if self.test is not None:
                raise RuntimeError(f"{self.test} is already running")
            self.expires = time.monotonic() + self.LEASE_S
            self.test, self.frac, self.msg = name, 0.0, "starting"
            self.points, self.result, self.error = [], None, None
            self._cancel = False

        def progress(frac, msg, **extra):
            self.frac, self.msg = frac, msg
            p = extra.get("point")
            if p is not None and len(self.points) < 4096:
                self.points.append(p)

        def cancelled():
            return self._cancel

        def work():
            try:
                r = fn(self.e.adc, progress=progress, cancel=cancelled, **params)
                self.result = r
            except DG.Cancelled:
                self.error = "cancelled"
            except Exception as ex:
                self.error = f"{type(ex).__name__}: {ex}"
            finally:
                # Whatever happened, the converter goes back to normal output
                # and the stored tap before anything else can use the board.
                try:
                    if not self.e.mock:
                        DG.restore_adc(self.e.adc)
                except Exception:
                    pass
                self.test = None
                self.frac = 1.0
                self.e._dirty.set()

        self._th = threading.Thread(target=work, daemon=True)
        self._th.start()
        return self.state()

    def cancel(self, token=None):
        if token is not None and token != self.owner:
            raise RuntimeError("not the diagnostics owner")
        self._cancel = True
        return self.state()

    def state(self):
        r = self.result
        return {"active": self.active, "owner": bool(self.owner),
                "test": self.test, "frac": self.frac, "msg": self.msg,
                "error": self.error, "npoints": len(self.points),
                "lease_s": max(0.0, self.expires - time.monotonic())
                           if self.owner else 0.0,
                "result": r.to_json() if r is not None else None}


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

    Samples are NOT decimated by Speed_Set, matching the HDL: that divider
    gates only the capture FSM's sample counter, never the FIFO write enable,
    so the data is always full rate (NOTES.md #26)."""
    path = "mock_synth"
    F0 = 25e6

    def __init__(self, adc=None):
        self._adc = adc
        self._phase = 0
        self._rng = np.random.default_rng(0)

    def read_into(self, out, nsamples, addr=0):
        t = self._phase + np.arange(nsamples, dtype=np.float64)
        self._phase += nsamples
        sig = (8192.0 + 1638.0 * np.sin(2 * np.pi * self.F0 / A.BASE_CLOCK_HZ * t)
               + self._rng.normal(0, 6, nsamples))
        out[:nsamples] = np.clip(sig, 0, 16383).astype(np.uint16)
        return nsamples

    def close(self):
        pass


class MockInterferometer:
    """Synthetic 3x3-coupler Michelson on channel 3, for the phase-noise tab.

    Emits the interleaved dual-channel format the FPGA produces (word 2i =
    A, 2i+1 = B, both 14-bit two's complement in a uint16) carrying two
    photocurrents from a laser with a KNOWN Lorentzian linewidth: phi is a
    random walk of step variance 2 pi dnu / fs, and dphi = phi(t) - phi(t-tau)
    with the delay interpolated, since tau is 24.48 samples at 250 MS/s and
    rounding it would move the sin^2 nulls the analysis has to undo.

    The operating point is also swept slowly across whole fringes. That is
    not decoration: with two photodiodes the ellipse calibration is only
    determined once the fringe has been traversed, so a mock that sat at one
    operating point would exercise the refusal path and nothing else. Real
    interferometers drift like this on their own; `--mock-drift 0` reproduces
    a stuck one on purpose.
    """
    path = "mock_interferometer"

    def __init__(self, adc=None, dnu_hz=50e3, length_m=10.0, psi_deg=118.0,
                 drift_hz=40.0, seed=0):
        self._adc = adc
        self.tau = PN.tau_from_length(length_m)
        self.dnu = dnu_hz
        self.psi = np.radians(psi_deg)
        self.drift_hz = drift_hz
        self._rng = np.random.default_rng(seed)
        ts = self.tau * A.BASE_CLOCK_HZ
        self._k = int(np.floor(ts))
        self._a = ts - self._k
        self._tail = np.zeros(self._k + 2)       # walk continuity across frames
        self._phi0 = 0.0
        self._n = 0

    def _photocurrents(self, n):
        """n samples of the interferometer's two photocurrents (ia, ib),
        continuing the phase walk from the previous call."""
        k, a = self._k, self._a
        steps = self._rng.normal(0.0, np.sqrt(2 * np.pi * self.dnu /
                                              A.BASE_CLOCK_HZ), n)
        phi = np.concatenate((self._tail, self._phi0 + np.cumsum(steps)))
        self._tail = phi[-(k + 2):].copy()
        self._phi0 = float(phi[-1])
        cur = phi[k + 2:]
        dly = ((1 - a) * phi[2:2 + n] + a * phi[1:1 + n])
        dphi = cur - dly
        t = (self._n + np.arange(n)) / A.BASE_CLOCK_HZ
        self._n += n
        drift = 2 * np.pi * self.drift_hz * t
        noise_a = self._rng.normal(0, 3.0, n)
        noise_b = self._rng.normal(0, 3.0, n)
        ia = -300.0 + 2600.0 * np.cos(dphi + drift) + noise_a
        ib = 450.0 + 2100.0 * np.cos(dphi + drift + self.psi) + noise_b
        return ia, ib

    def read_into(self, out, nsamples, addr=0):
        # `nsamples` means two different things depending on Channel_Set,
        # exactly as it does for the real DMA path (see server.py's _one()):
        # doubled 16-bit WORDS in dual mode, plain samples otherwise. This
        # class used to assume dual mode unconditionally, so selecting
        # Channel 1/2 alone (e.g. to look at one photodiode, or by an
        # accidental UI click) silently fed single-channel code a still-
        # interleaved A/B stream -- wrong data, and reported as "looks
        # different" rather than as the bug it was.
        ch = self._adc.rd(A.REG_CHANNEL) if self._adc is not None else A.CH_BOTH
        if ch == A.CH_BOTH:
            n = nsamples // 2
            ia, ib = self._photocurrents(n)
            w = np.empty(2 * n, np.uint16)
            w[0::2] = np.clip(np.round(ia), -8192, 8191).astype(np.int16).view(np.uint16) & 0x3FFF
            w[1::2] = np.clip(np.round(ib), -8192, 8191).astype(np.int16).view(np.uint16) & 0x3FFF
        elif ch in (A.CH_A, A.CH_B):
            ia, ib = self._photocurrents(nsamples)
            sel = ia if ch == A.CH_A else ib
            w = np.clip(np.round(sel), -8192, 8191).astype(np.int16).view(np.uint16) & 0x3FFF
        else:  # CH_TEST_RAMP: a free-running counter, like the real FPGA's
            base = self._n
            self._n += nsamples
            w = (np.arange(base, base + nsamples, dtype=np.uint32) & 0x3FFF).astype(np.uint16)
        m = min(out.size, w.size)
        out[:m] = w[:m]
        return m

    def close(self):
        pass


# ------------------------------------------------------------ acquisition
class Engine:
    def __init__(self, nsamples=1 << 20, channel=1, speed=0, nfft=8192,
                 max_frames=64, trace_width=1024, avg=4, min_period=0.005,
                 tap=None,
                 fast_dma=False, mock=False, mock_opts=None):
        self.fast_dma = fast_dma
        self.mock = mock                  # False | True | "interferometer"
        self.mock_opts = mock_opts or {}
        self.cfg = dict(nsamples=nsamples, channel=channel, speed=speed,
                        nfft=nfft, max_frames=max_frames, avg=avg,
                        trace_samples=-1,     # -1: trace shows the whole record
                                              # (explicit count still accepted)
                        classify=0,           # 1: run peak/noise classification
                        pfa_exp=6,            # detection sensitivity: Pfa=1e-N
                        readback=-1,   # -1 auto, 0 full N, >0 explicit
                        disp_log=1,    # spectrum x axis: 1 = log, 0 = linear
                        # --- acquisition mode -------------------------------
                        # 0 = block (legacy): arm, wait for Adc_Finish, DMA
                        #     one record. The frame rate is whatever a whole
                        #     capture+readback cycle costs, so it VARIES with
                        #     record length (238 fps at 1 M samples, 1.5 at
                        #     16 M).
                        # 1 = stream: the FPGA fills the 16-segment DDR ring
                        #     continuously and a reader thread drains it into
                        #     a RAM ring. The display then samples the NEWEST
                        #     segment at a fixed cadence, so the frame rate is
                        #     constant and independent of record length -- what
                        #     changes with load is how much data each frame
                        #     skips over, not how often frames arrive.
                        mode=0,
                        fps=20,        # display cadence in stream mode; also a
                                       # cap in block mode. 0 = as fast as able
                        stream_ram_gb=4,
                        stream_channels=2,
                        # Optional bound on the samples the time-trace
                        # envelope spans per frame in stream mode. 0 = off:
                        # `nsamples` means what it says and the frame rate is
                        # whatever that record length allows (still CONSTANT,
                        # just lower). Set it >0 to trade record length for a
                        # higher fixed rate. It defaulted to 2 M briefly and
                        # that was wrong: silently showing 32 ms when 268 ms
                        # was asked for is worse than an honest lower rate.
                        stream_window=0,
                        # 0 = time axis spans the RECORD (nsamples)
                        # 1 = time axis spans the WHOLE RAM RING, drawn from
                        #     per-segment summaries built as segments arrive.
                        #     Lets the axis cover seconds without pulling
                        #     gigabytes through the CPU every frame.
                        trace_span_ring=0)
        self.trace_width = trace_width
        self.tap = tap                    # IDELAY tap to apply at open
        self._ramp_busy = False
        self._ramp_result = None
        self._ctr_result = None
        self._eye_result = None
        self.diag = None                  # DiagSession, created after startup
        self.stream = None                # stream.Stream while mode=1
        self._skey = None                 # (channel, ram_gb, nchan) in use
        self.stream_seg = -1              # newest segment the display used
        self.stream_span = (0, 0)         # (first, last) segments in the frame
        self.stream_err = None
        self.min_period = min_period      # floor on loop period; leaves the
                                          # scheduler room for networking
        self.running = True
        self.lock = threading.Lock()
        self.new_frame = threading.Condition(self.lock)
        self.frame = None
        self.err = None
        self.sysmon = SysMon()
        # SysMon reads /proc/net/wireless (~1.5 ms) every call and shells out
        # to `iw` (~4 ms) every 5 s. Called per frame that is ~10% of a 15 ms
        # frame plus a periodic hiccup, all of it inside the capture loop.
        # Sample it on its own timer instead and hand the loop a cached dict.
        self._sys_cache = self.sysmon.sample()
        self._sys_th = None
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
        self.raw = None                 # recent raw samples, for /raw
        self._pending = None            # de-interleaved channels in dual mode
        self._raw_req = 0.0
        self.dual = None                # per-channel stats when ch_sel=3
        self._ch_fail = 0               # consecutive timeouts on this channel
        self.classify_period = 1.0
        self.pfa = 1e-6

        # ---- phase noise (see phasenoise.py). Its own worker for the same
        # reason classification has one: a full demodulate + Welch over a
        # 67 M-sample dual record is ~1 s of CPU, and the capture loop must
        # not wait for it. Unlike classification it also carries STATE across
        # captures -- the ellipse calibration is accumulated from the drift
        # of the operating point over many records, because one record from a
        # narrow-linewidth laser moves dphi by milliradians and determines
        # nothing on its own.
        self.pn_cfg = dict(
            enabled=0,
            length_m=10.0,           # delay fibre in the long arm
            n_group=PN.GROUP_INDEX,
            double_pass=1,           # Michelson: the fibre is traversed twice
            tau_ns=0.0,              # >0 overrides the geometry above
            decim=0,                 # 0 = auto from tau
            nperseg=0,               # 0 = the whole record in one segment
            window="hann",
            f_max_hz=0.0,            # 0 = 1/(2 tau)
            npts=600,
            predecimate=1,
            cal_mode="auto",         # auto | hold | nominal
            psi_deg=120.0,           # nominal 3x3 hybrid angle
            require_cal=1,
            period_s=0.0,            # min seconds between analyses
        )
        self.pn = None               # latest result, JSON-ready
        self.pn_cal = None           # PN.Cal carried across captures
        self.pn_sid = 0
        self._pn_acc = None          # (N,2) float32 ring of calibration points
        self._pn_acc_n = 0
        self._pn_lock = threading.Lock()
        self._pn_cv = threading.Condition(self._pn_lock)
        self._pn_req = None
        self._pn_last = 0.0
        self._dual_stage = None      # landing buffer for interleaved A/B reads
        self._an_lock = threading.Lock()
        self._an_cv = threading.Condition(self._an_lock)
        self._an_req = None
        self._an_last = 0.0
        self._sid = 0          # structure id: bumped per completed analysis

    # ---- lifecycle
    def start(self):
        if self.mock:
            self.adc = MockAdc()
            self.dma = (MockInterferometer(self.adc, **self.mock_opts)
                        if self.mock == "interferometer"
                        else MockDma(self.adc))
        else:
            # tap=None -> apply the stored IDELAY tap. The FPGA loses it on
            # every reconfiguration, so the server is one of the places that
            # has to re-apply it (see ad9643.load_stored_tap).
            self.adc = A.Adc(tap=self.tap)
            self.dma = DmaReader(self.fast_dma)
        self.sp = gpu.Spectrum(self.cfg["nfft"],
                               max_samples=max(1 << 22, self.cfg["nsamples"]),
                               trace_width=self.trace_width)
        self.diag = DiagSession(self)
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        self.an_th = threading.Thread(target=self._analysis_loop, daemon=True)
        self.an_th.start()
        self._sys_th = threading.Thread(target=self._sys_loop, daemon=True)
        self._sys_th.start()
        self.pn_th = threading.Thread(target=self._pn_loop, daemon=True)
        self.pn_th.start()

    def stop(self):
        self._stop.set()
        if self.diag is not None:
            try:
                self.diag.leave()
            except Exception:
                pass
        # The FPGA ring keeps writing until reg0 is cleared, and the helper
        # subprocesses outlive this process unless they are told to quit.
        self._teardown_stream()
        # One in-flight cycle at max samples/FFT takes ~2.3s (1.05s capture +
        # ~1s DMA + ~0.3s GPU); 3s left too little margin -- a single Ctrl-C
        # could still be waiting on join() when the terminal least expects it,
        # inviting an impatient second Ctrl-C mid-shutdown. 20s comfortably
        # covers worst case plus Spectrum teardown of the largest buffers.
        with self._an_lock:
            self._an_cv.notify_all()
        with self._pn_lock:
            self._pn_cv.notify_all()
        self.th.join(timeout=20)
        if getattr(self, "an_th", None):
            self.an_th.join(timeout=5)
        if getattr(self, "pn_th", None):
            self.pn_th.join(timeout=10)
        self.sp.close(); self.dma.close(); self.adc.close()

    def configure(self, **kw):
        # While the diagnostics tab holds the board, acquisition settings are
        # read-only: a capture starting underneath a tap sweep or a test
        # pattern change would produce data that matches neither.
        if self.diag is not None and self.diag.active:
            if kw.get("running") or kw.get("trigger"):
                raise RuntimeError(
                    "diagnostics active: acquisition is locked. Leave the "
                    "ADC/FPGA tab to release it.")
        trig = bool(kw.pop("trigger", False))
        UNSET = object()
        zoom = kw.pop("zoom", UNSET)
        with self.lock:
            for k, v in kw.items():
                if k == "running":
                    self.running = bool(v)
                elif k == "speed":
                    if int(v) != 0:
                        raise ValueError(
                            "Speed_Set must be 0: it does not decimate (fs is "
                            "always 250 Msps) and a non-zero value corrupts "
                            "block lengths on this bitstream (spec 8.1)")
                    self.cfg[k] = 0
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

    # ---- phase noise
    PN_STR_KEYS = ("window", "cal_mode")
    PN_INT_KEYS = ("enabled", "double_pass", "decim", "nperseg", "npts",
                   "predecimate", "require_cal")
    PN_CAL_POINTS = 1 << 12      # sampled from each capture into the ring
    PN_CAL_CAP = 1 << 19         # ring capacity: ~128 captures of history

    def pn_configure(self, **kw):
        """Phase-noise settings. Separate from configure() because these are
        floats and strings (fibre length, hybrid angle, window name) and the
        capture config is deliberately int-only."""
        with self.lock:
            for k, v in kw.items():
                if k not in self.pn_cfg:
                    raise KeyError(f"unknown phase-noise setting {k!r}")
                if k in self.PN_STR_KEYS:
                    self.pn_cfg[k] = str(v)
                elif k in self.PN_INT_KEYS:
                    self.pn_cfg[k] = int(v)
                else:
                    self.pn_cfg[k] = float(v)
            if self.pn_cfg["window"] not in ("hann", "blackmanharris",
                                             "flattop", "boxcar"):
                self.pn_cfg["window"] = "hann"
            if self.pn_cfg["cal_mode"] not in ("auto", "hold", "nominal"):
                self.pn_cfg["cal_mode"] = "auto"
            if self.pn_cfg["enabled"]:
                # a two-port interferogram needs both photodiodes from the
                # SAME sample clock, which is what ch_sel=3 delivers
                # (NOTES.md #33) -- nothing else can produce this measurement
                self.cfg["channel"] = A.CH_BOTH
        self._dirty.set()

    def pn_recalibrate(self, clear=True):
        """Throw away the accumulated ellipse and start again.

        The right thing after anything that moves the operating point or the
        fringe amplitude: touching the fibre, changing laser power, swapping
        a photodiode. Stale points and fresh ones fitted together describe
        neither.
        """
        with self._pn_lock:
            if clear:
                self._pn_acc_n = 0
            self.pn_cal = None

    def pn_tau(self):
        c = self.pn_cfg
        if c["tau_ns"] > 0:
            return c["tau_ns"] * 1e-9
        return PN.tau_from_length(c["length_m"], c["n_group"],
                                  bool(c["double_pass"]))

    def _pn_push_cal(self, xa, xb):
        """Add a stride-sampled slice of this capture to the calibration ring.

        Strided, not contiguous: what the conic needs is COVERAGE of the
        fringe, and within one 67 ms record the operating point barely moves,
        so a contiguous block and a strided one carry the same information --
        but the strided one keeps a record's worth of amplitude statistics.
        Coverage arrives across captures, as the interferometer drifts.
        """
        step = max(1, xa.size // self.PN_CAL_POINTS)
        pts = np.column_stack((xa[::step][:self.PN_CAL_POINTS],
                               xb[::step][:self.PN_CAL_POINTS])).astype(np.float32)
        with self._pn_lock:
            if self._pn_acc is None:
                self._pn_acc = np.zeros((self.PN_CAL_CAP, 2), np.float32)
            k = pts.shape[0]
            n = self._pn_acc_n
            if n + k <= self.PN_CAL_CAP:
                self._pn_acc[n:n + k] = pts
                self._pn_acc_n = n + k
            else:
                # FIFO: drop the oldest. Old points are not just surplus,
                # they are wrong once the photodiode DC has drifted.
                keep = self.PN_CAL_CAP - k
                self._pn_acc[:keep] = self._pn_acc[self.PN_CAL_CAP - keep:]
                self._pn_acc[keep:] = pts
                self._pn_acc_n = self.PN_CAL_CAP

    def _pn_calibration(self, cfg):
        """The Cal to demodulate this capture with, under the current mode.

        'auto' refits from the ring every time but only ADOPTS a fit that
        passes its own checks -- a momentarily worse fit must not evict a
        good calibration, or the readout would flicker between right and
        wrong every time the operating point paused.
        """
        if cfg["cal_mode"] == "nominal":
            return None                       # analyse() builds it per record
        if cfg["cal_mode"] == "hold":
            return self.pn_cal
        with self._pn_lock:
            n = self._pn_acc_n
            pts = None if n < 4096 else self._pn_acc[:n].copy()
        if pts is None:
            return self.pn_cal
        cal = PN.fit_ellipse(pts[:, 0], pts[:, 1])
        if cal is not None and (cal.trustworthy or self.pn_cal is None):
            self.pn_cal = cal
        return self.pn_cal

    def _pn_loop(self):
        while not self._stop.is_set():
            with self._pn_lock:
                while self._pn_req is None and not self._stop.is_set():
                    self._pn_cv.wait(timeout=0.5)
                req = self._pn_req
            if req is None:
                continue
            try:
                xa, xb, fs, cfg = req
                # Sign-extend ONCE, here, and use the same array for both the
                # calibration ring and the analysis. Feeding the ring raw
                # uint16 fits an ellipse in a coordinate system that WRAPS at
                # code 0 (NOTES.md #27): every negative excursion jumps to
                # ~16383, the Lissajous shatters into fragments, and the fit
                # comes back with amplitudes of 10000 codes on a 2600-code
                # fringe -- confidently, since fragments still admit a conic.
                xa = A.adc_signed(xa).astype(np.int16)
                xb = A.adc_signed(xb).astype(np.int16)
                self._pn_push_cal(xa, xb)
                cal = self._pn_calibration(cfg)
                r = PN.analyse(
                    xa, xb, fs, self.pn_tau(),
                    cal=cal, cal_mode=cfg["cal_mode"], psi_deg=cfg["psi_deg"],
                    decim=cfg["decim"], nperseg=cfg["nperseg"],
                    window=cfg["window"], f_max=cfg["f_max_hz"],
                    npts=cfg["npts"], predecimate=bool(cfg["predecimate"]),
                    require_cal=bool(cfg["require_cal"]))
                self.pn_sid += 1
                r["sid"] = self.pn_sid
                r["ts"] = time.time()
                r["volt_scale"] = A.VOLT_SCALE
                r["cal_points"] = self._pn_acc_n
                self.pn = r
            except Exception as e:
                self.pn = {"error": f"{type(e).__name__}: {e}",
                           "sid": self.pn_sid, "ts": time.time()}
            finally:
                with self._pn_lock:
                    self._pn_req = None

    def _pn_submit(self, chans, fs, cfg):
        """Hand a dual-channel capture to the worker, if it is idle.

        Idle-only, like the classifier: the analysis is slower than the
        capture loop, and queueing would build an ever-growing backlog of
        stale records. Skipping is the correct behaviour -- every record is
        an equally valid sample of the same stationary process.
        """
        now = time.monotonic()
        if now - self._pn_last < cfg["period_s"]:
            return
        with self._pn_lock:
            if self._pn_req is not None:
                return
            self._pn_req = (chans["A"].copy(), chans["B"].copy(), fs, dict(cfg))
            self._pn_last = now
            self._pn_cv.notify()

    def _dual_words(self, nwords):
        """Landing buffer for an interleaved A/B read.

        NOT Spectrum.stage: that is sized for max_samples SAMPLES, but a dual
        capture is two 16-bit words per sample clock. Reading N samples of
        dual data into it silently truncated to half the record at any
        N > max_samples/2 -- ddr_read_into clips to len(out) and reports the
        short count, so the spectrum was of half a record and nothing said
        so. Phase noise made that visible: a record that stops halfway is a
        step discontinuity in dphi.
        """
        if self._dual_stage is None or self._dual_stage.size < nwords:
            self._dual_stage = np.empty(nwords, np.uint16)
        return self._dual_stage

    # ------------------------------------------------------ hardware report
    # reg -> (name, decoder). The decoders exist so the tab shows what a bit
    # MEANS, not just its value: a raw 0x00000005 in reg10 is the difference
    # between a healthy front end and a dead one, and nobody should have to
    # remember which bit is which to see that.
    REG_INFO = [
        (A.REG_START,     "reg0  start/stream",
         lambda v: ("stopped" if not (v & 3) else
                    ("streaming" if v & 2 else "block") +
                    (", started" if v & 1 else ", idle"))),
        (A.REG_SPEED,     "reg1  Speed_Set",
         lambda v: "0 (required)" if v == 0 else f"{v} -- MUST BE 0"),
        (A.REG_CHANNEL,   "reg2  ChannelSel",
         lambda v: {0: "0 = FPGA counter", 1: "1 = ADC A", 2: "2 = ADC B",
                    3: "3 = A+B"}.get(v & 3, str(v))),
        (A.REG_NSAMPLES,  "reg3  DataNum",     lambda v: f"{v:,} sample clocks"),
        (A.REG_FINISH,    "reg4  Adc_Finish",
         lambda v: "complete" if v & 1 else "busy"),
        (A.REG_SEGCNT,    "reg5  seg_count",   lambda v: f"{v:,} segments"),
        (A.REG_FLAGS,     "reg6  stream flags",
         lambda v: (", ".join(x for x, c in
                              (("overrun", v & 1), ("fifo_overflow", v & 2)) if c)
                    or "clear")),
        (A.REG_SEGACK,    "reg7  host_seg_ack", lambda v: f"{v:,}"),
        (A.REG_SPI_CMD,   "reg8  SPI command",
         lambda v: f"{'read' if v >> 31 else 'write'} "
                   f"addr 0x{(v >> 8) & 0x1FFF:02X} data 0x{v & 0xFF:02X}"),
        (A.REG_SPI_STATUS, "reg9  SPI status",
         lambda v: ("busy" if v & 1 else "idle") + f", last read 0x{(v >> 8) & 0xFF:02X}"),
        (A.REG_ADC_STATUS, "reg10 ADC status",
         lambda v: ", ".join(
             [("clk locked" if v & 1 else "CLK NOT LOCKED"),
              ("IDELAY ready" if v & 8 else "IDELAYCTRL NOT READY")]
             + (["overrange A"] if v & 2 else []) + (["overrange B"] if v & 4 else []))),
        (A.REG_RAMP_ERR_A, "reg11 ramp errors A",
         lambda v: "saturated" if v == 0xFFFFFFFF else f"{v:,}"),
        (A.REG_RAMP_ERR_B, "reg12 ramp errors B",
         lambda v: "saturated" if v == 0xFFFFFFFF else f"{v:,}"),
        (A.REG_DELAY,      "reg13 data delay",
         lambda v: f"requested {v & 0x1FF}, read back {(v >> 16) & 0x1FF}"),
        (A.REG_RSVD14,     "reg14 reserved",    lambda v: "0" if v == 0 else f"0x{v:08X}"),
        (A.REG_DESIGN_ID,  "reg15 design ID",
         lambda v: ("0x%08X" % v) + (" (match)" if v == A.DESIGN_ID
                                     else f" -- expected 0x{A.DESIGN_ID:08X}")),
    ]

    _LINK = "/sys/bus/pci/devices/0005:01:00.0"

    def hw_link(self):
        """PCIe link state from sysfs. Cheap, and it is the first thing that
        goes wrong after an FPGA reconfiguration: the BARs get wiped and every
        register reads 0xFFFFFFFF."""
        out = {}
        for k, f in (("speed", "current_link_speed"), ("width", "current_link_width"),
                     ("max_speed", "max_link_speed"), ("max_width", "max_link_width"),
                     ("enabled", "enable")):
            try:
                out[k] = open(os.path.join(self._LINK, f)).read().strip()
            except OSError:
                out[k] = None
        try:
            out["driver"] = os.path.basename(
                os.readlink(os.path.join(self._LINK, "driver")))
        except OSError:
            out["driver"] = None
        return out

    def hw_snapshot(self, dump=False):
        if self.mock:
            return {"mock": True, "regs": [], "adc": self.adc_health(),
                    "link": {}, "dma_path": "mock"}
        regs = []
        for off, name, dec in self.REG_INFO:
            v = self.adc.rd(off)
            try:
                d = dec(v)
            except Exception:
                d = ""
            regs.append({"offset": off, "name": name, "value": int(v), "decode": d})
        out = {"mock": False, "regs": regs, "adc": self.adc_health(),
               "link": self.hw_link(),
               "dma_path": getattr(self.dma, "path", "?"),
               "seg_bytes": A.SEG_BYTES, "nseg": A.NSEG,
               "alias_bytes": A.REG_ALIAS_BYTES,
               "counter_test": self._ctr_result,
               "eye_scan": self._eye_result}
        if dump:
            try:
                out["adc_registers"] = self.adc_dump()
            except Exception as e:
                out["adc_dump_error"] = f"{type(e).__name__}: {e}"
        return out

    # ------------------------------------------------------- hardware tests
    def _pause_for(self, fn, slot):
        """Run `fn` with acquisition stopped, restoring it afterwards."""
        def work():
            was = self.running
            try:
                self.running = False
                time.sleep(0.3)
                with self.lock:
                    r = fn()
                r["ts"] = time.time(); r["running"] = False
                setattr(self, slot, r)
            except Exception as e:
                setattr(self, slot, {"running": False, "ts": time.time(),
                                     "error": f"{type(e).__name__}: {e}"})
            finally:
                self.running = was
                self._dirty.set()
        setattr(self, slot, {"running": True})
        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    def counter_test(self, nsamples=1048576):
        """Spec 7.1 from the UI: the FPGA counter must be a clean ramp. This
        exercises FIFO, DDR writer, XDMA and the host decode WITHOUT the ADC,
        so it separates 'the board is broken' from 'the ADC path is broken'."""
        if self.mock:
            raise RuntimeError("no hardware in mock mode")
        n = int(nsamples)
        if not (256 <= n <= (1 << 24)) or n % 256:
            raise ValueError("nsamples must be a multiple of 256, 256..16777216")

        def run():
            d = self.adc.capture(n, channel=A.CH_TEST_RAMP)
            a = (d & 0x3FFF).astype(np.uint16)
            diff = np.empty(a.size - 1, np.uint16)
            np.subtract(a[1:], a[:-1], out=diff)
            np.bitwise_and(diff, 0x3FFF, out=diff)
            bad = int(np.count_nonzero(diff != 1))
            return {"samples": int(a.size), "violations": bad,
                    "high_bits": bool((d & 0xC000).any()),
                    "first": a[:8].tolist(), "pass": bad == 0}
        return self._pause_for(run, "_ctr_result")

    def eye_scan(self, step=8, dwell=0.01, nsamp=65536):
        """Spec 9.4 sweep, run in the background with the ADC on its ramp."""
        if self.mock:
            raise RuntimeError("no hardware in mock mode")
        self.adc.require_design_id("the eye scan")
        step = max(1, min(64, int(step)))
        nsamp = max(4096, min(1 << 20, int(nsamp)))
        nsamp -= nsamp % A.SAMPLE_GRANULARITY_DUAL

        def run():
            tap0 = self.adc.get_tap()["requested"]
            rows = []
            try:
                with self.adc.ramp_mode():
                    for tap in range(0, A.DELAY_TAPS, step):
                        self.adc.wr(A.REG_DELAY, tap)
                        time.sleep(50e-6)
                        # reg11/reg12 since the 2026-09-29 bitstream: no
                        # capture per tap, so the sweep is faster and the FPGA
                        # sees every sample rather than a block. The host
                        # method is still reachable from the diagnostics tab
                        # for comparison.
                        self.adc.ramp_clear()
                        time.sleep(dwell)
                        ea, eb = self.adc.ramp_errors()
                        rows.append([tap, int(ea), int(eb)])
            finally:
                self.adc.wr(A.REG_DELAY, tap0)
            best = cur = None
            for tap, ea, eb in rows:
                if ea == 0 and eb == 0:
                    cur = (tap, tap) if cur is None else (cur[0], tap)
                    if best is None or (cur[1] - cur[0]) > (best[1] - best[0]):
                        best = cur
                else:
                    cur = None
            res = {"step": step, "samples": nsamp, "rows": rows, "tap0": tap0,
                   "ps_per_tap": A.PS_PER_TAP, "ui_taps": A.UI_TAPS}
            if best:
                res.update({"found": True, "lo": best[0], "hi": best[1],
                            "width": best[1] - best[0] + step,
                            "centre": (best[0] + best[1]) // 2,
                            "ps": (best[1] - best[0] + step) * A.PS_PER_TAP,
                            "edge": best[0] <= 0 or best[1] >= A.DELAY_TAPS - step})
            else:
                res["found"] = False
            return res
        return self._pause_for(run, "_eye_result")

    # ----------------------------------------------------------- ADC health
    def adc_health(self):
        """Cheap per-frame ADC/front-end health (design ID 0xAD964302).

        Three MMIO reads. Worth doing every frame because these are exactly
        the bits that say "the numbers on screen are meaningless": on
        2026-09-29 the ADC LVDS capture was dead (every sample constant) with
        reg10 bit3 IDELAYCTRL-ready low, and nothing in the UI said so -- the
        spectrum just looked like a very quiet input.
        """
        if self.mock:
            return {"present": False, "reason": "mock"}
        try:
            if not self.adc.has_adc_ctl:
                return {"present": False,
                        "design_id": self.adc.design_id,
                        "reason": f"design ID 0x{self.adc.design_id:08X} "
                                  f"(needs 0x{A.DESIGN_ID:08X})"}
            st = self.adc.adc_status()
            tp = self.adc.get_tap()
            return {"present": True,
                    "design_id": self.adc.design_id,
                    "clk_locked": st["clk_locked"],
                    "overrange_a": st["overrange_a"],
                    "overrange_b": st["overrange_b"],
                    "idelay_ready": st["idelay_ready"],
                    "tap_requested": tp["requested"],
                    "tap_readback": tp["readback"],
                    "applied_tap": self.adc.applied_tap,
                    "output_mode": self.adc.output_mode,
                    "output_invert": self.adc.output_invert,
                    "ramp_test": self._ramp_result}
        except Exception as e:
            return {"present": False, "reason": f"{type(e).__name__}: {e}"}

    def adc_dump(self, lo=0x00, hi=0x3A):
        """Full ADC register dump. On demand only: ~3.3 us per SPI transfer."""
        if self.mock:
            raise RuntimeError("no ADC in mock mode")
        self.adc.require_design_id("the ADC register dump")
        with self.lock:
            return {f"0x{a:02X}": self.adc.adc_rd(a) for a in range(lo, hi + 1)}

    def adc_set_tap(self, tap, save=False):
        if self.mock:
            raise RuntimeError("no ADC in mock mode")
        self.adc.require_design_id("the ADC data delay")
        with self.lock:
            r = self.adc.set_tap(int(tap))
        if save:
            A.save_stored_tap(int(tap))
        return r

    def adc_clear_flags(self):
        if self.mock:
            raise RuntimeError("no ADC in mock mode")
        self.adc.require_design_id("the ADC status flags")
        self.adc.ramp_clear()
        return self.adc.adc_status()

    def ramp_test(self, seconds=5.0):
        """Run a ramp check in the background, pausing acquisition.

        The ADC has to emit its ramp for this, so any capture running at the
        same time would be recording a sawtooth. Acquisition is therefore
        stopped for the duration and restored afterwards -- including if the
        test raises.
        """
        if self.mock:
            raise RuntimeError("no ADC in mock mode")
        self.adc.require_design_id("the ramp checker")
        if self._ramp_busy:
            raise RuntimeError("a ramp test is already running")

        def work():
            was_running = self.running
            self._ramp_busy = True
            self._ramp_result = {"running": True, "seconds": seconds}
            try:
                self.running = False
                time.sleep(0.3)                 # let the loop finish a frame
                with self.lock:
                    with self.adc.ramp_mode():
                        # reg11/reg12 ARE the verdict as of the 2026-09-29
                        # bitstream: the checker now tests the second
                        # difference, which this converter's ramp satisfies.
                        # The host-side count is kept alongside as a cross
                        # check -- when the two disagreed before, only the
                        # host one was right, and knowing that immediately is
                        # worth one extra pass over data already in RAM.
                        self.adc.ramp_clear()
                        t0 = time.monotonic()
                        a = b = 0
                        n = 0
                        while time.monotonic() - t0 < seconds:
                            d = self.adc.capture(1 << 18, channel=A.CH_BOTH)
                            a += A.ramp_deviations(d[0::2])["errors"]
                            b += A.ramp_deviations(d[1::2])["errors"]
                            n += (1 << 18)
                        fa, fb = self.adc.ramp_errors()
                        st = self.adc.adc_status()
                self._ramp_result = {
                    "running": False, "seconds": seconds,
                    "errors_a": int(fa), "errors_b": int(fb),
                    "host_errors_a": int(a), "host_errors_b": int(b),
                    "samples": n,
                    "pass": (fa == 0 and fb == 0),
                    "agree": (fa == 0) == (a == 0) and (fb == 0) == (b == 0),
                    "status": st, "ts": time.time()}
            except Exception as e:
                self._ramp_result = {"running": False,
                                     "error": f"{type(e).__name__}: {e}",
                                     "ts": time.time()}
            finally:
                self._ramp_busy = False
                self.running = was_running
                self._dirty.set()

        threading.Thread(target=work, daemon=True).start()
        return {"started": True, "seconds": seconds}

    # ------------------------------------------------------------ streaming
    def _stream_key(self, cfg):
        return (cfg["channel"], float(cfg["stream_ram_gb"]),
                int(cfg["stream_channels"]))

    # ---------------------------------------------------- block acquisition
    def _arm_block(self, cfg, N, ch):
        """Legacy one-shot arm sequence (spec section 5)."""
        self.adc.wr(A.REG_START, 0)          # stopped + block mode (bit1 low)
        self.adc.wr(A.REG_SPEED, 0)          # spec 8.1: must be 0
        self.adc.wr(A.REG_CHANNEL, ch)       # spec 8.2: only while stopped
        # DataNum counts sample CLOCKS in every mode -- measured 2026-09-28,
        # see ad9643.DUAL_DATANUM_IN_WORDS. The old `N * 2` here made every
        # dual capture run twice as long as asked and, past 65,536,000 pairs,
        # wrap the 500 MB window onto its own record: the phase-noise size of
        # 67,108,864 pairs overran by 12,582,912 B, so the record read back
        # from address 0 was spliced 3,145,728 pairs in.
        self.adc.wr(A.REG_NSAMPLES, A.datanum_for(N, ch))
        self.adc.wr(A.REG_START, A.START_BIT)
        # reg4 idles HIGH and keeps the previous capture's state for well
        # under a microsecond after the start edge, so polling immediately
        # reads a stale "finished" (spec section 5).
        time.sleep(10e-6)

    def _wait_block(self, ch, N, expect):
        """Poll Adc_Finish. Returns (elapsed, suspect).

        Adc_Finish NOW ASSERTS IN DUAL MODE. On the previous bitstream it
        never did at any depth (NOTES.md #30) and this blind-waited expect*4
        for ch_sel=3 -- four times longer than the capture, which is why
        dual-channel frame rates were a third of single-channel. Verified on
        the 2026-09-28 streaming build: ch_sel=3 at 1,048,576 and 67,108,864
        samples both asserted, elapsed/expected = 1.000.
        """
        timeout = max(0.5, expect * 4 + 0.5)
        t0 = time.monotonic()
        while not self.adc.finished:
            if time.monotonic() - t0 > timeout:
                st = self.adc.regs(); self.adc.recover()
                raise A.CaptureTimeout(
                    f"Adc_Finish low after {timeout:.3f}s regs={st}")
            time.sleep(1e-4)      # never busy-spin on MMIO reads
        t_cap = time.monotonic() - t0
        if ch == A.CH_BOTH:
            # Adc_Finish can lead the last few bursts (<= 8 x 512 B) into DDR
            # by a few microseconds (spec section 5, known FPGA issue).
            time.sleep(20e-6)
        # Physics check: the FSM cannot digitise N samples faster than N/fs,
        # so a completion far short of that means Adc_Finish was still high
        # from the previous run and we are about to read a buffer that was
        # never filled. Flag it rather than publishing a plausible-looking
        # spectrum built from stale DDR.
        return t_cap, t_cap < 0.5 * expect

    def _stream_meta(self, streaming):
        """Per-frame stream telemetry. `mode` is what the UI switches on; the
        rest is what tells the operator whether the display is keeping up --
        `lost` and the reg6 flags are the ones that matter, because in stream
        mode a display that falls behind shows OLDER data rather than fewer
        frames, which is otherwise invisible."""
        st = self.stream
        if not streaming or st is None:
            return {"mode": "block", "err": self.stream_err}
        ring = st.ring
        per_sample = 4 if st.channel == A.CH_BOTH else 2
        seg_s = A.SEG_BYTES / (A.BASE_CLOCK_HZ * per_sample)
        lo, hi = (ring.resident if ring else (0, 0))
        return {
            "mode": "stream",
            "err": self.stream_err,
            "segments": st.n_read,
            "lost": st.n_lost,
            "first_overrun_at": st.first_overrun_at,
            "flags": int(st.flags_seen),
            "overrun": bool(st.flags_seen & A.FLAG_OVERRUN),
            "fifo_overflow": bool(st.flags_seen & A.FLAG_FIFO_OVF),
            "gb_read": st.bytes_read / 1e9,
            "rate_gbps": (st.bytes_read / (time.monotonic() - st.t_start) / 1e9)
                         if st.t_start else 0.0,
            "needed_gbps": A.SEG_BYTES / seg_s / 1e9,
            "seg_period_ms": seg_s * 1e3,
            "ring_segments": ring.nslots if ring else 0,
            "ring_gb": (ring.nslots * A.SEG_BYTES / 1e9) if ring else 0.0,
            "ring_seconds": (ring.nslots * seg_s) if ring else 0.0,
            "resident": [lo, hi],
            "evicted": ring.evicted if ring else 0,
            "shown_segment": self.stream_seg,
            "shown_span": list(self.stream_span),
            "shown_segments": max(0, self.stream_span[1] - self.stream_span[0] + 1),
            "age_ms": ((hi - self.stream_seg) * seg_s * 1e3)
                      if self.stream_seg >= 0 else 0.0,
            "frames_late": getattr(self, "frames_late", 0),
        }

    def _maybe_leave_stream(self, cfg):
        """Block capture and streaming cannot share the FPGA: stream mode
        leaves reg0 bit1 set and the writer free-running, and arming a block
        capture on top of that would race the ring writer."""
        if not cfg.get("mode", 0) and self.stream is not None:
            self._teardown_stream()

    def _ensure_stream(self, cfg):
        """Bring the background stream up, or reconfigure it if the channel /
        ring size / C2H fan-out changed. ChannelSel may only be written while
        stopped (spec 8.2), so a channel change is a full restart."""
        key = self._stream_key(cfg)
        if self.stream is not None and self.stream.alive and self._skey == key:
            return self.stream
        self._teardown_stream()
        nslots = max(3, int(cfg["stream_ram_gb"] * 1000**3) // A.SEG_BYTES)
        st = ST.Stream(self.adc, cfg["channel"], nchan=int(cfg["stream_channels"]),
                       nslots=nslots, ram_ring=True, envelope=True)
        st.start_background()
        self.stream, self._skey, self.stream_err = st, key, None
        return st

    def _teardown_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop_background()
            except Exception as e:
                self.stream_err = f"{type(e).__name__}: {e}"
            finally:
                try:
                    self.stream.close()
                except Exception:
                    pass
        self.stream, self._skey = None, None

    def _acquire_stream(self, cfg, ch, want_samples):
        """Fill the staging buffers from the newest resident segment.

        Returns (raw_src, n_samples) with exactly the layout the block path
        produces, so everything downstream -- de-interleave, GPU, /raw, phase
        noise -- is untouched.
        """
        st = self._ensure_stream(cfg)
        seg = st.newest()
        if seg is None:
            return None, 0
        self.stream_seg = seg
        # A record may span SEVERAL consecutive ring segments. Capping it at
        # one segment silently limited every record to 32.768 ms (mode 3) or
        # 65.5 ms (single channel) however large `nsamples` was.
        per_seg = A.seg_samples(ch)
        usable = max(1, st.ring.n_resident - 4)   # keep clear of eviction
        n = min(int(want_samples), per_seg * usable)
        if ch == A.CH_BOTH:
            nw = n * 2                            # two uint16 per sample clock
            buf = self._dual_words(nw)
            got, first = st.ring.read_span(seg, nw, buf[:nw])
            self.stream_span = (first, seg)
            return buf[:got.size], got.size // 2
        got, first = st.ring.read_span(seg, n, self.sp.stage[:n])
        self.stream_span = (first, seg)
        return self.sp.stage[:got.size], got.size

    # ---- the loop
    def _loop(self):
        while not self._stop.is_set():
            if self.diag is not None and self.diag.active:
                time.sleep(0.05); continue
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
                # Dual-channel (ch_sel=3) does not complete on this
                # bitstream: Adc_Finish never asserts, at every record
                # length, with or without the vendor's doubled count (see
                # NOTES.md #29). Rather than stall the display half a second
                # per frame forever, fall back and say why.
                time.sleep(0.2)
            except Exception as e:
                self.err = f"{type(e).__name__}: {e}"
                time.sleep(0.3)
            if self.running:               # no artificial delay after a single shot
                # CONSTANT FRAME RATE.
                # Block mode is capture-bound: a frame costs a whole
                # arm+wait+DMA cycle, so the rate falls with record length
                # (238 fps at 1 M samples, 1.5 fps at 16 M) and `fps` can only
                # act as a ceiling.
                # Stream mode is not: the FPGA fills the ring continuously and
                # a frame only COPIES the newest segment, so the display can
                # hold a fixed period regardless of record length. Pacing off
                # an absolute schedule rather than sleeping a fixed slack
                # keeps it from drifting when one frame runs long -- the next
                # frame simply takes newer data, which is the whole point:
                # what varies under load is how much data each frame skips,
                # not how often frames arrive.
                fps = int(cfg.get("fps", 0) or 0)
                period = (1.0 / fps) if fps > 0 else self.min_period
                period = max(period, self.min_period)
                nxt = getattr(self, "_next_frame_at", 0.0)
                now = time.monotonic()
                if nxt <= 0.0 or now - nxt > 1.0:
                    nxt = now              # first frame, or we fell far behind
                nxt += period
                self._next_frame_at = nxt
                slack = nxt - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    self.frames_late = getattr(self, "frames_late", 0) + 1
            else:
                self._next_frame_at = 0.0

    # The structure (which frequencies belong to which family) is fetched
    # separately via /noise and only when it changes, so these can be
    # generous without bloating the per-frame payload.
    MAX_WIRE_FAMILIES = 12
    MAX_WIRE_MEMBERS = 4096
    MAX_WIRE_SPURS = 20000     # mark every detected peak; /noise is fetched
                               # once per structure change, not per frame

    def _sys_loop(self):
        while not self._stop.is_set():
            try:
                self._sys_cache = self.sysmon.sample()
            except Exception:
                pass
            self._stop.wait(1.0)

    def _analysis_loop(self):
        while not self._stop.is_set():
            with self._an_lock:
                while self._an_req is None and not self._stop.is_set():
                    self._an_cv.wait(timeout=0.5)
                req = self._an_req
            if req is None:
                continue
            try:
                spec, bin_hz, fs, K, nfft, ch, pfa = req
                r = noise.analyse(spec, bin_hz, fs, nframes=K, pfa=pfa,
                                  max_peaks=self.MAX_WIRE_SPURS)
                self._sid += 1
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

        offs = np.arange(-2, 3, dtype=np.int64)

        def peak_level(freqs):
            """Max level over +-2 bins about each frequency, vectorised.

            A single rounded bin can miss the true peak: the fundamental is
            refitted only once per structural pass, so a drifting switcher
            moves between passes and rounding can land on a shoulder.

            This runs on EVERY frame for every family member, and families
            may now hold thousands of members, so it must not be a Python
            loop -- one gather over an (n x 5) index matrix instead."""
            if len(freqs) == 0:
                return None
            idx = np.rint(np.asarray(freqs, dtype=np.float64) / bin_hz)
            idx = np.clip(idx.astype(np.int64)[:, None] + offs[None, :],
                          0, nb - 1)
            return float(shown[idx].max())


        # Frequencies are the bulk of the payload and change only when the
        # structure does, so they ride on /noise?sid=N instead of every
        # frame. Dots are drawn at the polyline's own value, so the client
        # needs no per-frame levels at all -- only the card's peak_db.
        out = {k: an[k] for k in ("sid", "ts", "fs_hz", "channel", "floor_db",
                                  "thr_db", "n_peaks", "n_spurs_total", "pfa")
               if k in an}
        out["families"] = [{k: f[k] for k in
                            ("id", "f0_hz", "label", "why", "n_members",
                             "density", "significance") if k in f}
                           | {"peak_db": (peak_level(f.get("freqs", []))
                                          or f.get("peak_db"))}
                           for f in an.get("families", [])]
        out["structure_nfft"] = an.get("nfft")
        out["nfft"] = nfft
        out["live"] = True
        return out

    THR_CURVE_PTS = 512

    @staticmethod
    def _thr_curve(floor_db, thr_db, npts):
        """Detection threshold sampled on the same log-frequency grid the
        display uses, so the client can draw it over the spectrum. Smooth by
        construction, so a few hundred points suffice."""
        nb = len(floor_db)
        if nb < 4:
            return []
        idx = np.power(10.0, np.linspace(0.0, np.log10(nb - 1), npts + 1))
        starts = np.maximum.accumulate(
            np.clip(np.floor(idx[:-1]).astype(np.int64), 1, nb - 2)) - 1
        vals = floor_db[np.clip(starts, 0, nb - 1)] + thr_db
        return [round(float(v), 1) for v in vals]

    def _shape_analysis(self, r, fs, nfft, ch):
        """Trim to what the plot needs -- a full peak list can be thousands
        of entries and would dwarf the spectrum payload itself."""
        fams = []
        for i, f in enumerate(r["families"][:self.MAX_WIRE_FAMILIES]):
            # by prominence, not absolute level -- see noise.detect_peaks
            mem = sorted(f["members"],
                         key=lambda m: -m.get("prominence_db", m["db"]))
            mem = mem[:self.MAX_WIRE_MEMBERS]
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
        spurs.sort(key=lambda p: -p.get("prominence_db", p["db"]))
        why_by_label = {}
        for p in spurs:
            why_by_label.setdefault(p.get("label", "?"), p.get("why", ""))
        return {
            "sid": self._sid,
            "ts": time.time(), "fs_hz": fs, "nfft": nfft, "channel": ch,
            "floor_db": r["floor_db"],
            "thr_db": r["threshold_db_over_floor"],
            "n_peaks": r["n_peaks"],
            "n_spurs_total": len(spurs),
            "pfa": r.get("pfa"),
            "thr_curve": self._thr_curve(r["floor_curve"],
                                         r["threshold_db_over_floor"],
                                         self.THR_CURVE_PTS),
            "families": fams,
            # `why` is a whole sentence and repeats across hundreds of spurs;
            # send it once per label instead
            "why_by_label": why_by_label,
            "spurs": [{"freq_hz": p["freq_hz"], "db": p["db"],
                       "prom": p.get("prominence_db", 0.0),
                       "label": p["label"]}
                      for p in spurs[:self.MAX_WIRE_SPURS]],
        }

    def _one(self, cfg):
        t_loop0 = time.monotonic()
        N, ch, sp_, nfft = cfg["nsamples"], cfg["channel"], cfg["speed"], cfg["nfft"]

        # nfft > N cannot fill even one FFT frame: adc_process returns -1 and
        # the capture raises, EVERY iteration, so the engine stops producing
        # frames entirely and every connected client sits on its last good
        # one looking frozen. Clamp instead of failing, because this state is
        # reachable innocently: a client that changes record length and FFT
        # length in two separate /control requests (which web/index.html
        # does) is briefly in exactly this configuration, and a capture
        # landing in that window should degrade, not wedge the engine until
        # someone notices and sets a valid pair by hand.
        if nfft > N:
            nfft = max(1 << 10, 1 << int(N).bit_length() - 1)
            nfft = min(nfft, N)
            cfg["nfft"] = nfft

        # rebuild the GPU context if FFT size or capacity changed
        if nfft != self.sp.nfft or N > self.sp.max_samples:
            self.sp.close()
            self.sp = gpu.Spectrum(nfft, max_samples=max(1 << 22, N),
                                   trace_width=self.trace_width)
            self._acc = None; self._acc_n = 0

        A.Adc._validate(N, ch, sp_,
                        streaming=bool(cfg.get("mode", 0)) and not self.mock)

        # STREAM MODE: the FPGA ring is already free-running, so there is
        # nothing to arm and no Adc_Finish to wait for -- the whole arm/poll
        # block below is skipped and the samples come from the RAM ring
        # instead of a fresh DMA. Everything after acquisition (GPU, traces,
        # phase noise, frame build) is shared, deliberately: the two modes
        # must not be able to drift apart in how they present data.
        streaming = bool(cfg.get("mode", 0)) and not self.mock
        self._maybe_leave_stream(cfg)
        if streaming:
            t_cap, suspect, expect = 0.0, False, N / A.BASE_CLOCK_HZ
        else:
            self._arm_block(cfg, N, ch)
            expect = N / A.BASE_CLOCK_HZ
            t_cap, suspect = self._wait_block(ch, N, expect)

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
        if streaming and tr_req < 0 and int(cfg.get("stream_window", 0)) > 0:
            # CONSTANT FRAME RATE depends on the per-frame work being bounded.
            # "trace = the whole record" is the right default in block mode,
            # where a frame IS one record. In stream mode the record is
            # whatever `nsamples` says while the data arrives continuously, so
            # letting the envelope span it makes the frame cost scale with
            # `nsamples` again -- exactly the variable rate streaming is meant
            # to remove. The FFT only ever consumes max_frames*nfft samples
            # (524,288 by default) no matter how large `nsamples` is, so the
            # envelope is the only thing pulling the whole record through the
            # CPU. Bound it, and let stream_window raise it deliberately.
            trace_n = min(trace_n, max(need, int(cfg["stream_window"])))
        rb = cfg.get("readback", -1)
        want = max(need, trace_n, rb) if rb > 0 else (N if rb == 0
                                                      else max(need, trace_n))
        if self.pn_cfg["enabled"] and ch == A.CH_BOTH:
            # phase noise needs every sample: the record length IS the
            # frequency resolution (1/T), and a partial read would also put
            # a discontinuity in the middle of the demodulated phase
            want = N
        read_n = min(N, want)
        g = A.SAMPLE_GRANULARITY
        read_n = max(g, -(-read_n // g) * g)     # round UP to granularity
        read_n = min(read_n, N)
        trace_n = min(trace_n, read_n)

        # Readback via the vendor dma_from_device CLI (subprocess + tempfile) —
        # NOT gpu.FastC2H's raw os.readv(), which bisect_dma.py showed wedges
        # the SoC regardless of destination buffer type. See module docstring.
        t1 = time.monotonic()
        if ch == A.CH_BOTH:
            # every sample clock emits BOTH channels as one 32-bit word, so
            # the record is twice as many uint16 words
            nbytes = read_n * 4
            if streaming:
                # Already in RAM: the stream reader DMA'd this segment while
                # the previous frame was being drawn, so there is no transfer
                # on the display path at all -- which is exactly why the frame
                # rate stops depending on the record length.
                raw, read_n = self._acquire_stream(cfg, ch, read_n)
                if raw is None:
                    return                    # ring not primed yet
                nbytes = read_n * 4
            else:
                buf = self._dual_words(read_n * 2)
                n = self.dma.read_into(buf, read_n * 2)
                raw = buf[:n]
            # vendor client: raw[2i] is channel A, raw[2i+1] is channel B.
            # De-interleave into CONTIGUOUS arrays first, then sign-extend
            # ONCE per channel and take all four statistics off that one
            # array. This used to call to_signed() four times per channel on
            # the STRIDED view, each promoting to int32 -- eight conversions
            # and ~262 MB of allocation per frame at 8.192 M samples, which
            # was 175 ms of the ~190 ms frame time and was being reported as
            # "DMA". Both modes pay this path, so block mode got faster too.
            self._pending = {"A": np.ascontiguousarray(raw[0::2]),
                             "B": np.ascontiguousarray(raw[1::2])}
            # self.dual is filled from the GPU further down. It used to be
            # computed here with four numpy passes per channel -- mean, std,
            # min and max each traversing the whole record. At the maximum
            # record that was 2478 ms of a 3499 ms frame, 71% of the time, for
            # four numbers in a readout. k_stats already computes exactly
            # those on the device, in one pass, as part of work the frame does
            # anyway, so the CPU passes were pure duplication.
            self.dual = None
            raw_src = raw
            if self.pn_cfg["enabled"] and self._pending["A"].size:
                self._pn_submit(self._pending, A.sample_rate(sp_),
                                dict(self.pn_cfg))
            n = read_n
            got = nbytes
        else:
            self.dual = None
            self._pending = None
            if streaming:
                src, read_n = self._acquire_stream(cfg, ch, read_n)
                if src is None:
                    return                    # ring not primed yet
                n = read_n
            else:
                n = self.dma.read_into(self.sp.stage, read_n)
            nbytes = read_n * 2
            self.sp.load(n * 2)
            got = n * 2
            raw_src = self.sp.stage[:n]
        trace_n = min(trace_n, read_n)   # stream mode caps read_n at one
                                         # segment, so re-clamp after acquiring
        if time.monotonic() - self._raw_req < 5.0:
            # Must come from the buffer THIS branch actually filled. Dual mode
            # lands in _dual_stage, not sp.stage (see _dual_words), so sourcing
            # /raw from sp.stage served the PREVIOUS frame's channel B -- the
            # last thing the trace loop copied in. Anything that de-interleaves
            # /raw (tools/channel_skew.py, diag_input.py) was then splitting one
            # channel's record even/odd and calling the halves A and B.
            self.raw = np.array(raw_src[:min(raw_src.size, 1 << 16)], copy=True)
        t_dma = time.monotonic() - t1

        # CUDA -- once per trace. Dual mode draws both channels, so both get
        # analysed; a GPU pass is a few ms, cheaper than making the user pick.
        t2 = time.monotonic()
        # ChannelSel 0 is the FPGA's own counter, generated in fabric: it does
        # NOT pass through the ADC's output inverter, so the kernel must not
        # flip it. Every other channel carries converter data and must.
        want_inv = bool(getattr(A, "OUTPUT_INVERT", False)) and ch != A.CH_TEST_RAMP
        if getattr(self.sp, "invert", None) != want_inv:
            try:
                self.sp.set_invert(want_inv)
            except Exception:
                pass

        traces = []
        if self._pending:
            for name in ("A", "B"):
                arr = self._pending[name]
                np.copyto(self.sp.stage[:arr.size], arr)
                self.sp.load(arr.size * 2)
                sp_i = self.sp.process(arr.size, max_frames=cfg["max_frames"],
                                       trace_n=min(trace_n, arr.size)).copy()
                traces.append({"name": name, "spec": sp_i,
                               "tmin": self.sp.tmin.copy(),
                               "tmax": self.sp.tmax.copy(),
                               "tmean": self.sp.tmean.copy(),
                               "stats": self.sp.stats.copy(),
                               "nframes": int(self.sp.nframes)})
        else:
            sp_i = self.sp.process(read_n, max_frames=cfg["max_frames"],
                                   trace_n=trace_n)
            traces.append({"name": "", "spec": sp_i,
                           "tmin": self.sp.tmin, "tmax": self.sp.tmax,
                           "tmean": self.sp.tmean, "stats": self.sp.stats,
                           "nframes": int(self.sp.nframes)})
        # RING TIME AXIS. The spectrum still comes from the record (it has to
        # -- resolution is 1/T of the analysed block), but the time trace can
        # be swapped for the whole-ring envelope, which is already built.
        ring_span_s = 0.0
        if streaming and int(cfg.get("trace_span_ring", 0)) and \
           self.stream is not None and self.stream.env is not None:
            env, ring = self.stream.env, self.stream.ring
            lo, hi = ring.resident
            for ci, t in enumerate(traces):
                r = env.span(lo, hi, min(ci, env.nch - 1), self.trace_width)
                if r is None:
                    break
                mn, mx, me, nseg = r
                t["tmin"], t["tmax"], t["tmean"] = mn, mx, me
                ring_span_s = nseg * env.seconds_per_seg

        # Per-channel statistics, straight off the device: k_stats_fin writes
        # [min, max, mean, std, std] over exactly the samples each channel's
        # process() pass consumed, already decoded through s14() with the
        # output-inversion flag applied.
        if self._pending and traces:
            self.dual = {}
            for t in traces:
                st = t.get("stats")
                if st is None or not t["name"]:
                    continue
                self.dual[t["name"]] = {"min": int(st[0]), "max": int(st[1]),
                                        "mean": float(st[2]), "std": float(st[3])}

        spec = traces[0]["spec"]
        t_gpu_wall = time.monotonic() - t2

        # exponential spectrum averaging
        navg = max(1, cfg["avg"])
        # configure() resets the accumulator, but a frame produced under the
        # OLD cfg can still land afterwards and re-seed it; spectra from
        # different fs/nfft/channel share bin indices, so blending them shows
        # stale peaks at wrong frequencies (seen: a 6.25 MHz peak from
        # speed=7 displayed as 50 MHz after switching to speed=0). Tag the
        # accumulator with the producing config and reseed on any mismatch.
        tag = (fs := A.sample_rate(sp_), nfft, ch, len(traces))
        if (self._acc is None or len(self._acc) != len(traces)
                or self._acc[0].shape != spec.shape or self._acc_tag != tag):
            self._acc = [t["spec"].copy() for t in traces]
            self._acc_n = 1
            self._acc_tag = tag
        elif navg == 1:
            for a_, t in zip(self._acc, traces):
                np.copyto(a_, t["spec"])
            self._acc_n = 1
        else:
            k = np.float32(1.0 / navg)
            for a_, t in zip(self._acc, traces):
                a_ *= (1 - k); a_ += k * t["spec"]
            self._acc_n = min(self._acc_n + 1, navg)
        shown = self._acc[0]

        self._ch_fail = 0          # a completed capture clears the streak
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
            # The snapshot is a full-spectrum copy (134 MB at nfft=2^27) taken
            # in the capture loop, so its period scales with size: hold it to
            # roughly 1% of a frame's own budget rather than a fixed 1 Hz.
            period = max(self.classify_period, len(shown) * 4 / 1e9 * 100)
            if now - self._an_last >= period:
                with self._an_lock:
                    if self._an_req is None:          # only when idle
                        self._an_req = (shown.copy(), bin_hz, fs,
                                        max(1, self.sp.nframes * max(1, self._acc_n)),
                                        nfft, ch,
                                        10.0 ** -max(1, min(12, cfg.get("pfa_exp", 6))))
                        self._an_last = now
                        self._an_cv.notify()
        elif self.analysis is not None:
            self.analysis = None

        want_log = bool(cfg.get("disp_log", 1))
        _disp_fn = log_display if want_log else lin_display
        disp, disp_f0, disp_f1 = _disp_fn(shown, bin_hz, DISP_BINS)
        # one display-decimated spectrum per trace; trace 0 is `shown`, the
        # EMA-averaged one that peaks and classification are taken from
        disps = [disp] + [_disp_fn(self._acc[i], bin_hz, DISP_BINS)[0]
                          for i in range(1, len(traces))]

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
            "disp_log": want_log,   # True: points log-spaced disp_f0..disp_f1
                                    # False: linearly spaced. The client's axis
                                    # MUST match, or every frequency is wrong.
            "disp_f0": disp_f0,
            "disp_f1": disp_f1,
            "zoom": zoom_meta,
            "trace_width": int(self.trace_width),
            "traces": [t["name"] for t in traces],
            "n_traces": len(traces),
            "nframes": int(self.sp.nframes),
            "bin_hz": bin_hz,
            "stream": self._stream_meta(streaming),
            "adc": self.adc_health(),
            "acq": {
                "capture_ms": t_cap * 1e3,
                "capture_theory_ms": expect * 1e3,
                "capture_suspect": bool(suspect),
                "capture_samples": N,
                "read_samples": read_n,
                "read_pct": 100.0 * read_n / N if N else 0.0,
                # With the ring axis the trace no longer covers `trace_n`
                # samples of the record -- it covers the whole resident ring.
                # The UI derives the time axis from this, so it has to report
                # the span actually drawn or the axis would lie.
                "trace_samples": (int(ring_span_s * A.BASE_CLOCK_HZ)
                                  if ring_span_s > 0 else trace_n),
                "trace_span_ring": bool(ring_span_s > 0),
                "trace_span_s": ring_span_s if ring_span_s > 0
                                else (trace_n / A.BASE_CLOCK_HZ),
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
            "dual": self.dual,
            "pn": {"enabled": bool(self.pn_cfg["enabled"]),
                   "sid": self.pn_sid,
                   "waiting": bool(self.pn_cfg["enabled"]
                                   and ch != A.CH_BOTH)},
            "sys": self._sys_cache,
            "analysis": self._live_analysis(shown, bin_hz, fs, nfft, ch)
                        if cfg.get("classify") else None,
            "err": self.err,
        }
        # Halve the wire size: the spectrum goes as int16 hundredths of a dB
        # (0.01 dB steps against a display that resolves ~0.1 dB) and the
        # envelope as uint16, which is EXACT because those are 14-bit ADC
        # codes. 26.5 kB/frame at 66 fps is 14 Mbit/s on a 7 Mbit/s link;
        # this brings it to ~3.4 Mbit/s at 30 fps.
        m["wire"] = 5          # per-trace array groups
        m["tmean_scale"] = 4.0
        m["volt_scale"] = A.VOLT_SCALE
        m["disp_scale"] = 100.0
        hdr = json.dumps(m).encode()
        z16 = (np.clip(np.frombuffer(zoom_bytes, dtype=np.float32),
                       -320.0, 40.0) * 100.0).astype(np.int16) \
              if zoom_bytes else np.zeros(0, np.int16)
        i16 = lambda a: np.clip(a, -32768, 32767).astype(np.int16).tobytes()
        # layout: per trace -> disp, [zoom, on trace 0 only], tmin, tmax,
        # tmean. The zoom slice is a crop of the primary spectrum, so it goes
        # once rather than per trace.
        parts = [struct.pack("<I", len(hdr)), hdr]
        for i, t in enumerate(traces):
            parts.append(i16(np.clip(disps[i], -320.0, 40.0) * 100.0))
            if i == 0:
                parts.append(z16.tobytes())
            parts += [i16(t["tmin"]), i16(t["tmax"]), i16(t["tmean"] * 4.0)]
        blob = b"".join(parts)
        with self.new_frame:
            self.frame = blob
            self.new_frame.notify_all()   # wake every long-poll waiter
        self.err = None       # a completed capture clears any stale error


def json_safe(o):
    """Deep-convert a result dict to something json.dumps will accept.

    The masked FSR nulls are NaN on purpose -- "the interferometer sees
    nothing here" -- and JSON has no NaN. json.dumps' own options are both
    wrong: allow_nan=True emits a bare `NaN` token that JSON.parse rejects,
    and allow_nan=False raises rather than calling `default` (which fires
    only for types json does not already know, and it knows float). So the
    substitution has to happen before dumps is called. `null` is what a
    canvas plot already knows to lift the pen for.
    """
    if isinstance(o, np.ndarray):
        if o.dtype.kind == "f":
            out = o.astype(object)
            out[~np.isfinite(o)] = None
            return out.tolist()
        return o.tolist()
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


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
        try:
            self._send_inner(code, body, ctype)
        except (BrokenPipeError, ConnectionResetError):
            # the client went away (reload, navigation, or a dropped link
            # while parked on a long-poll). Normal; not worth a traceback.
            self.close_connection = True

    def _send_inner(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # CORS: the Flutter client (flutter_client/) is a separate origin --
        # served by `flutter run -d chrome`'s own dev port, or wherever its
        # `flutter build web` output ends up -- and fetches this API cross-
        # origin. This host is already only reachable on a private/VPN
        # network (see README's NetBird section), so an open origin costs
        # nothing that network access didn't already grant.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        # CORS preflight for POST /control and /pn/control: a JSON body
        # (Content-Type: application/json) is not a "simple" request, so the
        # browser sends this first and refuses the real POST without a
        # matching answer here.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

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
                deadline = time.monotonic() + 10.0
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
        if p == "/raw":
            # Raw samples straight from the last capture, for diagnosing what
            # the converter is actually producing (stuck bits, dropouts,
            # interleave order) without stopping the server to take the lock.
            e = self.engine
            e._raw_req = time.monotonic()
            r = e.raw
            if r is None:
                return self._send(503, b"no raw yet - retry in a moment",
                                  "text/plain")
            return self._send(200, r.tobytes(), "application/octet-stream")
        if p == "/noise":
            # Full structure (family member frequencies, spur list). Fetched
            # by the client only when the frame's `sid` changes.
            an = self.engine.analysis
            if not an or an.get("error"):
                return self._send(503, b'{"sid":0,"families":[],"spur_f":[]}',
                                  "application/json")
            labels = {}
            body = json.dumps({
                "sid": an.get("sid", 0),
                "fs_hz": an.get("fs_hz"), "channel": an.get("channel"),
                "why_by_label": an.get("why_by_label", {}),
                "thr_curve": an.get("thr_curve", []),
                "families": [{"id": f["id"],
                              "freqs": [round(x, 1) for x in f.get("freqs", [])]}
                             for f in an.get("families", [])],
                # parallel arrays, not per-peak objects: at a few thousand
                # peaks the repeated JSON keys and label strings dominate
                "spur_f": [int(round(s["freq_hz"])) for s in an.get("spurs", [])],
                "spur_p": [round(s.get("prom", 0.0), 1) for s in an.get("spurs", [])],
                "spur_l": [labels.setdefault(s.get("label", ""), len(labels))
                           for s in an.get("spurs", [])],
                "label_names": None,   # filled below
            })
            body = body.replace('"label_names": null',
                                '"label_names": ' + json.dumps(
                                    [k for k, _ in sorted(labels.items(),
                                                          key=lambda kv: kv[1])]))
            return self._send(200, body.encode(), "application/json")
        if p == "/pn":
            # Latest phase-noise result. Fetched on its own poll rather than
            # ridden along on /frame: it updates at the worker's rate (~1 Hz
            # or slower on a long record), not the capture rate, and it is
            # ~40 kB of JSON that most viewers never open the tab for.
            e = self.engine
            r = e.pn
            if r is None:
                body = json.dumps({
                    "pending": True,
                    "enabled": bool(e.pn_cfg["enabled"]),
                    "channel": e.cfg["channel"],
                    "cfg": e.pn_cfg,
                    "tau_s": e.pn_tau(),
                    "cal_points": e._pn_acc_n,
                })
                return self._send(200, body.encode(), "application/json")
            out = json_safe(r)
            out["cfg"] = e.pn_cfg
            out["channel"] = e.cfg["channel"]
            out["enabled"] = bool(e.pn_cfg["enabled"])
            out["cal_points"] = e._pn_acc_n
            return self._send(200, json.dumps(out).encode(),
                              "application/json")
        if p == "/diag/status":
            e = self.engine
            try:
                st = {"session": e.diag.state(), "adc": e.adc_health()}
                if not e.mock and e.adc.has_adc_ctl:
                    st["regs"] = {
                        "reg10": e.adc.rd(A.REG_ADC_STATUS),
                        "reg11": e.adc.rd(A.REG_RAMP_ERR_A),
                        "reg12": e.adc.rd(A.REG_RAMP_ERR_B),
                        "reg13": e.adc.rd(A.REG_DELAY),
                        "reg6":  e.adc.rd(A.REG_FLAGS),
                        "reg15": e.adc.rd(A.REG_DESIGN_ID)}
                    # SPI reads are ~3.3 us each; three is cheap at 2 Hz
                    st["adc_regs"] = {
                        "0x0D": e.adc.adc_rd(A.ADC_TEST_MODE),
                        "0x14": e.adc.adc_rd(A.ADC_OUTPUT_MODE),
                        "0x17": e.adc.adc_rd(A.ADC_DCO_DELAY)}
                    st["stored_tap"] = A.load_stored_tap()
                    st["patterns"] = DG.PATTERNS
            except Exception as ex:
                st = {"error": f"{type(ex).__name__}: {ex}"}
            return self._send(200, json.dumps(json_safe(st)).encode(),
                              "application/json")
        if p == "/diag/progress":
            d = self.engine.diag
            out = d.state()
            try:
                since = int(self.path.split("since=")[1].split("&")[0])
            except Exception:
                since = 0
            out["points"] = d.points[since:since + 2048]
            out["point_base"] = since
            return self._send(200, json.dumps(json_safe(out)).encode(),
                              "application/json")
        if p in ("/diag/result.csv", "/diag/result.json"):
            r = self.engine.diag.result
            if r is None:
                return self._send(404, b"no result yet", "text/plain")
            if p.endswith(".csv"):
                body = r.to_csv().encode()
                if not body:
                    return self._send(404, b"this result has no table",
                                      "text/plain")
                return self._send(200, body, "text/csv")
            return self._send(200, json.dumps(json_safe(r.to_json())).encode(),
                              "application/json")
        if p == "/hw":
            q = {}
            if "?" in self.path:
                for kv in self.path.split("?", 1)[1].split("&"):
                    k, _, v = kv.partition("=")
                    q[k] = v
            try:
                out = self.engine.hw_snapshot(dump=(q.get("dump") == "1"))
            except Exception as ex:
                out = {"error": f"{type(ex).__name__}: {ex}"}
            return self._send(200, json.dumps(json_safe(out)).encode(),
                              "application/json")
        if p == "/adc":
            e = self.engine
            q = {}
            if "?" in self.path:
                for kv in self.path.split("?", 1)[1].split("&"):
                    k, _, v = kv.partition("=")
                    q[k] = v
            out = {"health": e.adc_health()}
            if q.get("dump") == "1":
                try:
                    out["registers"] = {k: v for k, v in e.adc_dump().items()}
                except Exception as ex:
                    out["dump_error"] = f"{type(ex).__name__}: {ex}"
            return self._send(200, json.dumps(json_safe(out)).encode(),
                              "application/json")
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
        if p.startswith("/diag/"):
            n = int(self.headers.get("Content-Length", 0))
            try:
                cfg = json.loads(self.rfile.read(n) or b"{}")
                d = self.engine.diag
                tok = cfg.get("token") or "anon"
                act = p[len("/diag/"):]
                if act == "enter":
                    out = d.enter(tok)
                elif act == "keepalive":
                    out = d.keepalive(tok)
                elif act == "leave":
                    out = d.leave(tok)
                elif act == "cancel":
                    out = d.cancel(tok)
                elif act == "run":
                    out = d.start(tok, cfg.get("test"), cfg.get("params") or {})
                elif act == "pattern":
                    if not d.active or d.owner != tok:
                        raise RuntimeError(
                            "diagnostics lease lapsed or held by another"
                            " client -- re-enter the tab")
                    DG.set_pattern(self.engine.adc, int(cfg["pattern"]))
                    out = {"pattern": self.engine.adc.adc_rd(A.ADC_TEST_MODE)}
                elif act == "reg":
                    if not d.active or d.owner != tok:
                        raise RuntimeError(
                            "diagnostics lease lapsed or held by another"
                            " client -- re-enter the tab")
                    adc = self.engine.adc
                    addr = int(cfg["addr"])
                    if "value" in cfg:
                        # goes through the driver guards: 0x09 bit0 and 0x14
                        adc.adc_wr_transfer(addr, int(cfg["value"]),
                                            force=bool(cfg.get("force")))
                    if cfg.get("transfer"):
                        adc.adc_transfer()
                    out = {"addr": addr, "value": adc.adc_rd(addr)}
                elif act == "clear":
                    if not d.active or d.owner != tok:
                        raise RuntimeError(
                            "diagnostics lease lapsed or held by another"
                            " client -- re-enter the tab")
                    self.engine.adc.ramp_clear()              # reg10: counters
                    self.engine.adc.wr(A.REG_FLAGS, 0)        # reg6: stream flags
                    out = {"cleared": True}
                elif act == "tap":
                    if not d.active or d.owner != tok:
                        raise RuntimeError(
                            "diagnostics lease lapsed or held by another"
                            " client -- re-enter the tab")
                    t = int(cfg["tap"])
                    r = self.engine.adc.set_tap(t)
                    if cfg.get("save"):
                        A.save_stored_tap(t)
                    out = {"tap": r, "saved": bool(cfg.get("save"))}
                else:
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, json.dumps(json_safe(out)).encode(),
                                  "application/json")
            except Exception as ex:
                return self._send(400, json.dumps(
                    {"error": f"{type(ex).__name__}: {ex}"}).encode(),
                    "application/json")
        if p == "/hw/control":
            n = int(self.headers.get("Content-Length", 0))
            try:
                cfg = json.loads(self.rfile.read(n) or b"{}")
                out = {"ok": True}
                if "counter_test" in cfg:
                    out["counter"] = self.engine.counter_test(cfg["counter_test"])
                if "eye_scan" in cfg:
                    e = cfg["eye_scan"] or {}
                    out["eye"] = self.engine.eye_scan(
                        step=e.get("step", 8), dwell=e.get("dwell", 0.01),
                        nsamp=e.get("samples", 65536))
                return self._send(200, json.dumps(json_safe(out)).encode(),
                                  "application/json")
            except Exception as ex:
                return self._send(400, json.dumps(
                    {"error": f"{type(ex).__name__}: {ex}"}).encode(),
                    "application/json")
        if p == "/adc/control":
            n = int(self.headers.get("Content-Length", 0))
            try:
                cfg = json.loads(self.rfile.read(n) or b"{}")
                out = {"ok": True}
                if "tap" in cfg:
                    out["tap"] = self.engine.adc_set_tap(
                        cfg["tap"], save=bool(cfg.get("save_tap", False)))
                if cfg.get("clear_flags"):
                    out["status"] = self.engine.adc_clear_flags()
                if "ramp_test" in cfg:
                    secs = float(cfg["ramp_test"])
                    if not (0.1 <= secs <= 120.0):
                        raise ValueError("ramp_test seconds must be 0.1..120")
                    out["ramp"] = self.engine.ramp_test(secs)
                out["health"] = self.engine.adc_health()
                return self._send(200, json.dumps(json_safe(out)).encode(),
                                  "application/json")
            except Exception as ex:
                return self._send(400, json.dumps(
                    {"error": f"{type(ex).__name__}: {ex}"}).encode(),
                    "application/json")
        if p == "/pn/control":
            n = int(self.headers.get("Content-Length", 0))
            try:
                cfg = json.loads(self.rfile.read(n) or b"{}")
                if cfg.pop("recalibrate", False):
                    self.engine.pn_recalibrate()
                self.engine.pn_configure(**cfg)
                return self._send(200, json.dumps(
                    {"ok": True, "cfg": self.engine.pn_cfg,
                     "tau_s": self.engine.pn_tau()}).encode(),
                    "application/json")
            except Exception as e:
                return self._send(400, json.dumps({"error": str(e)}).encode(),
                                  "application/json")
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
    ap.add_argument("--tap", type=int, default=None,
                    help="IDELAY tap to load at start-up; default is the value "
                         "stored by tools/eye_scan.py --save. The FPGA does not "
                         "retain it across reconfiguration.")
    ap.add_argument("--fast-dma", action="store_true",
                    help="use the resident DMA helper (native/xdma_shm_reader)"
                         " instead of spawning dma_from_device per frame."
                         " Validate on hardware with validate_fast_dma.py"
                         " before trusting it")
    ap.add_argument("--mock", action="store_true",
                    help="no hardware: synthetic 25 MHz tone, never opens"
                         " /dev/*. For UI/server development")
    ap.add_argument("--mock-interferometer", action="store_true",
                    help="mock a 3x3-coupler Michelson on channel 3 instead"
                         " of the tone: two photocurrents from a laser of"
                         " --mock-linewidth, for the phase-noise tab."
                         " Implies --mock")
    ap.add_argument("--mock-linewidth", type=float, default=50e3,
                    help="Lorentzian FWHM the mock laser is built to have,"
                         " in Hz (default 50k). The phase-noise tab should"
                         " read this number back")
    ap.add_argument("--mock-drift", type=float, default=40.0,
                    help="fringes per second the mock operating point"
                         " sweeps. 0 reproduces a stuck interferometer, the"
                         " condition under which two photodiodes cannot be"
                         " calibrated at all")
    a = ap.parse_args()
    if a.mock_interferometer:
        a.mock = True

    lock = None
    if a.mock:
        print("MOCK MODE - synthetic data, hardware untouched")
        if a.mock_interferometer:
            print(f"  interferometer: {a.mock_linewidth/1e3:g} kHz linewidth,"
                  f" {a.mock_drift:g} fringes/s drift")
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
    eng = Engine(nsamples=a.nsamples,
                 channel=A.CH_BOTH if a.mock_interferometer else a.channel,
                 nfft=a.nfft,
                 min_period=a.min_period, fast_dma=a.fast_dma, tap=a.tap,
                 mock="interferometer" if a.mock_interferometer else a.mock,
                 mock_opts=dict(dnu_hz=a.mock_linewidth,
                                drift_hz=a.mock_drift))
    if a.mock_interferometer:
        eng.pn_configure(enabled=1)
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
