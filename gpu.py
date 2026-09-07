"""CUDA spectrum engine (cuFFT) + DMA readback.

FastC2H IS UNSAFE — DO NOT USE FOR READBACK
--------------------------------------------
The original theory here was that FastC2H.read_into() (raw os.readv() on the
XDMA C2H character device) was only unsafe when the destination was CUDA
pinned memory (cudaHostAlloc → nvmap, already mapped into the GPU's SMMU
domain — a PCIe DMA into it writes to a stream the SMMU hasn't mapped for
that client, wedging the interconnect: a hard SoC freeze with nothing logged).
The fix below (route through `Spectrum.stage`, a plain malloc'd array) was
written on that assumption.

bisect_dma.py DISPROVED it: run against this exact code, it froze the machine
at stage S4 — FastC2H.read_into() into a PLAIN array — without ever reaching
S6 (the CUDA-pinned case). See bisect_marker.txt (still reads "S4" — no run
has gotten further) and NOTES.md. So the raw read_into() path is unsafe
regardless of destination buffer; the pinned-vs-plain distinction was never
actually the dividing line.

server.py therefore does not use FastC2H at all. It reads back via
ad9643.ddr_read_samples() — the vendor `dma_from_device` CLI run as a
subprocess into a tempfile — which is the one readback path bisect_dma.py
confirmed completes (stage S3) without freezing. `Spectrum.stage` /
`Spectrum.load()` are kept as the landing buffer for that data before it's
copied into the pinned buffer; that copy is a plain CPU memcpy, not an FPGA
DMA target, so it carries none of this risk.

ADC_DIRECT_DMA=1 still exists below for anyone deliberately re-testing
FastC2H via bisect_dma.py. Do not wire it back into server.py without first
getting a clean bisect_dma.py run past S6 — S4 alone is reason enough to
leave FastC2H out of the live server.

A real fix, if someone wants FPGA-DMA-speed readback later, needs the vendor
XDMA driver/kernel-module behavior understood first (why does its own
`dma_from_device` tool survive where a Python read() into equivalent memory
does not?) rather than another guess at the CUDA memory theory.
"""
import ctypes, os
import numpy as np

DIRECT_DMA = os.environ.get("ADC_DIRECT_DMA", "0") == "1"

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda", "libadcfft.so")
_l = ctypes.CDLL(_LIB)

_l.adc_create.restype = ctypes.c_void_p
_l.adc_create.argtypes = [ctypes.c_int]*3
_l.adc_hostbuf.restype = ctypes.POINTER(ctypes.c_uint16)
_l.adc_hostbuf.argtypes = [ctypes.c_void_p]
_l.adc_nbins.argtypes = [ctypes.c_void_p]; _l.adc_nbins.restype = ctypes.c_int
_l.adc_maxframes.argtypes = [ctypes.c_void_p]; _l.adc_maxframes.restype = ctypes.c_int
_F = ctypes.POINTER(ctypes.c_float)
_l.adc_process.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int,
                           _F, _F, _F, _F, _F, ctypes.POINTER(ctypes.c_int)]
_l.adc_process.restype = ctypes.c_int
_l.adc_set_window.argtypes = [ctypes.c_void_p, ctypes.c_int]
_l.adc_set_window.restype = ctypes.c_int
_l.adc_win_enbw.argtypes = [ctypes.c_void_p]
_l.adc_win_enbw.restype = ctypes.c_float
_l.adc_destroy.argtypes = [ctypes.c_void_p]
_l.adc_meminfo.argtypes = [ctypes.POINTER(ctypes.c_size_t)]*2


def gpu_mem():
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    _l.adc_meminfo(ctypes.byref(f), ctypes.byref(t))
    return f.value, t.value


# window ids must match the WIN_* defines in cuda/adcfft.cu
WINDOWS = {"hann": 0, "blackman-harris": 1, "flattop": 2, "rect": 3}
WINDOW_NAMES = {v: k for k, v in WINDOWS.items()}


class Spectrum:
    """Owns the pinned host buffer plus a plain staging buffer for DMA."""

    def __init__(self, nfft=8192, max_samples=1 << 24, trace_width=1024):
        self.nfft, self.max_samples, self.tw = nfft, max_samples, trace_width
        self.h = _l.adc_create(nfft, max_samples, trace_width)
        if not self.h:
            raise RuntimeError("adc_create failed (CUDA init?)")
        self.nbins = _l.adc_nbins(self.h)
        self.max_frames = _l.adc_maxframes(self.h)
        buf = _l.adc_hostbuf(self.h)
        self.host = np.ctypeslib.as_array(buf, shape=(max_samples,))   # pinned

        # Ordinary malloc'd memory. This is what DMA writes into — never
        # let the FPGA write directly into `host`. See the module docstring.
        self.stage = np.empty(max_samples, np.uint16)

        self.spec  = np.zeros(self.nbins, np.float32)
        self.tmin  = np.zeros(trace_width, np.float32)
        self.tmax  = np.zeros(trace_width, np.float32)
        self.stats = np.zeros(5, np.float32)
        self.times = np.zeros(5, np.float32)
        self.window = 0
        self.enbw = float(_l.adc_win_enbw(self.h))

    @property
    def dma_target(self):
        """The array a DMA read should write into."""
        return self.host if DIRECT_DMA else self.stage

    def load(self, nbytes):
        """Move staged DMA data into the pinned buffer. No-op in direct mode."""
        if DIRECT_DMA:
            return
        n = nbytes // 2
        np.copyto(self.host[:n], self.stage[:n])

    def set_window(self, win):
        """win: name or id from WINDOWS. Rebuilds the window table and the
        amplitude normalisation (coherent gain) on the device."""
        wid = WINDOWS.get(win, win) if isinstance(win, str) else int(win)
        if wid not in WINDOW_NAMES:
            raise ValueError(f"unknown window {win!r}; use {list(WINDOWS)}")
        if _l.adc_set_window(self.h, wid) != 0:
            raise RuntimeError("adc_set_window failed")
        self.window = wid
        self.enbw = float(_l.adc_win_enbw(self.h))
        return self.enbw

    def _p(self, a):
        return a.ctypes.data_as(_F)

    def process(self, nsamples, max_frames=64, trace_n=0):
        """trace_n: samples the min/max envelope covers (0 = all of nsamples).
        Decoupled from the FFT so changing Welch depth no longer changes the
        time-trace span or shading."""
        nf = ctypes.c_int()
        rc = _l.adc_process(self.h, nsamples, max_frames, trace_n,
                            self._p(self.spec), self._p(self.tmin), self._p(self.tmax),
                            self._p(self.stats), self._p(self.times), ctypes.byref(nf))
        if rc != 0:
            raise RuntimeError(f"adc_process rc={rc}")
        self.nframes = nf.value
        return self.spec

    def close(self):
        if getattr(self, "h", None):
            self.host = None
            self.stage = None
            _l.adc_destroy(self.h)
            self.h = None


class FastC2H:
    """Direct read() on the XDMA C2H node into a preallocated buffer.

    UNSAFE — freezes the machine even with a plain/ordinary destination
    buffer (bisect_dma.py stage S4). Not used by server.py; kept only for
    bisect_dma.py A/B testing against the known-good ad9643.ddr_read_samples()
    path. See module docstring before using this anywhere else.
    """

    def __init__(self, dev="/dev/xdma0_c2h_0"):
        self.fd = os.open(dev, os.O_RDONLY)

    def read_into(self, arr, nbytes, addr=0):
        if nbytes > arr.nbytes:
            raise ValueError(
                f"read of {nbytes} B into a {arr.nbytes} B buffer would "
                f"overrun — the DMA engine does not bounds-check")
        os.lseek(self.fd, addr, os.SEEK_SET)
        mv = memoryview(arr).cast("B")[:nbytes]
        got, off = 0, 0
        while got < nbytes:
            n = os.readv(self.fd, [mv[off:]])
            if n <= 0:
                break
            got += n; off += n
        return got

    def close(self):
        os.close(self.fd)
