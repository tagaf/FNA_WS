"""
AD9643 block-capture driver for the Puzhi xcku040 / XDMA design.

Memory map
----------
  /dev/xdma0_user  == BAR0 == AXI 0x40000000 == AXI_CMD register 0
  Captured samples land in DDR4 at AXI 0x0, read via /dev/xdma0_c2h_*

Findings established empirically on this hardware (the HDL sources are not
present on this machine; every constant below was measured, see NOTES.md):

  Speed_Set    fs = 250 MHz / (Speed_Set + 1).  0 => full rate, 250 Msps.
  Channel_Set  0 = internal test ramp (NOT the ADC)
               1 = ADC channel A
               2 = ADC channel B
               3 = invalid, hangs the capture FSM
  set_sample_num  1 count = one 16-bit sample = 2 bytes.
                  MUST be a multiple of 256 samples (= 512 B = 32 AXI beats).
                  Other values hang or silently return a rounded length.
  Packing      14-bit unsigned, right-aligned in 16-bit little-endian words,
               bits 15:14 always 0. Samples are strictly sequential in address
               order (8 per 128-bit beat). No channel interleaving - the
               channel is selected per capture.
"""

import mmap, os, time, subprocess, tempfile
import numpy as np

USER_DEV = "/dev/xdma0_user"
C2H_DEV  = "/dev/xdma0_c2h_0"
H2C_DEV  = "/dev/xdma0_h2c_0"
TOOLS    = os.path.expanduser("~/dma_ip_drivers/XDMA/linux-kernel/tools")

REG_START    = 0x00   # W  bit0, edge-triggered (0 -> 1)
REG_SPEED    = 0x04   # W
REG_CHANNEL  = 0x08   # W  bits[1:0]
REG_NSAMPLES = 0x0C   # W
REG_FINISH   = 0x10   # R  bit0 = ddr_wr_finish

BYTES_PER_SAMPLE = 2
SAMPLE_GRANULARITY = 256              # samples per 512-byte AXI burst
WR_WINDOW_BYTES = 524288000           # 500 MB linear window, no wrap
MAX_SAMPLES = WR_WINDOW_BYTES // BYTES_PER_SAMPLE
BASE_CLOCK_HZ = 250e6

CH_TEST_RAMP, CH_A, CH_B = 0, 1, 2
VALID_CHANNELS = (CH_TEST_RAMP, CH_A, CH_B)


class CaptureTimeout(RuntimeError):
    """Adc_Finish never asserted. Usually a bad set_sample_num/Channel_Set,
    or no DCO back from the ADC mezzanine."""


class DmaTimeout(RuntimeError):
    """The vendor dma_from_device/dma_to_device CLI did not finish in time.

    Seen intermittently on very large (near-MAX_SAMPLES) transfers during
    testing 2026-08-26: the same capture succeeded in ~2.7s on one run and
    did not return within 90+ seconds (no error, no child process, no CPU
    use -- see NOTES.md) on two others, with no reliable repro found. Root
    cause is not understood. This bounds the wait so a recurrence fails
    loudly and recoverably instead of wedging the acquisition loop forever."""


def sample_rate(speed=0):
    return BASE_CLOCK_HZ / (speed + 1)


class Adc:
    def __init__(self, user_dev=USER_DEV, c2h_dev=C2H_DEV):
        self.c2h_dev = c2h_dev
        self._fd = os.open(user_dev, os.O_RDWR | os.O_SYNC)
        self._mm = mmap.mmap(self._fd, 4096, mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE, offset=0)
        self._r = np.frombuffer(self._mm, dtype='<u4')

    def close(self):
        self._r = None                      # drop view before unmapping
        self._mm.close()
        os.close(self._fd)

    def __enter__(self):    return self
    def __exit__(self, *a): self.close()

    # ------------------------------------------------------------------ regs
    def rd(self, off):      return int(self._r[off >> 2])
    def wr(self, off, val): self._r[off >> 2] = np.uint32(val)

    @property
    def finished(self):     return bool(self.rd(REG_FINISH) & 1)

    def regs(self):
        return {n: self.rd(o) for n, o in (
            ("start", REG_START), ("speed", REG_SPEED),
            ("channel", REG_CHANNEL), ("nsamples", REG_NSAMPLES),
            ("finish", REG_FINISH))}

    # --------------------------------------------------------------- capture
    @staticmethod
    def _validate(nsamples, channel, speed):
        if channel not in VALID_CHANNELS:
            raise ValueError(
                f"Channel_Set={channel} invalid; use 1=A, 2=B, 0=test ramp "
                f"(3 hangs the capture FSM)")
        if speed < 0 or speed > 0xFFFFFFFF:
            raise ValueError("speed out of range")
        if nsamples < SAMPLE_GRANULARITY:
            raise ValueError(
                f"nsamples must be >= {SAMPLE_GRANULARITY}; smaller captures "
                f"never fill an AXI burst and hang")
        if nsamples % SAMPLE_GRANULARITY:
            raise ValueError(
                f"nsamples must be a multiple of {SAMPLE_GRANULARITY} "
                f"(got {nsamples}); other values hang or silently truncate")
        if nsamples > MAX_SAMPLES:
            raise ValueError(
                f"nsamples exceeds the {WR_WINDOW_BYTES} B capture window "
                f"(max {MAX_SAMPLES})")

    def recover(self):
        """Re-arm on a known-good setting to walk the FSM out of a hang."""
        self.wr(REG_SPEED, 0); self.wr(REG_CHANNEL, CH_TEST_RAMP)
        self.wr(REG_NSAMPLES, SAMPLE_GRANULARITY)
        self.wr(REG_START, 0); self.wr(REG_START, 1)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:
            if self.finished:
                return True
        return False

    def capture(self, nsamples, channel=CH_A, speed=0, timeout=None,
                addr=0, raw=False):
        """Run one block capture and return the samples as a numpy array.

        Returns uint16 with the 14-bit code right-aligned (raw=True), or
        float64 centred codes if raw=False is extended later. Currently both
        return the raw 14-bit codes - no calibration data exists to scale to
        volts (no analogue front end is fitted).
        """
        self._validate(nsamples, channel, speed)
        expected = nsamples * (speed + 1) / BASE_CLOCK_HZ
        if timeout is None:
            timeout = max(0.5, expected * 4 + 0.5)

        self.wr(REG_SPEED, speed)
        self.wr(REG_CHANNEL, channel)
        self.wr(REG_NSAMPLES, nsamples)
        self.wr(REG_START, 0)               # edge-triggered: force 0 -> 1
        self.wr(REG_START, 1)

        t0 = time.monotonic()
        while not self.finished:
            if time.monotonic() - t0 > timeout:
                st = self.regs()
                self.recover()
                raise CaptureTimeout(
                    f"Adc_Finish low after {timeout:.3f}s "
                    f"(expected ~{expected*1e3:.3f} ms), regs={st}. "
                    f"FSM stuck in ADC_SAMPLE - no DCO from the mezzanine, "
                    f"or an invalid nsamples/channel. FSM was reset.")
        self.elapsed = time.monotonic() - t0

        return ddr_read_samples(nsamples, addr=addr, dev=self.c2h_dev)


# ------------------------------------------------------------------ DDR4 I/O
def _tool(name, dev, addr, nbytes, path, write=False, timeout=None):
    # Measured throughput on this path is 0.3-0.7 GB/s; floor the estimate
    # far below that (50 MB/s) so the bound stays generous for legitimate
    # large transfers while still catching a genuine stall in finite time.
    if timeout is None:
        timeout = max(5.0, nbytes / 50e6 + 5.0)
    try:
        subprocess.run([f"{TOOLS}/{name}", "-d", dev, "-a", str(addr),
                        "-s", str(nbytes), "-f", path],
                       check=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise DmaTimeout(
            f"{name} did not finish within {timeout:.1f}s for {nbytes:,} B "
            f"on {dev}. See ad9643.DmaTimeout docstring / NOTES.md.")

def ddr_read_samples(nsamples, addr=0, dev=C2H_DEV):
    nbytes = nsamples * BYTES_PER_SAMPLE
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        _tool("dma_from_device", dev, addr, nbytes, f.name)
        d = np.fromfile(f.name, dtype='<u2')
    return d

def ddr_write(data, addr=0, dev=H2C_DEV):
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        data.tofile(f.name)
        _tool("dma_to_device", dev, addr, data.nbytes, f.name)
