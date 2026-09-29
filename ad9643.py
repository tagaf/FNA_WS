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

import contextlib, mmap, os, signal, subprocess, tempfile, time
import numpy as np

USER_DEV = "/dev/xdma0_user"
C2H_DEV  = "/dev/xdma0_c2h_0"
H2C_DEV  = "/dev/xdma0_h2c_0"
TOOLS    = os.path.expanduser("~/dma_ip_drivers/XDMA/linux-kernel/tools")

REG_START    = 0x00   # RW bit0 start (edge 0->1), bit1 stream mode
REG_SPEED    = 0x04   # RW Speed_Set -- must be 0, see sample_rate()
REG_CHANNEL  = 0x08   # RW bits[1:0] ChannelSel; write only while stopped
REG_NSAMPLES = 0x0C   # RW DataNum; never 0
REG_FINISH   = 0x10   # RO bit0 = ddr_wr_finish (Adc_Finish)
# --- added by the streaming bitstream (2026-09-28). Verified on hardware the
# day it landed: reg5 counts at exactly 65.5 ms/segment, reg6 bit0 asserts as
# the ring passes 16 unacked segments, and clearing reg0 freezes reg5.
REG_SEGCNT   = 0x14   # RO segments completely written since the last start
REG_FLAGS    = 0x18   # RO/W1C bit0 stream_overrun, bit1 fifo_overflow
REG_SEGACK   = 0x1C   # RW segments the host has finished reading

# reg0 bits
START_BIT  = 0x1      # rising edge arms a capture
STREAM_BIT = 0x2      # 0 = block capture, 1 = continuous streaming

# reg6 bits
FLAG_OVERRUN  = 0x1   # sticky; host fell behind. ANY write to 0x18 clears it
FLAG_FIFO_OVF = 0x2   # sticky; samples dropped before DDR. Cleared by a start

# --- design ID 0xAD964302 (2026-09-29) adds reg8..reg15 -------------------
REG_SPI_CMD    = 0x20   # RW [31] read, [20:8] addr, [7:0] data; write starts
REG_SPI_STATUS = 0x24   # RO [0] busy, [15:8] read data
REG_ADC_STATUS = 0x28   # RO/W  status bits below; ANY write clears the ramp
                        #       error counters and the overrange flags
REG_RAMP_ERR_A = 0x2C   # RO ramp test-pattern errors, channel A (saturating)
REG_RAMP_ERR_B = 0x30   # RO ramp test-pattern errors, channel B (saturating)
REG_DELAY      = 0x34   # RW write [8:0] tap to load; read [8:0] last
                        #    requested, [24:16] tap read back from the line
REG_RSVD14     = 0x38   # RO 0
REG_DESIGN_ID  = 0x3C   # RO design ID

DESIGN_ID = 0xAD964302          # this interface revision

# reg10 status bits
ST_CLK_LOCKED   = 1 << 0
ST_OVERRANGE_A  = 1 << 1        # sticky
ST_OVERRANGE_B  = 1 << 2        # sticky
ST_IDELAY_READY = 1 << 3

# reg8 fields
SPI_READ_BIT = 1 << 31
SPI_BUSY     = 1 << 0
DELAY_TAPS   = 512              # IDELAY range, ~2.2 ps per tap

# The AXI-Lite file now decodes 6 address bits: measured 2026-09-29, reg16..31
# read back identical to reg0..reg15 (reg20 == reg4 == 1 while idle). It was
# 5 bits and aliased every 32 bytes on the previous build, which is why reg8+
# could not exist there. A future reg16+ needs another widening.
REG_ALIAS_BYTES = 64

# 14-bit TWO'S COMPLEMENT, right-aligned in a 16-bit little-endian word.
# Established from the vendor's own client (pcie_client_sw/mainwindow.cpp),
# which sign-extends with ((int16_t)(x<<2))>>2 and scales by
# ADC_FS_VOLTAGE/ADC_MAX_CODE. Reading the codes as unsigned makes a signal
# near zero appear to jump between ~0 and ~16383 at every zero crossing.
# AD9643 datasheet Table 11 (two's complement, the power-up DEFAULT):
#   VIN+ - VIN- = -0.875 V -> -8192
#                  0       ->     0
#                 +0.875 V -> +8191
# so 8192 codes correspond to 0.875 V PEAK; "1.75 V p-p input span" is the
# full differential swing. The vendor client uses 1.75/8192 and is therefore
# a factor of two out; this uses the datasheet value.
ADC_FS_VOLTS_PP = 1.75         # full-scale differential swing, peak-to-peak
ADC_FS_VOLTS = ADC_FS_VOLTS_PP / 2.0         # 0.875 V peak
ADC_MAX_CODE = 8192.0
VOLT_SCALE = ADC_FS_VOLTS / ADC_MAX_CODE     # 106.8 uV per code


# --------------------------------------------------------- output inversion
# MEASURED 2026-09-29. Register 0x14 reads 0x05 on this board -- the power-up
# default; nothing in this stack writes it (adc_wr refuses 0x14 outright).
# Bits[1:0]=01 is two's complement, and bit 2 is SET, which on this hardware
# means the ADC inverts its digital output. Confirmed with the converter's own
# reference patterns, which have known values:
#
#   0x0D   pattern           expected   raw sign-extend   -x-1
#   0x01   midscale short          0        -1              0
#   0x02   +full scale         +8191     -8192          +8191
#   0x03   -full scale         -8192     +8191          -8192
#
# All three match only under `-x - 1`. This also explains the "decrementing"
# ADC ramp reported on 2026-09-29: an INCREMENTING ramp reads backwards
# through an inverter.
#
# NOTE the datasheet register table says "Output invert: 1 = normal (default),
# 0 = inverted" -- the opposite polarity to what the part actually does. The
# measurement wins; see HOST_INTERFACE.md.
ADC_INVERT_BIT = 0x04          # 0x14 bit 2

# Set from 0x14 when an Adc is opened, so a differently-configured board is
# decoded correctly instead of assuming this one's state.
OUTPUT_INVERT = False


def to_signed(u):
    """uint16 DDR words -> signed 14-bit codes, WITHOUT output inversion.

    This is the raw converter-code decode and is deliberately left alone: the
    FPGA's internal test counter (ChannelSel 0) is generated in fabric and
    never passes through the ADC's output inverter, so it must NOT be flipped.
    For real ADC data use adc_signed().
    """
    import numpy as _np
    return ((_np.asarray(u).astype(_np.int32) ^ 0x2000) - 0x2000)


def adc_signed(u, invert=None):
    """uint16 DDR words -> signed codes as the ANALOG INPUT saw them.

    Applies the output inversion when the converter is configured for it.
    `invert=None` uses the flag read from 0x14 at open.
    """
    import numpy as _np
    inv = OUTPUT_INVERT if invert is None else bool(invert)
    if not inv:
        return to_signed(u)
    # -x-1 on the sign-extended value is exactly a bitwise NOT of the 14-bit
    # code, so invert the code and sign-extend once rather than doing both.
    return to_signed(_np.asarray(u).astype(_np.int32) ^ 0x3FFF)


def to_signed_i16(u, out=None, invert=None):
    """uint16 DDR words -> signed 14-bit codes, staying in int16.

    to_signed() promotes to int32, doubling the memory traffic and allocating
    4 bytes per sample. At display record lengths that is the difference
    between a few milliseconds and a hundred: the live server was calling
    to_signed() four times per channel (mean/std/min/max) on a STRIDED view,
    which at 8.192 M samples per channel meant ~262 MB of allocation and
    strided reads per frame -- measured as 175 ms of "DMA" time that was not
    DMA at all.

    Shift left 2 then arithmetic-shift right 2: bit 13 lands in the sign bit
    and comes back replicated, which is sign extension from bit 13 with no
    widening.
    """
    if (OUTPUT_INVERT if invert is None else bool(invert)):
        u = np.asarray(u) ^ 0x3FFF
    x = np.asarray(u).astype(np.int16, copy=True) if out is None else out
    if out is not None:
        np.copyto(out, np.asarray(u).view(np.int16))
    np.left_shift(x, 2, out=x)
    np.right_shift(x, 2, out=x)
    return x


def to_volts(u, invert=None):
    return adc_signed(u, invert) * VOLT_SCALE


BYTES_PER_SAMPLE = 2
SAMPLE_GRANULARITY = 256              # samples per 512-byte AXI burst
# Mode 3 packs one 32-bit word per sample clock, so 128 sample CLOCKS already
# fill the same 512-byte burst. Below the granularity the final partial burst
# is never issued and Adc_Finish never asserts.
SAMPLE_GRANULARITY_DUAL = 128

def granularity(channel):
    return SAMPLE_GRANULARITY_DUAL if channel == CH_BOTH else SAMPLE_GRANULARITY

# RESOLVED 2026-09-28 on hardware: DataNum counts sample CLOCKS in EVERY
# mode, including ch_sel=3. The spec (section 6.2) is right; server.py's
# "DataNum = samples*2 in dual mode" convention, carried since 2026-09-08,
# was wrong.
#
# Measured by timing block captures against both hypotheses (the FSM runs for
# DataNum sample clocks, so the elapsed time reads the units directly):
#
#   DataNum written   elapsed   if clocks   if words
#       1,048,576     4185 us     4194 us    2097 us
#       2,097,152     8365 us     8389 us    4194 us
#       4,194,304    16725 us    16777 us    8389 us
#   (single-channel control at the same DataNum: 4175 / 8345 us)
#
# Why it went unnoticed: doubling DataNum makes the FPGA capture twice as
# long, and server.py then reads back only the first half -- which IS the
# requested record, just at half the achievable frame rate. It only turns
# into corruption once 2*N*4 exceeds the 500 MB window and the writer wraps
# onto its own record (at N > 65,536,000 pairs), which the phase-noise
# capture size of 67,108,864 pairs did by 12,582,912 bytes.
DUAL_DATANUM_IN_WORDS = False

def datanum_for(nsamples, channel):
    """Value to write to reg3 for `nsamples` sample clocks on `channel`."""
    if channel == CH_BOTH and DUAL_DATANUM_IN_WORDS:
        return nsamples * 2
    return nsamples
# 500 MB DDR window. The FPGA writer WRAPS at the end (fifo_to_axi4.v:
#   else if(m_axi_awaddr >= WR_AXI_BYTE_ADDR_END)
#       m_axi_awaddr <= WR_AXI_BYTE_ADDR_BEGIN;
# so it is a ring buffer, not the linear one the original handoff described.
# Irrelevant for one-shot capture (the address is reset per trigger), but it
# matters for any future streaming mode -- see NOTES.md #34.
WR_WINDOW_BYTES = 524288000
MAX_SAMPLES = WR_WINDOW_BYTES // BYTES_PER_SAMPLE
BASE_CLOCK_HZ = 250e6

# ---------------------------------------------------------------- streaming
# The 500 MB window is carved into 16 equal segments. reg5 counts segments
# whose data is completely in DDR; the host acks with reg7.
SEG_BYTES = 32768000              # 0x01F40000, 4 KB aligned
NSEG = 16
SEG_ADDR = lambda n: (n % NSEG) * SEG_BYTES

# The writer runs up to 8 bursts ahead of the last COMPLETED segment, so it
# begins overwriting slot n%16 slightly before reg5 reaches n+16. A segment is
# therefore only trustworthy if reg5, re-read AFTER its DMA finishes, is at
# most n+MAX_LAG. Usable ring depth is 15, not 16. reg6 bit0 is a summary
# flag; this per-segment test is the authoritative one.
MAX_LAG = 14
RING_USABLE = 15

# Samples per segment, per ChannelSel. Mode 3 emits one 32-bit word (A+B) per
# sample clock, so a segment holds half as many sample CLOCKS.
def seg_samples(channel):
    return SEG_BYTES // 4 if channel == CH_BOTH else SEG_BYTES // 2

# ---------------------------------------------------------------- AD9643
ADC_CHIP_ID      = 0x01   # must read 0x82
ADC_CHIP_GRADE   = 0x02   # bits 5:4 = 00 for 250 MSPS
ADC_DCS          = 0x09   # bit0 duty-cycle stabiliser -- MUST stay enabled
ADC_TEST_MODE    = 0x0D   # 0x00 normal, 0x0F ramp
ADC_OUTPUT_MODE  = 0x14   # bits 1:0 format -- MUST stay two's complement
ADC_DCO_DELAY    = 0x17   # bit7 enable, bits 4:0 delay, (v+1)*100 ps
ADC_TRANSFER     = 0xFF   # write 0x01 to latch the shadowed 0x08..0x20, 0x3A

ADC_TEST_NORMAL  = 0x00
ADC_TEST_RAMP    = 0x0F

# Shadowed registers need a transfer to take effect (spec 9.1).
ADC_SHADOWED = set(range(0x08, 0x21)) | {0x3A}

CH_TEST_RAMP, CH_A, CH_B, CH_BOTH = 0, 1, 2, 3
VALID_CHANNELS = (CH_TEST_RAMP, CH_A, CH_B, CH_BOTH)

# ch_sel=3 drives ad_out_comb = {2'd0, ch_A, 2'd0, ch_B} into a separate
# 32-bit FIFO (fifo_comb), muxed into the same AXI writer -- so both channels
# are captured over the SAME time window, which is the point of the mode.
# Little-endian: the low 16 bits (channel B) land first, so as uint16 the
# stream is B,A,B,A,... EXPERIMENTAL: an earlier attempt at ch_sel=3 timed
# out, and the HDL's byte accounting for this mode is ambiguous
# (burst_num is computed the same as for 16-bit mode). Verify against
# hardware before trusting the de-interleave order.
BOTH_INTERLEAVE = ("A", "B")   # vendor client: raw[2i]=A, raw[2i+1]=B


# --------------------------------------------------- ADC ramp verification
# MEASURED 2026-09-29, not the spec's figure. The spec (section 9.4) says
# ~2.2 ps per IDELAY tap; sweeping the ADC's DCO output delay by a known
# amount and watching how far the transition band moved gives 6.25 ps, from
# three consistent intervals (100->400 ps: 48 taps; 400->800 ps: 64 taps;
# 100->800 ps: 112 taps). The difference matters: it is the conversion from
# eye width in taps to real timing margin, and at 2.2 it understates the
# board's margin by 2.8x.
PS_PER_TAP = 6.25
UI_TAPS = 2000.0 / PS_PER_TAP        # 250 MSPS DDR -> 2 ns unit interval


def ramp_deviations(codes):
    """Deviations from the ramp law the ADC ACTUALLY produces.

    A host-side cross-check of the FPGA's ramp checker, kept as a second
    opinion rather than a workaround.

    It was a workaround: the checker used to assert a +1-per-conversion ramp,
    and this ADC emits both channels identical, each value held for 2
    conversions, DECREMENTING by 1 -- perfectly regular (0 deviations in
    1,048,576 samples) but not what the checker tested, so reg11/reg12 read
    1.000 errors/sample at every tap. (A half-rate capture was ruled out as
    the cause: on real data lag-1 autocorrelation is -0.61/-0.15 and A==B on
    only 0.02% of samples, where duplication would give +0.5 and 100%.) The
    2026-09-29 bitstream rewrote the checker to test the second difference and
    the two now agree.

    This fits (hold, direction, phase) to the capture and counts mismatches,
    so it measures the same thing -- outside the eye the data stops being
    predictable -- without assuming which way the ramp runs.

    Returns {"errors", "samples", "hold", "dir", "phase"}.
    """
    import numpy as _np
    r = (_np.asarray(codes) & 0x3FFF).astype(_np.int32).ravel()
    if r.size < 8:
        return {"errors": 0, "samples": int(r.size), "hold": 0, "dir": 0,
                "phase": 0}
    i = _np.arange(r.size)
    best = None
    for hold in (2, 1):
        for dr in (-1, 1):
            for ph in range(hold):
                e = (int(r[0]) + dr * ((i + ph) // hold)) & 0x3FFF
                bad = int(_np.count_nonzero(e != r))
                if best is None or bad < best[0]:
                    best = (bad, hold, dr, ph)
    return {"errors": best[0], "samples": int(r.size), "hold": best[1],
            "dir": best[2], "phase": best[3]}


# ------------------------------------------------------------ stored tap
# The FPGA does NOT retain the IDELAY tap across reconfiguration -- after
# every bitstream load reg13 reverts to the build default. Whatever opens the
# device is therefore responsible for re-applying the measured value, or the
# capture silently runs at a sampling point nobody chose. tools/eye_scan.py
# --save writes it here; Adc.__init__ applies it.
CONFIG_PATH = os.path.expanduser("~/.adc_capture.conf")


def load_stored_tap(path=CONFIG_PATH):
    """Stored IDELAY tap, or None. Never raises: a missing or corrupt config
    must not stop the instrument opening."""
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                k, v = (x.strip() for x in line.split("=", 1))
                if k == "idelay_tap":
                    t = int(v, 0)
                    return t if 0 <= t < DELAY_TAPS else None
    except (OSError, ValueError):
        pass
    return None


def save_stored_tap(tap, path=CONFIG_PATH):
    if not (0 <= int(tap) < DELAY_TAPS):
        raise ValueError(f"tap {tap} out of range 0..{DELAY_TAPS-1}")
    lines, seen = [], False
    try:
        with open(path) as f:
            for line in f:
                if line.split("#", 1)[0].strip().startswith("idelay_tap"):
                    lines.append(f"idelay_tap = {int(tap)}\n"); seen = True
                else:
                    lines.append(line)
    except OSError:
        pass
    if not seen:
        lines.append("# ADC data-delay tap measured by tools/eye_scan.py.\n")
        lines.append("# The FPGA loses this on every reconfiguration.\n")
        lines.append(f"idelay_tap = {int(tap)}\n")
    with open(path, "w") as f:
        f.writelines(lines)
    return int(tap)


class AdcSpiTimeout(RuntimeError):
    """reg9 bit0 never cleared. The SPI engine is in the FPGA, so this means
    the fabric is wedged or the ADC is not clocking -- not a wiring fault,
    which shows up instead as a wrong chip ID."""


class CaptureTimeout(RuntimeError):
    """Adc_Finish never asserted. Usually a bad set_sample_num/Channel_Set,
    or no DCO back from the ADC mezzanine."""


class DmaTimeout(RuntimeError):
    """The vendor dma_from_device/dma_to_device CLI did not finish in time.

    Seen intermittently at ALL transfer sizes, not just large ones: first
    observed near MAX_SAMPLES during testing 2026-08-26 (same capture
    succeeded in ~2.7s once, didn't return within 90+ seconds twice, no
    error/child process/CPU use -- see NOTES.md), then again the same day
    on ordinary 2 MB default-config captures during normal continuous use
    (~3 times in ~3000 frames; a fresh 4500-frame run right after couldn't
    reproduce it on demand). So it's size-independent and genuinely rare
    -- not something tied to extreme settings. Root cause still not
    understood. This bounds the wait so a recurrence fails loudly and
    recoverably instead of wedging the acquisition loop forever."""


def sample_rate(speed=0):
    """Always 250 Msps. `Speed_Set` does NOT decimate -- see NOTES.md #26.

    Confirmed from the HDL: `speed_ctrl` divides `adc_data_en`, but that
    signal only gates the SAMPLE COUNTER in wr_ddr_ctrl. The FIFO write
    enable is `dvalid` (= `ad_sample_en`), which is high on every adc_clk
    during ADC_SAMPLE, so DDR receives full-rate samples either way. The
    capture merely takes (Speed_Set+1)x longer, because `write_ddr_done`
    additionally waits for `fifo_empty` and the FIFO keeps being fed until
    the sampling FSM stops.

    Treating fs as 250 MHz/(Speed_Set+1) put every frequency axis at
    Speed_Set>0 out by exactly that factor.
    """
    return BASE_CLOCK_HZ


class Adc:
    def __init__(self, user_dev=USER_DEV, c2h_dev=C2H_DEV, tap=None):
        """`tap`: None = use the stored config, an int = force that tap,
        False = do not touch the delay line at all."""
        self.c2h_dev = c2h_dev
        self._fd = os.open(user_dev, os.O_RDWR | os.O_SYNC)
        self._mm = mmap.mmap(self._fd, 4096, mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE, offset=0)
        self._r = np.frombuffer(self._mm, dtype='<u4')
        self.design_id = self.rd(REG_DESIGN_ID)
        self.output_invert = None
        self.output_mode = None
        if self.design_id == DESIGN_ID:
            try:
                self.output_mode = self.adc_rd(ADC_OUTPUT_MODE)
                self.output_invert = bool(self.output_mode & ADC_INVERT_BIT)
                globals()["OUTPUT_INVERT"] = self.output_invert
            except Exception:
                pass
        self.applied_tap = None
        if tap is not False:
            self.apply_stored_tap(tap)

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
            ("finish", REG_FINISH), ("segcnt", REG_SEGCNT),
            ("flags", REG_FLAGS), ("segack", REG_SEGACK))}

    # ------------------------------------------------------- design ID gate
    @property
    def has_adc_ctl(self):
        """True if reg8..reg15 (SPI, ramp checker, delay) exist."""
        return self.design_id == DESIGN_ID

    def require_design_id(self, what="this feature"):
        """reg8..reg15 read 0 on older bitstreams, so silence is NOT proof of
        absence: reg15 aliased onto reg7 (host_seg_ack) when the decoder was
        5 bits wide, and reg7 reads 0 while idle. Only the exact design ID
        distinguishes 'register present and zero' from 'register absent'."""
        if not self.has_adc_ctl:
            raise RuntimeError(
                f"{what} needs FPGA design ID 0x{DESIGN_ID:08X}; reg15 reads "
                f"0x{self.design_id:08X}. On the previous bitstream reg8..15 "
                f"aliased onto reg0..7, so using them would silently drive the "
                f"capture registers instead. Load the matching bitstream.")

    # ------------------------------------------------------------- ADC SPI
    SPI_TIMEOUT = 0.05          # a transfer is ~3.3 us; this is 15000x that

    def _spi_wait(self, why):
        t0 = time.monotonic()
        while self.rd(REG_SPI_STATUS) & SPI_BUSY:
            if time.monotonic() - t0 > self.SPI_TIMEOUT:
                raise AdcSpiTimeout(
                    f"ADC SPI busy for >{self.SPI_TIMEOUT*1e3:.0f} ms {why} "
                    f"(reg9=0x{self.rd(REG_SPI_STATUS):08X}). Check the FMC "
                    f"card is seated and the ADC has its clock.")

    def adc_spi(self, read, addr, data=0):
        """One 3-wire SPI byte transfer through reg8/reg9 (spec 9.1)."""
        self.require_design_id("ADC SPI access")
        if not (0 <= addr <= 0x1FFF):
            raise ValueError(f"ADC register address 0x{addr:X} out of range")
        if not (0 <= data <= 0xFF):
            raise ValueError(f"ADC data 0x{data:X} is not a byte")
        self._spi_wait("before the transfer")
        self.wr(REG_SPI_CMD, ((SPI_READ_BIT if read else 0)
                              | ((addr & 0x1FFF) << 8) | (data & 0xFF)))
        self._spi_wait("during the transfer")
        return (self.rd(REG_SPI_STATUS) >> 8) & 0xFF

    def adc_rd(self, addr):
        return self.adc_spi(1, addr)

    def adc_wr(self, addr, data, force=False):
        """Write one ADC register.

        Two registers are refused here rather than trusted to call sites,
        because getting either wrong silently invalidates everything else:

          0x09 bit0  duty-cycle stabiliser. The FPGA's static timing analysis
                     assumes it is on; clearing it moves the capture window
                     and every sample becomes suspect.
          0x14       output format. The host decodes two's complement
                     (to_signed/to_signed_i16 and section 4 of the spec);
                     changing it makes every code wrong without any error.

        `force` exists only so a deliberate experiment can say so explicitly.
        """
        if not force:
            if addr == ADC_DCS and not (data & 0x01):
                raise ValueError(
                    "refusing to clear the duty-cycle stabiliser (0x09 bit0): "
                    "the FPGA capture timing depends on it. Pass force=True "
                    "only if you mean to invalidate the timing.")
            if addr == ADC_OUTPUT_MODE:
                raise ValueError(
                    "refusing to write the output format register (0x14): the "
                    "host decodes two's complement. Change ad9643.to_signed* "
                    "and the spec section 4 decoding first, then pass "
                    "force=True.")
        return self.adc_spi(0, addr, data)

    def adc_transfer(self):
        """Latch the shadowed registers (0x08..0x20, 0x3A)."""
        return self.adc_spi(0, ADC_TRANSFER, 0x01)

    def adc_wr_transfer(self, addr, data, force=False):
        """Write and, if the register is shadowed, latch it."""
        r = self.adc_wr(addr, data, force=force)
        if addr in ADC_SHADOWED:
            self.adc_transfer()
        return r

    def adc_selftest(self):
        """Chip ID + speed grade. Confirms the SPI wiring (CSB=LA19_P,
        SCLK=LA19_N, SDIO=LA20_P from the FMC schematics)."""
        cid = self.adc_rd(ADC_CHIP_ID)
        grade = self.adc_rd(ADC_CHIP_GRADE)
        return {"chip_id": cid, "chip_id_ok": cid == 0x82,
                "grade_raw": grade, "grade_bits": (grade >> 4) & 0x3,
                "grade_ok": ((grade >> 4) & 0x3) == 0}

    # --------------------------------------------------------- ADC status
    def adc_status(self):
        v = self.rd(REG_ADC_STATUS)
        return {"raw": v,
                "clk_locked": bool(v & ST_CLK_LOCKED),
                "overrange_a": bool(v & ST_OVERRANGE_A),
                "overrange_b": bool(v & ST_OVERRANGE_B),
                "idelay_ready": bool(v & ST_IDELAY_READY)}

    def ramp_clear(self):
        """Any write to reg10 clears the ramp counters and overrange flags."""
        self.require_design_id("the ramp checker")
        self.wr(REG_ADC_STATUS, 0)

    def ramp_errors(self):
        self.require_design_id("the ramp checker")
        return (self.rd(REG_RAMP_ERR_A), self.rd(REG_RAMP_ERR_B))

    def apply_stored_tap(self, tap=None):
        """Load the configured IDELAY tap, if this bitstream has reg13.

        Silent no-op on an older design ID: the tap is an optimisation, and
        refusing to open the device over it would be worse than running at
        the build default.
        """
        if not self.has_adc_ctl:
            return None
        t = load_stored_tap() if tap is None else int(tap)
        if t is None:
            return None
        try:
            self.set_tap(t)
            self.applied_tap = t
            return t
        except Exception:
            return None

    # ------------------------------------------------------------ ramp mode
    @contextlib.contextmanager
    def ramp_mode(self):
        """Put the ADC into ramp test-pattern output for the duration.

        ALWAYS restores 0x0D = 0x00 -- on success, on exception, and on
        SIGINT/SIGTERM. Leaving the converter in ramp mode would make every
        subsequent capture a sawtooth that looks like a real (very periodic)
        signal, with nothing on screen to say why, so the restore is wired to
        the signals as well as to the `finally`.
        """
        self.require_design_id("the ramp test pattern")
        restored = [False]

        def restore(*_a):
            if not restored[0]:
                restored[0] = True
                try:
                    self.adc_wr_transfer(ADC_TEST_MODE, ADC_TEST_NORMAL)
                except Exception:
                    pass

        prev = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev[sig] = signal.getsignal(sig)
            except (ValueError, OSError):
                pass

        def handler(signum, frame):
            restore()
            old = prev.get(signum)
            if callable(old):
                old(signum, frame)
            else:
                raise KeyboardInterrupt

        for sig in list(prev):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                prev.pop(sig, None)
        try:
            self.adc_wr_transfer(ADC_TEST_MODE, ADC_TEST_RAMP)
            if self.adc_rd(ADC_TEST_MODE) != ADC_TEST_RAMP:
                raise RuntimeError(
                    f"0x0D did not take the ramp value (reads "
                    f"0x{self.adc_rd(ADC_TEST_MODE):02X}); is the transfer "
                    f"register working?")
            yield self
        finally:
            restore()
            for sig, old in prev.items():
                try:
                    signal.signal(sig, old)
                except (ValueError, OSError):
                    pass

    # ---------------------------------------------------------- data delay
    def set_tap(self, tap):
        """Load an IDELAY tap (0..511). The load takes ~1 us."""
        self.require_design_id("the ADC data delay")
        if not (0 <= tap < DELAY_TAPS):
            raise ValueError(f"tap {tap} out of range 0..{DELAY_TAPS-1}")
        self.wr(REG_DELAY, tap)
        time.sleep(20e-6)
        return self.get_tap()

    def get_tap(self):
        v = self.rd(REG_DELAY)
        return {"raw": v, "requested": v & 0x1FF, "readback": (v >> 16) & 0x1FF}

    # --------------------------------------------------------------- capture
    @staticmethod
    def _validate(nsamples, channel, speed, streaming=False):
        if channel not in VALID_CHANNELS:
            raise ValueError(
                f"Channel_Set={channel} invalid; use 1=A, 2=B, 3=both, "
                f"0=test ramp")
        if speed != 0:
            # Not merely useless -- actively harmful. It does not decimate
            # (NOTES.md #26.1 / sample_rate()), it only stretches the FSM's
            # sample counter, and the 2026-09-28 spec section 8.1 states a
            # non-zero value corrupts block lengths outright.
            raise ValueError(
                f"Speed_Set must be 0 (got {speed}). It does NOT decimate -- "
                f"fs is always 250 Msps -- and a non-zero value corrupts "
                f"block lengths on this bitstream.")
        g = granularity(channel)
        if nsamples <= 0:
            raise ValueError("nsamples must be > 0 (DataNum = 0 never completes)")
        if nsamples < g:
            raise ValueError(
                f"nsamples must be >= {g} for channel {channel}; smaller "
                f"captures never fill an AXI burst and hang")
        if nsamples % g:
            raise ValueError(
                f"nsamples must be a multiple of {g} for channel {channel} "
                f"(got {nsamples}); the last partial burst is never written "
                f"and Adc_Finish never asserts")
        if streaming:
            # The 500 MB DDR window is a RING in stream mode, not the record:
            # the FPGA overwrites it every 16 segments while the host drains
            # it into RAM, so a record is bounded by the RAM ring the host
            # allocated, not by DDR. Applying the block-capture cap here made
            # any stream record longer than 131,072,000 pairs (0.52 s) fail
            # with a message about a window that is not the limit. The engine
            # clamps to what is actually resident in the ring.
            return
        cap = MAX_SAMPLES // 2 if channel == CH_BOTH else MAX_SAMPLES
        if nsamples > cap:
            # A+B emits one 32-bit word per sample clock into the SAME
            # 500 MiB window (NOTES.md #26.3), so a dual capture reaches the
            # end of it at half the sample count. Past that the FPGA wraps
            # and the readback is a record spliced onto itself -- which looks
            # entirely plausible in a spectrum, and is a step discontinuity
            # in a demodulated phase.
            raise ValueError(
                f"nsamples exceeds the {WR_WINDOW_BYTES} B capture window "
                f"(max {cap}" + (" in dual-channel mode, where every sample "
                                 "clock writes two words)" if channel == CH_BOTH
                                 else ")"))

    def recover(self):
        """Re-arm on a known-good setting to walk the FSM out of a hang.

        Clears STREAM_BIT first: a recover() that left streaming armed would
        restart the ring instead of running the one short block capture this
        is meant to be."""
        self.wr(REG_START, 0)
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
        expected = nsamples / BASE_CLOCK_HZ          # speed is forced to 0
        if timeout is None:
            timeout = max(0.5, expected * 4 + 0.5)

        # ChannelSel/Speed_Set/DataNum are sampled without synchronisation
        # (spec section 8.2), so they must be written while stopped -- which
        # clearing REG_START below also guarantees for the restart edge.
        self.wr(REG_START, 0)               # stopped, block mode (bit1 clear)
        self.wr(REG_SPEED, 0)
        self.wr(REG_CHANNEL, channel)
        self.wr(REG_NSAMPLES, datanum_for(nsamples, channel))
        self.wr(REG_START, START_BIT)       # edge-triggered: 0 -> 1

        # reg4 keeps the PREVIOUS capture's state for well under a
        # microsecond after the start edge, and it idles HIGH -- so polling
        # immediately reads a stale "finished" and hands back a buffer that
        # was never filled. Measured 2026-09-28: with this wait in place a
        # 1,048,576-sample counter capture reported 4216 us against a
        # predicted 4194 us, i.e. the flag was correctly low on first poll.
        time.sleep(10e-6)

        t0 = time.monotonic()
        self.finish_asserted = True
        while not self.finished:
            if time.monotonic() - t0 > timeout:
                if channel == CH_BOTH:
                    # Known defect on the PREVIOUS bitstream: Adc_Finish never
                    # asserted in dual mode at any depth, though the data was
                    # fine (NOTES.md #30; the vendor client blind-waits too).
                    # The 2026-09-28 spec says it does assert, merely a few us
                    # early. Rather than pick a side, fall back to the proven
                    # blind wait and RECORD which path ran, so the behaviour
                    # can be reported back to the FPGA side.
                    self.finish_asserted = False
                    time.sleep(max(0.02, expected * 4 + 0.02))
                    break
                st = self.regs()
                self.recover()
                raise CaptureTimeout(
                    f"Adc_Finish low after {timeout:.3f}s "
                    f"(expected ~{expected*1e3:.3f} ms), regs={st}. "
                    f"FSM stuck in ADC_SAMPLE - no DCO from the mezzanine, "
                    f"or an invalid nsamples/channel. FSM was reset.")
        self.elapsed = time.monotonic() - t0

        if channel == CH_BOTH:
            # Adc_Finish can lead the last few bursts (at most 8 x 512 B) into
            # DDR by a few microseconds (spec section 5). Reading immediately
            # returns a tail of stale bytes.
            time.sleep(20e-6)

        # One 32-bit word per sample clock in dual mode: twice the uint16s.
        nwords = nsamples * 2 if channel == CH_BOTH else nsamples
        return ddr_read_samples(nwords, addr=addr, dev=self.c2h_dev)


# ------------------------------------------------------------------ DDR4 I/O
def _tool(name, dev, addr, nbytes, path, write=False, timeout=None):
    # Measured throughput on this path is 0.3-0.7 GB/s; floor the estimate
    # far below that (50 MB/s) so the bound stays generous for legitimate
    # large transfers while still catching a genuine stall in finite time.
    # The +5.0s flat margin this used to carry was sized for the largest
    # transfers and made small/typical captures (a few ms normally) wait a
    # full 5s before recovering from a stall -- a very visible stutter in a
    # ~50 fps continuous loop. +1.5s is still >100x the normal small-capture
    # time; 1.0s absolute floor in case nbytes is tiny.
    if timeout is None:
        timeout = max(1.0, nbytes / 50e6 + 1.5)
    try:
        subprocess.run([f"{TOOLS}/{name}", "-d", dev, "-a", str(addr),
                        "-s", str(nbytes), "-f", path],
                       check=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise DmaTimeout(
            f"{name} did not finish within {timeout:.1f}s for {nbytes:,} B "
            f"on {dev}. See ad9643.DmaTimeout docstring / NOTES.md.")

# Persistent transfer file on tmpfs. /dev/shm keeps the vendor tool's output
# entirely in RAM: with the old NamedTemporaryFile on /tmp (disk-backed here),
# every frame pushed megabytes through the page cache to storage, and periodic
# writeback flushes showed up as multi-hundred-ms stalls in the live loop.
_XFER_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
_XFER_PATH = os.path.join(_XFER_DIR, f"adc_dma_{os.getpid()}.bin")

def _xfer_cleanup():
    try:
        os.unlink(_XFER_PATH)
    except OSError:
        pass


def _xfer_reap_stale():
    """Remove transfer files left by processes that no longer exist.

    These live on tmpfs (RAM). A SIGTERM (systemctl stop) or the /restart
    button's os._exit() both bypass atexit, so each restart used to strand
    one file sized to that session's largest capture - up to 500 MB each,
    accumulating until reboot."""
    import glob, re
    for p in glob.glob(os.path.join(_XFER_DIR, "adc_dma_*.bin")) + \
             glob.glob(os.path.join(_XFER_DIR, "adc_helper_*.buf")):
        m = re.search(r"_(\d+)\.(bin|buf)$", p)
        if not m:
            continue
        pid = int(m.group(1))
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)          # still alive: leave it alone
        except ProcessLookupError:
            try:
                os.unlink(p)
            except OSError:
                pass
        except PermissionError:
            pass                     # someone else's live process

import atexit
atexit.register(_xfer_cleanup)


def ddr_read_samples(nsamples, addr=0, dev=C2H_DEV):
    nbytes = nsamples * BYTES_PER_SAMPLE
    _tool("dma_from_device", dev, addr, nbytes, _XFER_PATH)
    return np.fromfile(_XFER_PATH, dtype='<u2', count=nsamples)


def ddr_read_into(out, nsamples, addr=0, dev=C2H_DEV):
    """Like ddr_read_samples but into a preallocated uint16 array -
    no per-frame allocation. Returns samples actually read."""
    nbytes = nsamples * BYTES_PER_SAMPLE
    _tool("dma_from_device", dev, addr, nbytes, _XFER_PATH)
    with open(_XFER_PATH, "rb", buffering=0) as f:
        got = f.readinto(memoryview(out[:nsamples]).cast("B") if out.ndim == 1
                         else memoryview(out).cast("B")[:nbytes])
    return got // BYTES_PER_SAMPLE

def ddr_write(data, addr=0, dev=H2C_DEV):
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        data.tofile(f.name)
        _tool("dma_to_device", dev, addr, data.nbytes, f.name)
