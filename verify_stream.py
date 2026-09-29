#!/usr/bin/env python3
"""Automatic verification of the PCIE_AD9643 streaming build (spec section 7).

Items 1-3 run unattended and set the exit code; items 4-5 need a signal
generator and live in tools/sine_check.py.

  1  block capture, ChannelSel=0: the 14-bit counter increments by 1 mod 16384
     everywhere. Validates the writer and the section 4 byte ordering.
  2  streaming, ChannelSel=0: the same continuity check ACROSS segment
     boundaries, reg6 clear throughout, sustained rate reported. Also reads
     segments back out of the RAM ring to prove the ring is coherent.
  3  overrun: stall the reader for >1 s and confirm BOTH the section 6.4
     validity rule and reg6 bit0 report the loss, then that a write to 0x18
     clears bit0.

Usage:  python3 verify_stream.py [-t SECONDS] [--channels N]
Exit 0 = all pass.  Stop the adc-capture service first; this takes the same
single-instance lock the server does.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad9643 as A
from stream import Stream, CounterCheckSink, summarise

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"\n  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


# ------------------------------------------------------------------ item 1
def test_block_counter(adc, nsamples=1048576):
    print(f"\n=== 1. block capture, ChannelSel=0, DataNum={nsamples:,} ===")
    d = adc.capture(nsamples, channel=A.CH_TEST_RAMP)
    if d.size != nsamples:
        return record("block counter", False, f"got {d.size} of {nsamples} samples")
    a = d & 0x3FFF
    diff = np.empty(a.size - 1, np.uint16)
    np.subtract(a[1:], a[:-1], out=diff)
    np.bitwise_and(diff, 0x3FFF, out=diff)
    bad = int(np.count_nonzero(diff != 1))
    high = int((d & 0xC000).any())
    print(f"  elapsed {adc.elapsed*1e6:.0f} us (expected {nsamples/A.BASE_CLOCK_HZ*1e6:.0f})")
    print(f"  Adc_Finish asserted: {adc.finish_asserted}")
    print(f"  ramp violations {bad}, bits 15:14 set: {bool(high)}, first codes {a[:6].tolist()}")
    return record("block counter (spec 7.1)", bad == 0 and not high,
                  f"{nsamples:,} samples, {bad} violations")


# ------------------------------------------------------------------ item 2
def test_stream_counter(adc, seconds, nchan, ram_gb):
    print(f"\n=== 2. streaming, ChannelSel=0, {seconds:.0f} s ===")
    nslots = max(3, int(ram_gb * 1000**3) // A.SEG_BYTES)
    sink = CounterCheckSink()
    st = Stream(adc, A.CH_TEST_RAMP, nchan=nchan, nslots=nslots, ram_ring=True)
    try:
        st.run(sink=sink, duration=seconds)
        ok = summarise(st, sink, label="streaming counter")
        # the ring must hand back what it took in
        ring_ok, checked = True, 0
        if st.ring is not None and st.ring.n_resident >= 2:
            lo, hi = st.ring.resident
            for seg in (lo, (lo + hi) // 2, hi):
                a = st.ring.read_segment(seg) & 0x3FFF
                d = np.empty(a.size - 1, np.uint16)
                np.subtract(a[1:], a[:-1], out=d); np.bitwise_and(d, 0x3FFF, out=d)
                if int(np.count_nonzero(d != 1)):
                    ring_ok = False
                checked += 1
            # consecutive segments must join
            j = st.ring.read_segment(lo)[-1] & 0x3FFF
            k = st.ring.read_segment(lo + 1)[0] & 0x3FFF
            if (int(k) - int(j)) & 0x3FFF != 1:
                ring_ok = False
            print(f"  ring readback      {checked} segments re-checked + "
                  f"segment join {lo}->{lo+1}: {'ok' if ring_ok else 'BROKEN'}")
        return record("streaming counter (spec 7.2)", ok and ring_ok,
                      f"{sink.checked:,} samples, {sink.violations} violations, "
                      f"{st.n_lost} segments lost")
    finally:
        st.close()


# ------------------------------------------------------------------ item 3
class StallSink:
    """Sleeps once, part-way in, to make the host fall behind on purpose.

    The sink runs inline on the reader thread, so sleeping here IS a stalled
    reader -- no extra machinery needed. 1.5 s against a 0.98 s ring slack
    (15 usable segments x 65.5 ms) is comfortably past the cliff.
    """

    def __init__(self, at_segment=6, seconds=1.5):
        self.at, self.seconds, self.done = at_segment, seconds, False

    def __call__(self, views, n):
        if n >= self.at and not self.done:
            self.done = True
            print(f"  ... stalling the reader {self.seconds}s at segment {n}")
            time.sleep(self.seconds)


def test_overrun(adc, nchan):
    print(f"\n=== 3. overrun: stall the reader past the ring slack ===")
    sink = StallSink()
    st = Stream(adc, A.CH_TEST_RAMP, nchan=nchan, nslots=4, ram_ring=False)
    try:
        st.run(sink=sink, duration=6.0)
        summarise(st, None, label="overrun probe")
        rule = st.n_lost > 0                       # spec 6.4 per-segment rule
        flag = bool(st.flags_seen & A.FLAG_OVERRUN)  # reg6 bit0
        print(f"  section 6.4 validity rule detected loss : {rule} "
              f"({st.n_lost} segments)")
        print(f"  reg6 bit0 (stream_overrun) asserted     : {flag}")
        before = adc.rd(A.REG_FLAGS)
        adc.wr(A.REG_FLAGS, 0)                     # any write clears bit0
        after = adc.rd(A.REG_FLAGS)
        cleared = bool(before & A.FLAG_OVERRUN) and not (after & A.FLAG_OVERRUN)
        print(f"  reg6 {before:#06x} -> write 0x18 -> {after:#06x}   "
              f"bit0 cleared: {cleared}")
        if not sink.done:
            return record("overrun (spec 7.3)", False, "the stall never ran")
        return record("overrun (spec 7.3)", rule and flag and cleared,
                      f"validity rule={rule}, reg6 bit0={flag}, clear={cleared}")
    finally:
        st.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-t", "--seconds", type=float, default=10.0,
                   help="duration of the streaming test (default 10)")
    p.add_argument("--channels", type=int, default=2, choices=(1, 2, 4),
                   help="C2H channels per segment (default 2)")
    p.add_argument("--ram-gb", type=float, default=4.0,
                   help="RAM ring for test 2 (default 4)")
    p.add_argument("--tap", type=int, default=None,
                   help="IDELAY tap to load first (default: stored config)")
    p.add_argument("--only", type=int, choices=(1, 2, 3),
                   help="run a single test")
    a = p.parse_args()

    with A.Adc(tap=a.tap) as adc:
        if a.only in (None, 1):
            test_block_counter(adc)
        if a.only in (None, 2):
            test_stream_counter(adc, a.seconds, a.channels, a.ram_gb)
        if a.only in (None, 3):
            test_overrun(adc, a.channels)

    print("\n" + "=" * 62)
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:38} {detail}")
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("=" * 62)
    print(f"  {len(RESULTS)-nfail}/{len(RESULTS)} passed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
