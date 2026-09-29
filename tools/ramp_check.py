#!/usr/bin/env python3
"""AD9643 ramp test-pattern check (spec 9.3), design ID 0xAD964302.

The FPGA checks EVERY sample of BOTH channels against a +1 (mod 16384) step
while the ADC outputs its ramp, and counts mismatches in reg11/reg12. No
capture has to be running. This is a far stronger test than a sine fit: it
looks at every bit of every sample rather than at a residual.

  python3 tools/ramp_check.py -t 60          check for 60 s
  python3 tools/ramp_check.py -t 10 --tap 96 check at a specific IDELAY tap

The ADC is returned to normal output on every exit path, including Ctrl-C.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ad9643 as A
import diag


def show_status(adc, label):
    s = adc.adc_status()
    print(f"  {label}: reg10=0x{s['raw']:08X}  clk_locked={s['clk_locked']}  "
          f"overrange A/B={s['overrange_a']}/{s['overrange_b']}  "
          f"idelay_ready={s['idelay_ready']}")
    return s


def run(adc, seconds, progress=True):
    """Delegates to diag.ramp_check so the CLI and the web server cannot
    diverge on the pass criterion. Returns (errA, errB, status)."""
    def cb(frac, msg, **kw):
        if progress and "t=" in msg:
            print(f"    {msg}", flush=True)
    r = diag.ramp_check(adc, seconds=seconds, progress=cb)
    return r.data["errors_a"], r.data["errors_b"], r.data["status"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-t", "--seconds", type=float, default=10.0,
                   help="how long to accumulate (default 10)")
    p.add_argument("--tap", type=int, default=None,
                   help="load this IDELAY tap first (0..511)")
    a = p.parse_args()

    with A.Adc() as adc:
        try:
            adc.require_design_id("tools/ramp_check.py")
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        st = adc.adc_selftest()
        if not st["chip_id_ok"]:
            print(f"error: ADC chip ID reads 0x{st['chip_id']:02X}, not 0x82 -- "
                  f"SPI wiring is not confirmed, a ramp test would be "
                  f"meaningless", file=sys.stderr)
            return 2

        if a.tap is not None:
            t = adc.set_tap(a.tap)
            print(f"  loaded tap {a.tap} -> readback {t['readback']}")
        tap = adc.get_tap()
        print(f"  IDELAY tap: requested {tap['requested']}, "
              f"readback {tap['readback']}")
        show_status(adc, "before")

        print(f"\n  ramp on, accumulating for {a.seconds:.0f} s "
              f"({a.seconds*250e6:,.0f} samples per channel)...")
        errA, errB, stat = run(adc, a.seconds)

        print(f"\n  channel A errors: {errA:,}")
        print(f"  channel B errors: {errB:,}")
        print(f"  reg10=0x{stat['raw']:08X}  clk_locked={stat['clk_locked']}  "
              f"overrange A/B={stat['overrange_a']}/{stat['overrange_b']}  "
              f"idelay_ready={stat['idelay_ready']}")
        print(f"  0x0D restored to 0x{adc.adc_rd(A.ADC_TEST_MODE):02X} "
              f"(0x00 = normal ADC data)")

        ok = (errA == 0 and errB == 0)
        if not ok:
            n = a.seconds * 250e6
            print(f"\n  error rate A {errA/n:.3e}/sample, B {errB/n:.3e}/sample"
                  + ("   (0xFFFFFFFF = counter saturated)"
                     if 0xFFFFFFFF in (errA, errB) else ""))
        print(f"\n  {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
