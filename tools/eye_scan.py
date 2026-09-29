#!/usr/bin/env python3
"""ADC capture-timing eye scan (spec 9.4), design ID 0xAD964302.

Sweeps the FPGA IDELAY (reg13) with the ADC emitting its ramp, counting
errors per channel at each tap. The widest run of taps with ZERO errors on
both channels is the eye; its width x 6.25 ps (measured) is the real capture-timing
margin of the board, and its centre is the tap to keep.

  python3 tools/eye_scan.py                       step 4, 10 ms dwell
  python3 tools/eye_scan.py --step 2 --dwell 20   finer, slower
  python3 tools/eye_scan.py --apply               write the centre tap at the end
  python3 tools/eye_scan.py --dco-sweep           also sweep the ADC DCO delay

The tap is NOT retained by the FPGA across reconfiguration -- `--save` writes
it to the config the capture tools read at start-up (see ad9643.stored_tap).

The ADC is returned to normal output on every exit path, including Ctrl-C.
"""
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ad9643 as A
import diag


def sweep(adc, step, dwell, lo=0, hi=A.DELAY_TAPS, quiet=False):
    """FPGA-checker sweep via diag.eye_scan."""
    return _sweep(adc, "fpga", step, lo, hi, dwell=dwell, quiet=quiet)


def host_sweep(adc, step, nsamples, lo=0, hi=A.DELAY_TAPS, quiet=False):
    """Captured-data sweep via diag.eye_scan."""
    return _sweep(adc, "host", step, lo, hi, samples=nsamples, quiet=quiet)


def _sweep(adc, method, step, lo, hi, dwell=0.010, samples=65536, quiet=False):
    state = {"n": 0}

    def cb(frac, msg, **kw):
        state["n"] += 1
        if not quiet and state["n"] % 16 == 0:
            print(f"    {msg}", flush=True)
    r = diag.eye_scan(adc, method=method, lo=lo, hi=hi, step=step,
                      dwell=dwell, samples=samples, progress=cb)
    _sweep.last = r
    return [tuple(row) for row in r.rows]


def widest_zero_run(rows, idx):
    """(start_tap, end_tap, n_points) of the widest zero-error run."""
    best = cur = None
    for tap, a, b in rows:
        z = (rows and (a, b)[idx - 1] == 0) if False else ((a if idx == 1 else b) == 0)
        if z:
            cur = (tap, tap, 1) if cur is None else (cur[0], tap, cur[2] + 1)
            if best is None or cur[2] > best[2]:
                best = cur
        else:
            cur = None
    return best


def both_zero_run(rows):
    best = cur = None
    for tap, a, b in rows:
        if a == 0 and b == 0:
            cur = (tap, tap, 1) if cur is None else (cur[0], tap, cur[2] + 1)
            if best is None or cur[2] > best[2]:
                best = cur
        else:
            cur = None
    return best


def plot(rows, step):
    """One character per swept tap: '.' zero errors, digits = log10(errors)."""
    def row(sel):
        s = ""
        for _, a, b in rows:
            e = a if sel == "A" else b
            if e == 0:
                s += "."
            elif e >= 0xFFFFFFFF:
                s += "#"
            else:
                d = min(9, max(1, len(str(e))))
                s += str(d)
        return s
    lo, hi = rows[0][0], rows[-1][0]
    print(f"\n  taps {lo}..{hi} step {step}   '.'=no errors  1-9=log10(errors)  '#'=saturated")
    for sel in ("A", "B"):
        r = row(sel)
        for off in range(0, len(r), 100):
            tag = f"  {sel} {rows[off][0]:>3}" if off == 0 else f"      {rows[off][0]:>3}"
            print(f"{tag} |{r[off:off+100]}|")


def report(rows, step, tap0):
    plot(rows, step)
    print()
    for name, idx in (("A", 1), ("B", 2)):
        r = widest_zero_run(rows, idx)
        if r:
            print(f"  channel {name}: widest zero-error run taps {r[0]}..{r[1]} "
                  f"({r[2]} points, {r[1]-r[0]+step} taps, "
                  f"~{(r[1]-r[0]+step)*A.PS_PER_TAP:.0f} ps), centre {(r[0]+r[1])//2}")
        else:
            print(f"  channel {name}: NO tap with zero errors")
    both = both_zero_run(rows)
    print()
    if both:
        width = both[1] - both[0] + step
        centre = (both[0] + both[1]) // 2
        print(f"  BOTH channels: taps {both[0]}..{both[1]}  width {width} taps "
              f"~{width*A.PS_PER_TAP:.0f} ps  centre {centre}")
        edge = (both[0] <= 0) or (both[1] >= A.DELAY_TAPS - step)
        if edge:
            print(f"  WARNING: the run touches the end of the IDELAY range, so the")
            print(f"  eye centre is outside it. Shift it with the ADC DCO delay")
            print(f"  (0x17: bit7 enable, bits4:0 delay, (v+1)*100 ps) and re-scan.")
        return centre, width, edge
    print("  BOTH channels: NO tap with zero errors on both -- no eye at all.")
    return None, 0, False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--dwell", type=float, default=0.010,
                   help="seconds per tap (default 0.010 = 2.5 M samples/channel)")
    p.add_argument("--lo", type=int, default=0)
    p.add_argument("--hi", type=int, default=A.DELAY_TAPS)
    p.add_argument("--csv", default="eye_scan.csv")
    p.add_argument("--apply", action="store_true", help="load the centre tap at the end")
    p.add_argument("--save", action="store_true",
                   help="also store the centre tap so capture tools apply it at start-up")
    p.add_argument("--host-check", action="store_true",
                   help="count errors from CAPTURED DATA instead of the FPGA's "
                        "reg11/reg12. Since the 2026-09-29 bitstream both work "
                        "and agree to ~2 taps; this is the cross-check.")
    p.add_argument("--host-samples", type=int, default=65536,
                   help="samples per tap in --host-check mode (default 65536)")
    p.add_argument("--dco-sweep", action="store_true",
                   help="sweep the ADC DCO delay (0x17) too, 100 ps per step")
    a = p.parse_args()

    with A.Adc() as adc:
        try:
            adc.require_design_id("tools/eye_scan.py")
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr); return 2
        st = adc.adc_selftest()
        if not st["chip_id_ok"]:
            print(f"error: chip ID 0x{st['chip_id']:02X} != 0x82", file=sys.stderr)
            return 2

        tap0 = adc.get_tap()
        stat = adc.adc_status()
        print(f"  default tap before the scan: requested {tap0['requested']}, "
              f"readback {tap0['readback']}")
        print(f"  reg10 = 0x{stat['raw']:08X}  clk_locked={stat['clk_locked']} "
              f"idelay_ready={stat['idelay_ready']}")
        if not stat["idelay_ready"]:
            print("  WARNING: IDELAYCTRL is NOT ready. The delay line cannot load "
                  "taps in this state, so every tap below is the same physical "
                  "sampling point and the scan cannot find an eye.")
        npts = len(range(a.lo, a.hi, a.step))
        print(f"\n  sweeping {npts} taps, {a.dwell*1e3:.0f} ms each "
              f"(~{npts*a.dwell:.0f} s)\n")

        rows = []
        try:
            with adc.ramp_mode():
                if a.host_check:
                    rows = host_sweep(adc, a.step, a.host_samples, a.lo, a.hi)
                else:
                    rows = sweep(adc, a.step, a.dwell, a.lo, a.hi)
                if a.dco_sweep:
                    print("\n  DCO delay sweep (ADC 0x17):")
                    for dco in range(0, 32, 4):
                        adc.adc_wr_transfer(A.ADC_DCO_DELAY, 0x80 | dco)
                        time.sleep(0.01)
                        r = sweep(adc, max(a.step, 8), a.dwell, quiet=True)
                        z = sum(1 for _, x, y in r if x == 0 and y == 0)
                        print(f"    0x17=0x{0x80|dco:02X} "
                              f"({(dco+1)*100:>4} ps): {z} taps with zero errors")
                    adc.adc_wr_transfer(A.ADC_DCO_DELAY, 0x00)
        finally:
            if rows:
                with open(a.csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["tap", "errors_a", "errors_b"])
                    w.writerows(rows)
                print(f"\n  wrote {a.csv} ({len(rows)} rows)")
            adc.wr(A.REG_DELAY, tap0["requested"])

        if not rows:
            return 1
        centre, width, edge = report(rows, a.step, tap0)
        if centre is not None and a.apply:
            t = adc.set_tap(centre)
            print(f"\n  loaded centre tap {centre} -> readback {t['readback']}")
        if centre is not None and a.save:
            A.save_stored_tap(centre)
            print(f"  saved tap {centre} to {A.CONFIG_PATH}")
        print(f"\n  0x0D = 0x{adc.adc_rd(A.ADC_TEST_MODE):02X} (0x00 = normal)")
        return 0 if centre is not None else 1


if __name__ == "__main__":
    sys.exit(main())
