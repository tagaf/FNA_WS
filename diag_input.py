#!/usr/bin/env python3
"""Characterise what the ADC actually returns, for a known input condition.

Run with the server STOPPED (it holds the same single-instance lock):

    sudo systemctl stop adc-capture
    python3 diag_input.py --label "shorted"
    python3 diag_input.py --label "floating"
    python3 diag_input.py --label "50R terminated"
    sudo systemctl start adc-capture

Each run saves baselines/input_<label>.json so conditions can be compared.

Why these particular statistics: the internal ramp (channel 0) proves the
DDR/DMA path end to end, but it is generated INSIDE the FPGA, so it says
nothing about the ADC -> FPGA LVDS capture. A bit-alignment error there
would leave the ramp perfect while corrupting real samples. The checks below
are chosen to expose that:

* code histogram -- a noisy input should fill a smooth, roughly Gaussian
  band. Regularly spaced gaps ("missing codes") are the classic signature of
  a stuck or mis-ordered bit (IEEE 1241 histogram test).
* per-bit toggle rate -- for a small signal near a DC level, low bits should
  toggle ~50% and high bits should be static. A high bit toggling wildly, or
  a low bit frozen, points at capture alignment rather than at the analogue
  side.
* even/odd sample split -- the AD9643 is DDR: samples arrive on both clock
  edges. If the capture picks the wrong edge or swaps the two lanes, the
  even and odd subsequences differ systematically.
* lag-1/2 autocorrelation -- interleave errors show up as strong alternation
  that white converter noise does not have.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ad9643 as A


def analyse(d, fs):
    d = d.astype(np.int64)
    n = d.size
    out = {"n": int(n), "min": int(d.min()), "max": int(d.max()),
           "mean": float(d.mean()), "std": float(d.std()),
           "median": float(np.median(d))}
    out["vs_midscale"] = out["mean"] - 8192.0

    # per-bit activity
    bits = []
    for b in range(16):
        frac = float(((d >> b) & 1).mean())
        bits.append(round(frac, 4))
    out["bit_one_fraction"] = bits
    out["bits_stuck_low"] = [b for b in range(16) if bits[b] == 0.0]
    out["bits_stuck_high"] = [b for b in range(16) if bits[b] == 1.0]
    out["bits_active"] = [b for b in range(16) if 0.02 < bits[b] < 0.98]

    # histogram over the occupied span: look for missing codes
    lo, hi = int(d.min()), int(d.max())
    span = hi - lo + 1
    if span <= 1 << 16:
        h = np.bincount(d - lo, minlength=span)
        occupied = int((h > 0).sum())
        out["code_span"] = span
        out["codes_occupied"] = occupied
        out["codes_missing_pct"] = round(100.0 * (1 - occupied / span), 2)
        # a stuck/mis-ordered bit leaves gaps at a fixed period
        gaps = np.flatnonzero(h == 0)
        if gaps.size > 2:
            dg = np.diff(gaps)
            vals, cnts = np.unique(dg, return_counts=True)
            out["gap_period_mode"] = int(vals[np.argmax(cnts)])
            out["gap_period_share"] = round(float(cnts.max() / dg.size), 3)

    # DDR edge / interleave
    ev, od = d[0::2], d[1::2]
    out["even_mean"], out["odd_mean"] = float(ev.mean()), float(od.mean())
    out["even_odd_delta"] = out["even_mean"] - out["odd_mean"]
    out["even_std"], out["odd_std"] = float(ev.std()), float(od.std())
    x = (d - d.mean()).astype(np.float64)
    den = float((x * x).sum())
    for lag in (1, 2, 4):
        out[f"acf_lag{lag}"] = round(float((x[:-lag] * x[lag:]).sum() / den), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="input condition, e.g. 'shorted'")
    ap.add_argument("-n", "--nsamples", type=int, default=1 << 20)
    ap.add_argument("-c", "--channels", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--speed", type=int, default=0)
    ap.add_argument("--save-raw", action="store_true",
                    help="also write the raw samples as .npy")
    a = ap.parse_args()

    try:
        adc = A.Adc()
    except A.DeviceMissing as e:
        print(e, file=sys.stderr)
        return 1

    res = {"label": a.label, "ts": time.time(), "nsamples": a.nsamples,
           "speed": a.speed, "channels": {}}
    try:
        for ch in a.channels:
            d = adc.capture(a.nsamples, channel=ch, speed=a.speed)
            r = analyse(d, A.sample_rate(a.speed))
            res["channels"][str(ch)] = r
            print(f"\n=== channel {ch} ({a.label}) ===")
            print(f"  mean {r['mean']:9.2f}   std {r['std']:7.2f}   "
                  f"min {r['min']:6d}   max {r['max']:6d}")
            print(f"  vs mid-scale (8192): {r['vs_midscale']:+.1f} codes")
            print(f"  bits active {r['bits_active']}  stuck-low {r['bits_stuck_low']}"
                  f"  stuck-high {r['bits_stuck_high']}")
            if "codes_missing_pct" in r:
                print(f"  code span {r['code_span']}, occupied {r['codes_occupied']}"
                      f" ({r['codes_missing_pct']}% missing)"
                      + (f", gaps mostly every {r['gap_period_mode']}"
                         f" ({100*r['gap_period_share']:.0f}%)"
                         if "gap_period_mode" in r else ""))
            print(f"  even/odd mean {r['even_mean']:.2f} / {r['odd_mean']:.2f}"
                  f"  (delta {r['even_odd_delta']:+.2f})")
            print(f"  acf lag1 {r['acf_lag1']:+.4f}  lag2 {r['acf_lag2']:+.4f}"
                  f"  lag4 {r['acf_lag4']:+.4f}")
            if a.save_raw:
                p = os.path.join(HERE, "baselines",
                                 f"raw_{a.label.replace(' ','_')}_ch{ch}.npy")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, d)
                print(f"  raw samples -> {p}")
    finally:
        adc.close()

    out = os.path.join(HERE, "baselines",
                       f"input_{a.label.replace(' ', '_')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved -> {out}")
    print("\nWhat to look for:")
    print("  * shorted should sit near a stable DC with a few LSB of noise;")
    print("    which DC it is tells you the coding (8192 => offset binary)")
    print("  * regular gaps in the histogram, a stuck low bit, or a large")
    print("    even/odd delta all point at the ADC->FPGA capture, not the")
    print("    analogue input")
    return 0


if __name__ == "__main__":
    sys.exit(main())
