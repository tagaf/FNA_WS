#!/usr/bin/env python3
"""Capture a block of AD9643 samples and save / summarise them."""
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad9643 as A


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--nsamples", type=int, default=65536,
                   help="samples to capture; multiple of 256 (default 65536)")
    p.add_argument("-c", "--channel", type=int, default=1,
                   help="1=ADC A (default), 2=ADC B, 0=internal test ramp")
    p.add_argument("-s", "--speed", type=int, default=0,
                   help="rate divider: fs = 250 MHz/(speed+1); 0 = full rate")
    p.add_argument("-o", "--out", help="write samples to .npy or .bin")
    p.add_argument("--fft", action="store_true", help="report the FFT peak")
    p.add_argument("--timeout", type=float, default=None)
    a = p.parse_args()

    try:
        with A.Adc() as adc:
            d = adc.capture(a.nsamples, channel=a.channel, speed=a.speed,
                            timeout=a.timeout)
            el = adc.elapsed
    except (ValueError, A.CaptureTimeout) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    fs = A.sample_rate(a.speed)
    print(f"captured {len(d)} samples  ch={a.channel}  "
          f"fs={fs/1e6:.3f} Msps  in {el*1e3:.3f} ms")
    print(f"  codes: min {d.min()}  max {d.max()}  mean {d.mean():.2f}  "
          f"std {d.std():.2f}   (14-bit, 0..16383)")
    if a.channel != 0 and (d.max() >= 16383 or d.min() == 0):
        print("  warning: input may be clipping the converter")

    if a.fft:
        x = d.astype(np.float64) - d.mean()
        w = np.hanning(len(x))
        S = np.abs(np.fft.rfft(x * w))
        k = int(np.argmax(S[1:]) + 1)
        f = k * fs / len(x)
        print(f"  FFT peak: bin {k} -> {f/1e6:.6f} MHz  "
              f"({20*np.log10(S[k]/S[1:].sum()):.1f} dB rel. total)")

    if a.out:
        if a.out.endswith(".npy"):
            np.save(a.out, d)
        else:
            d.tofile(a.out)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
