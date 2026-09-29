#!/usr/bin/env python3
"""Spec section 7 items 4-5: channel identity and capture quality, from a sine.

Needs a signal generator. Feed one channel at a time, capture in ChannelSel=3,
and this reports which half of the interleave the tone actually landed in --
that is the item-4 answer, and it is the only way to tie "channel A" in the
data to a physical SMA on the mezzanine.

  item 4  channel identity: a tone on ONE input must appear in ONE half of the
          A0,B0,A1,B1 interleave and leave the other at noise. A tone in both
          means the de-interleave is wrong or the inputs are crosstalking.
  item 5  capture quality: least-squares sine fit; residual should be ~1-2 LSB
          rms with NO isolated outliers. Outliers point at ADC capture timing
          (spec section 8.3: this build misses its worst-case analysis by
          0.1-0.15 ns, and NOTES.md #37 has the local evidence).

Usage:
    # capture first, e.g.
    python3 stream.py -c 3 -t 10 --ram-gb 4 --dump /dev/shm/sine.bin
    # then
    python3 tools/sine_check.py /dev/shm/sine.bin --dual --fs 250e6
"""
import argparse, os, sys
import numpy as np
from scipy.signal.windows import blackmanharris

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ad9643 as A


def refine_freq(x, fs, f0, span_bins=1.5, steps=61, rounds=3):
    """Sub-bin frequency refinement by maximising |DTFT| near f0.

    The raw FFT bin is not good enough to fit against: at fs=250 MHz and a
    4 Mpoint transform a bin is 59.6 Hz, and 59.6 Hz of frequency error drifts
    1.6 rad of phase across a 1 Mpoint fit window -- which showed up as a
    1438 LSB rms residual on synthetic data whose true noise was 1.5 LSB.
    Three golden-ish refinement rounds bring the error to ~1e-3 bins, at which
    point the residual reflects the converter and not the estimator.

    Evaluated on a decimated copy: the peak is smooth on this scale, so a
    stride of 16 costs nothing in accuracy and makes the search ~16x cheaper.
    """
    step = max(1, x.size // (1 << 16))
    xs = x[::step].astype(np.float64)
    n = np.arange(xs.size, dtype=np.float64) * step
    df = fs / x.size * span_bins
    for _ in range(rounds):
        grid = np.linspace(f0 - df, f0 + df, steps)
        mag = np.abs(np.exp(-2j * np.pi * np.outer(grid / fs, n)) @ xs)
        f0 = float(grid[int(np.argmax(mag))])
        df *= 2.5 / steps
    return f0


def sine_fit(x, fs, f0):
    """Least-squares fit of b*cos + c*sin + a at f0. Returns (amp, resid)."""
    n = np.arange(x.size, dtype=np.float64)
    w = 2 * np.pi * f0 / fs
    M = np.column_stack((np.cos(w * n), np.sin(w * n), np.ones(x.size)))
    coef, *_ = np.linalg.lstsq(M, x, rcond=None)
    return float(np.hypot(coef[0], coef[1])), x - M @ coef


EXCL = 10          # bins excluded either side of a tone (Blackman-Harris
                   # main lobe is 8 bins wide; 10 covers it with margin)


def analyse(x, fs, name, fit_n=1 << 20, nharm=5):
    """FFT peak, SNR/SINAD/THD/SFDR, and a sine-fit residual.

    Blackman-Harris rather than Hanning, and +/-10 bins excluded rather than
    +/-3: with Hanning at +/-3 an off-bin tone leaks enough into the "noise"
    band to read 41.6 dB on synthetic data whose true SNR was 63 dB. The
    window choice is not cosmetic -- it sets the noise floor this tool can
    measure at all.
    """
    x = x.astype(np.float64)
    x -= x.mean()
    n = min(x.size, 1 << 22)
    n -= n % 2
    w = blackmanharris(n)
    S = np.abs(np.fft.rfft(x[:n] * w)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / fs)
    S[:EXCL] = 0.0                                  # kill DC and its skirt
    k = int(np.argmax(S))

    def band(centre):
        lo, hi = max(0, centre - EXCL), min(S.size, centre + EXCL + 1)
        return lo, hi

    lo, hi = band(k)
    sig = S[lo:hi].sum()
    rest = S.copy(); rest[lo:hi] = 0.0
    # SFDR is conventionally the largest spur INCLUDING harmonics, so take it
    # before the harmonic bins are removed from `rest` for the SNR split.
    sfdr = 10 * np.log10(sig / rest.max()) if rest.max() > 0 else float("inf")

    # harmonics fold about fs/2
    hpow, hbins = 0.0, []
    for m in range(2, nharm + 1):
        fh = (m * freqs[k]) % fs
        if fh > fs / 2:
            fh = fs - fh
        kh = int(round(fh / (fs / n)))
        if kh <= EXCL or kh >= S.size - EXCL or abs(kh - k) <= EXCL:
            continue
        a, b = band(kh)
        hpow += rest[a:b].sum(); rest[a:b] = 0.0
        hbins.append(kh)

    noise = rest.sum()
    snr = 10 * np.log10(sig / noise) if noise > 0 else float("inf")
    sinad = 10 * np.log10(sig / (noise + hpow)) if (noise + hpow) > 0 else float("inf")
    thd = 10 * np.log10(hpow / sig) if hpow > 0 else float("-inf")
    enob = (sinad - 1.76) / 6.02
    tone = snr > 10.0        # below this there is no tone, only noise

    print(f"\n  {name}:")
    print(f"    rms {x.std():8.2f} codes   ptp {np.ptp(x):8.0f}")
    print(f"    FFT peak  bin {k}  ->  {freqs[k]/1e6:.6f} MHz")
    if not tone:
        print(f"    SNR       {snr:8.2f} dB   <- NO TONE HERE, this is noise "
              f"({x.std():.1f} codes rms); SINAD/THD/SFDR omitted")
    else:
        print(f"    SNR       {snr:8.2f} dB   (harmonics excluded)")
        print(f"    SINAD     {sinad:8.2f} dB   -> ENOB {enob:5.2f} bits")
        print(f"    THD       {thd:8.2f} dB   ({len(hbins)} harmonics found)")
        print(f"    SFDR      {sfdr:8.2f} dB")

    m = min(fit_n, x.size)
    f_ref = refine_freq(x[:m], fs, freqs[k])
    print(f"    refined   {f_ref/1e6:.9f} MHz  "
          f"({(f_ref-freqs[k]):+.2f} Hz from the raw bin)")
    amp, resid = sine_fit(x[:m], fs, f_ref)
    r = resid.std()
    out = int(np.count_nonzero(np.abs(resid) > 6 * r))
    print(f"    sine fit  amplitude {amp:.1f} codes over {m:,} samples")
    print(f"    residual  {r:8.2f} LSB rms   max |{np.abs(resid).max():.0f}|")
    print(f"    outliers  {out:8d} beyond 6 sigma "
          f"({100*out/m:.4f}%)  <- item 5 wants ZERO isolated outliers")
    return {"freq": f_ref, "snr": snr, "sinad": sinad, "enob": enob,
            "resid_rms": r, "outliers": out, "rms": x.std(), "tone": tone}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="raw capture (uint16 little-endian)")
    p.add_argument("--dual", action="store_true",
                   help="ChannelSel=3 data: de-interleave A0,B0,A1,B1")
    p.add_argument("--fs", type=float, default=A.BASE_CLOCK_HZ)
    p.add_argument("--max-samples", type=int, default=1 << 24,
                   help="limit how much of the file is read (default 16 M)")
    a = p.parse_args()

    n = a.max_samples * (2 if a.dual else 1)
    d = np.fromfile(a.path, dtype='<u2', count=n)
    if d.size == 0:
        print(f"error: {a.path} is empty", file=sys.stderr)
        return 2
    print(f"{a.path}: {d.size:,} uint16 words, fs = {a.fs/1e6:.3f} MHz")
    if (d & 0xC000).any():
        print("  WARNING: bits 15:14 set -- not zero-padded 14-bit data")

    if a.dual:
        res = {"A": analyse(A.adc_signed(d[0::2]), a.fs, "channel A (even words)"),
               "B": analyse(A.adc_signed(d[1::2]), a.fs, "channel B (odd words)")}
        ra, rb = res["A"]["rms"], res["B"]["rms"]
        strong, weak = ("A", "B") if ra > rb else ("B", "A")
        ratio = max(ra, rb) / max(1e-9, min(ra, rb))
        print(f"\n  item 4 -- channel identity:")
        print(f"    tone is in channel {strong} ({max(ra,rb):.1f} codes rms); "
              f"{weak} sits at {min(ra,rb):.1f}")
        print(f"    amplitude ratio {ratio:.1f}x")
        if ratio < 10:
            print(f"    AMBIGUOUS: the tone appears in BOTH halves. Either the "
                  f"de-interleave is wrong, the inputs are crosstalking, or a "
                  f"splitter is feeding both.")
        else:
            print(f"    -> whichever SMA you drove is channel {strong} "
                  f"({'even' if strong=='A' else 'odd'} uint16 positions)")
        worst = max((r["outliers"] for r in res.values() if r["tone"]),
                    default=0)
    else:
        r = analyse(A.adc_signed(d), a.fs, "single channel")
        worst = r["outliers"] if r["tone"] else 0

    print(f"\n  item 5 -- capture quality: "
          f"{'PASS (no isolated outliers)' if worst == 0 else f'{worst} OUTLIERS -- see spec 8.3 / NOTES.md #37'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
