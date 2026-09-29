#!/usr/bin/env python3
"""GPU spectrum engine (cuda/adcfft.cu) against closed-form expectations.

Needs the CUDA library; skips cleanly if it will not load (no GPU, not built).

Three bugs this file exists to keep fixed, all found 2026-09-28:

  1. DC and Nyquist were multiplied by 2 like every other bin. A real-input
     FFT folds each positive frequency onto its negative twin, hence the 2 --
     but DC and Nyquist are their own mirror and have no twin. A full-scale
     Nyquist tone read -6.227 dBFS against a true -12.247, i.e. +6.02 dB.
  2. Variance came from E[x^2]-mean^2 accumulated in float32, which cancels
     catastrophically once |mean| >> std -- the shape of every DC-coupled
     detector signal. Measured: mean 8000, true std 2.02 -> reported 4.09
     (+103%); mean 4000, std 5.01 -> 6.62 (+32%).
  3. Welch detrended ONCE globally instead of per segment (scipy's
     detrend='constant'). On a drifting signal each frame kept its own
     offset, which landed in bin 1: -15.25 dBFS against a true -33.69.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    import gpu
    _SKIP = None
except Exception as e:                      # no CUDA / not built
    _SKIP = str(e)

FS = 250e6
FAILS = []


def chk(name, cond, info=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{info}]" if info else ""))
    if not cond:
        FAILS.append(f"{name}: {info}")


def codes(x):
    q = np.clip(np.rint(np.asarray(x)).astype(np.int32), -8192, 8191)
    return (q & 0x3FFF).astype(np.uint16)


def s14(u):
    return ((u.astype(np.int32) ^ 0x2000) - 0x2000).astype(np.float64)


def run(u, nfft, max_frames=1, window="hann"):
    n = u.size
    sp = gpu.Spectrum(nfft=nfft, max_samples=max(1 << 22, n), trace_width=1024)
    sp.set_window(window)
    sp.host[:n] = u
    spec = sp.process(n, max_frames=max_frames, trace_n=n).copy()
    out = (spec, sp.stats.copy(), sp.nframes,
           sp.tmin.copy(), sp.tmax.copy(), sp.tmean.copy())
    sp.close()
    return out


def main():
    if _SKIP:
        print(f"SKIP: CUDA spectrum engine unavailable ({_SKIP})")
        return 0

    print("tone amplitude, every window (exactly on a bin -> no scalloping)")
    nfft = 65536
    t = np.arange(nfft)
    for win in ("hann", "blackman-harris", "flattop", "rect"):
        for amp in (4096.0, 1024.0, 100.0):
            spec, *_ = run(codes(amp * np.cos(2 * np.pi * 1000 * t / nfft)),
                           nfft, 1, win)
            exp = 20 * np.log10(amp / 8192.0)
            got = float(spec.max())
            chk(f"{win} amplitude {amp:.0f} codes", abs(got - exp) < 0.05,
                f"{got:.3f} vs {exp:.3f} dBFS")

    print("\nDC and Nyquist must NOT be doubled")
    n2 = 8192
    amp = 2000.0
    spec, *_ = run(codes(amp * np.cos(np.pi * np.arange(n2))), n2, 1, "rect")
    exp = 20 * np.log10(amp / 8192.0)
    got = float(spec[n2 // 2])
    chk("Nyquist bin not doubled", abs(got - exp) < 0.05,
        f"{got:.3f} vs {exp:.3f} dBFS")

    print("\nvariance survives |mean| >> std")
    rng = np.random.default_rng(3)
    for mean, std in ((185, 7.2), (4000, 5.0), (8000, 2.0)):
        u = codes(rng.normal(mean, std, 1 << 22))
        x = s14(u)
        _, st, *_ = run(u, 8192)
        err = abs(st[3] - x.std()) / max(x.std(), 1e-9)
        chk(f"std at mean={mean}, std={std}", err < 0.01,
            f"{st[3]:.4f} vs {x.std():.4f} ({100*err:.2f}%)")

    print("\nWelch detrends per frame, not globally")
    nfft, frames = 8192, 16
    n = nfft * frames
    tt = np.arange(n)
    u = codes(2000 * np.sin(2 * np.pi * 1.3 * tt / n)
              + 200 * np.cos(2 * np.pi * (nfft // 5) * tt / nfft)
              + rng.normal(0, 5, n))
    spec, *_ = run(u, nfft, frames, "hann")
    x = s14(u).reshape(frames, nfft)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / (nfft - 1))
    P = (np.abs(np.fft.rfft((x - x.mean(axis=1, keepdims=True)) * w,
                            axis=1)) ** 2).mean(axis=0)
    a = 2 * np.sqrt(P) / (0.5 * nfft)
    a[0] *= 0.5; a[-1] *= 0.5
    ref = 20 * np.log10(np.maximum(a, 1e-9) / 8192.0)
    chk("bin 1 free of inter-frame DC", abs(spec[1] - ref[1]) < 0.1,
        f"{spec[1]:.2f} vs {ref[1]:.2f} dBFS")
    chk("whole spectrum matches the reference",
        np.abs(spec - ref).max() < 0.05, f"max {np.abs(spec-ref).max():.4f} dB")

    print("\ntime-trace envelope is exact")
    n = 1 << 20
    x = 3000 * np.sin(2 * np.pi * 50 * np.arange(n) / n) + rng.normal(0, 20, n)
    x[123456] = 8000.0                     # a lone spike must survive
    u = codes(x); xt = s14(u)
    _, _, _, tmin, tmax, tmean = run(u, 8192)
    tw = tmin.size
    idx = (np.arange(tw + 1).astype(np.int64) * n) // tw
    rmn = np.array([xt[idx[i]:idx[i+1]].min() for i in range(tw)])
    rmx = np.array([xt[idx[i]:idx[i+1]].max() for i in range(tw)])
    chk("envelope min/max exact",
        np.abs(tmin - rmn).max() == 0 and np.abs(tmax - rmx).max() == 0)
    chk("single-sample spike preserved", tmax.max() == 8000)

    print()
    if FAILS:
        print("FAILURES:")
        for f in FAILS:
            print("  -", f)
        return 1
    print("ALL FFT CHECKS PASSED")
    return 0


def test_fft():
    """pytest entry point. The suite collects `test_*` functions, so without
    this the file was importable, runnable by hand, and silently contributed
    ZERO tests to `pytest tests/` -- which is worse than not existing."""
    import pytest
    if _SKIP:
        pytest.skip(f"CUDA spectrum engine unavailable ({_SKIP})")
    assert main() == 0, "; ".join(FAILS)


if __name__ == "__main__":
    sys.exit(main())
