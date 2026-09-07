#!/usr/bin/env python3
"""Unit tests for noise.py against synthetic spectra with known ground truth.

No hardware, no CUDA. Run: python3 tests/test_noise.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import noise

fails = []
def chk(name, cond, info=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{info}]" if info else ""))
    if not cond:
        fails.append(f"{name}: {info}")

def welch_noise(n, K, mean_db=-120.0, rng=None):
    """Welch-averaged power of Gaussian noise: Gamma(K, 1/K), in dB."""
    rng = rng or np.random.default_rng(0)
    p = rng.gamma(shape=K, scale=1.0 / K, size=n)
    return mean_db + 10.0 * np.log10(p)


print("floor estimation recovers the true MEAN (not median) power")
for K in (1, 8, 64):
    s = welch_noise(1 << 16, K, -120.0, np.random.default_rng(K))
    est = np.median(noise.estimate_floor(s, nframes=K))
    chk(f"K={K} floor within 0.3 dB", abs(est + 120.0) < 0.3, f"{est:.2f} dBFS")

print("\nCFAR false-alarm rate matches the requested Pfa")
for K, pfa in ((1, 1e-5), (16, 1e-5), (64, 1e-6)):
    n = 1 << 20
    s = welch_noise(n, K, -120.0, np.random.default_rng(100 + K))
    fl = noise.estimate_floor(s, nframes=K)
    pk = noise.detect_peaks(s, fl, bin_hz=1.0, nframes=K, pfa=pfa, max_peaks=10**6)
    exp = pfa * n
    chk(f"K={K} Pfa={pfa:g}: {len(pk)} alarms vs ~{exp:.0f} expected",
        0.15 * exp <= max(len(pk), 0.5) <= 4 * exp + 3, f"{len(pk)} vs {exp:.1f}")

print("\nsub-bin refinement recovers a known fractional offset")
n = 1 << 14
for true_off in (0.0, 0.25, -0.4):
    k0 = 5000
    # real Hann-windowed DFT of a tone at bin k0+true_off
    m = np.arange(n)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * m / (n - 1))
    x = np.cos(2 * np.pi * (k0 + true_off) * m / n) * w
    S = np.abs(np.fft.rfft(x))
    s = 20 * np.log10(np.maximum(S, 1e-12))
    kpk = int(np.argmax(s))
    d, db = noise._refine(s, kpk)
    got = kpk + d - k0
    chk(f"offset {true_off:+.2f} -> {got:+.3f}", abs(got - true_off) < 0.06,
        f"err {got-true_off:+.3f} bin")

print("\nharmonic sieve: recovers f0 and resists GCD collapse")
bin_hz = 238.0
f0_true = 497_600.0
peaks = [{"freq_hz": h * f0_true, "db": -85.0 - 2 * h, "bin": 0,
          "snr_db": 20.0, "prominence_db": 10.0} for h in range(1, 11)]
fams = noise.find_families(peaks, bin_hz)
chk("one family found", len(fams) == 1, f"{len(fams)}")
if fams:
    f = fams[0]
    chk(f"f0 = {f['f0_hz']:.1f} Hz (true {f0_true:.0f})",
        abs(f["f0_hz"] - f0_true) < 2 * bin_hz, f"err {f['f0_hz']-f0_true:+.1f} Hz")
    chk("not collapsed to f0/2", abs(f["f0_hz"] - f0_true / 2) > bin_hz)
    chk("all 10 members claimed", f["n_members"] == 10, f"{f['n_members']}")
    chk("density 1.0", f["density"] > 0.95, f"{f['density']:.2f}")

print("\nharmonic sieve: fundamental BELOW the analysed band")
peaks = [{"freq_hz": h * f0_true, "db": -90.0, "bin": 0, "snr_db": 20.0,
          "prominence_db": 10.0} for h in range(6, 14)]
fams = noise.find_families(peaks, bin_hz)
ok = fams and abs(fams[0]["f0_hz"] - f0_true) < 2 * bin_hz
chk("recovered from spacing alone", bool(ok),
    f"{fams[0]['f0_hz']:.1f} Hz" if fams else "no family")

print("\nclassification labels")
fs = 250e6
peaks = [
    {"freq_hz": fs / 2, "db": -82.0, "bin": 0, "snr_db": 40, "prominence_db": 30},
    {"freq_hz": 37.1e6, "db": -95.0, "bin": 0, "snr_db": 25, "prominence_db": 15},
]
comb = [{"freq_hz": h * f0_true, "db": -85.0, "bin": 0, "snr_db": 20,
         "prominence_db": 10} for h in range(1, 8)]
mains = [{"freq_hz": h * 50.0, "db": -100.0, "bin": 0, "snr_db": 15,
          "prominence_db": 8} for h in range(1, 8)]
allp = peaks + comb + mains
fams = noise.find_families(allp, bin_hz=1.0)
fams, allp = noise.classify(allp, fams, fs, bin_hz=1.0)
labels = {f["label"] for f in fams}
chk("switching_regulator identified", "switching_regulator" in labels, str(labels))
chk("mains_harmonics identified", "mains_harmonics" in labels, str(labels))
byf = {round(p["freq_hz"]): p.get("label") for p in allp}
chk("fs/2 flagged as sampling artifact",
    byf.get(round(fs / 2)) == "sampling_artifact_fs_2", str(byf.get(round(fs/2))))
chk("isolated line -> unclassified_spur",
    byf.get(round(37.1e6)) == "unclassified_spur", str(byf.get(round(37.1e6))))

print("\nend-to-end analyse() on a synthetic spectrum")
n = 1 << 18; bh = 250e6 / 2 / n
s = welch_noise(n, 16, -125.0, np.random.default_rng(7))
for h in range(1, 9):                       # comb at 497.6 kHz
    k = int(round(h * f0_true / bh))
    if k < n: s[k] += 35 - 2 * h
s[int(round((fs / 2 - bh) / bh))] = -80     # near-Nyquist artifact
r = noise.analyse(s, bh, fs, nframes=16, pfa=1e-6)
sw = [f for f in r["families"] if f["label"] == "switching_regulator"]
chk("analyse() finds the switcher", bool(sw),
    f"{r['n_peaks']} peaks, {len(r['families'])} families")
if sw:
    chk(f"f0 = {sw[0]['f0_hz']/1e3:.2f} kHz", abs(sw[0]["f0_hz"] - f0_true) < 3 * bh,
        f"err {sw[0]['f0_hz']-f0_true:+.1f} Hz")

print()
if fails:
    print("FAILURES:")
    for f in fails: print("  -", f)
    sys.exit(1)
print("ALL NOISE TESTS PASSED")
