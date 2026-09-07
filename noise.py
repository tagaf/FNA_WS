"""Spectral peak detection and electrical-noise classification.

Pipeline, in the order the functions are meant to be called:

    floor  = estimate_floor(spec_db, nframes)      # robust local noise mean
    peaks  = detect_peaks(spec_db, floor, ...)     # CFAR + sub-bin refinement
    fams   = find_families(peaks, bin_hz)          # harmonic sieve
    labels = classify(peaks, fams, ctx)            # rule-based

Everything here is pure NumPy/SciPy on a magnitude spectrum in dBFS -- no
hardware, no CUDA -- so it is unit-testable against synthetic spectra
(tests/test_noise.py) and reusable on saved captures.

Design notes
------------
* Detection uses CFAR (radar practice) rather than a fixed dB margin, so the
  threshold has a stated false-alarm probability given the local noise. For
  a Welch average of K periodograms of Gaussian noise, the per-bin power is
  Gamma(K, mean/K); the threshold multiplier for a target Pfa follows from
  the inverse regularised incomplete gamma. A fixed "+9 dB" (what the ad-hoc
  EMI script used) has no such guarantee and behaves differently at every K.
* Sub-bin refinement is magnitude-only (parabolic on dB): Welch power
  averaging discards phase, so the complex estimators (Candan, Jacobsen,
  Quinn) are unavailable. Parabolic-on-dB is good to ~0.01-0.05 bin for
  Hann-like main lobes, which is far finer than the family tolerances.
* The harmonic sieve must resist "GCD collapse": a comb at f0 is also
  explained by f0/2 with every other member missing. Families are therefore
  scored on *density* (observed / predicted members) as well as count, and
  ties break toward the larger fundamental.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammainccinv, gammaincinv

CAND_PEAK_LIMIT = 400   # peaks used to seed pairwise-difference candidates

__all__ = ["estimate_floor", "cfar_alpha_db", "detect_peaks",
           "find_families", "classify", "analyse"]


# --------------------------------------------------------------- noise floor
def estimate_floor(spec_db, nframes=1, block=256):
    """Robust per-bin estimate of the *mean* noise power, in dB.

    Block-median rather than a rolling median filter: a rolling filter over
    tens of millions of bins is prohibitively slow, while a block median plus
    linear interpolation is O(n) and easily smooth enough for a noise floor.

    A median is not a mean. For Gamma(K, 1/K) power the median sits below the
    mean by a factor that depends on K, so the bias is corrected explicitly --
    without this the CFAR threshold is systematically ~1.6 dB too low at K=1
    and the false-alarm rate is not what was asked for.
    """
    spec_db = np.asarray(spec_db, dtype=np.float64)
    n = spec_db.size
    if n == 0:
        return np.zeros(0)
    block = max(8, min(int(block), max(8, n // 4)))
    nb = max(1, n // block)
    trimmed = spec_db[:nb * block].reshape(nb, block)
    med = np.median(trimmed, axis=1)
    if nb == 1:
        floor = np.full(n, med[0])
    else:
        centres = (np.arange(nb) + 0.5) * block
        floor = np.interp(np.arange(n), centres, med)
    # median -> mean correction for Gamma(K, 1/K)
    K = max(1, int(nframes))
    med_over_mean = gammaincinv(K, 0.5) / K
    return floor - 10.0 * np.log10(med_over_mean)


def cfar_alpha_db(nframes=1, pfa=1e-6):
    """dB above the local mean noise power for a target false-alarm rate.

    Welch-averaged power of Gaussian noise is Gamma(K, mean/K); normalising
    by the mean, P(X > a) = Pfa gives a = gammainccinv(K, Pfa)/K.
    """
    K = max(1, int(nframes))
    a = gammainccinv(K, float(pfa)) / K
    return 10.0 * np.log10(a)


# ------------------------------------------------------------ peak detection
def _refine(spec_db, k):
    """Parabolic vertex on log-magnitude -> (sub-bin offset, corrected dB)."""
    if k <= 0 or k >= spec_db.size - 1:
        return 0.0, float(spec_db[k])
    ym, y0, yp = float(spec_db[k - 1]), float(spec_db[k]), float(spec_db[k + 1])
    denom = ym - 2.0 * y0 + yp
    if abs(denom) < 1e-12:
        return 0.0, y0
    d = 0.5 * (ym - yp) / denom
    d = max(-0.5, min(0.5, d))
    return d, y0 - 0.25 * (ym - yp) * d


def detect_peaks(spec_db, floor_db, bin_hz, nframes=1, pfa=1e-6,
                 f_offset=0.0, max_peaks=4000, exclude_dc_bins=2,
                 min_sep_bins=3):
    """CFAR-detect local maxima. Returns a list of dicts sorted by frequency.

    Each peak: {bin, freq_hz, db, prominence_db, snr_db}. `f_offset` is the
    frequency of bin 0 (nonzero for a zoom slice).

    Ranking when trimming to `max_peaks` is by PROMINENCE above the local
    threshold, not absolute level. Absolute level just picks whichever band
    sits highest -- on real data every retained peak came from one 103-112
    MHz hump while genuinely isolated lines elsewhere were dropped. A modest
    line standing clear of its neighbourhood is the more notable feature.

    `min_sep_bins` collapses detections that are really one broad peak with a
    dip in it, keeping the most prominent of each cluster.
    """
    spec_db = np.asarray(spec_db, dtype=np.float64)
    floor_db = np.asarray(floor_db, dtype=np.float64)
    n = spec_db.size
    if n < 3:
        return []
    thr = floor_db + cfar_alpha_db(nframes, pfa)
    over = spec_db > thr
    if exclude_dc_bins > 0:
        over[:min(exclude_dc_bins, n)] = False
    # local maxima among the bins that cleared the threshold
    idx = np.flatnonzero(over)
    idx = idx[(idx > 0) & (idx < n - 1)]
    if idx.size == 0:
        return []
    keep = (spec_db[idx] >= spec_db[idx - 1]) & (spec_db[idx] > spec_db[idx + 1])
    idx = idx[keep]
    if idx.size == 0:
        return []
    prom_all = spec_db[idx] - thr[idx]
    if min_sep_bins > 1 and idx.size > 1:
        keep_mask = np.ones(idx.size, dtype=bool)
        last = -10 ** 9
        last_i = -1
        for i in range(idx.size):
            if idx[i] - last < min_sep_bins:
                if prom_all[i] > prom_all[last_i]:
                    keep_mask[last_i] = False
                    last, last_i = idx[i], i
                else:
                    keep_mask[i] = False
            else:
                last, last_i = idx[i], i
        idx = idx[keep_mask]
        prom_all = prom_all[keep_mask]
    if idx.size > max_peaks:            # keep the most PROMINENT
        sel = np.argsort(prom_all)[::-1][:max_peaks]
        idx = np.sort(idx[sel])
    peaks = []
    for k in idx.tolist():
        d, db = _refine(spec_db, k)
        peaks.append({
            "bin": int(k),
            "freq_hz": float(f_offset + (k + d) * bin_hz),
            "db": float(db),
            "snr_db": float(db - floor_db[k]),
            "prominence_db": float(db - thr[k]),
        })
    return peaks


# --------------------------------------------------------- harmonic families
def _tol_hz(f, bin_hz, rel=5e-4, nbins=3.0):
    """Frequency tolerance: resolution-limited at low f, drift-limited high."""
    return max(nbins * bin_hz, rel * abs(f))


def find_families(peaks, bin_hz, min_members=3, max_harmonic=64,
                  min_density=0.25, rel_tol=5e-4, min_significance=3.0):
    """Group peaks into harmonic families (fundamental + integer multiples).

    Three defences against a sieve's natural tendency to overfit:

    1. Two-pass fit. Members are gathered with a loose tolerance, f0 is
       re-fitted by least squares through the origin, then members are
       re-gathered with a tight tolerance about the refined f0. Without this
       the tolerance must stay wide enough to absorb fundamental error times
       the harmonic number, which at n=40 admits almost anything.
    2. Chance correction. With P peaks spread over bandwidth B, a predicted
       harmonic lands within +-t of one by luck with probability ~2tP/B, so a
       smaller f0 (denser predicted grid) collects accidental members. Score
       on excess over chance in units of its own standard deviation, not on
       raw member count -- otherwise f0/2 always beats f0.
    3. Post-merge. Families whose fitted f0 agree, or that are integer
       multiples of a more significant family, are folded together; one
       physical switcher must not be reported as five sources.
    """
    if len(peaks) < min_members:
        return []
    freqs = np.array([p["freq_hz"] for p in peaks], dtype=np.float64)
    amps = np.array([p["db"] for p in peaks], dtype=np.float64)
    order = np.argsort(freqs)
    freqs, amps = freqs[order], amps[order]
    peaks = [peaks[i] for i in order]
    fmin, fmax = float(freqs[0]), float(freqs[-1])
    band = max(fmax - fmin, bin_hz)
    npk = len(freqs)

    # Candidates: peak frequencies themselves, plus pairwise differences (a
    # comb whose fundamental sits below the band still shows as a constant
    # spacing). Deduplicate onto a resolution grid -- raw rounding to 1 mHz
    # keeps thousands of candidates that differ by far less than the matching
    # tolerance and cost a full evaluate() each.
    def _grid(f):
        step = max(bin_hz, abs(f) * 1e-4)
        return round(float(f) / step) * step

    cands = set()
    for f in freqs:
        if f > 0:
            cands.add(_grid(f))
    strong = np.argsort(amps)[::-1][:CAND_PEAK_LIMIT]
    strong = np.sort(strong)
    for a in range(len(strong)):
        for b in range(a + 1, min(a + 12, len(strong))):
            d = float(freqs[strong[b]] - freqs[strong[a]])
            if d > max(2 * bin_hz, 1.0):
                cands.add(_grid(d))

    def collect(f0, tol_fn, claimed=None):
        """Vectorised nearest-peak lookup for every harmonic at once.

        This was a Python loop doing argmin over the whole peak array per
        harmonic: O(candidates x harmonics x peaks). Fine for the tens of
        peaks in the unit tests, hopeless on real spectra -- a live capture
        yields ~1500 peaks and ~18k candidates, which is billions of
        comparisons and minutes per analysis. searchsorted on the (already
        sorted) frequencies makes it O(harmonics log peaks)."""
        if f0 <= 0:
            return []
        nmax = int(min(max_harmonic, np.floor(fmax / f0)))
        if nmax < 1:
            return []
        h = np.arange(1, nmax + 1, dtype=np.float64)
        ft = h * f0
        j = np.searchsorted(freqs, ft)
        lo = np.clip(j - 1, 0, npk - 1)
        hi = np.clip(j, 0, npk - 1)
        dl = np.abs(freqs[lo] - ft)
        dh = np.abs(freqs[hi] - ft)
        k = np.where(dl <= dh, lo, hi)
        ok = np.minimum(dl, dh) <= tol_fn(h, ft)
        if claimed is not None:
            ok &= ~claimed[k]
        return list(zip(h[ok].astype(np.int64).tolist(), k[ok].tolist()))

    def evaluate(f0, claimed=None):
        """Two-pass fit -> (members, f0_fit, density, significance)."""
        loose = lambda h, ft: np.maximum(3.0 * bin_hz, rel_tol * ft)
        m = collect(f0, loose, claimed)
        if len(m) < min_members:
            return None
        mf = np.array([freqs[k] for _, k in m])
        mh = np.array([h for h, _ in m], dtype=np.float64)
        f0_fit = float(np.sum(mf * mh) / np.sum(mh * mh))
        if f0_fit <= 0:
            return None
        tight = lambda h, ft: np.maximum(3.0 * bin_hz, 5e-5 * ft)
        m = collect(f0_fit, tight, claimed)
        if len(m) < min_members:
            return None
        h_max = max(h for h, _ in m)
        density = len(m) / float(h_max)
        # expected accidental members among the h_max predicted positions
        tol_typ = max(3.0 * bin_hz, 5e-5 * f0_fit * (1 + h_max) / 2.0)
        p_chance = min(1.0, 2.0 * tol_typ * npk / band)
        exp_chance = h_max * p_chance
        sig = (len(m) - exp_chance) / max(np.sqrt(max(exp_chance, 1e-9)), 1e-9)
        return m, f0_fit, density, float(sig)

    ranked = []
    for f0 in cands:
        if f0 <= 0:
            continue
        r = evaluate(f0)
        if r is None:
            continue
        m, f0_fit, density, sig = r
        if density >= min_density and sig >= min_significance:
            ranked.append((sig, len(m), f0_fit))
    ranked.sort(key=lambda r: (-r[0], -r[1], -r[2]))

    claimed = np.zeros(npk, dtype=bool)
    families = []
    for _, _, f0 in ranked:
        r = evaluate(f0, claimed)
        if r is None:
            continue
        m, f0_fit, density, sig = r
        if len(m) < min_members or density < min_density or sig < min_significance:
            continue
        for _, k in m:
            claimed[k] = True
        families.append({
            "f0_hz": f0_fit,
            "n_members": len(m),
            "density": float(density),
            "significance": float(sig),
            "harmonics": [int(h) for h, _ in m],
            "members": [peaks[k] for _, k in m],
            "peak_db": float(max(amps[k] for _, k in m)),
            "strongest_harmonic": int(max(m, key=lambda x: amps[x[1]])[0]),
        })

    families = _merge_families(families, bin_hz)
    families.sort(key=lambda f: (-f["significance"], -f["n_members"]))
    return families


# A comb mistaken for its own subharmonic grid is off by a SMALL integer
# (2, 3, occasionally 4). Allowing arbitrary integer ratios is meaningless:
# 497.6 kHz is exactly 9952 x 50 Hz, so an unbounded test folds a switching
# regulator into the mains family.
MAX_MERGE_RATIO = 8


def _merge_families(families, bin_hz, rel=2e-3):
    """Fold together families that describe one physical source: equal
    fundamentals, or one a SMALL integer multiple of another (a comb seen as
    its own 2nd/3rd subharmonic grid)."""
    families = sorted(families, key=lambda f: -f["significance"])
    kept = []
    for fam in families:
        f0 = fam["f0_hz"]
        merged = False
        for k in kept:
            g0 = k["f0_hz"]
            tol = max(3.0 * bin_hz, rel * max(f0, g0))
            ratio = f0 / g0 if g0 else 0.0
            inv = g0 / f0 if f0 else 0.0
            near_int = (2 <= round(ratio) <= MAX_MERGE_RATIO
                        and abs(ratio - round(ratio)) * g0 <= tol)
            near_inv = (2 <= round(inv) <= MAX_MERGE_RATIO
                        and abs(inv - round(inv)) * f0 <= tol)
            if abs(f0 - g0) <= tol or near_int or near_inv:
                seen = {id(m) for m in k["members"]}
                for m in fam["members"]:
                    if id(m) not in seen:
                        k["members"].append(m)
                k["n_members"] = len(k["members"])
                k["peak_db"] = max(k["peak_db"], fam["peak_db"])
                k["merged"] = k.get("merged", 0) + 1
                merged = True
                break
        if not merged:
            kept.append(fam)
    return kept


# --------------------------------------------------------------- classifying
MAINS = (50.0, 60.0)


def classify(peaks, families, fs_hz, bin_hz, signal_hz=None):
    """Rule-based labels. Physics gives strong priors and labelled data is
    scarce, so rules beat ML here; the features are chosen so an ML stage
    could later consume the same table.

    Returns (families, peaks) annotated in place with 'label' and 'why'.
    """
    nyq = fs_hz / 2.0

    for fam in families:
        f0 = fam["f0_hz"]
        tol = _tol_hz(f0, bin_hz, rel=2e-3)
        if any(abs(f0 - m) <= max(tol, 0.5) for m in MAINS):
            fam["label"] = "mains_harmonics"
            fam["why"] = f"fundamental within tolerance of {f0:.1f} Hz mains"
        elif signal_hz and abs(f0 - signal_hz) <= _tol_hz(f0, bin_hz):
            fam["label"] = "harmonic_distortion"
            fam["why"] = ("harmonics of the applied tone -- ADC/front-end "
                          "nonlinearity (HD2, HD3, ...)")
        elif 1e4 <= f0 <= 5e6:
            fam["label"] = "switching_regulator"
            fam["why"] = (f"comb of {fam['n_members']} harmonics on "
                          f"{f0/1e3:.1f} kHz, in the DC-DC switching band")
        elif f0 < 1e4:
            fam["label"] = "low_freq_comb"
            fam["why"] = "sub-10 kHz comb (mains-adjacent or slow switcher)"
        else:
            fam["label"] = "hf_comb"
            fam["why"] = "harmonic comb above the usual switching band"

    claimed = {id(m) for fam in families for m in fam["members"]}
    for p in peaks:
        if id(p) in claimed:
            p.setdefault("label", "family_member")
            continue
        f = p["freq_hz"]
        if abs(f - nyq) <= _tol_hz(nyq, bin_hz, rel=1e-4):
            p["label"] = "sampling_artifact_fs_2"
            p["why"] = ("pinned at fs/2 -- verify by changing fs: an artifact "
                        "tracks Nyquist, a real signal does not")
        elif abs(f - nyq / 2) <= _tol_hz(nyq / 2, bin_hz, rel=1e-4):
            p["label"] = "sampling_artifact_fs_4"
            p["why"] = "at fs/4 -- likely sampling-chain, confirm with an fs sweep"
        elif signal_hz and abs(f - signal_hz) <= _tol_hz(f, bin_hz):
            p["label"] = "applied_signal"
            p["why"] = "matches the applied tone"
        else:
            p["label"] = "unclassified_spur"
            p["why"] = ("isolated line -- sweep fs to test for aliasing, "
                        "terminate the input to test for field pickup")
    return families, peaks


def analyse(spec_db, bin_hz, fs_hz, nframes=1, pfa=1e-6, f_offset=0.0,
            signal_hz=None, min_members=3, max_peaks=20000,
            family_peaks=1200):
    """Convenience wrapper: floor -> CFAR -> families -> labels.

    `max_peaks` bounds what is DETECTED (and therefore markable);
    `family_peaks` bounds what is fed to the harmonic sieve, whose cost grows
    with peak count. Marking every peak and sieving every peak are different
    jobs: the display wants completeness, the sieve wants the prominent lines
    that actually define a comb. Sieve input is a slice of the same list, so
    member identity (used by classify) is preserved.
    """
    floor = estimate_floor(spec_db, nframes)
    peaks = detect_peaks(spec_db, floor, bin_hz, nframes, pfa,
                         f_offset=f_offset, max_peaks=max_peaks)
    fam_in = peaks
    if len(peaks) > family_peaks:
        fam_in = sorted(peaks, key=lambda p: -p.get("prominence_db", p["db"]))
        fam_in = sorted(fam_in[:family_peaks], key=lambda p: p["freq_hz"])
    fams = find_families(fam_in, bin_hz, min_members=min_members)
    fams, peaks = classify(peaks, fams, fs_hz, bin_hz, signal_hz)
    return {
        "floor_db": float(np.median(floor)),
        "floor_curve": floor,
        "threshold_db_over_floor": cfar_alpha_db(nframes, pfa),
        "pfa": float(pfa),
        "n_peaks": len(peaks),
        "peaks": peaks,
        "families": fams,
    }
