"""Laser phase- and frequency-noise from a delay-line interferometer.

The optical front end is an unbalanced Michelson built from a 3x3 fibre
coupler and two Faraday rotator mirrors (Xu et al., Opt. Express 23, 22386
(2015)). Light that took the long arm interferes with light that took the
short one, so every output port carries

    I_n(t) = D_n + V_n cos( dphi(t) + psi_n ),
    dphi(t) = phi(t) - phi(t - tau),

the laser phase *differenced* over the interferometer delay tau. The 3x3
coupler acts as a 120-degree hybrid: psi_n are ~0, -120, +120 degrees, and
with all three ports the demodulation is a linear un-mixing (Xu eq. 2).

THIS PROTOTYPE HAS TWO PHOTODIODES, NOT THREE, so that route is unavailable
and the two-port geometry is what this module implements instead.

Two ports still determine dphi unambiguously, because (I_1, I_2) traced
against each other is an ELLIPSE -- a general conic, five degrees of freedom,
which is exactly the number of unknowns (D_1, D_2, V_1, V_2, psi). Fitting
that ellipse and normalising by it recovers a clean quadrature pair

    I = cos dphi,   Q = sin dphi = (I cos psi - v) / sin psi

from which dphi = atan2(Q, I), unwrapped. This is the Heydemann correction
used in displacement interferometry, and it makes the measurement immune to
the coupler's real (non-ideal) splitting ratios and to unequal photodiode
responsivities -- the same robustness Xu et al. get from calibrating eta/
zeta/xi, obtained here from the data itself.

WHAT TWO PORTS COST. Three ports over-determine the ellipse every sample, so
they calibrate instantaneously and additionally let common-mode intensity
(laser RIN) be cancelled. Two ports need the operating point to travel around
the fringe before the conic is determined, i.e. the calibration has to be
accumulated over seconds of interferometer drift rather than read off one
record, and RIN is not cancelled. `Cal.span` reports how much of the fringe
the calibration data actually covers; below ~0.35 the fit is not trustworthy
and `nominal_cal()` (assume psi, take D/V from the signal extremes) is the
honest fallback. The sign of psi is NOT recoverable from a conic -- swapping
the two photodiodes conjugates dphi -- which is harmless here because every
quantity below depends on |dphi|^2.

From dphi(t), with S_x(f) denoting a one-sided PSD (Xu eqs. 3-5):

    S_dnu(f) = S_dphi(f) / (2 pi tau)^2          differential frequency
    S_phi(f) = S_dphi(f) / (4 sin^2(pi f tau))   instantaneous phase
    S_nu(f)  = f^2 S_phi(f)                      instantaneous frequency
    L(f)     = S_phi(f) / 2                      SSB phase noise

The 1/sin^2 factor is the whole point of the delay line and also its limit:
it diverges at every multiple of the free spectral range 1/tau, where the
interferometer is blind. Those neighbourhoods are masked, and the useful
band is taken to end at 1/(2 tau) by default.

Linewidth follows Di Domenico, Schilt & Thomann, Appl. Opt. 49, 4801 (2010):
only frequency-noise components above the beta-separation line
S_nu = 8 ln2 f / pi^2 broaden the line, so

    A(f_lo) = integral over {f > f_lo, S_nu > beta line} of S_nu df
    FWHM    = sqrt(8 ln2 A)

which depends on f_lo -- i.e. on how long you look. A single linewidth number
is meaningless without its integration limit, so `linewidth_curve()` returns
the whole FWHM-versus-observation-time relation.

Pure NumPy/SciPy: no hardware, no CUDA. tests/test_phasenoise.py exercises it
against synthesised interferograms with known linewidth.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window, resample_poly, welch

__all__ = ["Cal", "tau_from_length", "fit_ellipse", "nominal_cal",
           "fringe_span", "demodulate", "auto_decim", "psd_dphi", "log_bin",
           "convert", "linewidth_curve", "white_linewidth", "analyse",
           "GROUP_INDEX", "C_VAC"]

C_VAC = 2.99792458e8
GROUP_INDEX = 1.4682         # SMF-28 group index at 1550 nm
BETA_K = 8.0 * np.log(2.0) / np.pi ** 2      # beta-separation line slope

# How close to a multiple of the FSR the 1/sin^2 correction is refused. At
# 3% of an FSR the correction is already +24 dB and every systematic in the
# measurement is amplified with it; past that it is not a measurement.
NULL_GUARD = 0.03

# Calibration acceptance.
#
# MAX_RESID catches a cloud that is not an ellipse at all -- clipping, a dead
# channel, a cycle slip. It is set two orders above a clean fit (0.001 on
# synthetic fringes; real photodiode amplitude noise adds a few percent).
#
# MAX_INSTABILITY is the gate that actually matters, and it exists because
# the obvious one does not work. Fringe COVERAGE cannot be measured with the
# fitted ellipse: a short arc admits a whole family of ellipses that fit it
# equally well, and the one the algebra picks is elongated in a way that
# smears the arc right around the unit circle -- so `span` reads 0.4, or at
# realistic noise 0.99, while the amplitudes are wrong by 60%. Residual does
# not catch it either (0.003: the arc genuinely lies on that ellipse). The
# family is only visible by asking whether the fit MOVES: fit the first half
# of the calibration history and the second half separately, and compare. A
# determined ellipse gives the same answer twice; an under-determined one
# gives two different members of the family.
#
# Measured across drift from 0 to 3 fringes and detector noise from 2 to 40
# codes on a 2600-code fringe: every case below 0.002 had parameters good to
# 1.7% or better, every case above it was wrong by 11% or more.
MAX_RESID = 0.15
MAX_INSTABILITY = 0.002


def tau_from_length(length_m, n_group=GROUP_INDEX, double_pass=True):
    """Interferometer delay from the extra fibre in one arm.

    A Michelson is double-pass: light traverses the delay fibre on the way
    out and again on the way back, so 10 m of fibre is 20 m of path. Set
    double_pass=False for a Mach-Zehnder, where it is traversed once.
    """
    return (2.0 if double_pass else 1.0) * n_group * float(length_m) / C_VAC


# ------------------------------------------------------------- calibration
class Cal:
    """Interferometer quadrature calibration.

        I_1 = dc1 + a1 cos(dphi)
        I_2 = dc2 + a2 cos(dphi + psi)

    `span` is the fraction of the fringe the calibrating data covered, and is
    the number to look at before believing anything downstream.
    """

    __slots__ = ("dc1", "dc2", "a1", "a2", "psi", "resid", "span", "npts",
                 "source", "stability")

    def __init__(self, dc1, dc2, a1, a2, psi, resid=float("nan"),
                 span=float("nan"), npts=0, source="ellipse",
                 stability=float("nan")):
        self.dc1, self.dc2 = float(dc1), float(dc2)
        self.a1, self.a2 = float(a1), float(a2)
        self.psi = float(psi)
        self.resid, self.span, self.npts = float(resid), float(span), int(npts)
        self.source = source
        self.stability = float(stability)

    @property
    def ok(self):
        """Geometrically valid: positive amplitudes, a hybrid angle that is
        not degenerate (psi -> 0 or pi means the two ports carry the same
        projection and Q cannot be formed at all)."""
        return (np.isfinite([self.dc1, self.dc2, self.a1, self.a2, self.psi]).all()
                and self.a1 > 0 and self.a2 > 0
                and 0.05 < abs(np.sin(self.psi)))

    @property
    def why_not(self):
        """Empty string if this calibration can be believed, else the reason.

        Note what is NOT here: `span`. Fringe coverage measured through the
        fitted ellipse is exactly the quantity a bad fit inflates, so gating
        on it is circular -- see MAX_INSTABILITY above. It is still reported,
        because when the fit IS determined it tells the operator how much
        margin they have.
        """
        if not self.ok:
            return "degenerate geometry"
        if self.source == "nominal":
            # min/max only mean anything if a whole fringe was traversed, and
            # for a nominal fit span is the only handle there is
            if not (self.span >= 0.9):
                return ("nominal calibration needs a full fringe on both "
                        "photodiodes; only %.0f%% was covered"
                        % (100 * self.span))
            return ""
        if not (self.resid <= MAX_RESID):
            return ("the data does not lie on an ellipse at all (residual "
                    "%.2f of a fringe) - check for clipping or a dead channel"
                    % self.resid)
        if not (self.stability <= MAX_INSTABILITY):
            return ("the ellipse is not determined: the first and second "
                    "halves of the calibration history disagree by %.1f%% - "
                    "the operating point has not travelled far enough round "
                    "the fringe" % (100 * self.stability))
        return ""

    @property
    def trustworthy(self):
        return self.why_not == ""

    def as_dict(self):
        return {"dc1": self.dc1, "dc2": self.dc2, "a1": self.a1, "a2": self.a2,
                "psi_deg": np.degrees(self.psi), "resid": self.resid,
                "span": self.span, "npts": self.npts, "source": self.source,
                "stability": self.stability,
                "ok": bool(self.ok), "trustworthy": bool(self.trustworthy),
                "why_not": self.why_not}


def _conic_to_cal(A, B, C, D, E, F):
    """Conic coefficients -> (dc1, dc2, a1, a2, psi), or None if not an ellipse.

    Eliminating dphi from the model gives, with u=(x-dc1)/a1, v=(y-dc2)/a2,

        u^2 - 2 u v cos psi + v^2 = sin^2 psi,

    so cos psi = -B / (2 sqrt(A C)) (the centre and the overall scale drop
    out of that ratio), and the semi-axes follow from the centred constant.
    """
    det = 4.0 * A * C - B * B
    if not np.isfinite(det) or det <= 0:          # parabola/hyperbola
        return None
    # The conic comes from an eigenvector, whose overall sign is arbitrary;
    # 4AC > B^2 already forces A and C to share a sign, so fixing A > 0 fixes
    # the whole thing. Without this the fit fails half the time, at random.
    if A < 0:
        A, B, C, D, E, F = -A, -B, -C, -D, -E, -F
    x0 = (B * E - 2.0 * C * D) / det
    y0 = (B * D - 2.0 * A * E) / det
    g = -(F + 0.5 * (D * x0 + E * y0))            # A X^2 + B XY + C Y^2 = g
    if not (g > 0 and A > 0 and C > 0):
        return None
    cpsi = -B / (2.0 * np.sqrt(A * C))
    if not np.isfinite(cpsi) or abs(cpsi) >= 1.0:
        return None
    s2 = 1.0 - cpsi * cpsi
    if s2 <= 1e-6:
        return None
    a1 = np.sqrt(g / (A * s2))
    a2 = np.sqrt(g / (C * s2))
    # psi in (0, pi): a conic fixes only cos psi. The other root conjugates
    # dphi, which no PSD here can tell apart.
    return x0, y0, a1, a2, float(np.arccos(cpsi))


def fit_ellipse(x, y, iters=2, trim=0.04, stability=True):
    """Direct least-squares ellipse fit (Halir & Flusser 1998) -> Cal.

    PASS THE POINTS IN ACQUISITION ORDER. With stability=True the array is
    split down the middle and each half fitted separately, and the two are
    compared; that split is only a determinacy test if the halves come from
    different times, because what has to be ruled out is a family of
    ellipses that all fit the same arc. Interleaving the split instead
    measures the noise and misses the family entirely (measured: 0.004
    disagreement on a fit whose amplitudes were wrong by 59%).

    Halir & Flusser rather than plain Fitzgibbon because the 6x6 scatter
    matrix of raw ADC codes is hopelessly conditioned; splitting it into the
    quadratic and linear blocks and inverting only the 3x3 keeps it stable.
    Inputs are additionally centred and scaled per axis first -- an affine
    per-axis map, which the parametrisation absorbs exactly (dc and a scale
    with it, psi is invariant), so nothing is approximated by doing so.

    `iters` rounds of trimming the worst `trim` fraction by algebraic
    residual: a cycle slip or a dropout otherwise drags the conic visibly.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 32 or x.size != y.size:
        return None
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(), y.std()
    if not (sx > 0 and sy > 0):
        return None
    u, v = (x - mx) / sx, (y - my) / sy

    keep = np.ones(u.size, bool)
    conic = None
    for it in range(max(1, iters)):
        uu, vv = u[keep], v[keep]
        if uu.size < 32:
            break
        D1 = np.column_stack((uu * uu, uu * vv, vv * vv))
        D2 = np.column_stack((uu, vv, np.ones_like(uu)))
        S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
        try:
            T = -np.linalg.solve(S3, S2.T)
        except np.linalg.LinAlgError:
            return None
        M = S1 + S2 @ T
        # premultiply by inv(C1) for the ellipse constraint 4ac - b^2 = 1
        M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
        w, vec = np.linalg.eig(M)
        cond = 4.0 * vec[0] * vec[2] - vec[1] ** 2
        idx = np.flatnonzero(np.isfinite(cond) & (cond > 0))
        if idx.size == 0:
            return None
        a1v = np.real(vec[:, idx[0]])
        conic = np.concatenate((a1v, T @ a1v))       # A B C D E F, scaled coords
        if it + 1 < max(1, iters):
            r = np.abs(_algebraic(u, v, conic))
            thr = np.quantile(r, 1.0 - trim)
            keep = r <= thr
    if conic is None:
        return None

    par = _conic_to_cal(*conic)
    if par is None:
        return None
    x0s, y0s, a1s, a2s, psi = par
    cal = Cal(mx + sx * x0s, my + sy * y0s, sx * a1s, sy * a2s, psi,
              npts=int(keep.sum()), source="ellipse")
    if not cal.ok:
        return None
    cal.resid = _radial_resid(x, y, cal)
    cal.span = fringe_span(x, y, cal)
    if stability:
        cal.stability = _stability(x, y, iters, trim)
    return cal


def _stability(x, y, iters, trim):
    """Disagreement between the ellipse fitted to the first half of the data
    and the ellipse fitted to the second half, as a fraction of the fringe.

    Reported as the worst of the five parameters, each normalised by the
    fringe amplitude it belongs to so the number means one thing.
    """
    h = x.size // 2
    if h < 64:
        return float("nan")           # cannot verify -> not trustworthy
    a = fit_ellipse(x[:h], y[:h], iters, trim, stability=False)
    b = fit_ellipse(x[h:], y[h:], iters, trim, stability=False)
    if a is None or b is None:
        return float("inf")
    A, B = max(a.a1, b.a1), max(a.a2, b.a2)
    return float(max(abs(a.a1 - b.a1) / A, abs(a.a2 - b.a2) / B,
                     abs(a.psi - b.psi) / np.pi,
                     abs(a.dc1 - b.dc1) / A, abs(a.dc2 - b.dc2) / B))


def _algebraic(u, v, c):
    A, B, C, D, E, F = c
    return A * u * u + B * u * v + C * v * v + D * u + E * v + F


def _radial_resid(x, y, cal):
    """RMS of |(I,Q)| - 1 after normalisation: 0 = the data lies on the fitted
    ellipse, and it is in units of the fringe amplitude, so it compares
    directly against the phase noise being measured."""
    I, Q = _iq(x, y, cal, np.float64)
    r = np.hypot(I, Q)
    return float(np.sqrt(np.mean((r - 1.0) ** 2)))


def fringe_span(x, y, cal, nbins=72):
    """Fraction of the fringe (0..1) the sample cloud covers.

    Occupancy of 72 angular bins rather than a peak-to-peak angle: drift
    sweeps the operating point back and forth, so the cloud is often several
    disjoint arcs, and what matters for conditioning the conic is how much of
    the circle has data anywhere on it.
    """
    I, Q = _iq(x, y, cal, np.float64)
    a = np.arctan2(Q, I)
    h, _ = np.histogram(a, bins=nbins, range=(-np.pi, np.pi))
    return float(np.count_nonzero(h)) / nbins


def nominal_cal(x, y, psi_deg=120.0, lo=0.2, hi=99.8):
    """Fallback when the fringe is not covered: assume the coupler's nominal
    hybrid angle and take each channel's offset/amplitude from its extremes.

    Percentiles, not min/max, so a single ADC glitch does not set the scale.
    This is only correct if BOTH channels swing a full fringe within the
    record; `span` is still reported so the caller can see when they do not.
    """
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    xl, xh = np.percentile(x, [lo, hi])
    yl, yh = np.percentile(y, [lo, hi])
    a1, a2 = (xh - xl) / 2.0, (yh - yl) / 2.0
    if not (a1 > 0 and a2 > 0):
        return None
    cal = Cal((xh + xl) / 2.0, (yh + yl) / 2.0, a1, a2,
              np.radians(psi_deg), npts=x.size, source="nominal")
    cal.resid = _radial_resid(x, y, cal)
    cal.span = fringe_span(x, y, cal)
    return cal


# ----------------------------------------------------------- demodulation
def _iq(x, y, cal, dtype=np.float32):
    """Normalised quadrature pair. float32 by default because these are the
    full-rate arrays -- a 268 ms record is 67 M samples per channel, and
    float64 would cost 1 GB per intermediate for no benefit: I and Q are of
    order 1, so float32 resolves them to ~1e-7, two decades finer than the
    ADC's own 1/2600 of a fringe."""
    u = (np.asarray(x, dtype) - dtype(cal.dc1)) / dtype(cal.a1)
    v = (np.asarray(y, dtype) - dtype(cal.dc2)) / dtype(cal.a2)
    s, c = np.sin(cal.psi), np.cos(cal.psi)
    return u, (u * dtype(c) - v) / dtype(s)


def auto_decim(fs, tau, headroom=1.5):
    """Largest power-of-two decimation that keeps Nyquist above the usable
    band. The measurement dies at the first FSR null (f = 1/tau) and is
    normally read to 1/(2 tau), so anything past ~1/tau is only cost."""
    if not (tau > 0 and fs > 0):
        return 1
    d = int(2 ** np.floor(np.log2(max(1.0, fs * tau / headroom))))
    return max(1, d)


def demodulate(x, y, cal, decim=1, predecimate=True):
    """(I_1, I_2) -> unwrapped dphi(t), optionally decimated.

    predecimate=True low-passes and decimates the QUADRATURE PAIR before the
    arctangent instead of after it. I and Q are linear in the photocurrents,
    so this is the ordinary I/Q downconversion and it is exact whenever the
    analytic signal exp(j dphi) fits below the new Nyquist. For a delay-line
    interferometer that is the normal regime -- dphi is milliradians, so
    exp(j dphi) ~ 1 + j dphi carries the same band as dphi itself -- and it
    moves the arctangent and the unwrap off the full-rate stream, which is
    where essentially all the CPU goes (a 1 s record at 250 MS/s is 262 M
    samples). Set it False and the arctangent runs at full rate: correct for
    arbitrarily large phase excursions, ~10x slower.

    Returns (dphi, fs_factor, slips) where slips counts samples whose wrapped
    phase step exceeded 0.8 pi -- the unwrap is guessing at those.
    """
    I, Q = _iq(x, y, cal)
    d = max(1, int(decim))
    if d > 1 and predecimate:
        I = resample_poly(I, 1, d)
        Q = resample_poly(Q, 1, d)
    p = np.arctan2(Q, I).astype(np.float64)
    step = np.diff(p)
    step -= np.round(step / (2 * np.pi)) * (2 * np.pi)
    slips = int(np.count_nonzero(np.abs(step) > 0.8 * np.pi))
    dphi = np.empty(p.size, np.float64)
    dphi[0] = p[0]
    np.cumsum(step, out=dphi[1:])
    dphi[1:] += p[0]
    if d > 1 and not predecimate:
        dphi = resample_poly(dphi - dphi.mean(), 1, d)
    return dphi, d, slips


# -------------------------------------------------------------------- PSD
def psd_dphi(dphi, fs, nperseg=0, window="hann", overlap=0.5):
    """One-sided Welch PSD of the differential phase, in rad^2/Hz.

    nperseg=0 means "the whole record in one segment": the resolution the
    record actually bought (1/T), with no averaging. Shorter segments trade
    that resolution for variance; the log-binning below recovers most of the
    variance at high f without giving up resolution at low f, so a single
    full-length segment is usually the right choice here.

    detrend='linear' is not cosmetic. Interferometer drift puts tens of
    radians of ramp on dphi, ~100 dB above the noise being measured; leaving
    it in would bury the first decades under leakage.
    """
    n = dphi.size
    seg = n if nperseg <= 0 else min(int(nperseg), n)
    seg = max(256, seg)
    win = get_window(window, seg, fftbins=True)
    f, P = welch(dphi, fs=fs, window=win, nperseg=seg,
                 noverlap=int(seg * overlap), detrend="linear",
                 return_onesided=True, scaling="density")
    navg = 1 if seg >= n else max(1, int((n - seg) / max(1, seg * (1 - overlap))) + 1)
    return f, P, seg, navg


def log_bin(f, S, npts=600, f_lo=0.0, f_hi=0.0):
    """Average a linear-frequency PSD into log-spaced buckets.

    A phase-noise plot is log-log, and a linear PSD puts almost every bin in
    the top decade where it is drawn on top of itself. Averaging within a
    bucket is also free variance reduction that grows with f exactly where
    the estimator is noisiest -- the same benefit multi-rate PSD stitching is
    built for, without a second pass over the data. Averaging is done in
    linear power (the unbiased thing to do); `n` per bucket is returned so
    the client can say how much confidence each point carries.
    """
    f = np.asarray(f, np.float64)
    S = np.asarray(S, np.float64)
    good = np.isfinite(S) & (f > 0)
    f, S = f[good], S[good]
    if f.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    lo = f_lo if f_lo > 0 else f[0]
    hi = f_hi if f_hi > 0 else f[-1]
    if not (hi > lo):
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    edges = np.logspace(np.log10(lo), np.log10(hi), int(npts) + 1)
    idx = np.searchsorted(f, edges)
    idx = np.clip(idx, 0, f.size)
    keep = idx[1:] > idx[:-1]
    starts, stops = idx[:-1][keep], idx[1:][keep]
    n = (stops - starts).astype(np.int64)
    csum = np.concatenate(([0.0], np.cumsum(S)))
    cf = np.concatenate(([0.0], np.cumsum(f)))
    return (cf[stops] - cf[starts]) / n, (csum[stops] - csum[starts]) / n, n


def convert(f, S_dphi, tau, null_guard=NULL_GUARD):
    """S_dphi -> the four PSDs of Xu eq. 3-5 plus L(f).

    Masks a +-null_guard fraction of an FSR around every multiple of 1/tau
    with k >= 1, where sin(pi f tau) -> 0 and the interferometer genuinely
    sees nothing. Only DC itself is dropped at the low end -- see below.
    """
    f = np.asarray(f, np.float64)
    S_dphi = np.asarray(S_dphi, np.float64)
    s = np.sin(np.pi * f * tau)
    frac = f * tau
    k = np.round(frac)
    # Only the nulls at k >= 1 are blind spots. k = 0 is NOT one: as f -> 0
    # the correction tends to 1/(2 pi f tau)^2, the ordinary
    # frequency-discriminator response, and S_dphi tends to a constant with
    # it -- so S_phi -> h0/f^2, large but perfectly well determined. Guarding
    # around k = 0 as well would blank everything below null_guard/tau, which
    # at a 10 m delay is 300 kHz: the entire decade range a laser's flicker
    # noise lives in.
    near_null = (k >= 1) & (np.abs(frac - k) < null_guard)
    bad = near_null | (f <= 0) | ~np.isfinite(S_dphi)
    denom = np.where(bad, np.nan, 4.0 * s * s)
    S_phi = S_dphi / denom
    return {
        "S_dphi": np.where(f > 0, S_dphi, np.nan),          # rad^2/Hz
        "S_dnu": np.where(f > 0, S_dphi, np.nan) / (2 * np.pi * tau) ** 2,  # Hz^2/Hz
        "S_phi": S_phi,                                      # rad^2/Hz
        "S_nu": f * f * S_phi,                               # Hz^2/Hz
        "L": S_phi / 2.0,                                    # rad^2/Hz (SSB)
        "mask": ~bad,
    }


# -------------------------------------------------------------- linewidth
def _beta_line(f):
    return BETA_K * np.asarray(f, np.float64)


def beta_crossing(f, S_nu):
    """Highest f at which S_nu still rises above the beta-separation line.

    Above it the frequency noise only modulates the carrier (it contributes
    sidebands, not width), so it is where the linewidth integral stops
    gaining and the natural top of the FWHM-vs-observation-time curve.
    """
    f = np.asarray(f, np.float64)
    S = np.asarray(S_nu, np.float64)
    ok = np.isfinite(S) & (f > 0) & (S > _beta_line(f))
    return float(f[ok].max()) if np.any(ok) else float("nan")


def linewidth_curve(f, S_nu, f_lo_list=None, f_hi=0.0):
    """FWHM versus integration lower limit (Di Domenico 2010).

    Only the part of S_nu that lies ABOVE the beta-separation line
    S_nu = 8 ln2 f / pi^2 contributes to the linewidth; everything below it
    only shifts the line's centre on a timescale slower than you are looking.
    Because 1/f^a noise keeps adding area as f_lo drops, the answer grows
    with observation time T_obs = 1/f_lo -- which is why this returns a curve
    and not a number.
    """
    f = np.asarray(f, np.float64)
    S = np.asarray(S_nu, np.float64)
    ok = np.isfinite(S) & np.isfinite(f) & (f > 0)
    if f_hi > 0:
        ok &= f <= f_hi
    f, S = f[ok], S[ok]
    if f.size < 4:
        return np.zeros(0), np.zeros(0)
    above = S > _beta_line(f)
    contrib = np.where(above, S, 0.0)
    # cumulative area from the TOP down, so every f_lo is one lookup
    seg = np.concatenate(([0.0], np.cumsum(np.diff(f) * 0.5 *
                                           (contrib[1:] + contrib[:-1]))))
    total = seg[-1]
    if f_lo_list is None:
        f_lo_list = np.logspace(np.log10(f[0]), np.log10(max(f[0] * 10, f[-1] / 10)), 40)
    f_lo_list = np.asarray(f_lo_list, np.float64)
    area = total - np.interp(f_lo_list, f, seg)
    return f_lo_list, np.sqrt(8.0 * np.log(2.0) * np.maximum(area, 0.0))


def white_linewidth(f, S_nu, band=None):
    """Intrinsic (Lorentzian) linewidth from the white part of S_nu.

    A frequency-noise floor S_nu = h_0 gives a Lorentzian of FWHM pi h_0.
    h_0 is taken as the MEDIAN of S_nu over the top of the analysed band --
    median, because that band is where residual spurs live and one mains
    harmonic would drag a mean.
    """
    f = np.asarray(f, np.float64)
    S = np.asarray(S_nu, np.float64)
    ok = np.isfinite(S) & (f > 0)
    if band:
        ok &= (f >= band[0]) & (f <= band[1])
    if np.count_nonzero(ok) < 8:
        return float("nan"), float("nan")
    h0 = float(np.median(S[ok]))
    return np.pi * h0, h0


# ------------------------------------------------------------------ driver
def analyse(x, y, fs, tau, *, cal=None, cal_mode="auto", psi_deg=120.0,
            decim=0, nperseg=0, window="hann", f_max=0.0, npts=600,
            predecimate=True, trace_cols=1024, liss_pts=1200,
            null_guard=NULL_GUARD, require_cal=True):
    """Two photocurrent records -> everything the phase-noise tab shows.

    `cal` is an already-established calibration to reuse. Pass one: a single
    record rarely covers a whole fringe (a 2 kHz laser moves dphi by tens of
    MILLIradians over a 98 ns delay), so the conic is determined by
    accumulating the operating point's drift across many captures, not from
    one. cal=None fits from this record alone, which works only when
    something is genuinely sweeping the fringe.

    cal_mode: 'auto' reuse/fit an ellipse, 'refit' ignore `cal`, 'nominal'
    assume the coupler's hybrid angle and take D/V from the signal extremes.

    require_cal=True refuses to report numbers built on a calibration that
    failed its own checks. That refusal is the feature: an under-determined
    ellipse still produces a smooth, plausible, WRONG linewidth, and a phase
    noise analyser that quietly does that is worse than one that stops.

    Arrays come back log-binned and small: the caller ships them as JSON.
    """
    x = np.asarray(x); y = np.asarray(y)
    n = min(x.size, y.size)
    if n < 4096:
        return {"error": f"need >= 4096 samples per channel, got {n}"}
    x, y = x[:n], y[:n]
    if not (tau > 0):
        return {"error": "delay tau must be > 0"}

    # --- calibration
    warn = []
    sub = max(1, n // 200000)          # the conic needs coverage, not samples
    if cal_mode == "nominal":
        cal = nominal_cal(x[::sub], y[::sub], psi_deg)
    elif cal is None or cal_mode == "refit":
        cal = fit_ellipse(x[::sub], y[::sub])
    if cal is None or not cal.ok:
        cal = nominal_cal(x[::sub], y[::sub], psi_deg)
        if cal is None:
            return {"error": "no fringe amplitude on one or both channels - "
                             "check that both photodiodes see light"}
        if cal_mode != "nominal":
            warn.append("ellipse fit failed; fell back to the nominal "
                        "%.0f deg hybrid angle" % psi_deg)
    if not cal.trustworthy:
        msg = "calibration not trustworthy: " + cal.why_not
        if require_cal:
            return {"error": msg + ". Let the interferometer drift (or warm "
                                   "one arm) until the Lissajous closes into "
                                   "a full ellipse, then recalibrate.",
                    "cal": cal.as_dict()}
        warn.append(msg + " - absolute levels below are provisional")

    # --- demodulate
    d = auto_decim(fs, tau) if decim <= 0 else max(1, int(decim))
    dphi, d, slips = demodulate(x, y, cal, decim=d, predecimate=predecimate)
    fs_d = fs / d
    if dphi.size < 1024:
        return {"error": "decimation left too few samples; reduce it"}
    if slips:
        warn.append("%d phase steps exceeded 0.8 pi - the unwrap is guessing "
                    "there; check the calibration or lower the decimation"
                    % slips)

    # --- spectra. Everything downstream uses the LOG-BINNED PSD, not the raw
    # periodogram. Each bucket is the mean of the bins inside it, which is
    # unbiased, whereas a single Welch bin is exponentially distributed and
    # any robust statistic over raw bins inherits that distribution's bias
    # (the median of an exponential sits ln2 = -1.59 dB below its mean, which
    # is exactly the error this used to make). Bucket mean x bucket width
    # also preserves area, so the linewidth integral is unaffected by binning.
    f, S, seg, navg = psd_dphi(dphi, fs_d, nperseg=nperseg, window=window)
    fmax = f_max if f_max > 0 else 0.5 / tau
    fmax = min(fmax, 0.45 * fs_d)
    fb, Sb, nb = log_bin(f, S, npts=npts, f_lo=f[1] if f.size > 1 else 0.0,
                         f_hi=fmax)
    if fb.size < 4:
        return {"error": "empty spectrum after log-binning"}
    conv = convert(fb, Sb, tau, null_guard=null_guard)
    S_nu = conv["S_nu"]

    # --- linewidth
    t_lo = max(fb[0], fs_d / dphi.size)          # 1/T_record: no lower limit
    f_cross = beta_crossing(fb, S_nu)
    # stop the curve where the beta line stops being crossed: past that every
    # point is identically zero, which is true but plots as a hole
    t_hi = max(t_lo * 10, min(fmax / 10, f_cross if np.isfinite(f_cross)
                              else fmax / 10))
    f_lo_grid = np.logspace(np.log10(t_lo), np.log10(t_hi), 40)
    f_lo_grid, fwhm = linewidth_curve(fb, S_nu, f_lo_grid, f_hi=fmax)
    dnu_white, h0 = white_linewidth(fb, S_nu, band=(fmax / 10.0, fmax))

    # --- statistics on the DETRENDED phase, matching what the PSD sees.
    # Raw dphi is dominated by drift: the operating point walking a fringe
    # across the record is radians of ramp on top of milliradians of laser
    # noise, so an undetrended "rms phase" would just be restating the drift.
    # The slope is worth reporting on its own -- as an apparent laser
    # frequency drift rate, which is what an operator can act on.
    t = np.arange(dphi.size, dtype=np.float64)
    tm = t - t.mean()
    slope = float(tm @ (dphi - dphi.mean()) / (tm @ tm))     # rad per sample
    resid = dphi - (slope * tm + dphi.mean())
    # slope is rad per DECIMATED sample: rad/s is slope * fs_d, and the
    # interferometer converts rad to Hz of laser frequency by 1/(2 pi tau)
    drift_hz_s = slope * fs_d / (2 * np.pi * tau)

    # --- display extras
    step = max(1, dphi.size // trace_cols)
    ncol = dphi.size // step
    blk = dphi[:ncol * step].reshape(ncol, step)
    lstep = max(1, n // liss_pts)

    return {
        "tau_s": tau,
        "fsr_hz": 1.0 / tau,
        "fs_hz": fs,
        "fs_demod_hz": fs_d,
        "decim": d,
        "predecimate": bool(predecimate),
        "nsamples": n,
        "record_s": n / fs,
        "nperseg": seg,
        "navg": navg,
        "res_hz": fs_d / seg,
        "f_max_hz": fmax,
        "window": window,
        "slips": slips,
        "cal": cal.as_dict(),
        "warn": warn,
        # spectra, log-binned
        "f": fb,
        "n_per_bin": nb,
        "S_dphi": conv["S_dphi"],
        "S_dnu": conv["S_dnu"],
        "S_phi": conv["S_phi"],
        "S_nu": S_nu,
        "L": conv["L"],
        "beta_line": _beta_line(fb),
        # linewidth
        "lw_f_lo": f_lo_grid,
        "lw_fwhm": fwhm,
        "lw_white_hz": dnu_white,
        "lw_h0": h0,
        # time domain / diagnostics
        "dphi_min": blk.min(axis=1),
        "dphi_max": blk.max(axis=1),
        "dphi_span_s": ncol * step / fs_d,
        "f_beta_cross_hz": f_cross,
        "dphi_rms": float(np.std(resid)),           # detrended
        "dphi_p2p": float(np.ptp(dphi)),
        "fringes_swept": float(np.ptp(dphi)) / (2 * np.pi),
        "dnu_rms_hz": float(np.std(resid)) / (2 * np.pi * tau),
        "drift_hz_per_s": drift_hz_s,
        "liss_x": np.asarray(x[::lstep][:liss_pts], np.float32),
        "liss_y": np.asarray(y[::lstep][:liss_pts], np.float32),
        "ch1": {"min": float(x.min()), "max": float(x.max()),
                "mean": float(x.mean()), "std": float(x.std())},
        "ch2": {"min": float(y.min()), "max": float(y.max()),
                "mean": float(y.mean()), "std": float(y.std())},
    }
