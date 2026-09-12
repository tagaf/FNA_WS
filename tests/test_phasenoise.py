#!/usr/bin/env python3
"""phasenoise.py against synthetic interferograms with a KNOWN answer.

The point of these tests is that the truth is set by construction: the laser
phase is a Wiener process whose step variance fixes the Lorentzian linewidth
exactly (var = 2 pi dnu / fs), so S_nu must come back flat at dnu/pi and the
recovered FWHM must match the number that went in. Nothing here touches
hardware or CUDA.
"""
import os, sys, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phasenoise as P

FS = 250e6
TAU = P.tau_from_length(10.0)          # 10 m Michelson -> ~97.9 ns


def make_interferogram(n, dnu_hz=50e3, tau=TAU, fs=FS, psi_deg=118.0,
                       dc=(-300.0, 450.0), amp=(2600.0, 2100.0),
                       drift_fringes=3.0, det_noise=2.0, seed=0):
    """Two photocurrents from a delay-line interferometer, in ADC codes.

    phi is a random walk with the step variance that makes the Lorentzian
    FWHM exactly dnu_hz; dphi is phi differenced by tau with a fractional
    delay (tau is 24.48 samples at 250 MS/s -- not an integer, and rounding
    it would move the sin^2 nulls the correction has to undo). `drift`
    sweeps the operating point across whole fringes so the ellipse fit has
    something to fit, exactly as slow interferometer drift does in the lab.
    """
    rng = np.random.default_rng(seed)
    ts = tau * fs
    k = int(np.floor(ts)); a = ts - k
    steps = rng.normal(0.0, np.sqrt(2 * np.pi * dnu_hz / fs), n + k + 2)
    phi = np.cumsum(steps)
    # phi(t) - phi(t - tau) with the delay interpolated, both from one walk
    dphi = phi[-n:] - ((1 - a) * phi[-n - k:len(phi) - k]
                       + a * phi[-n - k - 1:len(phi) - k - 1])
    drift = 2 * np.pi * drift_fringes * np.arange(n) / n
    psi = np.radians(psi_deg)
    x = dc[0] + amp[0] * np.cos(dphi + drift) + rng.normal(0, det_noise, n)
    y = dc[1] + amp[1] * np.cos(dphi + drift + psi) + rng.normal(0, det_noise, n)
    return x, y, dphi


class TestDelay(unittest.TestCase):
    def test_michelson_is_double_pass(self):
        # 10 m of fibre is 20 m of path in a Michelson
        self.assertAlmostEqual(P.tau_from_length(10.0), 97.94e-9, delta=0.1e-9)
        self.assertAlmostEqual(P.tau_from_length(10.0, double_pass=False),
                               P.tau_from_length(10.0) / 2)

    def test_fsr(self):
        self.assertAlmostEqual(1.0 / TAU / 1e6, 10.21, delta=0.05)


class TestEllipse(unittest.TestCase):
    def test_recovers_planted_parameters(self):
        x, y, _ = make_interferogram(1 << 18, drift_fringes=4.0)
        cal = P.fit_ellipse(x[::4], y[::4])
        self.assertIsNotNone(cal)
        self.assertAlmostEqual(cal.dc1, -300.0, delta=8)
        self.assertAlmostEqual(cal.dc2, 450.0, delta=8)
        self.assertAlmostEqual(cal.a1, 2600.0, delta=25)
        self.assertAlmostEqual(cal.a2, 2100.0, delta=25)
        self.assertAlmostEqual(np.degrees(cal.psi), 118.0, delta=1.0)
        self.assertGreater(cal.span, 0.9)          # 4 fringes = full coverage
        self.assertLess(cal.resid, 0.01)

    def test_sign_of_psi_is_not_observable(self):
        # -118 deg is the mirror geometry; a conic cannot tell them apart, and
        # the demodulated phase simply comes out conjugated
        x, y, _ = make_interferogram(1 << 17, psi_deg=-118.0, drift_fringes=4.0)
        cal = P.fit_ellipse(x, y)
        self.assertAlmostEqual(abs(np.degrees(cal.psi)), 118.0, delta=1.5)

    def test_stuck_operating_point_is_rejected(self):
        """The failure this module exists to not make silently.

        A cloud confined to a short arc gets fitted by an absurd ellipse
        (measured: a1 = 3 codes where the truth is 2600), and normalising by
        it smears the arc right around the unit circle -- so `span` reads a
        perfect 1.0. Only the residual gives it away.
        """
        x, y, _ = make_interferogram(1 << 17, dnu_hz=1e3, drift_fringes=0.0)
        cal = P.fit_ellipse(x, y)
        self.assertIsNotNone(cal)
        self.assertGreater(cal.resid, 0.2)
        self.assertFalse(cal.trustworthy)
        self.assertIn("does not lie on an ellipse", cal.why_not)

    def test_partial_coverage_is_caught_by_instability_not_by_span(self):
        """The case that broke the first version of this gate.

        A short arc admits a family of ellipses; the one the algebra returns
        fits it beautifully (low residual) and, once normalised by it, smears
        the arc right around the circle -- so `span` reads high and `resid`
        reads low while the amplitudes are wrong by tens of percent. Only
        splitting the history in time and refitting exposes the family.
        """
        x, y, _ = make_interferogram(1 << 18, drift_fringes=0.05)
        cal = P.fit_ellipse(x, y)
        self.assertLess(cal.resid, 0.05)           # the fit "looks" fine
        wrong = max(abs(cal.a1 - 2600) / 2600, abs(cal.a2 - 2100) / 2100)
        self.assertGreater(wrong, 0.2)             # but it is 38% out
        self.assertGreater(cal.stability, P.MAX_INSTABILITY)
        self.assertFalse(cal.trustworthy)
        self.assertIn("not determined", cal.why_not)

    def test_a_swept_fringe_is_stable_and_accurate(self):
        # ~one fringe traversed is where the gate opens, which is exactly the
        # physical requirement: a conic needs the whole conic
        for drift in (0.8, 1.5, 3.0):
            cal = P.fit_ellipse(*make_interferogram(1 << 18,
                                                    drift_fringes=drift)[:2])
            self.assertLess(cal.stability, P.MAX_INSTABILITY, f"drift={drift}")
            self.assertTrue(cal.trustworthy)
            self.assertAlmostEqual(cal.a1, 2600, delta=30)

    def test_the_gate_is_conservative_not_optimistic(self):
        # at 0.6 of a fringe the fit happens to be right, and the gate still
        # refuses it. That is the correct direction to be wrong in: the cost
        # is waiting for more drift, not publishing a wrong linewidth.
        cal = P.fit_ellipse(*make_interferogram(1 << 18, drift_fringes=0.6)[:2])
        self.assertFalse(cal.trustworthy)

    def test_the_gate_tracks_the_real_error_across_noise(self):
        """Threshold calibration, kept as a test so it cannot silently rot.

        Every configuration the gate accepts must have parameters good to a
        few percent, and every one it rejects must actually be bad. Two
        detector noise levels, because the first gate that was tried here
        passed at 2 codes of noise and read span = 0.99 on a 95%-wrong fit at
        40 codes.
        """
        for noise in (2.0, 40.0):
            for drift, in [(0.0,), (0.1,), (0.3,), (0.8,), (3.0,)]:
                xs, ys = [], []
                for c in range(24):
                    x, y, _ = make_interferogram(1 << 14, drift_fringes=drift,
                                                 det_noise=noise, seed=c)
                    xs.append(x[::8]); ys.append(y[::8])
                X, Y = np.concatenate(xs), np.concatenate(ys)
                cal = P.fit_ellipse(X, Y)
                err = max(abs(cal.a1 - 2600) / 2600, abs(cal.a2 - 2100) / 2100,
                          abs(np.degrees(cal.psi) - 118) / 118)
                if cal.trustworthy:
                    self.assertLess(err, 0.05,
                                    f"accepted a {err:.0%} error at "
                                    f"drift={drift} noise={noise}")
                else:
                    self.assertGreater(err, 0.05,
                                       f"rejected a good fit ({err:.1%}) at "
                                       f"drift={drift} noise={noise}")

    def test_nominal_fallback(self):
        x, y, _ = make_interferogram(1 << 17, psi_deg=120.0, drift_fringes=4.0)
        cal = P.nominal_cal(x, y, 120.0)
        self.assertAlmostEqual(cal.a1, 2600.0, delta=60)
        self.assertAlmostEqual(cal.dc1, -300.0, delta=60)
        self.assertEqual(cal.source, "nominal")


class TestDemodulation(unittest.TestCase):
    def test_recovers_dphi_waveform(self):
        n = 1 << 18
        x, y, dphi = make_interferogram(n, drift_fringes=4.0, det_noise=0.5)
        cal = P.fit_ellipse(x, y)
        got, d, slips = P.demodulate(x, y, cal, decim=1)
        self.assertEqual(d, 1)
        self.assertEqual(slips, 0)
        drift = 2 * np.pi * 4.0 * np.arange(n) / n
        truth = dphi + drift
        err = (got - got.mean()) - (truth - truth.mean())
        self.assertLess(np.std(err), 0.01)         # rad

    def test_predecimation_matches_full_rate(self):
        # I/Q decimation before the arctangent is exact in the small-excursion
        # regime; if that ever stops being true this is where it shows
        x, y, _ = make_interferogram(1 << 19, drift_fringes=2.0, det_noise=0.5)
        cal = P.fit_ellipse(x, y)
        fast, _, _ = P.demodulate(x, y, cal, decim=16, predecimate=True)
        slow, _, _ = P.demodulate(x, y, cal, decim=16, predecimate=False)
        m = min(fast.size, slow.size)
        a = fast[100:m - 100] - fast[100:m - 100].mean()
        b = slow[100:m - 100] - slow[100:m - 100].mean()
        self.assertLess(np.std(a - b) / np.std(b), 0.02)

    def test_auto_decim_keeps_nyquist_above_the_fsr(self):
        d = P.auto_decim(FS, TAU)
        self.assertGreaterEqual(FS / d / 2, 0.5 / TAU)   # covers 1/(2 tau)
        self.assertEqual(d & (d - 1), 0)                 # power of two


class TestConversions(unittest.TestCase):
    def test_xu_relations_hold_pointwise(self):
        f = np.linspace(1e3, 4e6, 5000)
        S_dphi = np.full_like(f, 1e-12)
        c = P.convert(f, S_dphi, TAU)
        m = c["mask"]
        np.testing.assert_allclose(c["S_dnu"][m],
                                   S_dphi[m] / (2 * np.pi * TAU) ** 2, rtol=1e-9)
        np.testing.assert_allclose(c["S_nu"][m], f[m] ** 2 * c["S_phi"][m],
                                   rtol=1e-9)
        np.testing.assert_allclose(c["L"][m], c["S_phi"][m] / 2, rtol=1e-9)
        # Xu's low-frequency limit: for tau -> 0, S_nu -> S_dnu
        lo = f < 1e5
        np.testing.assert_allclose(c["S_nu"][lo & m], c["S_dnu"][lo & m],
                                   rtol=0.02)

    def test_dc_end_of_the_band_survives_the_null_guard(self):
        """Regression: the guard once treated k = 0 as a null like any other.

        4 sin^2(pi f tau) does vanish at f = 0, but so does nothing else --
        S_dphi tends to a constant there, so S_phi -> h0/f^2 is finite and
        correct. Guarding around k = 0 blanked everything below
        null_guard/tau = 300 kHz at a 10 m delay, which is most of a laser
        phase-noise plot.
        """
        f = np.logspace(0, np.log10(5e6), 4000)
        c = P.convert(f, np.full_like(f, 1e-12), TAU)
        self.assertTrue(np.all(c["mask"]))
        self.assertTrue(np.all(np.isfinite(c["S_phi"])))
        # and the low-f limit is the frequency-discriminator response
        lo = f < 1e4
        np.testing.assert_allclose(c["S_phi"][lo],
                                   1e-12 / (2 * np.pi * f[lo] * TAU) ** 2,
                                   rtol=1e-3)

    def test_fsr_nulls_are_masked_not_amplified(self):
        f = np.linspace(1e3, 3.5 / TAU, 20000)
        c = P.convert(f, np.full_like(f, 1e-12), TAU)
        for k in (1, 2, 3):
            near = np.abs(f - k / TAU) < 0.01 / TAU
            self.assertTrue(np.all(np.isnan(c["S_phi"][near])),
                            f"FSR null {k}/tau not masked")
        self.assertGreater(np.count_nonzero(c["mask"]), 0.8 * f.size)


class TestLinewidth(unittest.TestCase):
    def test_beta_line_recovers_a_planted_lorentzian(self):
        # flat S_nu = h0 over [f_lo, f_hi]; the beta line crosses it at
        # f* = h0/BETA_K, so only f < f* contributes: A = h0 f*, and
        # FWHM = sqrt(8 ln2 h0 f*) = h0 sqrt(8 ln2 / BETA_K) = pi h0.
        h0 = 1e4                                   # Hz^2/Hz -> dnu = pi h0
        f = np.logspace(0, 7, 20000)
        _, fwhm = P.linewidth_curve(f, np.full_like(f, h0), np.array([1.0]))
        self.assertAlmostEqual(fwhm[0] / (np.pi * h0), 1.0, delta=0.02)

    def test_flicker_makes_linewidth_grow_with_observation_time(self):
        f = np.logspace(0, 7, 20000)
        S = 1e4 + 1e9 / f                          # white + 1/f FM
        f_lo, fwhm = P.linewidth_curve(f, S, np.logspace(0, 4, 20))
        self.assertTrue(np.all(np.diff(fwhm) <= 1e-9))   # falls with f_lo
        self.assertGreater(fwhm[0], 2 * fwhm[-1])

    def test_white_linewidth(self):
        f = np.logspace(3, 7, 5000)
        dnu, h0 = P.white_linewidth(f, np.full_like(f, 2e4), band=(1e6, 1e7))
        self.assertAlmostEqual(h0, 2e4, delta=1)
        self.assertAlmostEqual(dnu, np.pi * 2e4, delta=10)


class TestEndToEnd(unittest.TestCase):
    """The whole chain: photocurrents in, linewidth out."""

    def _run(self, dnu, n=1 << 21, **kw):
        x, y, _ = make_interferogram(n, dnu_hz=dnu, drift_fringes=3.0, **kw)
        return P.analyse(x, y, FS, TAU, npts=400)

    def test_recovers_50khz_linewidth(self):
        r = self._run(50e3)
        self.assertNotIn("error", r)
        self.assertAlmostEqual(r["lw_white_hz"] / 50e3, 1.0, delta=0.15)
        self.assertAlmostEqual(r["cal"]["psi_deg"], 118.0, delta=1.5)
        self.assertEqual(r["slips"], 0)

    def test_recovers_500khz_linewidth(self):
        r = self._run(500e3)
        self.assertAlmostEqual(r["lw_white_hz"] / 500e3, 1.0, delta=0.15)

    def test_S_nu_is_flat_for_a_white_fm_laser(self):
        r = self._run(200e3)
        f, S = r["f"], r["S_nu"]
        band = np.isfinite(S) & (f > 3e4) & (f < r["f_max_hz"])
        db = 10 * np.log10(S[band])
        self.assertLess(db.std(), 2.0)             # dB, over 2+ decades
        self.assertAlmostEqual(10 * np.log10(np.median(S[band]) /
                                             (200e3 / np.pi)), 0.0, delta=1.0)

    def test_the_sin2_correction_is_what_makes_it_flat(self):
        # S_dphi itself is NOT flat -- it carries the interferometer's
        # 4 sin^2(pi f tau) transfer function. If someone plots S_dnu and
        # calls it frequency noise, this is the error they are making.
        r = self._run(200e3)
        f = r["f"]
        fmax = r["f_max_hz"]
        lo = np.isfinite(f) & (f > 1e5) & (f < 3e5)
        hi = np.isfinite(f) & (f > 3e6) & (f <= fmax)

        def tilt(S):
            return 10 * np.log10(np.median(S[lo]) / np.median(S[hi]))

        # S_dnu carries the interferometer's sinc^2(pi f tau) response and
        # droops by ~3 dB by the time it reaches 1/(2 tau); S_nu, which has
        # had it divided out, does not. Anyone plotting S_dnu and calling it
        # frequency noise is reading that droop as laser physics.
        sinc2 = lambda fr: (np.sin(np.pi * fr * TAU) / (np.pi * fr * TAU)) ** 2
        want = 10 * np.log10(sinc2(np.median(f[lo])) / sinc2(np.median(f[hi])))
        self.assertGreater(want, 2.0)                      # sanity on the bands
        self.assertAlmostEqual(tilt(r["S_dnu"]), want, delta=0.7)
        self.assertLess(abs(tilt(r["S_nu"])), 1.0)

    def test_refuses_to_report_on_an_uncalibrated_fringe(self):
        x, y, _ = make_interferogram(1 << 19, dnu_hz=1e3, drift_fringes=0.0)
        r = P.analyse(x, y, FS, TAU)
        self.assertIn("error", r)
        self.assertIn("not trustworthy", r["error"])
        self.assertIn("cal", r)                    # still says WHY
        # and can be overridden deliberately
        r2 = P.analyse(x, y, FS, TAU, require_cal=False)
        self.assertNotIn("error", r2)
        self.assertTrue(any("provisional" in w for w in r2["warn"]))

    def test_phase_noise_falls_as_one_over_f_squared(self):
        # white FM => S_phi = h0/f^2 => -20 dB/decade, and L(f) = S_phi/2
        r = self._run(200e3, n=1 << 22)
        f, S = r["f"], r["S_phi"]
        m = np.isfinite(S) & (f > 1e4) & (f < 1e6)
        slope = np.polyfit(np.log10(f[m]), 10 * np.log10(S[m]), 1)[0]
        self.assertAlmostEqual(slope, -20.0, delta=1.5)
        np.testing.assert_allclose(r["L"][m], S[m] / 2, rtol=1e-9)

    def test_reports_a_short_record_honestly(self):
        r = P.analyse(np.zeros(1000), np.zeros(1000), FS, TAU)
        self.assertIn("error", r)

    def test_flat_input_is_an_error_not_a_number(self):
        r = P.analyse(np.zeros(1 << 16), np.zeros(1 << 16), FS, TAU)
        self.assertIn("error", r)

    def test_drift_rate_is_a_laser_frequency_rate(self):
        # the generator sweeps `drift_fringes` fringes across the record; one
        # fringe is 2 pi of dphi, which is 1/tau of apparent laser frequency
        n = 1 << 21
        x, y, _ = make_interferogram(n, drift_fringes=3.0)
        r = P.analyse(x, y, FS, TAU)
        want = 3.0 / TAU / (n / FS)              # Hz per second
        self.assertAlmostEqual(abs(r["drift_hz_per_s"]) / want, 1.0, delta=0.02)

    def test_display_arrays_are_wire_sized(self):
        r = self._run(100e3)
        self.assertLessEqual(r["f"].size, 400)
        self.assertLessEqual(r["liss_x"].size, 1200)
        self.assertLessEqual(r["dphi_min"].size, 1024)
        self.assertEqual(r["dphi_min"].size, r["dphi_max"].size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
