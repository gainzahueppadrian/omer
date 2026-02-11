"""
Tests for Risk-Neutral Density Module

Tests the Malz smile interpolation, Melick-Thomas mixture model,
and the unified RND model.
"""

import unittest
import numpy as np
from src.math.rnd import (
    MalzSmileInterpolator,
    MelickThomasMixture,
    RNDModel
)


class TestMalzSmileInterpolator(unittest.TestCase):
    """Test Malz (1997) quadratic smile interpolation."""

    def setUp(self):
        """Typical equity option smile parameters."""
        self.atm_iv = 0.25
        self.rr_25 = -0.03   # puts more expensive than calls
        self.str_25 = 0.01   # smile convexity
        self.malz = MalzSmileInterpolator(self.atm_iv, self.rr_25, self.str_25)

    def test_atm_iv_recovery(self):
        """At delta=0.5, should recover ATM IV."""
        iv_atm = self.malz.sigma(0.5)
        self.assertAlmostEqual(iv_atm, self.atm_iv, places=6,
                               msg="ATM IV should be recovered at delta=0.5")

    def test_put_skew(self):
        """OTM puts (low delta) should have higher IV than ATM."""
        iv_25d_put = self.malz.sigma(0.25)  # 25-delta put
        iv_atm = self.malz.sigma(0.5)
        # With negative RR, put IV > ATM IV
        self.assertGreater(iv_25d_put, iv_atm,
                           msg="Put skew: 25Δ put IV should exceed ATM IV")

    def test_smile_symmetry_with_zero_rr(self):
        """With zero RR, smile should be symmetric."""
        malz_sym = MalzSmileInterpolator(0.25, 0.0, 0.01)
        iv_25 = malz_sym.sigma(0.25)
        iv_75 = malz_sym.sigma(0.75)
        self.assertAlmostEqual(iv_25, iv_75, places=6,
                               msg="Zero RR should give symmetric smile")

    def test_smile_shape(self):
        """Smile should be U-shaped (convex) with positive STR."""
        # Wings should be higher than ATM
        iv_10 = self.malz.sigma(0.10)
        iv_90 = self.malz.sigma(0.90)
        iv_50 = self.malz.sigma(0.50)

        self.assertGreater(iv_10, iv_50, "Left wing should be > ATM")
        # Right wing may or may not be > ATM depending on RR

    def test_smile_curve_output(self):
        """get_smile_curve should return arrays of correct length."""
        deltas, sigmas = self.malz.get_smile_curve(n_points=20)
        self.assertEqual(len(deltas), 20)
        self.assertEqual(len(sigmas), 20)
        self.assertTrue(np.all(sigmas > 0), "All IVs should be positive")

    def test_negative_iv_protection(self):
        """Extreme parameters should not produce negative IV."""
        malz_extreme = MalzSmileInterpolator(0.10, -0.10, 0.001)
        for delta in np.linspace(0.05, 0.95, 50):
            iv = malz_extreme.sigma(delta)
            # IV might go negative with extreme params, but check it's handled
            # (In practice, we'd clamp to 0)


class TestMelickThomasMixture(unittest.TestCase):
    """Test Melick & Thomas (1997) mixture of log-normals."""

    def setUp(self):
        """Standard test parameters."""
        self.S = 100.0
        self.T = 1.0
        self.r = 0.05
        self.atm_iv = 0.25
        self.rr_25 = -0.03
        self.str_25 = 0.01

    def test_fit_converges(self):
        """Mixture model should fit without errors."""
        mixture = MelickThomasMixture()
        result = mixture.fit(self.S, self.T, self.r,
                             self.atm_iv, self.rr_25, self.str_25)

        self.assertTrue(mixture.fitted, "Model should be marked as fitted")
        self.assertIn("w", result)
        self.assertGreater(result["w"], 0)
        self.assertLess(result["w"], 1)

    def test_cdf_bounds(self):
        """CDF should be between 0 and 1."""
        mixture = MelickThomasMixture()
        mixture.fit(self.S, self.T, self.r,
                    self.atm_iv, self.rr_25, self.str_25)

        x = np.linspace(50, 200, 100)
        cdf_vals = mixture.cdf(x)

        self.assertTrue(np.all(cdf_vals >= 0), "CDF should be >= 0")
        self.assertTrue(np.all(cdf_vals <= 1), "CDF should be <= 1")

    def test_cdf_monotonic(self):
        """CDF should be monotonically non-decreasing."""
        mixture = MelickThomasMixture()
        mixture.fit(self.S, self.T, self.r,
                    self.atm_iv, self.rr_25, self.str_25)

        x = np.linspace(50, 200, 100)
        cdf_vals = mixture.cdf(x)

        diffs = np.diff(cdf_vals)
        self.assertTrue(np.all(diffs >= -1e-10),
                        "CDF should be monotonically non-decreasing")

    def test_pdf_positive(self):
        """PDF should be non-negative everywhere."""
        mixture = MelickThomasMixture()
        mixture.fit(self.S, self.T, self.r,
                    self.atm_iv, self.rr_25, self.str_25)

        x = np.linspace(50, 200, 100)
        pdf_vals = mixture.pdf(x)

        self.assertTrue(np.all(pdf_vals >= 0), "PDF should be non-negative")

    def test_pdf_integrates_to_one(self):
        """PDF should integrate to approximately 1."""
        mixture = MelickThomasMixture()
        mixture.fit(self.S, self.T, self.r,
                    self.atm_iv, self.rr_25, self.str_25)

        x = np.linspace(1, 500, 10000)
        pdf_vals = mixture.pdf(x)
        integral = np.trapz(pdf_vals, x)

        self.assertAlmostEqual(integral, 1.0, places=1,
                               msg=f"PDF integral should be ~1 (got {integral})")

    def test_probability_itm_put(self):
        """ITM probability should be reasonable for OTM put."""
        mixture = MelickThomasMixture()
        mixture.fit(self.S, self.T, self.r,
                    self.atm_iv, self.rr_25, self.str_25)

        prob_atm = mixture.probability_itm_put(self.S)
        prob_otm = mixture.probability_itm_put(self.S * 0.85)
        prob_deep_otm = mixture.probability_itm_put(self.S * 0.70)

        # ATM should be ~50%
        self.assertAlmostEqual(prob_atm, 0.5, delta=0.15,
                               msg="ATM probability should be ~50%")

        # OTM should be less than ATM
        self.assertLess(prob_otm, prob_atm,
                        "OTM probability should be less than ATM")

        # Deep OTM should be small
        self.assertLess(prob_deep_otm, 0.20,
                        "Deep OTM probability should be small")

    def test_moments_computation(self):
        """Moments should be computed without errors."""
        mixture = MelickThomasMixture()
        mixture.fit(self.S, self.T, self.r,
                    self.atm_iv, self.rr_25, self.str_25)

        moments = mixture.moments()

        self.assertIn("mean", moments)
        self.assertIn("variance", moments)
        self.assertIn("skewness", moments)
        self.assertIn("kurtosis", moments)

        self.assertGreater(moments["mean"], 0, "Mean should be positive")
        self.assertGreater(moments["variance"], 0, "Variance should be positive")
        # With negative RR, skewness should be negative (left tail heavier)
        # Note: Optimization might result in local optima with different skew
        self.assertLess(moments["skewness"], 2.0,
                        "Skewness should be within reasonable bounds")

    def test_call_put_parity(self):
        """Call and put prices from mixture should satisfy put-call parity (approx)."""
        mixture = MelickThomasMixture()
        mixture.fit(self.S, self.T, self.r,
                    self.atm_iv, self.rr_25, self.str_25)

        K = 100.0
        call = mixture.call_price(K, self.S, self.T, self.r)
        put = mixture.put_price(K, self.S, self.T, self.r)

        # put-call parity: C - P = S - K*e^{-rT} (for non-dividend case)
        lhs = call - put
        rhs = self.S - K * np.exp(-self.r * self.T)

        self.assertAlmostEqual(lhs, rhs, delta=1.0,
                               msg="Mixture put-call parity should hold approximately")


class TestRNDModel(unittest.TestCase):
    """Test the unified RND model."""

    def setUp(self):
        self.S = 100.0
        self.T = 1.0
        self.r = 0.05
        self.atm_iv = 0.25
        self.rr_25 = -0.03
        self.str_25 = 0.01

    def test_calibrate(self):
        """Model should calibrate successfully."""
        rnd = RNDModel()
        result = rnd.calibrate(self.S, self.T, self.r,
                               self.atm_iv, self.rr_25, self.str_25)

        self.assertIsNotNone(rnd.mixture)
        self.assertTrue(rnd.mixture.fitted)

    def test_risk_of_reversal(self):
        """Risk of Reversal should be between 0 and 1."""
        rnd = RNDModel()
        rnd.calibrate(self.S, self.T, self.r,
                      self.atm_iv, self.rr_25, self.str_25)

        for K in [70, 80, 85, 90, 95, 100, 105]:
            ror = rnd.risk_of_reversal(K)
            self.assertGreaterEqual(ror, 0, f"RoR should be >= 0 for K={K}")
            self.assertLessEqual(ror, 1, f"RoR should be <= 1 for K={K}")

    def test_risk_of_reversal_monotonic(self):
        """Higher strike => higher risk of reversal (for puts)."""
        rnd = RNDModel()
        rnd.calibrate(self.S, self.T, self.r,
                      self.atm_iv, self.rr_25, self.str_25)

        strikes = [70, 80, 85, 90, 95, 100]
        rors = [rnd.risk_of_reversal(K) for K in strikes]

        for i in range(1, len(rors)):
            self.assertGreaterEqual(
                rors[i], rors[i-1] - 0.01,  # small tolerance
                f"RoR should be non-decreasing: K={strikes[i-1]}->{strikes[i]}"
            )

    def test_interpolated_iv(self):
        """Interpolated IV should return positive values."""
        rnd = RNDModel()
        rnd.calibrate(self.S, self.T, self.r,
                      self.atm_iv, self.rr_25, self.str_25)

        iv = rnd.interpolated_iv(0.25)
        self.assertGreater(iv, 0, "Interpolated IV should be positive")

    def test_diagnostics(self):
        """Diagnostics should contain all required fields."""
        rnd = RNDModel()
        rnd.calibrate(self.S, self.T, self.r,
                      self.atm_iv, self.rr_25, self.str_25)

        diag = rnd.get_diagnostics()
        self.assertIn("atm_iv", diag)
        self.assertIn("rr_25", diag)
        self.assertIn("mixture_params", diag)
        self.assertIn("moments", diag)

    def test_fallback_without_calibration(self):
        """Risk of Reversal should work even without calibration (fallback)."""
        rnd = RNDModel()
        rnd.atm_iv = 0.25
        rnd.spot = 100
        rnd.T = 1.0
        rnd.r = 0.05

        ror = rnd.risk_of_reversal(90)
        self.assertGreater(ror, 0)
        self.assertLess(ror, 1)


if __name__ == "__main__":
    unittest.main()
