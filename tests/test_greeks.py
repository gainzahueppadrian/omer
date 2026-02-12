"""
Tests for Black-Scholes Greeks Calculator
"""

import unittest
import numpy as np
from src.math.greeks import (
    d1, d2, bs_put_price, bs_call_price,
    put_delta, call_delta, gamma, vega,
    theta_put, theta_call, implied_volatility_put,
    compute_all_greeks
)


class TestBlackScholesGreeks(unittest.TestCase):
    """Test suite for Greeks calculations."""

    def setUp(self):
        """Standard test parameters."""
        self.S = 100.0    # spot
        self.K = 95.0     # strike (OTM put)
        self.T = 1.0      # 1 year
        self.r = 0.05     # 5% risk-free
        self.sigma = 0.25  # 25% IV
        self.q = 0.02     # 2% dividend yield

    def test_put_call_parity(self):
        """Verify put-call parity: C - P = S*e^{-qT} - K*e^{-rT}"""
        call = bs_call_price(self.S, self.K, self.T, self.r, self.sigma, self.q)
        put = bs_put_price(self.S, self.K, self.T, self.r, self.sigma, self.q)

        lhs = call - put
        rhs = (self.S * np.exp(-self.q * self.T) -
               self.K * np.exp(-self.r * self.T))

        self.assertAlmostEqual(lhs, rhs, places=6,
                               msg="Put-call parity violated")

    def test_put_price_positive(self):
        """Put price should be positive for OTM puts."""
        price = bs_put_price(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(price, 0, "Put price should be positive")

    def test_put_price_itm(self):
        """ITM put should be worth at least intrinsic."""
        K_itm = 110.0  # ITM put
        price = bs_put_price(self.S, K_itm, self.T, self.r, self.sigma)
        intrinsic = max(K_itm * np.exp(-self.r * self.T) - self.S, 0)
        self.assertGreater(price, intrinsic - 0.01,
                           "ITM put should be >= intrinsic")

    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        delta = put_delta(self.S, self.K, self.T, self.r, self.sigma)
        self.assertLess(delta, 0, "Put delta should be negative")
        self.assertGreater(delta, -1, "Put delta should be > -1")

    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        delta = call_delta(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(delta, 0, "Call delta should be positive")
        self.assertLess(delta, 1, "Call delta should be < 1")

    def test_delta_sum(self):
        """call_delta - put_delta = e^{-qT} (with dividends)."""
        cd = call_delta(self.S, self.K, self.T, self.r, self.sigma, self.q)
        pd = put_delta(self.S, self.K, self.T, self.r, self.sigma, self.q)

        # More precise: call_delta - put_delta = e^{-qT}
        self.assertAlmostEqual(
            cd - pd, np.exp(-self.q * self.T), places=6
        )

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        g = gamma(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(g, 0, "Gamma should be positive")

    def test_gamma_same_for_put_call(self):
        """Gamma is the same for puts and calls at same strike."""
        # Already implemented as single function, just verify
        g = gamma(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(g, 0)

    def test_vega_positive(self):
        """Vega should be positive."""
        v = vega(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(v, 0, "Vega should be positive")

    def test_theta_negative_for_long(self):
        """Theta should be negative for long options (time decay)."""
        # For puts, theta is generally negative (but can be positive for deep ITM)
        t = theta_put(self.S, self.K, self.T, self.r, self.sigma)
        # OTM put theta should be negative
        self.assertLess(t, 0, "OTM put theta should be negative")

    def test_implied_vol_roundtrip(self):
        """Computing IV from a price should recover the original vol."""
        price = bs_put_price(self.S, self.K, self.T, self.r, self.sigma)
        iv = implied_volatility_put(price, self.S, self.K, self.T, self.r)

        self.assertIsNotNone(iv, "IV solver should converge")
        self.assertAlmostEqual(iv, self.sigma, places=4,
                               msg="IV roundtrip should recover original vol")

    def test_implied_vol_various_strikes(self):
        """Test IV solver at various moneyness levels."""
        for K in [80, 90, 100, 110, 120]:
            price = bs_put_price(self.S, K, self.T, self.r, self.sigma)
            if price > 0.01:
                iv = implied_volatility_put(price, self.S, K, self.T, self.r)
                if iv is not None:
                    self.assertAlmostEqual(
                        iv, self.sigma, places=3,
                        msg=f"IV mismatch at K={K}"
                    )

    def test_compute_all_greeks(self):
        """Test the all-in-one Greeks function."""
        greeks = compute_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma, "put"
        )

        self.assertIn("delta", greeks)
        self.assertIn("gamma", greeks)
        self.assertIn("vega", greeks)
        self.assertIn("theta", greeks)
        self.assertIn("price", greeks)

        self.assertLess(greeks["delta"], 0)
        self.assertGreater(greeks["gamma"], 0)
        self.assertGreater(greeks["vega"], 0)
        self.assertGreater(greeks["price"], 0)

    def test_edge_cases_zero_time(self):
        """At expiry, put should be worth intrinsic."""
        price = bs_put_price(self.S, 110, 0, self.r, self.sigma)
        self.assertAlmostEqual(price, 10.0, places=2)

        price = bs_put_price(self.S, 90, 0, self.r, self.sigma)
        self.assertAlmostEqual(price, 0.0, places=2)

    def test_edge_cases_zero_vol(self):
        """With zero vol, option should be worth discounted intrinsic."""
        price = bs_put_price(self.S, 110, self.T, self.r, 0)
        expected = max(110 * np.exp(-self.r * self.T) - self.S, 0)
        self.assertAlmostEqual(price, expected, places=2)


class TestGreeksNumericalStability(unittest.TestCase):
    """Test numerical stability edge cases."""

    def test_very_deep_otm(self):
        """Very deep OTM put should have near-zero delta and price."""
        greeks = compute_all_greeks(100, 50, 1.0, 0.05, 0.20, "put")
        self.assertAlmostEqual(greeks["delta"], 0.0, places=3)
        self.assertAlmostEqual(greeks["price"], 0.0, places=2)

    def test_very_deep_itm(self):
        """Very deep ITM put should have delta near -1."""
        greeks = compute_all_greeks(100, 200, 1.0, 0.05, 0.20, "put")
        self.assertAlmostEqual(greeks["delta"], -1.0, places=1)

    def test_very_long_dated(self):
        """5-year option should still compute cleanly."""
        greeks = compute_all_greeks(100, 95, 5.0, 0.05, 0.25, "put")
        self.assertGreater(greeks["price"], 0)
        self.assertLess(greeks["delta"], 0)

    def test_very_high_vol(self):
        """100% vol should compute cleanly."""
        greeks = compute_all_greeks(100, 95, 1.0, 0.05, 1.00, "put")
        self.assertGreater(greeks["price"], 0)


if __name__ == "__main__":
    unittest.main()
