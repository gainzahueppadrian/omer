import unittest
import numpy as np
from src.math.rnd import malz_smile, calculate_risk_of_reversal

class TestRND(unittest.TestCase):
    def test_malz_smile(self):
        # Test ATM
        # delta = 0.5
        # sigma = atm - 0 + 0 = atm
        iv = malz_smile(0.5, 0.20, -0.05, 0.02)
        self.assertAlmostEqual(iv, 0.20)

        # Test 25D Call (delta = 0.25)
        # sigma = 0.20 - 2*(-0.05)*(-0.25) + 16*0.02*(-0.25)^2
        # sigma = 0.20 - 0.025 + 0.02 = 0.195
        iv_25 = malz_smile(0.25, 0.20, -0.05, 0.02)
        expected = 0.20 - 2*(-0.05)*(0.25-0.5) + 16*0.02*(0.25-0.5)**2
        self.assertAlmostEqual(iv_25, expected)

    def test_risk_of_reversal_run(self):
        # Just check it runs and returns a valid prob
        S = 100
        K = 90
        T = 1.0
        r = 0.04
        atm = 0.20
        rr = -0.05
        strangle = 0.02

        prob = calculate_risk_of_reversal(S, K, T, r, atm, rr, strangle)
        print(f"Calculated Risk of Reversal (Prob < {K}): {prob:.4f}")
        self.assertTrue(0 <= prob <= 1)

        # Check that lower strike has lower prob
        prob_lower = calculate_risk_of_reversal(S, 80, T, r, atm, rr, strangle)
        self.assertLess(prob_lower, prob)

if __name__ == '__main__':
    unittest.main()
