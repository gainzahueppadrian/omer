"""
Tests for LEAPS Strategy Logic
"""

import unittest
from src.strategy.leaps_strategy import LEAPSStrategy
from src.math.rnd import RNDModel


class TestLEAPSStrategy(unittest.TestCase):
    """Test LEAPS strategy calculations."""

    def setUp(self):
        self.config = {
            "strategy": {
                "leaps": {
                    "min_dte": 270,
                    "max_dte": 548,
                    "target_dte": 365,
                    "min_delta": -0.30,
                    "max_delta": -0.45,
                    "min_annualized_return": 0.10,
                    "target_annualized_return": 0.14,
                    "max_risk_of_reversal": 0.50,
                },
                "bonus_calls": {
                    "enabled": True,
                    "max_premium_allocation": 0.50,
                    "call_delta_target": 0.40,
                },
            },
            "risk": {
                "margin_safety_factor": 1.5,
            },
        }
        self.strategy = LEAPSStrategy(self.config)

    def test_annualized_return_calculation(self):
        """Test annualized return calculation."""
        # $5 premium, $95 strike, 365 DTE
        ret = self.strategy.calculate_annualized_return(5.0, 95.0, 365)
        expected = (5.0 * 100) / (95.0 * 100) * (365.0 / 365)
        self.assertAlmostEqual(ret, expected, places=4)

        # Should be approximately 5.26%
        self.assertAlmostEqual(ret, 0.0526, places=3)

    def test_annualized_return_higher_for_shorter_dte(self):
        """Same premium, shorter DTE should give higher annualized return."""
        ret_365 = self.strategy.calculate_annualized_return(5.0, 95.0, 365)
        ret_180 = self.strategy.calculate_annualized_return(5.0, 95.0, 180)

        self.assertGreater(ret_180, ret_365,
                           "Shorter DTE should give higher annualized return")

    def test_break_even(self):
        """Test break-even calculation."""
        be = self.strategy.calculate_break_even(95.0, 5.0)
        self.assertEqual(be, 90.0)

    def test_margin_requirement(self):
        """Test margin requirement estimation."""
        margin = self.strategy.calculate_margin_requirement(
            strike=95.0, spot=100.0, premium=5.0, margin_safety=1.5
        )
        self.assertGreater(margin, 0, "Margin should be positive")
        # Should be reasonable (not astronomical)
        self.assertLess(margin, 50000, "Margin should be reasonable")

    def test_evaluate_candidate_viable(self):
        """Test evaluation of a viable candidate."""
        from datetime import datetime, timedelta
        expiry = (datetime.now() + timedelta(days=365)).strftime("%Y%m%d")

        result = self.strategy.evaluate_candidate(
            symbol="AAPL",
            spot=150.0,
            strike=130.0,
            expiry=expiry,
            premium=14.00,
            iv=0.28,
            delta=-0.35,
        )

        self.assertTrue(result["viable"], f"Should be viable: {result.get('reasons')}")
        self.assertGreater(result["annualized_return"], 0)
        self.assertGreater(result["break_even"], 0)
        self.assertLess(result["break_even"], result["strike"])

    def test_evaluate_candidate_rejected_low_return(self):
        """Candidate with low return should be rejected."""
        from datetime import datetime, timedelta
        expiry = (datetime.now() + timedelta(days=365)).strftime("%Y%m%d")

        result = self.strategy.evaluate_candidate(
            symbol="AAPL",
            spot=150.0,
            strike=130.0,
            expiry=expiry,
            premium=1.0,  # Very low premium
            iv=0.15,
            delta=-0.35,
        )

        self.assertFalse(result["viable"])

    def test_evaluate_candidate_rejected_high_delta(self):
        """Candidate with too high delta should be rejected."""
        from datetime import datetime, timedelta
        expiry = (datetime.now() + timedelta(days=365)).strftime("%Y%m%d")

        result = self.strategy.evaluate_candidate(
            symbol="AAPL",
            spot=150.0,
            strike=145.0,
            expiry=expiry,
            premium=15.0,
            iv=0.30,
            delta=-0.55,  # Too high
        )

        self.assertFalse(result["viable"])

    def test_evaluate_with_rnd_model(self):
        """Test evaluation with RND model for risk of reversal."""
        from datetime import datetime, timedelta
        expiry = (datetime.now() + timedelta(days=365)).strftime("%Y%m%d")

        rnd = RNDModel()
        rnd.calibrate(150.0, 1.0, 0.05, 0.28, -0.03, 0.01)

        result = self.strategy.evaluate_candidate(
            symbol="AAPL",
            spot=150.0,
            strike=130.0,
            expiry=expiry,
            premium=8.50,
            iv=0.28,
            delta=-0.35,
            rnd_model=rnd,
        )

        self.assertIn("risk_of_reversal", result)
        self.assertGreater(result["risk_of_reversal"], 0)
        self.assertLess(result["risk_of_reversal"], 1)

    def test_rank_candidates(self):
        """Test candidate ranking."""
        candidates = [
            {"viable": True, "return_per_risk": 1.5,
             "annualized_return": 0.12, "risk_of_reversal": 0.10, "iv": 0.25},
            {"viable": True, "return_per_risk": 2.0,
             "annualized_return": 0.15, "risk_of_reversal": 0.08, "iv": 0.30},
            {"viable": False, "return_per_risk": 3.0,
             "annualized_return": 0.20, "risk_of_reversal": 0.05, "iv": 0.35},
            {"viable": True, "return_per_risk": 0.5,
             "annualized_return": 0.08, "risk_of_reversal": 0.20, "iv": 0.20},
        ]

        ranked = self.strategy.rank_candidates(candidates)

        # Only viable candidates should be returned
        self.assertEqual(len(ranked), 3, "Only viable candidates should be ranked")

        # First should have highest composite score
        scores = [c["composite_score"] for c in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         "Candidates should be sorted by composite score")

    def test_bonus_strategy_assessment(self):
        """Test bonus strategy assessment in evaluation."""
        from datetime import datetime, timedelta
        expiry = (datetime.now() + timedelta(days=365)).strftime("%Y%m%d")

        result = self.strategy.evaluate_candidate(
            symbol="AAPL",
            spot=150.0,
            strike=130.0,
            expiry=expiry,
            premium=8.50,
            iv=0.28,
            delta=-0.35,
        )

        if result["viable"]:
            bonus = result.get("bonus_strategy", {})
            self.assertTrue(bonus.get("enabled"), "Bonus should be enabled")
            self.assertGreater(bonus.get("budget_per_contract", 0), 0)
            self.assertEqual(bonus.get("target_call_delta"), 0.40)


if __name__ == "__main__":
    unittest.main()
