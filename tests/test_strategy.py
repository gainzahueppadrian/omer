import unittest
from src.client.ib_client import IBClient
from src.agents.risk_agent import RiskAgent
from src.strategy.leaps_puts import LeapsPutsStrategy

class TestStrategy(unittest.TestCase):
    def test_scan_opportunities(self):
        client = IBClient(mock=True)
        # Setup mock data (Very High IV for LEAPS to meet >10% yield at <40% delta)
        client.set_mock_data("AAPL", 150.0, iv=0.50)

        risk_agent = RiskAgent(max_risk_of_reversal=0.40)
        strategy = LeapsPutsStrategy(client, risk_agent)

        opps = strategy.scan_opportunities(["AAPL"], expiry_months=12)

        self.assertTrue(len(opps) > 0)

        # Verify fields
        opp = opps[0]
        self.assertIn("symbol", opp)
        self.assertIn("strike", opp)
        self.assertIn("premium", opp)
        self.assertIn("roc_annualized", opp)

        # Check logic: Strike < Spot
        self.assertLess(opp['strike'], 150.0)

        # Check approved
        self.assertTrue(opp['allocation_mult'] > 0)

if __name__ == '__main__':
    unittest.main()
