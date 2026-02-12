"""
Tests for Multi-Agent System Components

Tests the message bus, agent communication patterns,
and agent logic (without requiring IBKR connection).
"""

import unittest
import threading
import time
from unittest.mock import MagicMock, patch

from src.agents.message_bus import (
    MessageBus, AgentMessage, MessagePriority
)
from src.agents.risk_agent import DynamicRiskAgent


class TestMessageBus(unittest.TestCase):
    """Test the message bus pub/sub system."""

    def setUp(self):
        self.bus = MessageBus()

    def tearDown(self):
        self.bus.shutdown()

    def test_subscribe_and_publish(self):
        """Messages should be delivered to subscribers."""
        received = []

        def callback(msg):
            received.append(msg)

        self.bus.subscribe("test_topic", callback)

        msg = AgentMessage(
            topic="test_topic",
            sender="test_sender",
            payload={"data": "hello"},
        )
        self.bus.publish(msg)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["data"], "hello")

    def test_multiple_subscribers(self):
        """Multiple subscribers should all receive the message."""
        received_a = []
        received_b = []

        self.bus.subscribe("topic", lambda m: received_a.append(m))
        self.bus.subscribe("topic", lambda m: received_b.append(m))

        self.bus.publish(AgentMessage(
            topic="topic", sender="s", payload={}
        ))

        self.assertEqual(len(received_a), 1)
        self.assertEqual(len(received_b), 1)

    def test_topic_isolation(self):
        """Messages on different topics should not cross."""
        received = []

        self.bus.subscribe("topic_a", lambda m: received.append(m))

        self.bus.publish(AgentMessage(
            topic="topic_b", sender="s", payload={}
        ))

        self.assertEqual(len(received), 0,
                         "Subscriber should not receive messages from other topics")

    def test_priority_ordering(self):
        """Higher priority messages should be delivered first in queues."""
        q = self.bus.create_agent_queue("test_agent")

        # Publish low priority first, then high
        self.bus.publish(AgentMessage(
            topic="test_agent", sender="s",
            payload={"order": 1},
            priority=MessagePriority.LOW,
        ))
        self.bus.publish(AgentMessage(
            topic="test_agent", sender="s",
            payload={"order": 2},
            priority=MessagePriority.CRITICAL,
        ))

        # Critical should come out first
        _, _, msg1 = q.get(timeout=1)
        _, _, msg2 = q.get(timeout=1)

        self.assertEqual(msg1.priority, MessagePriority.CRITICAL)
        self.assertEqual(msg2.priority, MessagePriority.LOW)

    def test_message_history(self):
        """Message history should be maintained."""
        for i in range(5):
            self.bus.publish(AgentMessage(
                topic="test", sender="s", payload={"i": i}
            ))

        history = self.bus.get_history(topic="test")
        self.assertEqual(len(history), 5)

    def test_agent_queue_creation(self):
        """Agent queues should be created correctly."""
        q = self.bus.create_agent_queue("my_agent")
        self.assertIsNotNone(q)

    def test_broadcast_topic(self):
        """Broadcast messages should reach all agent queues."""
        q1 = self.bus.create_agent_queue("agent1")
        q2 = self.bus.create_agent_queue("agent2")

        self.bus.publish(AgentMessage(
            topic="broadcast", sender="s", payload={"msg": "all"}
        ))

        self.assertFalse(q1.empty())
        self.assertFalse(q2.empty())

    def test_subscriber_error_handling(self):
        """Errors in subscribers should not crash the bus."""
        def bad_callback(msg):
            raise RuntimeError("Subscriber error")

        self.bus.subscribe("test", bad_callback)

        # Should not raise
        self.bus.publish(AgentMessage(
            topic="test", sender="s", payload={}
        ))


class TestDynamicRiskAgent(unittest.TestCase):
    """Test Risk Agent logic without IBKR connection."""

    def setUp(self):
        self.bus = MessageBus()
        self.config = {
            "risk": {
                "max_portfolio_delta": -500,
                "max_single_position_pct": 0.15,
                "max_total_notional_pct": 0.80,
                "max_gamma_exposure": 100,
                "base_allocation_pct": 0.10,
                "margin_safety_factor": 1.5,
                "dynamic_adjustments": {
                    "high_vol_threshold": 0.35,
                    "low_vol_threshold": 0.15,
                    "skew_warning_threshold": -0.03,
                    "kurtosis_warning_threshold": 4.0,
                },
            },
        }
        self.risk_agent = DynamicRiskAgent(self.bus, self.config)

    def tearDown(self):
        self.bus.shutdown()

    def test_regime_detection_high_vol(self):
        """High IV should trigger HIGH_VOL regime."""
        self.risk_agent._detect_regime(0.40, -0.05, -1.0, 4.0)
        self.assertEqual(self.risk_agent.market_regime, "HIGH_VOL")

    def test_regime_detection_low_vol(self):
        """Low IV should trigger LOW_VOL regime."""
        self.risk_agent._detect_regime(0.12, -0.01, -0.3, 3.0)
        self.assertEqual(self.risk_agent.market_regime, "LOW_VOL")

    def test_regime_detection_normal(self):
        """Normal IV should trigger NORMAL regime."""
        self.risk_agent._detect_regime(0.25, -0.03, -0.5, 3.2)
        self.assertEqual(self.risk_agent.market_regime, "NORMAL")

    def test_parameter_adjustment_high_vol(self):
        """In HIGH_VOL, allocation should decrease."""
        base_alloc = self.risk_agent.base_params["base_allocation_pct"]

        self.risk_agent.market_regime = "HIGH_VOL"
        self.risk_agent._adjust_parameters(0.40, -0.03, 0.01, -0.5, 3.5)

        adjusted_alloc = self.risk_agent.active_params["base_allocation_pct"]
        self.assertLess(adjusted_alloc, base_alloc,
                        "Allocation should decrease in high vol")

    def test_parameter_adjustment_skew_bonus(self):
        """Steep negative skew should provide allocation bonus."""
        self.risk_agent.market_regime = "NORMAL"

        # With moderate RR
        self.risk_agent._adjust_parameters(0.25, -0.02, 0.01, -0.5, 3.0)
        alloc_moderate = self.risk_agent.active_params["base_allocation_pct"]

        # With steep RR
        self.risk_agent._adjust_parameters(0.25, -0.06, 0.01, -0.5, 3.0)
        alloc_steep = self.risk_agent.active_params["base_allocation_pct"]

        self.assertGreater(alloc_steep, alloc_moderate,
                           "Steep skew should increase allocation")

    def test_evaluate_opportunity_approved(self):
        """Valid opportunity should be approved."""
        self.risk_agent.market_regime = "NORMAL"
        self.risk_agent.portfolio_state = {"account_value": 100000}

        opp = {
            "symbol": "AAPL",
            "risk_of_reversal": 0.15,
            "delta": -0.35,
            "gamma": 0.01,
            "premium": 5.0,
            "strike": 130.0,
            "spot": 150.0,
        }

        result = self.risk_agent._evaluate_opportunity(opp)
        self.assertTrue(result["approved"],
                        f"Should be approved: {result.get('reason')}")
        self.assertGreater(result["suggested_quantity"], 0)

    def test_evaluate_opportunity_rejected_high_ror(self):
        """High risk of reversal should be rejected."""
        self.risk_agent.market_regime = "NORMAL"

        opp = {
            "symbol": "AAPL",
            "risk_of_reversal": 0.30,  # Too high
            "delta": -0.35,
            "gamma": 0.01,
            "premium": 5.0,
            "strike": 130.0,
            "spot": 150.0,
        }

        result = self.risk_agent._evaluate_opportunity(opp)
        self.assertFalse(result["approved"])
        self.assertIn("Risk of Reversal", result["reason"])

    def test_evaluate_opportunity_rejected_high_delta(self):
        """Delta too high should be rejected."""
        self.risk_agent.market_regime = "NORMAL"

        opp = {
            "symbol": "AAPL",
            "risk_of_reversal": 0.15,
            "delta": -0.55,  # Too high
            "gamma": 0.01,
            "premium": 5.0,
            "strike": 145.0,
            "spot": 150.0,
        }

        result = self.risk_agent._evaluate_opportunity(opp)
        self.assertFalse(result["approved"])

    def test_evaluate_opportunity_rejected_low_premium(self):
        """Very low premium should be rejected."""
        self.risk_agent.market_regime = "NORMAL"

        opp = {
            "symbol": "AAPL",
            "risk_of_reversal": 0.15,
            "delta": -0.35,
            "gamma": 0.01,
            "premium": 0.50,  # Too low
            "strike": 130.0,
            "spot": 150.0,
        }

        result = self.risk_agent._evaluate_opportunity(opp)
        self.assertFalse(result["approved"])


if __name__ == "__main__":
    unittest.main()
