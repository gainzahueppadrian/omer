"""
Dynamic Risk Agent - Autonomous Risk Management

Responsibilities:
- Receive scan results from Scanner Agent
- Dynamically adjust allocation and risk thresholds based on market regime
- Calculate position-level and portfolio-level risk
- Approve or reject trade proposals
- Monitor existing positions for risk breaches

The Risk Agent is the gatekeeper: no trade executes without its approval.
"""

import threading
import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.agents.message_bus import MessageBus, AgentMessage, MessagePriority
from src.math.greeks import compute_all_greeks

logger = logging.getLogger(__name__)


class DynamicRiskAgent:
    """
    AI-driven risk management agent that adjusts parameters dynamically.

    Market Regime Detection:
    - HIGH_VOL: VIX > 35 equivalent -> reduce position sizes, widen strike selection
    - NORMAL: 15 < VIX < 35 -> standard parameters
    - LOW_VOL: VIX < 15 -> may increase size slightly, tighter strikes

    Skew Analysis:
    - Steep negative skew -> puts are rich, favorable for selling
    - Flat/positive skew -> caution, less edge in put selling

    Kurtosis Check:
    - High excess kurtosis -> fat tails, reduce size for safety
    - Normal kurtosis -> standard sizing
    """

    def __init__(self, message_bus: MessageBus, config: Dict):
        self.bus = message_bus
        self.config = config
        self.name = "risk_agent"
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Base risk parameters (will be dynamically adjusted)
        risk_cfg = config.get("risk", {})
        self.base_params = {
            "max_portfolio_delta": risk_cfg.get("max_portfolio_delta", -500),
            "max_single_position_pct": risk_cfg.get("max_single_position_pct", 0.15),
            "max_total_notional_pct": risk_cfg.get("max_total_notional_pct", 0.80),
            "max_gamma_exposure": risk_cfg.get("max_gamma_exposure", 100),
            "base_allocation_pct": risk_cfg.get("base_allocation_pct", 0.10),
            "margin_safety_factor": risk_cfg.get("margin_safety_factor", 1.5),
        }

        # Dynamic adjustment thresholds
        dyn_cfg = risk_cfg.get("dynamic_adjustments", {})
        self.high_vol_threshold = dyn_cfg.get("high_vol_threshold", 0.35)
        self.low_vol_threshold = dyn_cfg.get("low_vol_threshold", 0.15)
        self.skew_warning_threshold = dyn_cfg.get("skew_warning_threshold", -0.03)
        self.kurtosis_warning_threshold = dyn_cfg.get(
            "kurtosis_warning_threshold", 4.0
        )

        # Current adjusted parameters
        self.active_params = dict(self.base_params)
        self.market_regime = "NORMAL"
        self.portfolio_state: Dict = {}

        # Create message queue
        self._queue = self.bus.create_agent_queue(self.name)

        # Subscribe to relevant topics
        self.bus.subscribe("risk_check_request", self._handle_risk_check)

    def start(self):
        """Start the risk agent."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=self.name
        )
        self._thread.start()
        logger.info(f"{self.name} started")

    def stop(self):
        """Stop the risk agent."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info(f"{self.name} stopped")

    def _run_loop(self):
        """Process incoming messages."""
        while self._running:
            try:
                priority, timestamp, message = self._queue.get(timeout=1.0)
                self._process_message(message)
            except Exception:
                continue

    def _process_message(self, message: AgentMessage):
        """Route incoming messages to appropriate handlers."""
        payload = message.payload
        msg_type = payload.get("type", "")

        if msg_type == "scan_results":
            self._handle_scan_results(message)
        elif msg_type == "risk_check":
            self._handle_risk_check(message)
        elif msg_type == "portfolio_update":
            self._handle_portfolio_update(message)
        else:
            logger.debug(f"Risk agent: unhandled message type '{msg_type}'")

    def _handle_scan_results(self, message: AgentMessage):
        """
        Process scan results from Scanner Agent.

        1. Extract market conditions from opportunities
        2. Detect market regime
        3. Adjust risk parameters dynamically
        4. Filter and approve opportunities
        5. Publish approved trades to Execution Agent
        """
        opportunities = message.payload.get("opportunities", [])
        if not opportunities:
            return

        # Step 1: Analyze aggregate market conditions
        avg_atm_iv = sum(o["atm_iv"] for o in opportunities) / len(opportunities)
        avg_rr = sum(o["rr_25"] for o in opportunities) / len(opportunities)
        avg_str = sum(o["str_25"] for o in opportunities) / len(opportunities)

        # Get RND moments from first opportunity (representative)
        rnd_diag = opportunities[0].get("rnd_diagnostics", {})
        moments = rnd_diag.get("moments", {})
        market_skewness = moments.get("skewness", 0)
        market_kurtosis = moments.get("kurtosis", 3)

        # Step 2: Detect market regime
        self._detect_regime(avg_atm_iv, avg_rr, market_skewness, market_kurtosis)

        # Step 3: Adjust parameters
        self._adjust_parameters(avg_atm_iv, avg_rr, avg_str,
                                 market_skewness, market_kurtosis)

        # Step 4: Filter and approve
        approved = []
        for opp in opportunities:
            approval = self._evaluate_opportunity(opp)
            if approval["approved"]:
                opp["risk_approval"] = approval
                approved.append(opp)
            else:
                logger.info(
                    f"Risk REJECTED: {opp['symbol']} P{opp['strike']} - "
                    f"{approval['reason']}"
                )

        # Step 5: Publish approved trades
        if approved:
            self.bus.publish(AgentMessage(
                topic="execution_agent",
                sender=self.name,
                payload={
                    "type": "approved_trades",
                    "trades": approved,
                    "market_regime": self.market_regime,
                    "active_params": dict(self.active_params),
                    "timestamp": datetime.now().isoformat(),
                },
                priority=MessagePriority.HIGH,
            ))

            logger.info(
                f"Risk approved {len(approved)}/{len(opportunities)} trades "
                f"(regime={self.market_regime})"
            )

        # Publish risk assessment to orchestrator
        self.bus.publish(AgentMessage(
            topic="orchestrator",
            sender=self.name,
            payload={
                "type": "risk_assessment",
                "market_regime": self.market_regime,
                "avg_iv": avg_atm_iv,
                "avg_rr": avg_rr,
                "skewness": market_skewness,
                "kurtosis": market_kurtosis,
                "approved_count": len(approved),
                "rejected_count": len(opportunities) - len(approved),
                "active_params": dict(self.active_params),
            },
            priority=MessagePriority.NORMAL,
        ))

    def _detect_regime(self, avg_iv: float, avg_rr: float,
                       skewness: float, kurtosis: float):
        """Detect current market regime from aggregate metrics."""
        previous_regime = self.market_regime

        if avg_iv > self.high_vol_threshold:
            self.market_regime = "HIGH_VOL"
        elif avg_iv < self.low_vol_threshold:
            self.market_regime = "LOW_VOL"
        else:
            self.market_regime = "NORMAL"

        # Override: extreme skew or kurtosis signals stress
        if kurtosis > self.kurtosis_warning_threshold * 1.5:
            self.market_regime = "HIGH_VOL"  # treat as high vol
        if skewness < -2.0:
            self.market_regime = "HIGH_VOL"

        if self.market_regime != previous_regime:
            logger.warning(
                f"Market regime change: {previous_regime} -> {self.market_regime} "
                f"(IV={avg_iv:.2%}, RR={avg_rr:.4f}, Skew={skewness:.2f}, "
                f"Kurt={kurtosis:.2f})"
            )

    def _adjust_parameters(self, avg_iv: float, avg_rr: float,
                            avg_str: float, skewness: float,
                            kurtosis: float):
        """
        Dynamically adjust risk parameters based on market conditions.

        This is the core "AI" of the risk agent - it makes autonomous decisions
        about how much risk to take based on the current environment.
        """
        params = dict(self.base_params)

        if self.market_regime == "HIGH_VOL":
            # High vol: reduce size but premiums are rich
            params["base_allocation_pct"] *= 0.60  # 40% reduction
            params["max_single_position_pct"] *= 0.70
            params["margin_safety_factor"] *= 1.3

            # But if skew is very negative (puts are very rich), partially offset
            if avg_rr < self.skew_warning_threshold * 2:
                params["base_allocation_pct"] *= 1.15  # slight boost back

        elif self.market_regime == "LOW_VOL":
            # Low vol: premiums are thin, slight increase in allocation ok
            params["base_allocation_pct"] *= 1.15
            params["max_single_position_pct"] *= 1.10

        else:  # NORMAL
            # Standard parameters with minor adjustments
            pass

        # Kurtosis adjustment: fat tails = extra caution
        if kurtosis > self.kurtosis_warning_threshold:
            kurtosis_factor = 1.0 - min((kurtosis - 3.0) * 0.05, 0.30)
            params["base_allocation_pct"] *= max(kurtosis_factor, 0.50)
            logger.info(
                f"Kurtosis adjustment: factor={kurtosis_factor:.2f} "
                f"(kurtosis={kurtosis:.2f})"
            )

        # Skew adjustment: steep negative skew is favorable for put selling
        if avg_rr < self.skew_warning_threshold:
            # Negative RR means puts are more expensive - good for selling
            skew_bonus = min(abs(avg_rr) * 5, 0.20)  # up to 20% bonus
            params["base_allocation_pct"] *= (1.0 + skew_bonus)
            logger.info(f"Skew bonus: +{skew_bonus:.1%} (RR={avg_rr:.4f})")

        self.active_params = params

        logger.info(
            f"Active params: alloc={params['base_allocation_pct']:.2%}, "
            f"max_single={params['max_single_position_pct']:.2%}, "
            f"margin_safety={params['margin_safety_factor']:.2f}"
        )

    def _evaluate_opportunity(self, opp: Dict) -> Dict:
        """
        Evaluate a single opportunity against risk criteria.

        Returns approval dict with 'approved' boolean and 'reason' string.
        """
        symbol = opp["symbol"]
        ror = opp["risk_of_reversal"]
        delta = abs(opp["delta"])
        gamma = abs(opp.get("gamma", 0))
        premium = opp["premium"]
        strike = opp["strike"]
        spot = opp["spot"]

        # Check 1: Risk of Reversal threshold
        max_ror = 0.25
        if self.market_regime == "HIGH_VOL":
            max_ror = 0.20  # tighter in high vol
        elif self.market_regime == "LOW_VOL":
            max_ror = 0.30  # slightly more lenient

        if ror > max_ror:
            return {
                "approved": False,
                "reason": f"Risk of Reversal {ror:.1%} > max {max_ror:.1%}"
            }

        # Check 2: Delta bounds
        if delta > 0.50:
            return {
                "approved": False,
                "reason": f"Delta {delta:.3f} too high (max 0.50)"
            }

        # Check 3: Strike distance from spot
        distance_pct = (spot - strike) / spot
        min_distance = 0.05 if self.market_regime == "HIGH_VOL" else 0.03
        if distance_pct < min_distance:
            return {
                "approved": False,
                "reason": f"Strike too close to spot ({distance_pct:.1%} < {min_distance:.1%})"
            }

        # Check 4: Premium adequacy
        if premium < 1.0:
            return {
                "approved": False,
                "reason": f"Premium ${premium:.2f} too low"
            }

        # Check 5: Portfolio concentration (simplified - would need portfolio data)
        max_notional = self.active_params["max_single_position_pct"]

        # Calculate suggested position size
        notional_per_contract = strike * 100
        account_size = self.portfolio_state.get("account_value", 100000)
        max_contracts = int(
            (account_size * self.active_params["base_allocation_pct"]) /
            (notional_per_contract * self.active_params["margin_safety_factor"])
        )
        max_contracts = max(max_contracts, 1)

        return {
            "approved": True,
            "reason": "All checks passed",
            "suggested_quantity": min(max_contracts, 5),
            "max_ror": max_ror,
            "regime": self.market_regime,
            "allocation_pct": self.active_params["base_allocation_pct"],
        }

    def _handle_risk_check(self, message: AgentMessage):
        """Handle direct risk check requests."""
        trade = message.payload
        result = self._evaluate_opportunity(trade)

        self.bus.publish(AgentMessage(
            topic=message.reply_to or "orchestrator",
            sender=self.name,
            payload={"type": "risk_check_result", "result": result},
            priority=MessagePriority.HIGH,
            correlation_id=message.correlation_id,
        ))

    def _handle_portfolio_update(self, message: AgentMessage):
        """Update portfolio state for risk calculations."""
        self.portfolio_state.update(message.payload)

    def update_portfolio_state(self, account_value: float,
                                current_positions: List[Dict]):
        """
        Update the portfolio state used for risk calculations.
        Called by the orchestrator with real portfolio data.
        """
        self.portfolio_state = {
            "account_value": account_value,
            "positions": current_positions,
            "total_delta": sum(
                p.get("delta", 0) * p.get("position", 0) * 100
                for p in current_positions
                if p.get("sec_type") == "OPT"
            ),
            "total_notional": sum(
                p.get("strike", 0) * abs(p.get("position", 0)) * 100
                for p in current_positions
                if p.get("sec_type") == "OPT"
            ),
        }
