"""
Portfolio Agent - Autonomous Position Monitor

Responsibilities:
- Monitor existing positions for P&L, Greeks, and risk metrics
- Detect early close opportunities (IV crush, profit targets)
- Alert on risk breaches or upcoming events
- Track the bonus call positions
- Recommend position management actions
"""

import threading
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from src.ibkr.client import IBClient
from src.agents.message_bus import MessageBus, AgentMessage, MessagePriority
from src.math.greeks import compute_all_greeks

logger = logging.getLogger(__name__)


class PortfolioAgent:
    """
    Monitors portfolio positions and recommends management actions.

    Monitors for:
    1. Profit targets hit (e.g., 50% of max profit)
    2. IV normalization (vega profit)
    3. DTE thresholds for rolling
    4. Risk breaches (delta/gamma limits exceeded)
    5. Assignment risk warnings
    6. Bonus call position tracking
    """
    def __init__(self, ib_client: IBClient, message_bus: MessageBus,
                 config: Dict):
        self.ib_client = ib_client
        self.bus = message_bus
        self.config = config
        self.name = "portfolio_agent"
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Monitoring parameters
        self.profit_target_pct = 0.50       # Close at 50% of max profit
        self.loss_limit_pct = 2.0           # Close if loss > 2x premium received
        self.roll_dte_threshold = 60        # Consider rolling at 60 DTE
        self.delta_warning_threshold = -0.55  # Warn if delta exceeds this
        self.monitor_interval = 120         # Check every 2 minutes

        # Position tracking
        self.tracked_positions: Dict[str, Dict] = {}  # key: "SYMBOL_EXPIRY_STRIKE_RIGHT"
        self.portfolio_greeks: Dict[str, float] = {
            "total_delta": 0.0,
            "total_gamma": 0.0,
            "total_vega": 0.0,
            "total_theta": 0.0,
        }

        # Create message queue
        self._queue = self.bus.create_agent_queue(self.name)

    def start(self):
        """Start the portfolio monitoring agent."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=self.name
        )
        self._thread.start()
        logger.info(f"{self.name} started (interval={self.monitor_interval}s)")

    def stop(self):
        """Stop the portfolio agent."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info(f"{self.name} stopped")

    def _run_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Process any incoming messages first
                self._drain_queue()

                # Run monitoring cycle
                self._monitor_cycle()

            except Exception as e:
                logger.error(f"{self.name} cycle error: {e}", exc_info=True)

            # Heartbeat
            self.bus.publish(AgentMessage(
                topic="heartbeat",
                sender=self.name,
                payload={
                    "status": "alive",
                    "tracked_positions": len(self.tracked_positions),
                    "portfolio_greeks": dict(self.portfolio_greeks),
                },
                priority=MessagePriority.LOW,
            ))

            # Wait for next cycle
            for _ in range(self.monitor_interval):
                if not self._running:
                    return
                time.sleep(1)

    def _drain_queue(self):
        """Process all pending messages in the queue."""
        while True:
            try:
                priority, timestamp, message = self._queue.get_nowait()
                self._process_message(message)
            except Exception:
                break

    def _process_message(self, message: AgentMessage):
        """Handle incoming messages."""
        payload = message.payload
        msg_type = payload.get("type", "")

        if msg_type == "order_placed":
            self._track_new_position(payload)
        elif msg_type == "bonus_order_placed":
            self._track_bonus_position(payload)
        elif msg_type == "close_order_placed":
            self._handle_close_confirmation(payload)
        elif msg_type == "force_monitor":
            self._monitor_cycle()

    def _track_new_position(self, order_info: Dict):
        """Start tracking a newly placed position."""
        key = self._position_key(
            order_info["symbol"], order_info["expiry"],
            order_info["strike"], order_info["right"]
        )

        trade = order_info.get("trade", {})

        self.tracked_positions[key] = {
            "symbol": order_info["symbol"],
            "expiry": order_info["expiry"],
            "strike": order_info["strike"],
            "right": order_info["right"],
            "action": order_info["action"],
            "quantity": order_info["quantity"],
            "entry_price": order_info["limit_price"],
            "entry_time": datetime.now().isoformat(),
            "order_id": order_info["order_id"],
            "entry_spot": trade.get("spot", 0),
            "entry_iv": trade.get("atm_iv", 0),
            "entry_delta": trade.get("delta", 0),
            "entry_ror": trade.get("risk_of_reversal", 0),
            "position_type": "short_put",
            "status": "open",
            "max_profit": order_info["limit_price"] * order_info["quantity"] * 100,
        }

        logger.info(f"Tracking new position: {key}")

    def _track_bonus_position(self, order_info: Dict):
        """Track a bonus call position."""
        key = self._position_key(
            order_info["symbol"], order_info["expiry"],
            order_info["strike"], order_info["right"]
        )

        self.tracked_positions[key] = {
            "symbol": order_info["symbol"],
            "expiry": order_info["expiry"],
            "strike": order_info["strike"],
            "right": order_info["right"],
            "action": order_info["action"],
            "quantity": order_info["quantity"],
            "entry_price": order_info["limit_price"],
            "entry_time": datetime.now().isoformat(),
            "order_id": order_info["order_id"],
            "position_type": "bonus_call",
            "status": "open",
            "budget_used": order_info.get("budget_used", 0),
        }

        logger.info(f"Tracking bonus call: {key}")

    def _handle_close_confirmation(self, order_info: Dict):
        """Mark a position as closing."""
        for key, pos in self.tracked_positions.items():
            if (pos["symbol"] == order_info["symbol"] and
                    pos["status"] == "open"):
                pos["status"] = "closing"
                pos["close_order_id"] = order_info["order_id"]
                logger.info(f"Position {key} marked as closing")
                break

    def _position_key(self, symbol: str, expiry: str,
                       strike: float, right: str) -> str:
        """Generate a unique key for a position."""
        return f"{symbol}_{expiry}_{strike}_{right}"

    def _monitor_cycle(self):
        """
        Run one monitoring cycle across all tracked positions.

        For each position:
        1. Fetch current market data
        2. Compute P&L
        3. Check Greeks
        4. Evaluate management triggers
        5. Publish alerts/recommendations
        """
        if not self.tracked_positions:
            return

        # Reset portfolio Greeks
        self.portfolio_greeks = {
            "total_delta": 0.0,
            "total_gamma": 0.0,
            "total_vega": 0.0,
            "total_theta": 0.0,
        }

        alerts = []
        actions = []

        # Also fetch actual positions from IBKR for reconciliation
        ibkr_positions = self.ib_client.get_positions_sync()

        for key, pos in list(self.tracked_positions.items()):
            if pos["status"] != "open":
                continue

            try:
                result = self._monitor_position(pos, ibkr_positions)
                if result.get("alerts"):
                    alerts.extend(result["alerts"])
                if result.get("action"):
                    actions.append(result["action"])
            except Exception as e:
                logger.warning(f"Error monitoring {key}: {e}")

        # Publish portfolio summary
        self.bus.publish(AgentMessage(
            topic="orchestrator",
            sender=self.name,
            payload={
                "type": "portfolio_summary",
                "tracked_positions": len(self.tracked_positions),
                "open_positions": sum(
                    1 for p in self.tracked_positions.values()
                    if p["status"] == "open"
                ),
                "portfolio_greeks": dict(self.portfolio_greeks),
                "alerts": alerts,
                "timestamp": datetime.now().isoformat(),
            },
            priority=MessagePriority.NORMAL,
        ))

        # Execute recommended actions
        for action in actions:
            self._execute_action(action)

        # Update risk agent with portfolio state
        account_value = self.ib_client.get_account_value_sync("NetLiquidation")
        self.bus.publish(AgentMessage(
            topic="risk_agent",
            sender=self.name,
            payload={
                "type": "portfolio_update",
                "account_value": account_value,
                "positions": ibkr_positions,
                "portfolio_greeks": dict(self.portfolio_greeks),
            },
            priority=MessagePriority.NORMAL,
        ))

    def _monitor_position(self, pos: Dict,
                           ibkr_positions: List[Dict]) -> Dict:
        """Monitor a single position and return alerts/actions."""
        symbol = pos["symbol"]
        expiry = pos["expiry"]
        strike = pos["strike"]
        right = pos["right"]
        entry_price = pos["entry_price"]
        quantity = pos["quantity"]
        position_type = pos["position_type"]

        result = {"alerts": [], "action": None}

        # Get current spot price
        spot = self.ib_client.get_market_price_sync(symbol)
        if spot is None:
            return result

        # Calculate DTE
        try:
            exp_date = datetime.strptime(expiry, "%Y%m%d")
            dte = max((exp_date - datetime.now()).days, 0)
            T = max(dte / 365.0, 0.001)
        except ValueError:
            return result

        # Get current option market data
        opt_data = self.ib_client.get_option_market_data_sync(
            symbol, expiry, strike, right
        )

        current_price = (
            opt_data.get("mid") or
            opt_data.get("opt_price") or
            opt_data.get("last", 0)
        )

        current_iv = opt_data.get("iv", pos.get("entry_iv", 0.25))
        current_delta = opt_data.get("delta", 0)
        current_gamma = opt_data.get("gamma", 0)
        current_vega = opt_data.get("vega", 0)
        current_theta = opt_data.get("theta", 0)

        # Update portfolio Greeks
        multiplier = -1 if pos["action"] == "SELL" else 1
        self.portfolio_greeks["total_delta"] += (
            current_delta * quantity * 100 * multiplier
        )
        self.portfolio_greeks["total_gamma"] += (
            current_gamma * quantity * 100 * multiplier
        )
        self.portfolio_greeks["total_vega"] += (
            current_vega * quantity * 100 * multiplier
        )
        self.portfolio_greeks["total_theta"] += (
            current_theta * quantity * 100 * multiplier
        )

        # ─── P&L Calculation ───
        if position_type == "short_put":
            # For short put: profit = entry_premium - current_premium
            pnl_per_contract = (entry_price - current_price) * 100
            total_pnl = pnl_per_contract * quantity
            max_profit = pos.get("max_profit", entry_price * quantity * 100)
            pnl_pct = total_pnl / max_profit if max_profit > 0 else 0

            pos["current_price"] = current_price
            pos["current_pnl"] = total_pnl
            pos["pnl_pct"] = pnl_pct
            pos["current_delta"] = current_delta
            pos["current_iv"] = current_iv
            pos["dte"] = dte

            logger.debug(
                f"{symbol} P{strike}: P&L=${total_pnl:.0f} ({pnl_pct:.1%}), "
                f"Δ={current_delta:.3f}, IV={current_iv:.2%}, DTE={dte}"
            )

            # ─── Trigger Checks ───

            # Check 1: Profit target hit
            if pnl_pct >= self.profit_target_pct:
                result["alerts"].append({
                    "level": "INFO",
                    "message": (
                        f"{symbol} P{strike}: Profit target reached "
                        f"({pnl_pct:.0%} of max)"
                    ),
                })
                result["action"] = {
                    "type": "close_for_profit",
                    "symbol": symbol,
                    "expiry": expiry,
                    "strike": strike,
                    "right": right,
                    "quantity": quantity,
                    "current_action": "SELL",
                    "reason": f"Profit target {pnl_pct:.0%}",
                }

            # Check 2: Loss limit
            elif pnl_pct < -self.loss_limit_pct:
                result["alerts"].append({
                    "level": "CRITICAL",
                    "message": (
                        f"{symbol} P{strike}: LOSS LIMIT BREACHED "
                        f"(loss={pnl_pct:.0%})"
                    ),
                })
                result["action"] = {
                    "type": "close_for_loss",
                    "symbol": symbol,
                    "expiry": expiry,
                    "strike": strike,
                    "right": right,
                    "quantity": quantity,
                    "current_action": "SELL",
                    "reason": f"Loss limit {pnl_pct:.0%}",
                    "priority": "CRITICAL",
                }

            # Check 3: Delta warning (getting too deep ITM)
            if abs(current_delta) > abs(self.delta_warning_threshold):
                result["alerts"].append({
                    "level": "WARNING",
                    "message": (
                        f"{symbol} P{strike}: Delta warning "
                        f"(Δ={current_delta:.3f})"
                    ),
                })

            # Check 4: Roll trigger (low DTE)
            if dte <= self.roll_dte_threshold and pnl_pct > 0:
                result["alerts"].append({
                    "level": "INFO",
                    "message": (
                        f"{symbol} P{strike}: Consider rolling "
                        f"(DTE={dte}, P&L={pnl_pct:.0%})"
                    ),
                })
                # Don't auto-close for rolling, just alert
                if result["action"] is None:
                    result["action"] = {
                        "type": "roll_suggestion",
                        "symbol": symbol,
                        "expiry": expiry,
                        "strike": strike,
                        "right": right,
                        "quantity": quantity,
                        "current_action": "SELL",
                        "reason": f"DTE={dte}, consider rolling",
                    }

            # Check 5: IV crush opportunity
            entry_iv = pos.get("entry_iv", 0)
            if entry_iv > 0 and current_iv < entry_iv * 0.70:
                # IV has dropped 30%+ from entry
                result["alerts"].append({
                    "level": "INFO",
                    "message": (
                        f"{symbol} P{strike}: IV crush detected "
                        f"(entry={entry_iv:.2%}, now={current_iv:.2%})"
                    ),
                })

        elif position_type == "bonus_call":
            # For long call: profit = current_premium - entry_premium
            pnl_per_contract = (current_price - entry_price) * 100
            total_pnl = pnl_per_contract * quantity

            pos["current_price"] = current_price
            pos["current_pnl"] = total_pnl
            pos["dte"] = dte

            logger.debug(
                f"{symbol} C{strike} (bonus): P&L=${total_pnl:.0f}, "
                f"DTE={dte}"
            )

            # Bonus calls: take profit at 100%+ gain
            if current_price >= entry_price * 2.0:
                result["alerts"].append({
                    "level": "INFO",
                    "message": (
                        f"{symbol} C{strike} (bonus): 2x profit! "
                        f"Consider taking profits"
                    ),
                })
                result["action"] = {
                    "type": "close_bonus_profit",
                    "symbol": symbol,
                    "expiry": expiry,
                    "strike": strike,
                    "right": right,
                    "quantity": quantity,
                    "current_action": "BUY",
                    "reason": "Bonus call 2x profit",
                }

        return result

    def _execute_action(self, action: Dict):
        """Send an action to the execution agent."""
        action_type = action.get("type", "")
        priority = MessagePriority.CRITICAL if action.get(
            "priority") == "CRITICAL" else MessagePriority.HIGH

        if action_type in ("close_for_profit", "close_for_loss",
                           "close_bonus_profit"):
            self.bus.publish(AgentMessage(
                topic="execution_agent",
                sender=self.name,
                payload={
                    "type": "close_position",
                    **action,
                },
                priority=priority,
            ))
            logger.info(
                f"Action sent: {action_type} for {action['symbol']} "
                f"{action['right']}{action['strike']} - {action['reason']}"
            )

        elif action_type == "roll_suggestion":
            # Just publish as advisory; orchestrator or human decides
            self.bus.publish(AgentMessage(
                topic="orchestrator",
                sender=self.name,
                payload={
                    "type": "roll_suggestion",
                    **action,
                },
                priority=MessagePriority.NORMAL,
            ))

    def get_portfolio_summary(self) -> Dict:
        """Return current portfolio summary."""
        open_positions = {
            k: v for k, v in self.tracked_positions.items()
            if v["status"] == "open"
        }

        total_pnl = sum(
            p.get("current_pnl", 0) for p in open_positions.values()
        )

        return {
            "total_positions": len(open_positions),
            "total_pnl": total_pnl,
            "portfolio_greeks": dict(self.portfolio_greeks),
            "positions": {
                k: {
                    "symbol": v["symbol"],
                    "strike": v["strike"],
                    "right": v["right"],
                    "type": v["position_type"],
                    "pnl": v.get("current_pnl", 0),
                    "delta": v.get("current_delta", 0),
                    "dte": v.get("dte", 0),
                }
                for k, v in open_positions.items()
            },
        }
