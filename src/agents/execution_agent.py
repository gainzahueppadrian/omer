"""
Execution Agent - Autonomous Order Execution

Responsibilities:
- Receive approved trades from Risk Agent
- Determine optimal entry price (mid, limit, market)
- Execute orders via IBKR API
- Monitor fill status
- Report execution results
- Handle the bonus strategy (sell put + buy calls)

This agent ensures all approved trades are executed efficiently
with proper order management.
"""

import threading
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.ibkr.client import IBClient
from src.agents.message_bus import MessageBus, AgentMessage, MessagePriority

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    Autonomous execution agent with smart order routing.

    Features:
    - Limit order placement at favorable prices
    - Fill monitoring and timeout handling
    - Bonus strategy execution (put + call combo)
    - Trade journaling via message bus
    """

    def __init__(self, ib_client: IBClient, message_bus: MessageBus,
                 config: Dict):
        self.ib_client = ib_client
        self.bus = message_bus
        self.config = config
        self.name = "execution_agent"
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pending_orders: Dict[int, Dict] = {}

        self.bonus_config = config.get("strategy", {}).get("bonus_calls", {})

        # Create message queue
        self._queue = self.bus.create_agent_queue(self.name)

    def start(self):
        """Start the execution agent."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=self.name
        )
        self._thread.start()
        logger.info(f"{self.name} started")

    def stop(self):
        """Stop the execution agent."""
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
        """Route messages to handlers."""
        payload = message.payload
        msg_type = payload.get("type", "")

        if msg_type == "approved_trades":
            self._handle_approved_trades(message)
        elif msg_type == "close_position":
            self._handle_close_position(message)
        else:
            logger.debug(f"Execution agent: unhandled type '{msg_type}'")

    def _handle_approved_trades(self, message: AgentMessage):
        """
        Execute approved trades from the Risk Agent.

        For each approved trade:
        1. Get current bid/ask for optimal limit price
        2. Place sell put order
        3. If bonus strategy enabled, place buy call order
        4. Monitor and report
        """
        trades = message.payload.get("trades", [])
        regime = message.payload.get("market_regime", "NORMAL")

        logger.info(
            f"Executing {len(trades)} approved trades "
            f"(regime={regime})"
        )

        for trade in trades:
            try:
                self._execute_trade(trade, regime)
            except Exception as e:
                logger.error(
                    f"Execution error for {trade['symbol']}: {e}",
                    exc_info=True
                )

                # Report failure
                self.bus.publish(AgentMessage(
                    topic="orchestrator",
                    sender=self.name,
                    payload={
                        "type": "execution_failed",
                        "trade": trade,
                        "error": str(e),
                    },
                    priority=MessagePriority.HIGH,
                ))

    def _execute_trade(self, trade: Dict, regime: str):
        """Execute a single trade (put sell + optional call buy)."""
        symbol = trade["symbol"]
        expiry = trade["expiry"]
        strike = trade["strike"]
        quantity = trade.get("risk_approval", {}).get("suggested_quantity", 1)

        # Get fresh option market data for optimal pricing
        opt_data = self.ib_client.get_option_market_data_sync(
            symbol, expiry, strike, "P"
        )

        # Determine limit price
        bid = opt_data.get("bid", 0)
        ask = opt_data.get("ask", 0)
        mid = opt_data.get("mid", 0)

        if bid > 0 and ask > 0:
            # Place limit at natural (bid) or slightly above
            # For selling, we want at least the bid
            limit_price = round(bid + (ask - bid) * 0.3, 2)
        elif mid > 0:
            limit_price = round(mid, 2)
        else:
            limit_price = round(trade["premium"], 2)

        logger.info(
            f"EXECUTING: SELL {quantity}x {symbol} {expiry} P{strike} "
            f"@ ${limit_price:.2f} (bid={bid}, ask={ask})"
        )

        # Place the put sell order
        order_id = self.ib_client.place_option_order(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right="P",
            action="SELL",
            quantity=quantity,
            order_type="LMT",
            limit_price=limit_price,
        )

        self._pending_orders[order_id] = {
            "trade": trade,
            "type": "sell_put",
            "quantity": quantity,
            "limit_price": limit_price,
            "timestamp": datetime.now().isoformat(),
        }

        # Report execution
        self.bus.publish(AgentMessage(
            topic="orchestrator",
            sender=self.name,
            payload={
                "type": "order_placed",
                "order_id": order_id,
                "symbol": symbol,
                "expiry": expiry,
                "strike": strike,
                "right": "P",
                "action": "SELL",
                "quantity": quantity,
                "limit_price": limit_price,
                "trade": trade,
            },
            priority=MessagePriority.HIGH,
        ))

        # ─── Bonus Strategy: Buy LEAPS Calls ───
        if self.bonus_config.get("enabled", False):
            self._execute_bonus_calls(trade, quantity, limit_price)

    def _execute_bonus_calls(self, put_trade: Dict, put_qty: int,
                              put_premium: float):
        """
        Execute the bonus strategy: use put premium to fund LEAPS calls.

        Premium budget = put_premium * max_allocation * multiplier
        """
        max_alloc = self.bonus_config.get("max_premium_allocation", 0.50)
        call_delta_target = self.bonus_config.get("call_delta_target", 0.40)

        budget = put_premium * put_qty * 100 * max_alloc  # total $ budget

        symbol = put_trade["symbol"]
        expiry = put_trade["expiry"]
        spot = put_trade["spot"]
        atm_iv = put_trade["atm_iv"]

        # Find a call strike near the target delta
        # Approximate: ATM or slightly OTM call
        target_call_strike = round(spot * 1.05, 0)  # ~5% OTM

        # Get call market data
        call_data = self.ib_client.get_option_market_data_sync(
            symbol, expiry, target_call_strike, "C"
        )

        call_ask = call_data.get("ask", 0)
        call_mid = call_data.get("mid", 0)
        call_price = call_mid if call_mid > 0 else call_ask

        if call_price <= 0:
            logger.warning(f"Cannot price call for bonus strategy: {symbol}")
            return

        # How many calls can we buy with the budget?
        call_qty = max(int(budget / (call_price * 100)), 0)

        if call_qty <= 0:
            logger.info(
                f"Bonus calls: budget ${budget:.0f} insufficient for "
                f"calls @ ${call_price:.2f}"
            )
            return

        # Cap at put quantity to keep it balanced
        call_qty = min(call_qty, put_qty)

        limit_price = round(call_price, 2)

        logger.info(
            f"BONUS: BUY {call_qty}x {symbol} {expiry} C{target_call_strike} "
            f"@ ${limit_price:.2f} (funded by put premium)"
        )

        order_id = self.ib_client.place_option_order(
            symbol=symbol,
            expiry=expiry,
            strike=target_call_strike,
            right="C",
            action="BUY",
            quantity=call_qty,
            order_type="LMT",
            limit_price=limit_price,
        )

        self._pending_orders[order_id] = {
            "trade": put_trade,
            "type": "buy_call_bonus",
            "quantity": call_qty,
            "limit_price": limit_price,
            "funded_by": "put_premium",
        }

        self.bus.publish(AgentMessage(
            topic="orchestrator",
            sender=self.name,
            payload={
                "type": "bonus_order_placed",
                "order_id": order_id,
                "symbol": symbol,
                "expiry": expiry,
                "strike": target_call_strike,
                "right": "C",
                "action": "BUY",
                "quantity": call_qty,
                "limit_price": limit_price,
                "budget_used": call_qty * call_price * 100,
                "budget_total": budget,
            },
            priority=MessagePriority.NORMAL,
        ))

    def _handle_close_position(self, message: AgentMessage):
        """Handle request to close an existing position."""
        payload = message.payload
        symbol = payload["symbol"]
        expiry = payload["expiry"]
        strike = payload["strike"]
        right = payload.get("right", "P")
        quantity = payload.get("quantity", 1)

        # For a short put, closing = buying it back
        action = "BUY" if payload.get("current_action") == "SELL" else "SELL"

        opt_data = self.ib_client.get_option_market_data_sync(
            symbol, expiry, strike, right
        )

        ask = opt_data.get("ask", 0)
        mid = opt_data.get("mid", 0)
        limit_price = round(mid if mid > 0 else ask, 2)

        if limit_price <= 0:
            logger.warning(f"Cannot price close order for {symbol}")
            return

        order_id = self.ib_client.place_option_order(
            symbol=symbol, expiry=expiry, strike=strike,
            right=right, action=action, quantity=quantity,
            order_type="LMT", limit_price=limit_price,
        )

        logger.info(
            f"CLOSE: {action} {quantity}x {symbol} {expiry} "
            f"{right}{strike} @ ${limit_price:.2f}"
        )

        self.bus.publish(AgentMessage(
            topic="orchestrator",
            sender=self.name,
            payload={
                "type": "close_order_placed",
                "order_id": order_id,
                "symbol": symbol,
                "action": action,
            },
            priority=MessagePriority.HIGH,
        ))
