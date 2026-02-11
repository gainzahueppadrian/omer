"""
Orchestrator Agent - Central Coordination and Supervision

The Orchestrator is the "brain" of the multi-agent system.

Responsibilities:
- Start/stop all worker agents
- Coordinate workflows between agents
- Monitor agent health via heartbeats
- Log all activity for audit trail
- Handle system-level decisions
- Provide a unified interface for the main entry point
"""

import threading
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.ibkr.client import IBClient
from src.agents.message_bus import MessageBus, AgentMessage, MessagePriority
from src.agents.scanner_agent import ScannerAgent
from src.agents.risk_agent import DynamicRiskAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.portfolio_agent import PortfolioAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central orchestrator that manages the lifecycle and coordination
    of all trading agents.

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │                  ORCHESTRATOR                     │
    │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
    │  │ Scanner  │→│  Risk    │→│   Execution      │ │
    │  │  Agent   │ │  Agent   │ │    Agent          │ │
    │  └──────────┘ └──────────┘ └──────────────────┘ │
    │       ↑                           ↓              │
    │  ┌──────────────────────────────────────────┐   │
    │  │          Portfolio Agent                   │   │
    │  └──────────────────────────────────────────┘   │
    │                    ↕                             │
    │  ┌──────────────────────────────────────────┐   │
    │  │           Message Bus                     │   │
    │  └──────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self, config: Dict):
        self.config = config
        self.name = "orchestrator"
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Initialize message bus
        self.bus = MessageBus()

        # Initialize IBKR client
        ibkr_cfg = config.get("ibkr", {})
        self.ib_client = IBClient(
            host=ibkr_cfg.get("host", "127.0.0.1"),
            port=ibkr_cfg.get("port", 7497),
            client_id=ibkr_cfg.get("client_id", 1),
            timeout=ibkr_cfg.get("timeout_seconds", 10),
        )

        # Initialize agents
        self.scanner = ScannerAgent(self.ib_client, self.bus, config)
        self.risk_agent = DynamicRiskAgent(self.bus, config)
        self.execution_agent = ExecutionAgent(self.ib_client, self.bus, config)
        self.portfolio_agent = PortfolioAgent(self.ib_client, self.bus, config)

        # Create orchestrator message queue
        self._queue = self.bus.create_agent_queue(self.name)

        # Agent health tracking
        self._agent_heartbeats: Dict[str, datetime] = {}
        self._heartbeat_timeout = 120  # seconds

        # Subscribe to heartbeats
        self.bus.subscribe("heartbeat", self._handle_heartbeat)

        # Cycle interval
        self.cycle_seconds = config.get("agents", {}).get(
            "orchestrator_cycle_seconds", 60
        )

        # Activity log
        self._activity_log: List[Dict] = []

    def start(self):
        """
        Start the entire trading system.

        1. Connect to IBKR
        2. Start all worker agents
        3. Begin orchestration loop
        """
        logger.info("=" * 60)
        logger.info("LEAPS PUTS Multi-Agent Trading System Starting")
        logger.info("=" * 60)

        # Step 1: Connect to IBKR
        try:
            market_data_type = self.config.get("ibkr", {}).get(
                "market_data_type", 3
            )
            self.ib_client.connect_and_run(market_data_type=market_data_type)
        except ConnectionError as e:
            logger.critical(f"Cannot connect to IBKR: {e}")
            raise

        # Step 2: Verify connection with a test request
        try:
            account_value = self.ib_client.get_account_value_sync(
                "NetLiquidation"
            )
            logger.info(f"Account Net Liquidation: ${account_value:,.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch account value: {e}")

        # Step 3: Initialize risk agent with portfolio state
        try:
            positions = self.ib_client.get_positions_sync()
            self.risk_agent.update_portfolio_state(
                account_value if account_value else 100000,
                positions
            )
            logger.info(f"Loaded {len(positions)} existing positions")
        except Exception as e:
            logger.warning(f"Could not load positions: {e}")

        # Step 4: Start all agents
        self.scanner.start()
        self.risk_agent.start()
        self.execution_agent.start()
        self.portfolio_agent.start()

        # Step 5: Start orchestration loop
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=self.name
        )
        self._thread.start()

        self._log_activity("system_start", "All agents started successfully")

        logger.info("All agents started. System is running autonomously.")
        logger.info(
            f"Watchlist: {self.config.get('strategy', {}).get('watchlist', [])}"
        )
        logger.info(
            f"Mode: {self.config.get('ibkr', {}).get('mode', 'paper')}"
        )
        logger.info("=" * 60)

    def stop(self):
        """Gracefully stop all agents and disconnect."""
        logger.info("Shutting down trading system...")

        self._running = False

        # Stop agents in reverse dependency order
        self.portfolio_agent.stop()
        self.execution_agent.stop()
        self.risk_agent.stop()
        self.scanner.stop()

        # Disconnect IBKR
        self.ib_client.disconnect_gracefully()

        # Shutdown message bus
        self.bus.shutdown()

        self._log_activity("system_stop", "System shut down gracefully")

        logger.info("Trading system shut down complete")

    def _run_loop(self):
        """
        Main orchestration loop.

        Each cycle:
        1. Process incoming messages
        2. Check agent health
        3. Log system status
        """
        while self._running:
            try:
                # Process messages
                self._drain_queue()

                # Health check
                self._check_agent_health()

                # Periodic status log
                self._log_status()

            except Exception as e:
                logger.error(f"Orchestrator cycle error: {e}", exc_info=True)

            # Wait for next cycle
            for _ in range(self.cycle_seconds):
                if not self._running:
                    return
                time.sleep(1)

    def _drain_queue(self):
        """Process all pending orchestrator messages."""
        while True:
            try:
                priority, timestamp, message = self._queue.get_nowait()
                self._process_message(message)
            except Exception:
                break

    def _process_message(self, message: AgentMessage):
        """Handle messages directed to the orchestrator."""
        payload = message.payload
        msg_type = payload.get("type", "")

        if msg_type == "risk_assessment":
            self._handle_risk_assessment(payload)
        elif msg_type == "order_placed":
            self._handle_order_placed(payload)
        elif msg_type == "bonus_order_placed":
            self._handle_bonus_order(payload)
        elif msg_type == "execution_failed":
            self._handle_execution_failure(payload)
        elif msg_type == "portfolio_summary":
            self._handle_portfolio_summary(payload)
        elif msg_type == "roll_suggestion":
            self._handle_roll_suggestion(payload)
        else:
            logger.debug(f"Orchestrator: unhandled type '{msg_type}'")

    def _handle_heartbeat(self, message: AgentMessage):
        """Track agent heartbeats."""
        self._agent_heartbeats[message.sender] = datetime.now()

    def _check_agent_health(self):
        """Verify all agents are responsive."""
        now = datetime.now()
        expected_agents = [
            "scanner_agent", "portfolio_agent"
        ]  # risk and execution are event-driven

        for agent_name in expected_agents:
            last_beat = self._agent_heartbeats.get(agent_name)
            if last_beat is None:
                continue  # Agent may not have sent first heartbeat yet

            age = (now - last_beat).total_seconds()
            if age > self._heartbeat_timeout:
                logger.critical(
                    f"AGENT HEALTH: {agent_name} has not responded "
                    f"in {age:.0f}s (timeout={self._heartbeat_timeout}s)"
                )
                self._log_activity(
                    "agent_unresponsive",
                    f"{agent_name} timeout after {age:.0f}s"
                )

    def _handle_risk_assessment(self, payload: Dict):
        """Log risk assessment from Risk Agent."""
        self._log_activity("risk_assessment", (
            f"Regime={payload.get('market_regime')}, "
            f"IV={payload.get('avg_iv', 0):.2%}, "
            f"Approved={payload.get('approved_count')}, "
            f"Rejected={payload.get('rejected_count')}"
        ))

    def _handle_order_placed(self, payload: Dict):
        """Log order execution."""
        self._log_activity("order_placed", (
            f"{payload.get('action')} {payload.get('quantity')}x "
            f"{payload.get('symbol')} {payload.get('expiry')} "
            f"{payload.get('right')}{payload.get('strike')} "
            f"@ ${payload.get('limit_price', 0):.2f} "
            f"(order_id={payload.get('order_id')})"
        ))

    def _handle_bonus_order(self, payload: Dict):
        """Log bonus call order."""
        self._log_activity("bonus_order", (
            f"BUY {payload.get('quantity')}x "
            f"{payload.get('symbol')} {payload.get('expiry')} "
            f"C{payload.get('strike')} "
            f"@ ${payload.get('limit_price', 0):.2f} "
            f"(budget=${payload.get('budget_used', 0):.0f}/"
            f"${payload.get('budget_total', 0):.0f})"
        ))

    def _handle_execution_failure(self, payload: Dict):
        """Handle execution failures."""
        trade = payload.get("trade", {})
        error = payload.get("error", "unknown")
        logger.error(
            f"Execution FAILED for {trade.get('symbol', '?')}: {error}"
        )
        self._log_activity("execution_failed", (
            f"{trade.get('symbol', '?')}: {error}"
        ))

    def _handle_portfolio_summary(self, payload: Dict):
        """Process portfolio summary from Portfolio Agent."""
        alerts = payload.get("alerts", [])
        for alert in alerts:
            level = alert.get("level", "INFO")
            msg = alert.get("message", "")

            if level == "CRITICAL":
                logger.critical(f"PORTFOLIO ALERT: {msg}")
            elif level == "WARNING":
                logger.warning(f"Portfolio alert: {msg}")
            else:
                logger.info(f"Portfolio: {msg}")

            self._log_activity(f"alert_{level.lower()}", msg)

    def _handle_roll_suggestion(self, payload: Dict):
        """Handle roll suggestions from Portfolio Agent."""
        logger.info(
            f"Roll suggestion: {payload.get('symbol')} "
            f"{payload.get('right')}{payload.get('strike')} "
            f"- {payload.get('reason')}"
        )
        self._log_activity("roll_suggestion", (
            f"{payload.get('symbol')} {payload.get('right')}"
            f"{payload.get('strike')}: {payload.get('reason')}"
        ))

    def _log_status(self):
        """Log periodic system status."""
        portfolio = self.portfolio_agent.get_portfolio_summary()
        greeks = portfolio.get("portfolio_greeks", {})

        logger.info(
            f"STATUS | Positions: {portfolio.get('total_positions', 0)} | "
            f"P&L: ${portfolio.get('total_pnl', 0):,.0f} | "
            f"Δ={greeks.get('total_delta', 0):.0f} "
            f"Γ={greeks.get('total_gamma', 0):.2f} "
            f"V={greeks.get('total_vega', 0):.0f} "
            f"Θ={greeks.get('total_theta', 0):.0f}"
        )

    def _log_activity(self, event_type: str, details: str):
        """Add to activity log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details,
        }
        self._activity_log.append(entry)

        # Keep last 500 entries
        if len(self._activity_log) > 500:
            self._activity_log = self._activity_log[-500:]

    def get_activity_log(self, limit: int = 50) -> List[Dict]:
        """Return recent activity log entries."""
        return self._activity_log[-limit:]

    def run_forever(self):
        """
        Blocking call that keeps the system running until interrupted.
        Intended for use as the main entry point.
        """
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()
