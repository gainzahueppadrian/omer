"""
Scanner Agent - Autonomous Market Scanning Worker

Responsibilities:
- Scan watchlist for LEAPS put opportunities
- Fetch real-time market data and volatility smiles
- Filter candidates based on IV percentile, liquidity, and price range
- Publish opportunities to the message bus for the Risk Agent

This agent runs autonomously in its own thread, periodically scanning
and publishing results.
"""

import threading
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from src.ibkr.client import IBClient
from src.math.rnd import RNDModel
from src.math.greeks import compute_all_greeks, bs_put_price
from src.agents.message_bus import MessageBus, AgentMessage, MessagePriority

logger = logging.getLogger(__name__)


class ScannerAgent:
    """
    Autonomous agent that scans for LEAPS put selling opportunities.

    Workflow:
    1. Iterate through watchlist
    2. Get current price for each symbol
    3. Find LEAPS-eligible expirations (9-18 months out)
    4. Fetch volatility smile data
    5. Calibrate RND model
    6. Score and rank candidates
    7. Publish opportunities to message bus
    """

    def __init__(self, ib_client: IBClient, message_bus: MessageBus,
                 config: Dict):
        self.ib_client = ib_client
        self.bus = message_bus
        self.config = config
        self.name = "scanner_agent"
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.rnd_models: Dict[str, RNDModel] = {}

        # Extract config
        self.watchlist = config.get("strategy", {}).get("watchlist", [])
        self.leaps_config = config.get("strategy", {}).get("leaps", {})
        self.filter_config = config.get("strategy", {}).get("filters", {})
        self.scan_interval = config.get("agents", {}).get(
            "scan_interval_seconds", 300
        )

    def start(self):
        """Start the scanner agent in its own thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=self.name
        )
        self._thread.start()
        logger.info(f"{self.name} started (interval={self.scan_interval}s)")

    def stop(self):
        """Stop the scanner agent."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info(f"{self.name} stopped")

    def _run_loop(self):
        """Main scanning loop."""
        while self._running:
            try:
                self._scan_cycle()
            except Exception as e:
                logger.error(f"{self.name} scan cycle error: {e}", exc_info=True)

            # Heartbeat
            self.bus.publish(AgentMessage(
                topic="heartbeat",
                sender=self.name,
                payload={"status": "alive", "last_scan": datetime.now().isoformat()},
                priority=MessagePriority.LOW,
            ))

            # Wait for next cycle
            for _ in range(self.scan_interval):
                if not self._running:
                    return
                time.sleep(1)

    def _scan_cycle(self):
        """Execute one full scan cycle across the watchlist."""
        logger.info(f"Starting scan cycle for {len(self.watchlist)} symbols")
        opportunities = []

        for symbol in self.watchlist:
            if not self._running:
                break

            try:
                opps = self._scan_symbol(symbol)
                opportunities.extend(opps)
            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")
                continue

            # Brief pause between symbols to avoid rate limiting
            time.sleep(0.5)

        if opportunities:
            # Sort by score (higher is better)
            opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Publish top opportunities
            self.bus.publish(AgentMessage(
                topic="risk_agent",
                sender=self.name,
                payload={
                    "type": "scan_results",
                    "opportunities": opportunities[:10],  # Top 10
                    "total_scanned": len(self.watchlist),
                    "total_found": len(opportunities),
                    "timestamp": datetime.now().isoformat(),
                },
                priority=MessagePriority.NORMAL,
            ))

            logger.info(
                f"Scan complete: {len(opportunities)} opportunities found, "
                f"top 10 published"
            )
        else:
            logger.info("Scan complete: no opportunities found")

    def _scan_symbol(self, symbol: str) -> List[Dict]:
        """Scan a single symbol for LEAPS put opportunities."""
        opportunities = []

        # Step 1: Get current price
        spot = self.ib_client.get_market_price_sync(symbol)
        if spot is None or spot <= 0:
            logger.debug(f"{symbol}: no price data")
            return []

        # Apply price filters
        min_price = self.filter_config.get("min_underlying_price", 20)
        max_price = self.filter_config.get("max_underlying_price", 500)
        if not (min_price <= spot <= max_price):
            logger.debug(f"{symbol}: price ${spot:.2f} outside range")
            return []

        # Step 2: Find LEAPS-eligible expirations
        min_dte = self.leaps_config.get("min_dte", 270)
        max_dte = self.leaps_config.get("max_dte", 548)
        target_dte = self.leaps_config.get("target_dte", 365)

        min_date = datetime.now() + timedelta(days=min_dte)
        max_date = datetime.now() + timedelta(days=max_dte)

        expirations = self.ib_client.get_option_chain_expirations(symbol)
        leaps_expirations = []

        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, "%Y%m%d")
                if min_date <= exp_date <= max_date:
                    leaps_expirations.append(exp)
            except ValueError:
                continue

        if not leaps_expirations:
            logger.debug(f"{symbol}: no LEAPS expirations found")
            return []

        # Pick closest to target DTE
        target_date = datetime.now() + timedelta(days=target_dte)
        best_exp = min(
            leaps_expirations,
            key=lambda e: abs(
                (datetime.strptime(e, "%Y%m%d") - target_date).days
            )
        )

        exp_date = datetime.strptime(best_exp, "%Y%m%d")
        T = max((exp_date - datetime.now()).days / 365.0, 0.01)
        dte = (exp_date - datetime.now()).days

        # Step 3: Fetch volatility smile
        smile_data = self.ib_client.get_smile_data_sync(
            symbol, best_exp, spot
        )

        if not smile_data or smile_data.get("atm_iv", 0) <= 0:
            logger.debug(f"{symbol}: could not get smile data")
            return []

        atm_iv = smile_data["atm_iv"]
        rr_25 = smile_data.get("rr_25", 0)
        str_25 = smile_data.get("str_25", 0)

        # Step 4: Calibrate RND model
        rnd = RNDModel()
        r = self.config.get("strategy", {}).get("risk_free_rate", 0.05)  # risk-free rate
        try:
            rnd.calibrate(spot, T, r, atm_iv, rr_25, str_25)
            self.rnd_models[symbol] = rnd
        except Exception as e:
            logger.warning(f"{symbol}: RND calibration failed: {e}")
            return []

        # Step 5: Find optimal put strikes
        strikes = self.ib_client.get_option_strikes(symbol, best_exp, "P")
        if not strikes:
            return []

        # Filter strikes in delta range
        min_delta = abs(self.leaps_config.get("min_delta", -0.30))
        max_delta = abs(self.leaps_config.get("max_delta", -0.45))
        min_return = self.leaps_config.get("min_annualized_return", 0.10)
        max_ror = self.leaps_config.get("max_risk_of_reversal", 0.25)

        for K in strikes:
            if K >= spot:  # Only OTM/ATM puts
                continue
            if K / spot < 0.60:  # Too far OTM
                continue

            # Calculate Greeks
            greeks = compute_all_greeks(spot, K, T, r, atm_iv, "put")
            delta = abs(greeks["delta"])

            if not (min_delta <= delta <= max_delta):
                continue

            # Risk of Reversal (RND-based assignment probability)
            risk_of_reversal = rnd.risk_of_reversal(K)

            if risk_of_reversal > max_ror:
                continue

            # Premium and return calculation
            premium = greeks["price"]
            if premium <= 0:
                continue

            notional = K * 100  # Cash needed if assigned
            annualized_return = (premium * 100) / notional * (365.0 / dte)

            if annualized_return < min_return:
                continue

            # Score the opportunity
            score = self._score_opportunity(
                annualized_return, risk_of_reversal, atm_iv,
                greeks, rr_25, str_25
            )

            opp = {
                "symbol": symbol,
                "expiry": best_exp,
                "dte": dte,
                "strike": K,
                "spot": spot,
                "T": T,
                "premium": round(premium, 2),
                "annualized_return": round(annualized_return, 4),
                "risk_of_reversal": round(risk_of_reversal, 4),
                "delta": round(greeks["delta"], 4),
                "gamma": round(greeks["gamma"], 6),
                "vega": round(greeks["vega"], 4),
                "theta": round(greeks["theta"], 4),
                "atm_iv": round(atm_iv, 4),
                "rr_25": round(rr_25, 4),
                "str_25": round(str_25, 4),
                "score": round(score, 4),
                "smile_data": smile_data,
                "rnd_diagnostics": rnd.get_diagnostics(),
            }

            opportunities.append(opp)
            logger.debug(
                f"{symbol} {best_exp} P{K}: "
                f"prem=${premium:.2f} ret={annualized_return:.1%} "
                f"RoR={risk_of_reversal:.1%} score={score:.3f}"
            )

        return opportunities

    def _score_opportunity(
        self, annualized_return: float, risk_of_reversal: float,
        atm_iv: float, greeks: Dict, rr_25: float, str_25: float
    ) -> float:
        """
        Score an opportunity on multiple dimensions.

        Higher score = more attractive opportunity.

        Components:
        - Return attractiveness (higher return = better)
        - Safety (lower risk_of_reversal = better)
        - Vega attractiveness (higher vega * IV = more IV crush potential)
        - Skew favorability (more negative skew = puts are richer)
        """
        # Normalize components to 0-1 range
        return_score = min(annualized_return / 0.20, 1.0)  # cap at 20%
        safety_score = max(1.0 - risk_of_reversal / 0.30, 0.0)
        vega_score = min(abs(greeks.get("vega", 0)) * atm_iv / 0.10, 1.0)
        skew_score = min(abs(rr_25) / 0.05, 1.0) if rr_25 < 0 else 0.3

        # Weighted composite
        score = (
            0.35 * return_score +
            0.30 * safety_score +
            0.20 * vega_score +
            0.15 * skew_score
        )

        return score

    def scan_now(self) -> List[Dict]:
        """Trigger an immediate scan (for orchestrator use)."""
        logger.info("Immediate scan triggered")
        self._scan_cycle()
        return []  # Results published via message bus
