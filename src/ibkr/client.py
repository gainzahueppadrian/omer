"""
Interactive Brokers API Client

Provides synchronous wrappers around the asynchronous IBKR TWS API.
Handles real-time market data, option chains, and order execution
with zero-delay data prioritization.
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickAttrib, TickerId
import numpy as np

logger = logging.getLogger(__name__)


class IBClient(EWrapper, EClient):
    """
    Full-featured IBKR client with synchronous data-fetching wrappers.

    Design:
    - Runs the IBKR message loop on a background thread.
    - Exposes synchronous methods using threading.Event for blocking waits.
    - Handles tick data, option computations, contract details, and orders.
    - Implements retry logic and timeout handling for robustness.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1, timeout: float = 10.0):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout

        # Request ID management
        self._next_req_id = 1000
        self._req_id_lock = threading.Lock()

        # Data storage (thread-safe via events/locks)
        self._market_data: Dict[int, Dict] = defaultdict(dict)
        self._market_data_events: Dict[int, threading.Event] = {}
        self._option_data: Dict[int, Dict] = defaultdict(dict)
        self._option_events: Dict[int, threading.Event] = {}
        self._contract_details: Dict[int, List] = defaultdict(list)
        self._contract_events: Dict[int, threading.Event] = {}
        self._order_status: Dict[int, Dict] = {}
        self._positions: List[Dict] = []
        self._account_values: Dict[str, float] = {}
        self._account_event = threading.Event()

        # Connection state
        self._connected = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _get_req_id(self) -> int:
        with self._req_id_lock:
            rid = self._next_req_id
            self._next_req_id += 1
            return rid

    # ─── Connection Management ─────────────────────────────────

    def connect_and_run(self, market_data_type: int = 3):
        """Connect to TWS/Gateway and start the message processing thread."""
        self.connect(self.host, self.port, self.client_id)

        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

        # Wait for connection confirmation
        if not self._connected.wait(timeout=self.timeout):
            raise ConnectionError(
                f"Failed to connect to IBKR at {self.host}:{self.port} "
                f"within {self.timeout}s"
            )

        # Set market data type
        self.reqMarketDataType(market_data_type)
        logger.info(
            f"Connected to IBKR {self.host}:{self.port} "
            f"(client_id={self.client_id}, mkt_data_type={market_data_type})"
        )

    def disconnect_gracefully(self):
        """Clean disconnect."""
        try:
            self.disconnect()
        except Exception as e:
            logger.warning(f"Disconnect warning: {e}")
        logger.info("Disconnected from IBKR")

    # ─── EWrapper Callbacks ────────────────────────────────────

    def connectAck(self):
        logger.info("IBKR connection acknowledged")

    def nextValidId(self, orderId: int):
        """Called on connection; signals readiness."""
        with self._req_id_lock:
            self._next_req_id = max(self._next_req_id, orderId + 100)
        self._connected.set()
        logger.info(f"Next valid order ID: {orderId}")

    def error(self, reqId: TickerId, errorCode: int, errorString: str,
              advancedOrderRejectJson: str = ""):
        # Filter informational messages
        if errorCode in (2104, 2106, 2158, 2108):  # Market data farm messages
            logger.debug(f"IBKR info [{errorCode}]: {errorString}")
        elif errorCode == 200:  # No security found
            logger.warning(f"No security found for reqId={reqId}: {errorString}")
            self._signal_event(reqId)
        elif errorCode in (354, 10167, 10168):  # No market data permission
            logger.warning(f"Market data issue [{errorCode}] reqId={reqId}: {errorString}")
            self._signal_event(reqId)
        else:
            logger.error(f"IBKR error [{errorCode}] reqId={reqId}: {errorString}")
            if errorCode >= 1000:
                self._signal_event(reqId)

    def _signal_event(self, reqId: int):
        """Signal any waiting event for this request."""
        for events_dict in [self._market_data_events, self._option_events,
                            self._contract_events]:
            if reqId in events_dict:
                events_dict[reqId].set()

    # ── Market Data Callbacks ──

    def tickPrice(self, reqId: TickerId, tickType: int, price: float,
                  attrib: TickAttrib):
        """Handle price ticks (Last, Bid, Ask, Close)."""
        if price <= 0:
            return

        tick_names = {
            1: "bid", 2: "ask", 4: "last", 6: "high", 7: "low",
            9: "close", 14: "open",
            66: "delayed_bid", 67: "delayed_ask", 68: "delayed_last",
            72: "delayed_high", 73: "delayed_low", 75: "delayed_close",
        }

        name = tick_names.get(tickType)
        if name:
            self._market_data[reqId][name] = price
            # Signal on last/close or delayed equivalents
            if tickType in (4, 9, 68, 75):
                if reqId in self._market_data_events:
                    self._market_data_events[reqId].set()

    def tickSize(self, reqId: TickerId, tickType: int, size: int):
        """Handle size ticks (volume, open interest)."""
        size_names = {
            0: "bid_size", 3: "ask_size", 5: "last_size",
            8: "volume", 27: "option_call_oi", 28: "option_put_oi",
        }
        name = size_names.get(tickType)
        if name:
            self._market_data[reqId][name] = size

    def tickOptionComputation(self, reqId: TickerId, tickType: int,
                               tickAttrib: int, impliedVol: float,
                               delta: float, optPrice: float,
                               pvDividend: float, gamma: float,
                               vega: float, theta: float,
                               undPrice: float):
        """Handle option model computation ticks (IV, Greeks)."""
        # tickType 10=Bid, 11=Ask, 12=Last, 13=Model
        if impliedVol is not None and impliedVol > 0:
            self._option_data[reqId]["iv"] = impliedVol
        if delta is not None:
            self._option_data[reqId]["delta"] = delta
        if gamma is not None:
            self._option_data[reqId]["gamma"] = gamma
        if vega is not None:
            self._option_data[reqId]["vega"] = vega
        if theta is not None:
            self._option_data[reqId]["theta"] = theta
        if optPrice is not None and optPrice > 0:
            self._option_data[reqId]["opt_price"] = optPrice
        if undPrice is not None and undPrice > 0:
            self._option_data[reqId]["und_price"] = undPrice

        if tickType in (13, 53):  # Model or Delayed Model
            if reqId in self._option_events:
                self._option_events[reqId].set()

    def tickSnapshotEnd(self, reqId: int):
        """Snapshot complete."""
        self._signal_event(reqId)

    # ── Contract Details Callbacks ──

    def contractDetails(self, reqId: int, contractDetails):
        self._contract_details[reqId].append(contractDetails)

    def contractDetailsEnd(self, reqId: int):
        if reqId in self._contract_events:
            self._contract_events[reqId].set()

    # ── Order / Position Callbacks ──

    def orderStatus(self, orderId: int, status: str, filled: float,
                    remaining: float, avgFillPrice: float, permId: int,
                    parentId: int, lastFillPrice: float, clientId: int,
                    whyHeld: str, mktCapPrice: float = 0.0):
        self._order_status[orderId] = {
            "status": status, "filled": filled, "remaining": remaining,
            "avg_fill_price": avgFillPrice, "last_fill_price": lastFillPrice,
        }
        logger.info(
            f"Order {orderId}: {status} | filled={filled} "
            f"remaining={remaining} avg_price={avgFillPrice}"
        )

    def position(self, account: str, contract: Contract, position: float,
                 avgCost: float):
        self._positions.append({
            "account": account,
            "symbol": contract.symbol,
            "sec_type": contract.secType,
            "right": getattr(contract, "right", ""),
            "strike": getattr(contract, "strike", 0),
            "expiry": getattr(contract, "lastTradeDateOrContractMonth", ""),
            "position": position,
            "avg_cost": avgCost,
            "con_id": contract.conId,
        })

    def positionEnd(self):
        logger.info(f"Received {len(self._positions)} positions")

    def updateAccountValue(self, key: str, val: str, currency: str,
                           accountName: str):
        try:
            self._account_values[key] = float(val)
        except ValueError:
            pass

    def accountDownloadEnd(self, accountName: str):
        self._account_event.set()

    # ─── Synchronous Data Methods ──────────────────────────────

    def get_market_price_sync(self, symbol: str, sec_type: str = "STK",
                               exchange: str = "SMART",
                               currency: str = "USD") -> Optional[float]:
        """
        Get current market price synchronously.
        Prioritizes: Last > Close > Bid/Ask midpoint > Delayed versions
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency

        req_id = self._get_req_id()
        event = threading.Event()
        self._market_data_events[req_id] = event
        self._market_data[req_id] = {}

        self.reqMktData(req_id, contract, "", True, False, [])

        # Wait for data with timeout
        event.wait(timeout=self.timeout)

        # Cancel data subscription
        self.cancelMktData(req_id)

        data = self._market_data.get(req_id, {})

        # Priority-based price resolution
        price = (
            data.get("last") or
            data.get("close") or
            data.get("delayed_last") or
            data.get("delayed_close")
        )

        if price is None:
            bid = data.get("bid") or data.get("delayed_bid")
            ask = data.get("ask") or data.get("delayed_ask")
            if bid and ask and bid > 0 and ask > 0:
                price = (bid + ask) / 2.0

        # Cleanup
        self._market_data_events.pop(req_id, None)

        if price and price > 0:
            logger.debug(f"{symbol} price: ${price:.2f}")
            return price

        logger.warning(f"Could not get market price for {symbol}")
        return None

    def get_option_chain_expirations(self, symbol: str) -> List[str]:
        """Get available option expiration dates for a symbol."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"

        req_id = self._get_req_id()
        event = threading.Event()
        self._contract_events[req_id] = event
        self._contract_details[req_id] = []

        self.reqContractDetails(req_id, contract)
        event.wait(timeout=self.timeout * 2)

        expirations = set()
        for cd in self._contract_details.get(req_id, []):
            exp = cd.contract.lastTradeDateOrContractMonth
            if exp:
                expirations.add(exp)

        self._contract_events.pop(req_id, None)

        sorted_exp = sorted(expirations)
        logger.debug(f"{symbol} expirations: {sorted_exp[:5]}...")
        return sorted_exp

    def get_option_strikes(self, symbol: str, expiry: str,
                            right: str = "P") -> List[float]:
        """Get available strikes for a specific expiration."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.right = right

        req_id = self._get_req_id()
        event = threading.Event()
        self._contract_events[req_id] = event
        self._contract_details[req_id] = []

        self.reqContractDetails(req_id, contract)
        event.wait(timeout=self.timeout * 2)

        strikes = set()
        for cd in self._contract_details.get(req_id, []):
            strikes.add(cd.contract.strike)

        self._contract_events.pop(req_id, None)

        return sorted(strikes)

    def get_option_market_data_sync(
        self, symbol: str, expiry: str, strike: float,
        right: str = "P"
    ) -> Dict:
        """
        Get option market data including IV and Greeks synchronously.

        Returns dict with: iv, delta, gamma, vega, theta, opt_price, und_price,
                          bid, ask, last
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = right
        contract.multiplier = "100"

        req_id = self._get_req_id()
        opt_event = threading.Event()
        mkt_event = threading.Event()
        self._option_events[req_id] = opt_event
        self._market_data_events[req_id] = mkt_event
        self._option_data[req_id] = {}
        self._market_data[req_id] = {}

        # Request both price and option computation data
        self.reqMktData(req_id, contract, "100,101,104,106", True, False, [])

        # Wait for option model computation
        opt_event.wait(timeout=self.timeout)
        mkt_event.wait(timeout=min(self.timeout, 3.0))

        self.cancelMktData(req_id)

        result = {}
        result.update(self._option_data.get(req_id, {}))

        mkt = self._market_data.get(req_id, {})
        if "bid" in mkt:
            result["bid"] = mkt["bid"]
        if "ask" in mkt:
            result["ask"] = mkt["ask"]
        if "last" in mkt:
            result["last"] = mkt["last"]

        # Compute midpoint if we have bid/ask
        bid = result.get("bid", 0)
        ask = result.get("ask", 0)
        if bid > 0 and ask > 0:
            result["mid"] = (bid + ask) / 2.0

        # Cleanup
        self._option_events.pop(req_id, None)
        self._market_data_events.pop(req_id, None)

        return result

    def get_smile_data_sync(
        self, symbol: str, expiry: str, spot: float,
        r: float = 0.05
    ) -> Dict:
        """
        Fetch volatility smile data: ATM IV, 25-delta Risk Reversal, 25-delta Strangle.

        Strategy:
        1. Find ATM strike and get its IV
        2. Estimate 25-delta call and put strikes using BS approximation
        3. Fetch IVs at those strikes
        4. Compute RR = σ(25Δ call) - σ(25Δ put)
        5. Compute STR = [σ(25Δ call) + σ(25Δ put)] / 2 - σ_ATM

        Returns:
            Dict with atm_iv, rr_25, str_25, iv_25d_call, iv_25d_put, atm_strike
        """
        strikes = self.get_option_strikes(symbol, expiry, "P")
        if not strikes:
            logger.warning(f"No strikes found for {symbol} {expiry}")
            return {}

        # Parse expiry to get T
        try:
            exp_date = datetime.strptime(expiry, "%Y%m%d")
            T = max((exp_date - datetime.now()).days / 365.0, 0.01)
        except ValueError:
            T = 1.0

        # Find ATM strike
        atm_strike = min(strikes, key=lambda k: abs(k - spot))

        # Get ATM IV
        atm_data = self.get_option_market_data_sync(symbol, expiry, atm_strike, "P")
        atm_iv = atm_data.get("iv", 0)

        if atm_iv <= 0:
            # Try call side
            atm_data_call = self.get_option_market_data_sync(
                symbol, expiry, atm_strike, "C"
            )
            atm_iv = atm_data_call.get("iv", 0)

        if atm_iv <= 0:
            logger.warning(f"Could not determine ATM IV for {symbol} {expiry}")
            return {"atm_iv": 0, "rr_25": 0, "str_25": 0}

        # Estimate 25-delta strikes using BS approximation
        # For 25-delta put: K_put ≈ S · exp(-0.6745·σ·√T - 0.5·σ²·T)
        # For 25-delta call: K_call ≈ S · exp(0.6745·σ·√T + 0.5·σ²·T)
        sqrt_T = np.sqrt(T)
        z_25 = 0.6745  # norm.ppf(0.75) ≈ 0.6745

        K_25d_put_est = spot * np.exp(-z_25 * atm_iv * sqrt_T - 0.5 * atm_iv ** 2 * T)
        K_25d_call_est = spot * np.exp(z_25 * atm_iv * sqrt_T - 0.5 * atm_iv ** 2 * T)

        # Find nearest available strikes
        K_25d_put = min(strikes, key=lambda k: abs(k - K_25d_put_est))
        K_25d_call = min(strikes, key=lambda k: abs(k - K_25d_call_est))

        # Get IVs at 25-delta strikes
        put_25d_data = self.get_option_market_data_sync(
            symbol, expiry, K_25d_put, "P"
        )
        call_25d_data = self.get_option_market_data_sync(
            symbol, expiry, K_25d_call, "C"
        )

        iv_25d_put = put_25d_data.get("iv", atm_iv)
        iv_25d_call = call_25d_data.get("iv", atm_iv)

        # Fallback if IVs are zero
        if iv_25d_put <= 0:
            iv_25d_put = atm_iv * 1.05  # typical skew approximation
        if iv_25d_call <= 0:
            iv_25d_call = atm_iv * 0.98

        # Risk Reversal and Strangle
        rr_25 = iv_25d_call - iv_25d_put
        str_25 = (iv_25d_call + iv_25d_put) / 2.0 - atm_iv

        result = {
            "atm_iv": atm_iv,
            "rr_25": rr_25,
            "str_25": str_25,
            "iv_25d_call": iv_25d_call,
            "iv_25d_put": iv_25d_put,
            "atm_strike": atm_strike,
            "K_25d_put": K_25d_put,
            "K_25d_call": K_25d_call,
            "T": T,
        }

        logger.info(
            f"{symbol} smile: ATM_IV={atm_iv:.4f}, RR25={rr_25:.4f}, "
            f"STR25={str_25:.4f}"
        )

        return result

    # ─── Order Execution ───────────────────────────────────────

    def place_option_order(
        self, symbol: str, expiry: str, strike: float,
        right: str, action: str, quantity: int,
        order_type: str = "LMT", limit_price: float = 0.0
    ) -> int:
        """
        Place an option order.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date (YYYYMMDD)
            strike: Strike price
            right: "P" for put, "C" for call
            action: "SELL" or "BUY"
            quantity: Number of contracts
            order_type: "LMT", "MKT", "MID"
            limit_price: Limit price for LMT orders

        Returns:
            Order ID
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = right
        contract.multiplier = "100"

        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        order.tif = "GTC"

        if order_type == "LMT" and limit_price > 0:
            order.lmtPrice = round(limit_price, 2)
        elif order_type == "MID":
            order.orderType = "MIDPRICE"

        order_id = self._get_req_id()

        logger.info(
            f"Placing order {order_id}: {action} {quantity}x "
            f"{symbol} {expiry} {strike} {right} @ {order_type} "
            f"{'$' + str(limit_price) if limit_price else ''}"
        )

        self.placeOrder(order_id, contract, order)

        return order_id

    def get_positions_sync(self) -> List[Dict]:
        """Get all current positions."""
        self._positions = []
        self.reqPositions()
        time.sleep(2)  # Allow position data to flow in
        return list(self._positions)

    def get_account_value_sync(self, key: str = "NetLiquidation") -> float:
        """Get account value."""
        self._account_event.clear()
        req_id = self._get_req_id()
        self.reqAccountSummary(
            req_id, "All",
            "NetLiquidation,TotalCashValue,BuyingPower,GrossPositionValue"
        )
        self._account_event.wait(timeout=self.timeout)
        self.cancelAccountSummary(req_id)
        return self._account_values.get(key, 0.0)
        return self._account_values.get(key, 0.0)
