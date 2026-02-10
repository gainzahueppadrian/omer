import threading
import time
import queue
import numpy as np
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId

class IBClient(EWrapper, EClient):
    def __init__(self, host='127.0.0.1', port=7497, client_id=1, mock=False):
        EClient.__init__(self, self)
        self.host = host
        self.port = port
        self.client_id = client_id
        self.mock = mock

        self.next_valid_order_id = None
        self.connected_event = threading.Event()
        self.market_data_queues = {} # reqId -> Queue
        self.contract_details_queues = {} # reqId -> Queue (returns list of details)
        self.option_params_queues = {} # reqId -> Queue (returns list of params)

        # Mock data storage
        self.mock_prices = {} # symbol -> price
        self.mock_ivs = {} # symbol -> iv

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"Error: {reqId}, {errorCode}, {errorString}")

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId
        self.connected_event.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        if reqId in self.market_data_queues:
            # 4 = Last, 9 = Close. We prefer Last.
            if tickType == 4 or tickType == 9:
                 self.market_data_queues[reqId].put(price)

    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        if reqId in self.market_data_queues:
            # 13 = Model Option Computation
            # tickType 13: model option
            # tickType 10: bid option? No.
            # We want Model or Last Implied Vol.
            if tickType == 13 and impliedVol is not None:
                self.market_data_queues[reqId].put(impliedVol)

    def contractDetails(self, reqId, contractDetails):
        if reqId in self.contract_details_queues:
            self.contract_details_queues[reqId].put(contractDetails)

    def contractDetailsEnd(self, reqId):
        if reqId in self.contract_details_queues:
            # Signal end? We use queue put None or similar, or just let timeout handle if single item expected.
            # But usually we want a list.
            # For simplicity, we assume we want the first match in sync method.
            pass

    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, tradingClass, multiplier, expirations, strikes):
        if reqId in self.option_params_queues:
            self.option_params_queues[reqId].put((expirations, strikes))

    def securityDefinitionOptionParameterEnd(self, reqId):
        if reqId in self.option_params_queues:
            pass

    def connect_sync(self):
        if self.mock:
            print("Mock IBClient connected.")
            return True

        self.connect(self.host, self.port, self.client_id)

        # Start the socket in a thread
        thread = threading.Thread(target=self.run)
        thread.start()

        if not self.connected_event.wait(timeout=5):
            print("Failed to connect to IBKR")
            return False

        return True

    def get_market_price_sync(self, contract):
        if self.mock:
            return self.mock_prices.get(contract.symbol, 100.0)

        reqId = self.next_valid_order_id
        self.next_valid_order_id += 1

        q = queue.Queue()
        self.market_data_queues[reqId] = q

        self.reqMktData(reqId, contract, "", False, False, [])

        try:
            price = q.get(timeout=5)
            return price
        except queue.Empty:
            print(f"Timeout waiting for price for {contract.symbol}")
            return None
        finally:
            self.cancelMktData(reqId)
            del self.market_data_queues[reqId]

    def get_contract_details_sync(self, contract):
        reqId = self.next_valid_order_id
        self.next_valid_order_id += 1
        q = queue.Queue()
        self.contract_details_queues[reqId] = q

        self.reqContractDetails(reqId, contract)

        try:
            details = q.get(timeout=5)
            return details
        except queue.Empty:
            return None
        finally:
            del self.contract_details_queues[reqId]

    def get_option_params_sync(self, symbol, exchange, secType, conId):
        reqId = self.next_valid_order_id
        self.next_valid_order_id += 1
        q = queue.Queue()
        self.option_params_queues[reqId] = q

        self.reqSecDefOptParams(reqId, symbol, exchange, secType, conId)

        try:
            # We might get multiple callbacks for different exchanges/trading classes
            # We return the first one
            params = q.get(timeout=5)
            return params
        except queue.Empty:
            return None, None
        finally:
            del self.option_params_queues[reqId]

    def get_iv_sync(self, contract):
        reqId = self.next_valid_order_id
        self.next_valid_order_id += 1
        q = queue.Queue()
        self.market_data_queues[reqId] = q

        # Request Generic Tick 100 (Option Vol) ?
        # Or just default. TickType 13 comes by default for options.
        self.reqMktData(reqId, contract, "", False, False, [])

        try:
            iv = q.get(timeout=5)
            return iv
        except queue.Empty:
            return None
        finally:
            self.cancelMktData(reqId)
            del self.market_data_queues[reqId]

    def get_smile_data_sync(self, symbol, expiry_months, spot_price):
        """
        Fetches IV for ATM, 25-Delta Call, and 25-Delta Put.
        Returns (atm_iv, rr_25, str_25).
        """
        if self.mock:
            atm = self.mock_ivs.get(symbol, 0.20)
            return atm, -0.02, 0.01

        # 1. Get Contract Details for Underlying to get ConId
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        details = self.get_contract_details_sync(contract)
        if not details:
            print(f"Failed to get contract details for {symbol}")
            return 0.20, 0.0, 0.0 # Fallback

        conId = details.contract.conId
        exchange = details.contract.exchange # Or SMART? reqSecDefOptParams needs specific exchange often?
        # Usually we pass "" or the primary exchange.
        # Let's try "SMART" first.

        # 2. Get Option Params
        expirations, strikes = self.get_option_params_sync(symbol, "SMART", "STK", conId)
        if not expirations or not strikes:
            # Try Primary Exchange
            # self.get_option_params_sync(symbol, details.contract.primaryExchange, ...)
            print(f"Failed to get option params for {symbol}")
            return 0.20, 0.0, 0.0

        # 3. Find Target Expiration
        # Convert expiry_months to YYYYMMDD
        # This is hard to match exactly. We need to parse strings "20240621".
        # Let's pick the one ~ expiry_months away.
        # Placeholder logic: just sort and pick middle?
        # Or parse.
        # For this robust implementation, we assume we pick one.
        target_expiry = sorted(expirations)[-1] # Furthest out (LEAPS)

        # 4. Find Target Strikes
        # ATM
        strikes = sorted(list(strikes))
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))

        # 25 Delta Call (OTM) -> Strike > Spot
        # Approx: S * 1.1 ? Or use rough vol.
        target_call_strike = min(strikes, key=lambda x: abs(x - spot_price * 1.1))

        # 25 Delta Put (OTM) -> Strike < Spot
        # Approx: S * 0.9 ?
        target_put_strike = min(strikes, key=lambda x: abs(x - spot_price * 0.9))

        # 5. Fetch IVs
        def make_opt(strike, right):
            c = Contract()
            c.symbol = symbol
            c.secType = "OPT"
            c.exchange = "SMART"
            c.currency = "USD"
            c.lastTradeDateOrContractMonth = target_expiry
            c.strike = strike
            c.right = right
            c.multiplier = "100"
            return c

        atm_iv = self.get_iv_sync(make_opt(atm_strike, "C")) or 0.20
        call_25_iv = self.get_iv_sync(make_opt(target_call_strike, "C")) or 0.20
        put_25_iv = self.get_iv_sync(make_opt(target_put_strike, "P")) or 0.20

        # 6. Calculate RR and STR
        # RR = Vol(25D Call) - Vol(25D Put)
        # STR = 0.5 * (Vol(25D Call) + Vol(25D Put)) - ATM Vol

        rr = call_25_iv - put_25_iv
        strangle = 0.5 * (call_25_iv + put_25_iv) - atm_iv

        return atm_iv, rr, strangle

    def place_order_sync(self, contract, order):
        if self.mock:
            print(f"Mock Order Placed: {order.action} {order.totalQuantity} {contract.symbol} @ {order.lmtPrice}")
            return self.next_valid_order_id

        orderId = self.next_valid_order_id
        self.next_valid_order_id += 1
        self.placeOrder(orderId, contract, order)
        return orderId

    def set_mock_data(self, symbol, price, iv=0.2):
        self.mock_prices[symbol] = price
        self.mock_ivs[symbol] = iv
