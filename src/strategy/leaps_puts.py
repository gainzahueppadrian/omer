import numpy as np
from src.math.rnd import malz_smile, bs_price
from src.agents.risk_agent import RiskAgent

class LeapsPutsStrategy:
    def __init__(self, ib_client, risk_agent):
        self.ib = ib_client
        self.risk_agent = risk_agent
        self.min_annualized_return = 0.10 # 10%

    def scan_opportunities(self, symbols, expiry_months=12):
        opportunities = []

        for symbol in symbols:
            # 1. Get Spot Price
            contract = self.make_contract(symbol)
            spot = self.ib.get_market_price_sync(contract)
            if spot is None:
                continue

            # 2. Get Smile Data (ATM, RR, STR)
            # Expiry in years
            T = expiry_months / 12.0
            atm_iv, rr_25, str_25 = self.ib.get_smile_data_sync(symbol, expiry_months, spot)

            # 3. Iterate Potential Strikes (OTM Puts)
            # From 70% to 100% of spot
            strikes = np.linspace(spot * 0.70, spot * 1.0, 20)

            best_trade = None

            for K in strikes:
                # Calculate Delta (approx) to get IV from Malz
                # We need Delta for Malz input.
                # Delta depends on Vol. Iterative?
                # Use ATM vol to get approximate delta, get vol, refine delta?
                # Or just use the malz_smile function with delta derived from BS using ATM vol first.

                # First pass delta
                r = 0.04
                # d1 using atm_iv
                d1 = (np.log(spot / K) + (r + 0.5 * atm_iv**2) * T) / (atm_iv * np.sqrt(T))
                delta_approx = np.exp(-r*T) * 0.5 # Rough guess or use standard BS delta
                # Actually, let's use a helper or just use the d1 logic
                # Call Delta = N(d1)
                from scipy.stats import norm
                call_delta = norm.cdf(d1)

                # Get IV from Malz
                iv = malz_smile(call_delta, atm_iv, rr_25, str_25)

                # Calculate Option Price (Put)
                price = bs_price(spot, K, T, r, iv, 'put')

                # Calculate Annualized Return on Capital (Cash Secured)
                # Return = Premium / Strike
                # Annualized = Return * (1 / T)
                roc = (price / K) / T

                if roc < self.min_annualized_return:
                    continue

                # Check Risk Agent
                assessment = self.risk_agent.assess_opportunity(
                    symbol, spot, K, T, atm_iv, rr_25, str_25
                )

                if not assessment['approved']:
                    continue

                # Store candidate
                trade = {
                    "symbol": symbol,
                    "strike": K,
                    "expiry_months": expiry_months,
                    "premium": price,
                    "iv": iv,
                    "roc_annualized": roc,
                    "risk_of_reversal": assessment['risk_of_reversal'],
                    "allocation_mult": assessment['allocation_mult'],
                    "reason": assessment['reason']
                }

                # Prefer higher premium (closer to ATM) as long as it's approved?
                # Or prefer lower risk?
                # Strategy: "Sell a LEAPS put... correct strike... high probability"
                # Let's pick the one with highest Premium that is APPROVED.
                # Since we iterate from low strike to high strike, the last one is closest to ATM (highest premium).
                best_trade = trade

            if best_trade:
                opportunities.append(best_trade)

        return opportunities

    def make_contract(self, symbol):
        # Simplified contract creation
        from ibapi.contract import Contract
        c = Contract()
        c.symbol = symbol
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c
