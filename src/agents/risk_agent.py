from src.math.rnd import calculate_risk_of_reversal

class RiskAgent:
    def __init__(self, max_risk_of_reversal=0.40):
        self.max_risk_of_reversal = max_risk_of_reversal

    def assess_opportunity(self, symbol, spot, strike, expiry_years, atm_iv, rr_25, str_25):
        """
        Evaluates a potential LEAPS put trade based on Risk of Reversal and Market Conditions.

        Args:
            symbol (str): Ticker symbol.
            spot (float): Current spot price.
            strike (float): Put strike price.
            expiry_years (float): Time to expiration in years.
            atm_iv (float): ATM Implied Volatility.
            rr_25 (float): 25-Delta Risk Reversal (skew).
            str_25 (float): 25-Delta Strangle (kurtosis).

        Returns:
            dict: {
                "approved": bool,
                "allocation_mult": float,
                "risk_of_reversal": float,
                "reason": str
            }
        """
        # 1. Calculate Risk of Reversal (Assignment Probability)
        # Using a fixed risk-free rate of 4% for now, or fetch from client
        r = 0.04

        prob_assignment = calculate_risk_of_reversal(spot, strike, expiry_years, r, atm_iv, rr_25, str_25)

        allocation_mult = 1.0
        reasons = []

        # 2. Assignment Probability Check
        if prob_assignment > self.max_risk_of_reversal:
            return {
                "approved": False,
                "allocation_mult": 0.0,
                "risk_of_reversal": prob_assignment,
                "reason": f"Risk of Reversal too high: {prob_assignment:.2%}"
            }

        if prob_assignment > 0.20:
            allocation_mult *= 0.5
            reasons.append(f"High Assignment Risk ({prob_assignment:.2%})")

        # 3. Skew Check (Risk Reversal)
        # rr_25 is typically negative. If it's very negative (e.g. < -5%), market fears downside.
        if rr_25 < -0.05:
            allocation_mult *= 0.8
            reasons.append(f"High Downside Skew ({rr_25:.2%})")

        # 4. Kurtosis Check (Strangle)
        # str_25 measures curvature. High str = fat tails.
        if str_25 > 0.05:
            allocation_mult *= 0.8
            reasons.append(f"High Kurtosis/Fat Tails ({str_25:.2%})")

        return {
            "approved": True,
            "allocation_mult": allocation_mult,
            "risk_of_reversal": prob_assignment,
            "reason": ", ".join(reasons) if reasons else "Standard Risk"
        }
