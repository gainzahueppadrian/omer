"""
LEAPS Strategy Logic

Encapsulates the core strategy logic for LEAPS put selling with bonus calls.
Used by agents for strategy-specific calculations.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from src.math.greeks import (
    compute_all_greeks, bs_put_price, implied_volatility_put
)
from src.math.rnd import RNDModel

logger = logging.getLogger(__name__)


class LEAPSStrategy:
    """
    LEAPS Put Selling Strategy Calculator

    Strategy Overview:
    - Sell deep-OTM LEAPS puts on quality stocks
    - Target 30-45 delta, 9-18 month expiration
    - Minimum 10% annualized return
    - Maximum 25% risk of reversal (RND-based assignment probability)
    - Optionally buy LEAPS calls funded by put premium (bonus strategy)

    Edge Sources:
    1. Volatility risk premium: implied vol > realized vol historically
    2. Skew premium: OTM puts are systematically overpriced
    3. Time decay: long-dated options provide smooth theta collection
    4. RND alpha: using mixture models instead of BS for better risk assessment
    """

    def __init__(self, config: Dict):
        self.config = config
        self.leaps_cfg = config.get("strategy", {}).get("leaps", {})
        self.bonus_cfg = config.get("strategy", {}).get("bonus_calls", {})

    def calculate_annualized_return(
        self, premium: float, strike: float, dte: int
    ) -> float:
        """
        Calculate annualized return on risk for a short put.

        Return = (Premium / Notional) * (365 / DTE)

        Where Notional = Strike * 100 (cash needed if assigned)
        """
        if strike <= 0 or dte <= 0:
            return 0.0

        notional = strike * 100  # per contract
        return_on_risk = premium * 100 / notional
        annualized = return_on_risk * (365.0 / dte)

        return annualized

    def calculate_break_even(self, strike: float, premium: float) -> float:
        """
        Calculate the break-even price for a short put.

        Break-even = Strike - Premium
        (buyer doesn't profit until price drops below this)
        """
        return strike - premium

    def calculate_margin_requirement(
        self, strike: float, spot: float, premium: float,
        margin_safety: float = 1.5
    ) -> float:
        """
        Estimate margin requirement for a short put.

        Standard formula (approximate):
        Margin = max(
            20% of underlying - OTM amount + premium,
            10% of strike + premium
        ) * 100 * safety_factor
        """
        otm_amount = max(spot - strike, 0)

        method1 = (0.20 * spot - otm_amount + premium) * 100
        method2 = (0.10 * strike + premium) * 100

        base_margin = max(method1, method2)
        return base_margin * margin_safety

    def evaluate_candidate(
        self, symbol: str, spot: float, strike: float, expiry: str,
        premium: float, iv: float, delta: float,
        rnd_model: Optional[RNDModel] = None,
        r: float = 0.05
    ) -> Dict:
        """
        Comprehensive evaluation of a LEAPS put candidate.

        Returns a detailed evaluation dictionary with all metrics.
        """
        # Calculate DTE
        try:
            exp_date = datetime.strptime(expiry, "%Y%m%d")
            dte = max((exp_date - datetime.now()).days, 1)
            T = dte / 365.0
        except ValueError:
            return {"viable": False, "reason": "Invalid expiry date"}

        # Basic metrics
        annualized_return = self.calculate_annualized_return(
            premium, strike, dte
        )
        break_even = self.calculate_break_even(strike, premium)
        margin_req = self.calculate_margin_requirement(
            strike, spot, premium,
            self.config.get("risk", {}).get("margin_safety_factor", 1.5)
        )
        otm_pct = (spot - strike) / spot

        # Risk of Reversal
        if rnd_model is not None:
            risk_of_reversal = rnd_model.risk_of_reversal(strike)
            rnd_diagnostics = rnd_model.get_diagnostics()
        else:
            # Fallback: approximate from delta
            risk_of_reversal = abs(delta) * 1.1  # rough approximation
            rnd_diagnostics = {}

        # Viability checks
        min_return = self.leaps_cfg.get("min_annualized_return", 0.10)
        max_ror = self.leaps_cfg.get("max_risk_of_reversal", 0.25)
        min_dte = self.leaps_cfg.get("min_dte", 270)
        max_dte = self.leaps_cfg.get("max_dte", 548)
        min_delta = abs(self.leaps_cfg.get("min_delta", -0.30))
        max_delta = abs(self.leaps_cfg.get("max_delta", -0.45))

        viable = True
        reasons = []

        if dte < min_dte:
            viable = False
            reasons.append(f"DTE {dte} < min {min_dte}")
        if dte > max_dte:
            viable = False
            reasons.append(f"DTE {dte} > max {max_dte}")
        if abs(delta) < min_delta:
            viable = False
            reasons.append(f"|Δ| {abs(delta):.3f} < min {min_delta}")
        if abs(delta) > max_delta:
            viable = False
            reasons.append(f"|Δ| {abs(delta):.3f} > max {max_delta}")
        if annualized_return < min_return:
            viable = False
            reasons.append(
                f"Return {annualized_return:.1%} < min {min_return:.1%}"
            )
        if risk_of_reversal > max_ror:
            viable = False
            reasons.append(
                f"RoR {risk_of_reversal:.1%} > max {max_ror:.1%}"
            )

        # Bonus strategy assessment
        bonus_assessment = {}
        if self.bonus_cfg.get("enabled", False) and viable:
            max_alloc = self.bonus_cfg.get("max_premium_allocation", 0.50)
            call_budget = premium * max_alloc
            bonus_assessment = {
                "enabled": True,
                "budget_per_contract": round(call_budget, 2),
                "budget_total_per_contract": round(call_budget * 100, 2),
                "target_call_delta": self.bonus_cfg.get(
                    "call_delta_target", 0.40
                ),
            }

        return {
            "viable": viable,
            "reasons": reasons,
            "symbol": symbol,
            "spot": spot,
            "strike": strike,
            "expiry": expiry,
            "dte": dte,
            "T": T,
            "premium": round(premium, 2),
            "annualized_return": round(annualized_return, 4),
            "break_even": round(break_even, 2),
            "otm_pct": round(otm_pct, 4),
            "margin_requirement": round(margin_req, 2),
            "delta": round(delta, 4),
            "iv": round(iv, 4),
            "risk_of_reversal": round(risk_of_reversal, 4),
            "return_per_risk": round(
                annualized_return / max(risk_of_reversal, 0.01), 2
            ),
            "bonus_strategy": bonus_assessment,
            "rnd_diagnostics": rnd_diagnostics,
        }

    def rank_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Rank viable candidates by composite score.

        Score components:
        - Return/Risk ratio (40%)
        - Annualized return (25%)
        - Safety: inverse RoR (20%)
        - IV attractiveness (15%)
        """
        viable = [c for c in candidates if c.get("viable", False)]

        for c in viable:
            ret_risk = c.get("return_per_risk", 0)
            ann_ret = c.get("annualized_return", 0)
            ror = c.get("risk_of_reversal", 1)
            iv = c.get("iv", 0)

            # Normalize to 0-1
            ret_risk_score = min(ret_risk / 2.0, 1.0)
            return_score = min(ann_ret / 0.20, 1.0)
            safety_score = max(1.0 - ror / 0.30, 0.0)
            iv_score = min(iv / 0.40, 1.0)

            c["composite_score"] = round(
                0.40 * ret_risk_score +
                0.25 * return_score +
                0.20 * safety_score +
                0.15 * iv_score,
                4
            )

        viable.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        return viable
