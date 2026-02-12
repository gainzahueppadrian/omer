"""
Risk-Neutral Density (RND) Module

Implements:
1. Malz (1997) quadratic volatility smile interpolation
2. Melick & Thomas (1997) mixture of log-normals model
3. Risk of Reversal calculation (assignment probability from market skew/kurtosis)

These methods go beyond simple Black-Scholes delta to provide a market-implied
probability of the option finishing in-the-money, incorporating skew and fat tails.
"""

import numpy as np
from scipy.stats import norm, lognorm
from scipy.optimize import minimize, least_squares
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MalzSmileInterpolator:
    """
    Malz (1997) quadratic volatility smile interpolation.

    σ(Δ) = σ_ATM + 2·RR·(Δ - 0.5) + 16·STR·(Δ - 0.5)²

    Where:
        σ_ATM = at-the-money implied volatility
        RR = 25-delta risk reversal = σ(25Δ call) - σ(25Δ put)
        STR = 25-delta strangle = [σ(25Δ call) + σ(25Δ put)] / 2 - σ_ATM

    The factor adjustments (2 and 16) are the Malz coefficients that ensure
    the function exactly fits the three market quotes at Δ = 0.25, 0.50, 0.75.
    """

    def __init__(self, atm_iv: float, rr_25: float, str_25: float):
        """
        Args:
            atm_iv: ATM implied volatility (e.g., 0.25 for 25%)
            rr_25: 25-delta risk reversal (call IV - put IV)
            str_25: 25-delta strangle (avg of 25Δ call/put IV minus ATM IV)
        """
        self.atm_iv = atm_iv
        self.rr_25 = rr_25
        self.str_25 = str_25

        # Malz coefficients
        self.a0 = atm_iv
        self.a1 = 2.0 * rr_25
        self.a2 = 16.0 * str_25

    def sigma(self, delta: float) -> float:
        """
        Interpolate implied volatility at a given Black-Scholes delta.

        Args:
            delta: Black-Scholes delta (0 to 1 for calls, use |delta| for puts)

        Returns:
            Interpolated implied volatility
        """
        x = delta - 0.5
        return self.a0 + self.a1 * x + self.a2 * x ** 2

    def sigma_at_strike(self, K: float, S: float, T: float, r: float) -> float:
        """
        Get IV for a specific strike by first computing BS delta, then interpolating.
        Uses iterative approach since IV depends on delta and vice versa.
        """
        sigma_est = self.atm_iv

        for _ in range(20):
            d1 = (np.log(S / K) + (r + 0.5 * sigma_est ** 2) * T) / (sigma_est * np.sqrt(T))
            delta_call = norm.cdf(d1)
            sigma_new = self.sigma(delta_call)

            if abs(sigma_new - sigma_est) < 1e-6:
                break
            sigma_est = sigma_new

        return sigma_est

    def get_smile_curve(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Return (delta_array, sigma_array) for the fitted smile."""
        deltas = np.linspace(0.05, 0.95, n_points)
        sigmas = np.array([self.sigma(d) for d in deltas])
        return deltas, sigmas


class MelickThomasMixture:
    """
    Melick & Thomas (1997) Mixture of Log-Normals Model.

    The risk-neutral density is modeled as a mixture of two log-normal distributions:

        f(S_T) = w · LN(S_T; μ₁, σ₁) + (1-w) · LN(S_T; μ₂, σ₂)

    Where:
        w = mixing weight (0 < w < 1)
        μ₁, σ₁ = parameters of the first log-normal (normal/base scenario)
        μ₂, σ₂ = parameters of the second log-normal (crash/stress scenario)

    The model is fit to market-observed smile data (ATM, RR, STR) and can capture:
        - Skewness (asymmetric crash risk)
        - Excess kurtosis (fat tails)
        - Bimodal risk distributions (e.g., event risk)
    """

    def __init__(self):
        self.w: float = 0.5
        self.mu1: float = 0.0
        self.sigma1: float = 0.2
        self.mu2: float = 0.0
        self.sigma2: float = 0.4
        self.fitted: bool = False

    def _lognormal_pdf(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Log-normal probability density function."""
        if sigma <= 0:
            return np.zeros_like(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                x > 0,
                np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2) / (x * sigma * np.sqrt(2 * np.pi)),
                0.0
            )
        return result

    def _lognormal_cdf(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Log-normal cumulative distribution function."""
        if sigma <= 0:
            return np.where(x >= np.exp(mu), 1.0, 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                x > 0,
                norm.cdf((np.log(x) - mu) / sigma),
                0.0
            )
        return result

    def pdf(self, S_T: np.ndarray) -> np.ndarray:
        """Mixture probability density function."""
        return (self.w * self._lognormal_pdf(S_T, self.mu1, self.sigma1) +
                (1 - self.w) * self._lognormal_pdf(S_T, self.mu2, self.sigma2))

    def cdf(self, S_T: np.ndarray) -> np.ndarray:
        """
        Mixture cumulative distribution function.
        P(S_T <= x) = w · Φ((ln x - μ₁)/σ₁) + (1-w) · Φ((ln x - μ₂)/σ₂)
        """
        return (self.w * self._lognormal_cdf(S_T, self.mu1, self.sigma1) +
                (1 - self.w) * self._lognormal_cdf(S_T, self.mu2, self.sigma2))

    def call_price(self, K: float, S: float, T: float, r: float) -> float:
        """
        European call price under the mixture model.
        C = e^{-rT} · [w · E₁[max(S_T - K, 0)] + (1-w) · E₂[max(S_T - K, 0)]]

        For log-normal component i:
        E_i[max(S_T - K, 0)] = exp(μ_i + σ_i²/2) · Φ(d1_i) - K · Φ(d2_i)
        where d1_i = (μ_i + σ_i² - ln K) / σ_i
              d2_i = d1_i - σ_i
        """
        discount = np.exp(-r * T)

        def component_call(mu, sigma):
            if sigma <= 0:
                fwd = np.exp(mu)
                return max(fwd - K, 0.0)
            d1 = (mu + sigma ** 2 - np.log(K)) / sigma
            d2 = d1 - sigma
            return np.exp(mu + 0.5 * sigma ** 2) * norm.cdf(d1) - K * norm.cdf(d2)

        c1 = component_call(self.mu1, self.sigma1)
        c2 = component_call(self.mu2, self.sigma2)

        return discount * (self.w * c1 + (1 - self.w) * c2)

    def put_price(self, K: float, S: float, T: float, r: float) -> float:
        """European put via put-call parity: P = C - S + K·e^{-rT}."""
        call = self.call_price(K, S, T, r)
        fwd = S  # assuming forward = spot (simplification, or use S·e^{(r-q)T})
        return call - fwd + K * np.exp(-r * T)

    def probability_itm_put(self, K: float) -> float:
        """
        P(S_T < K) = CDF(K) = "Risk of Reversal" / Assignment Probability.
        This is the market-implied probability of assignment for a short put at strike K.
        It incorporates skew (crash risk) and kurtosis (tail risk) from the
        options market, unlike simple BS delta which assumes log-normal distribution.
        """
        return float(self.cdf(np.array([K]))[0])

    def fit(self, S: float, T: float, r: float,
            atm_iv: float, rr_25: float, str_25: float) -> Dict:
        """
        Fit the mixture model to market smile data.

        Uses the Malz interpolation to generate target IVs at multiple deltas,
        then optimizes mixture parameters to match those prices.

        Args:
            S: Current spot price
            T: Time to expiry (years)
            r: Risk-free rate
            atm_iv: ATM implied volatility
            rr_25: 25-delta risk reversal
            str_25: 25-delta butterfly/strangle

        Returns:
            Dict with fitted parameters and diagnostics
        """
        # Generate target smile from Malz interpolation
        malz = MalzSmileInterpolator(atm_iv, rr_25, str_25)

        # Compute forward price
        F = S * np.exp(r * T)

        # Generate strikes from deltas
        target_deltas = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
        strikes = []
        target_ivs = []

        for delta in target_deltas:
            iv = malz.sigma(delta)
            if iv > 0.01:
                # Strike from BS delta: K = F · exp(-d1·σ√T + 0.5·σ²T)
                # Approximate: K = F · exp(norm.ppf(1-delta) · iv · √T - 0.5·iv²·T)
                K = F * np.exp(-norm.ppf(delta) * iv * np.sqrt(T) + 0.5 * iv ** 2 * T)
                strikes.append(K)
                target_ivs.append(iv)

        strikes = np.array(strikes)
        target_ivs = np.array(target_ivs)

        # Target BS prices
        from src.math.greeks import bs_call_price
        target_prices = np.array([
            bs_call_price(S, K, T, r, iv) for K, iv in zip(strikes, target_ivs)
        ])

        # Initial guess: component 1 = base, component 2 = stress
        log_F = np.log(F)
        x0 = np.array([
            0.7,                           # w
            log_F - 0.5 * atm_iv ** 2 * T,  # mu1
            atm_iv * np.sqrt(T) * 0.8,      # sigma1
            log_F - 0.5 * (atm_iv * 1.5) ** 2 * T,  # mu2
            atm_iv * np.sqrt(T) * 1.5,      # sigma2
        ])

        def objective(x):
            w, mu1, s1, mu2, s2 = x
            self.w = np.clip(w, 0.01, 0.99)
            self.mu1 = mu1
            self.sigma1 = max(s1, 0.01)
            self.mu2 = mu2
            self.sigma2 = max(s2, 0.01)

            model_prices = np.array([
                self.call_price(K, S, T, r) for K in strikes
            ])
            residuals = (model_prices - target_prices) / (target_prices + 1e-8)
            return residuals

        try:
            result = least_squares(
                objective, x0,
                bounds=(
                    [0.01, -np.inf, 0.01, -np.inf, 0.01],
                    [0.99, np.inf, 5.0, np.inf, 5.0]
                ),
                method='trf',
                max_nfev=1000
            )

            self.w = np.clip(result.x[0], 0.01, 0.99)
            self.mu1 = result.x[1]
            self.sigma1 = max(result.x[2], 0.01)
            self.mu2 = result.x[3]
            self.sigma2 = max(result.x[4], 0.01)
            self.fitted = True

            logger.info(
                f"RND mixture fit: w={self.w:.3f}, "
                f"μ1={self.mu1:.4f}, σ1={self.sigma1:.4f}, "
                f"μ2={self.mu2:.4f}, σ2={self.sigma2:.4f}, "
                f"cost={result.cost:.6f}"
            )

            return {
                "w": self.w,
                "mu1": self.mu1, "sigma1": self.sigma1,
                "mu2": self.mu2, "sigma2": self.sigma2,
                "cost": result.cost,
                "success": result.success,
            }

        except Exception as e:
            logger.error(f"RND mixture fitting failed: {e}")
            # Fallback: single log-normal
            self.w = 1.0
            self.mu1 = log_F - 0.5 * atm_iv ** 2 * T
            self.sigma1 = atm_iv * np.sqrt(T)
            self.mu2 = self.mu1
            self.sigma2 = self.sigma1
            self.fitted = True
            return {"w": 1.0, "success": False, "error": str(e)}

    def moments(self) -> Dict[str, float]:
        """Compute mean, variance, skewness, kurtosis of the mixture."""
        def comp_moments(mu, sigma):
            m1 = np.exp(mu + 0.5 * sigma ** 2)
            m2 = np.exp(2 * mu + 2 * sigma ** 2)
            m3 = np.exp(3 * mu + 4.5 * sigma ** 2)
            m4 = np.exp(4 * mu + 8 * sigma ** 2)
            return m1, m2, m3, m4

        m1_a, m2_a, m3_a, m4_a = comp_moments(self.mu1, self.sigma1)
        m1_b, m2_b, m3_b, m4_b = comp_moments(self.mu2, self.sigma2)

        w = self.w
        mean = w * m1_a + (1 - w) * m1_b
        E_X2 = w * m2_a + (1 - w) * m2_b
        E_X3 = w * m3_a + (1 - w) * m3_b
        E_X4 = w * m4_a + (1 - w) * m4_b

        var = E_X2 - mean ** 2
        std = np.sqrt(max(var, 1e-12))

        # Central moments
        mu3 = E_X3 - 3 * mean * E_X2 + 2 * mean ** 3
        mu4 = E_X4 - 4 * mean * E_X3 + 6 * mean ** 2 * E_X2 - 3 * mean ** 4

        skew = mu3 / std ** 3 if std > 0 else 0.0
        kurt = mu4 / std ** 4 if std > 0 else 3.0

        return {
            "mean": mean,
            "variance": var,
            "std": std,
            "skewness": skew,
            "kurtosis": kurt,
            "excess_kurtosis": kurt - 3.0,
        }


class RNDModel:
    """
    Unified Risk-Neutral Density model combining Malz smile and Melick-Thomas mixture.

    This is the main interface for computing assignment probabilities
    (Risk of Reversal) from market data.
    """

    def __init__(self):
        self.malz: Optional[MalzSmileInterpolator] = None
        self.mixture: Optional[MelickThomasMixture] = None
        self.atm_iv: float = 0.0
        self.rr_25: float = 0.0
        self.str_25: float = 0.0
        self.spot: float = 0.0
        self.T: float = 0.0
        self.r: float = 0.05

    def calibrate(self, S: float, T: float, r: float,
                  atm_iv: float, rr_25: float, str_25: float) -> Dict:
        """
        Calibrate both Malz smile and mixture model from market data.

        Args:
            S: Spot price
            T: Time to expiry (years)
            r: Risk-free rate
            atm_iv: ATM implied vol
            rr_25: 25-delta risk reversal
            str_25: 25-delta strangle

        Returns:
            Calibration diagnostics
        """
        self.spot = S
        self.T = T
        self.r = r
        self.atm_iv = atm_iv
        self.rr_25 = rr_25
        self.str_25 = str_25

        # Malz smile
        self.malz = MalzSmileInterpolator(atm_iv, rr_25, str_25)

        # Mixture model
        self.mixture = MelickThomasMixture()
        fit_result = self.mixture.fit(S, T, r, atm_iv, rr_25, str_25)

        moments = self.mixture.moments()

        logger.info(
            f"RND calibrated: ATM_IV={atm_iv:.4f}, RR={rr_25:.4f}, STR={str_25:.4f} | "
            f"Skew={moments['skewness']:.3f}, Kurt={moments['kurtosis']:.3f}"
        )

        return {**fit_result, **moments}

    def risk_of_reversal(self, K: float) -> float:
        """
        Calculate the 'Risk of Reversal' = P(S_T < K) from the fitted RND.

        This is the market-implied probability of assignment for a short put at strike K.
        It incorporates skew (crash risk) and kurtosis (tail risk) from the
        options market, unlike simple BS delta which assumes log-normal distribution.

        Args:
            K: Strike price

        Returns:
            Probability of S_T finishing below K (0 to 1)
        """
        if self.mixture is None or not self.mixture.fitted:
            # Fallback to BS delta-based approximation
            if self.atm_iv > 0 and self.T > 0:
                from src.math.greeks import d2 as calc_d2
                d2_val = calc_d2(self.spot, K, self.T, self.r, self.atm_iv)
                return float(norm.cdf(-d2_val))
            return 0.5

        return self.mixture.probability_itm_put(K)

    def interpolated_iv(self, delta: float) -> float:
        """Get Malz-interpolated IV at a given delta."""
        if self.malz is None:
            return self.atm_iv
        return self.malz.sigma(delta)

    def get_diagnostics(self) -> Dict:
        """Return full model diagnostics."""
        result = {
            "atm_iv": self.atm_iv,
            "rr_25": self.rr_25,
            "str_25": self.str_25,
            "spot": self.spot,
            "T": self.T,
        }
        if self.mixture and self.mixture.fitted:
            result["mixture_params"] = {
                "w": self.mixture.w,
                "mu1": self.mixture.mu1, "sigma1": self.mixture.sigma1,
                "mu2": self.mixture.mu2, "sigma2": self.mixture.sigma2,
            }
            result["moments"] = self.mixture.moments()
        return result
