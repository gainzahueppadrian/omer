"""
Black-Scholes Greeks Calculator
Provides analytical Greeks for European options with dividend yield support.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate d1 in the Black-Scholes formula."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate d2 in the Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European put price."""
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)

    price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * np.exp(-q * T) * norm.cdf(-d1_val)
    return max(price, 0.0)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)

    price = S * np.exp(-q * T) * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
    return max(price, 0.0)


def put_delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Put option delta."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1_val = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * (norm.cdf(d1_val) - 1.0)


def call_delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Call option delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1_val = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(d1_val)


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Option gamma (same for puts and calls)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1_val = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1_val) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Option vega (per 1% IV change, divide by 100 for per-point)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1_val = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1_val) * np.sqrt(T) / 100.0


def theta_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Put theta (per calendar day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)

    term1 = -S * np.exp(-q * T) * norm.pdf(d1_val) * sigma / (2.0 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
    term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1_val)

    return (term1 + term2 + term3) / 365.0


def theta_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Call theta (per calendar day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)

    term1 = -S * np.exp(-q * T) * norm.pdf(d1_val) * sigma / (2.0 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
    term3 = q * S * np.exp(-q * T) * norm.cdf(d1_val)

    return (term1 + term2 + term3) / 365.0


def rho_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Put rho."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d2_val = d2(S, K, T, r, sigma, q)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2_val) / 100.0


def implied_volatility_put(
    price: float, S: float, K: float, T: float, r: float,
    q: float = 0.0, tol: float = 1e-6, max_iter: int = 100
) -> Optional[float]:
    """Newton-Raphson implied volatility solver for puts."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    if price < intrinsic:
        return None

    sigma_guess = 0.30

    for _ in range(max_iter):
        p = bs_put_price(S, K, T, r, sigma_guess, q)
        v = vega(S, K, T, r, sigma_guess, q) * 100.0  # undo the /100

        if abs(v) < 1e-12:
            break

        sigma_guess = sigma_guess - (p - price) / v

        if sigma_guess <= 0.001:
            sigma_guess = 0.001

        if abs(p - price) < tol:
            return sigma_guess

    return sigma_guess if abs(bs_put_price(S, K, T, r, sigma_guess, q) - price) < 0.01 else None


def compute_all_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "put", q: float = 0.0
) -> Dict[str, float]:
    """Compute all Greeks for a given option."""
    result = {
        "delta": 0.0,
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": 0.0,
        "rho": 0.0,
        "price": 0.0,
    }

    if option_type.lower() == "put":
        result["delta"] = put_delta(S, K, T, r, sigma, q)
        result["theta"] = theta_put(S, K, T, r, sigma, q)
        result["rho"] = rho_put(S, K, T, r, sigma, q)
        result["price"] = bs_put_price(S, K, T, r, sigma, q)
    else:
        result["delta"] = call_delta(S, K, T, r, sigma, q)
        result["theta"] = theta_call(S, K, T, r, sigma, q)
        result["price"] = bs_call_price(S, K, T, r, sigma, q)

    return result
