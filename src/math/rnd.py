import numpy as np
from scipy.stats import norm, lognorm
from scipy.optimize import minimize
import math

def malz_smile(delta, atm_vol, rr_25, str_25):
    """
    Interpolates implied volatility for a given delta using Malz (1997) formula.

    Args:
        delta (float): The delta of the option (0 to 1).
        atm_vol (float): At-the-money implied volatility.
        rr_25 (float): 25-delta Risk Reversal.
        str_25 (float): 25-delta Strangle.

    Returns:
        float: Interpolated implied volatility.
    """
    # Malz formula approximation
    # sigma(delta) = sigma_atm - 2 * RR * (delta - 0.5) + 16 * STR * (delta - 0.5)^2
    return atm_vol - 2 * rr_25 * (delta - 0.5) + 16 * str_25 * (delta - 0.5)**2

def bs_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculates Black-Scholes option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def delta_to_strike(delta, S, T, r, sigma, option_type='call'):
    """
    Converts a Delta to a Strike price using Black-Scholes formula.
    """
    # delta = N(d1) for call
    # delta = N(d1) - 1 for put? No, usually -N(-d1).
    # But for Malz, we usually work with absolute delta or call delta.
    # Assuming call delta for Malz interpolation input.

    if option_type == 'put':
        # Convert put delta to call delta equivalent for calculation if needed,
        # but here we just invert N(d1).
        # Put Delta is typically negative, so we use abs or N(d1)-1.
        # Let's assume input delta is always the relevant one, but Malz is usually defined on Call Delta 0-1.
        pass

    # We use Call Delta for the inversion
    # d1 = N^-1(delta)
    if delta <= 0 or delta >= 1:
        return S # Fallback

    d1 = norm.ppf(delta)
    # d1 = (ln(S/K) + (r + sigma^2/2)T) / (sigma*sqrt(T))
    # ln(S/K) = d1 * sigma * sqrt(T) - (r + sigma^2/2)T
    # K = S * exp( - (d1 * sigma * sqrt(T) - (r + sigma^2/2)T) )

    log_S_K = d1 * sigma * np.sqrt(T) - (r + 0.5 * sigma**2) * T
    K = S * np.exp(-log_S_K)
    return K

class MelickThomasMixture:
    def __init__(self):
        self.params = None # [w1, mu1, sigma1, mu2, sigma2]

    def _mixture_price(self, K, T, r, w1, mu1, sigma1, mu2, sigma2):
        # Price of a call option under mixture of lognormals
        # C = e^-rT * [ w1 * E1[(S-K)+] + (1-w1) * E2[(S-K)+] ]
        # E[(S-K)+] for lognormal(mu, sigma) is like BS but with mu instead of r?
        # Actually, if ln(S) ~ N(mu, sigma), then E[(S-K)+] = exp(mu + sigma^2/2) * N(d1) - K * N(d2)
        # where d1 = (mu - ln(K) + sigma^2) / sigma
        #       d2 = d1 - sigma

        def expected_payoff(mu, sigma):
            d1 = (mu - np.log(K) + sigma**2) / sigma
            d2 = d1 - sigma
            return np.exp(mu + 0.5 * sigma**2) * norm.cdf(d1) - K * norm.cdf(d2)

        val = w1 * expected_payoff(mu1, sigma1) + (1 - w1) * expected_payoff(mu2, sigma2)
        return np.exp(-r * T) * val

    def fit(self, strikes, prices, T, r, S_current):
        """
        Fits the mixture model to observed option prices.
        """
        def objective(params):
            w1, mu1, sigma1, mu2, sigma2 = params
            # Constraints
            if not (0 <= w1 <= 1) or sigma1 <= 0 or sigma2 <= 0:
                return 1e9

            # Martingale constraint: E[S] = S_current * e^rT
            # E[S] = w1 * exp(mu1 + sigma1^2/2) + (1-w1) * exp(mu2 + sigma2^2/2)
            # We can penalize deviation or enforce it.
            # For simplicity, let's penalize deviation from market prices AND martingale.

            model_prices = np.array([self._mixture_price(k, T, r, w1, mu1, sigma1, mu2, sigma2) for k in strikes])
            error = np.sum((model_prices - prices)**2)

            forward_price = S_current * np.exp(r * T)
            model_forward = w1 * np.exp(mu1 + 0.5 * sigma1**2) + (1 - w1) * np.exp(mu2 + 0.5 * sigma2**2)
            martingale_error = (model_forward - forward_price)**2

            return error + 10 * martingale_error

        # Initial guess
        # Assuming log-mean is roughly log(Forward) and vol is roughly ATM vol
        fwd_log = np.log(S_current) + r * T
        initial_guess = [0.5, fwd_log, 0.2 * np.sqrt(T), fwd_log, 0.4 * np.sqrt(T)]

        res = minimize(objective, initial_guess, method='Nelder-Mead', tol=1e-4)
        self.params = res.x
        return res.success

    def cdf(self, S_val):
        """
        Calculates the CDF of the mixture at S_val.
        P(S < S_val)
        If ln(S) ~ N(mu, sigma), then P(S < X) = P(ln(S) < ln(X)) = N( (ln(X) - mu) / sigma )
        """
        if self.params is None:
            raise ValueError("Model not fitted")

        w1, mu1, sigma1, mu2, sigma2 = self.params

        def lognorm_cdf(x, mu, sigma):
            return norm.cdf((np.log(x) - mu) / sigma)

        return w1 * lognorm_cdf(S_val, mu1, sigma1) + (1 - w1) * lognorm_cdf(S_val, mu2, sigma2)

def calculate_risk_of_reversal(S, K, T, r, atm_vol, rr_25, str_25):
    """
    Calculates the Risk of Reversal (Probability of Assignment) for a Put option at Strike K.

    Args:
        S (float): Current spot price.
        K (float): Strike price.
        T (float): Time to expiration (years).
        r (float): Risk-free rate.
        atm_vol (float): ATM Implied Volatility.
        rr_25 (float): 25-Delta Risk Reversal.
        str_25 (float): 25-Delta Strangle.

    Returns:
        float: Probability of S_T < K (Assignment probability for Put).
    """
    # 1. Generate synthetic market data using Malz
    deltas = np.linspace(0.05, 0.95, 10)
    strikes = []
    prices = []

    for d in deltas:
        # Get Vol
        vol = malz_smile(d, atm_vol, rr_25, str_25)
        # Get Strike
        strike = delta_to_strike(d, S, T, r, vol)
        # Get Price (Call)
        price = bs_price(S, strike, T, r, vol, 'call')

        strikes.append(strike)
        prices.append(price)

    strikes = np.array(strikes)
    prices = np.array(prices)

    # 2. Fit Melick & Thomas
    model = MelickThomasMixture()
    model.fit(strikes, prices, T, r, S)

    # 3. Calculate CDF at K
    prob_assignment = model.cdf(K)

    return prob_assignment
