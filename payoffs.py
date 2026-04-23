import numpy as np
from scipy.stats import norm


def euro_call(data: np.ndarray, K: float) -> np.ndarray:
    """
    European call option payoff: max(S_T - K, 0).

    Only the terminal price matters for a European option — the holder
    exercises if and only if S_T > K, earning S_T - K.

    Parameters
    ----------
    data : np.ndarray
        Either a 1D array (single path) or 2D array of shape
        (num_paths, num_steps). Only the final column is used.
    K : float
        Strike price.
    """
    if data.ndim == 1:
        return max(data[-1] - K, 0.0)
    return np.maximum(data[:, -1] - K, 0.0)


def asian_arith_call(data: np.ndarray, K: float) -> np.ndarray:
    """
    Arithmetic Asian call option payoff: max(A - K, 0).

    A is the arithmetic mean of the asset price at each time step.
    Averaging dampens the effect of extreme terminal prices, making
    Asian options cheaper than equivalent European options and less
    susceptible to price manipulation near expiry.

    No closed-form solution exists for this payoff under GBM because
    the sum of lognormal random variables does not have a tractable
    distribution. Monte Carlo is therefore necessary.

    Parameters
    ----------
    data : np.ndarray
        Either a 1D path or 2D array of shape (num_paths, num_steps).
    K : float
        Strike price.
    """
    if data.ndim == 1:
        return max(data.mean() - K, 0.0)
    return np.maximum(data.mean(axis=1) - K, 0.0)


def asian_geom_call(data: np.ndarray, K: float) -> np.ndarray:
    """
    Geometric Asian call option payoff: max(G - K, 0).

    G is the geometric mean of the asset price at each time step,
    computed as exp(mean(log(S_i))). This is numerically stable and
    equivalent to the nth root of the product of all prices.

    The geometric mean is always <= the arithmetic mean (AM-GM inequality),
    so geometric Asian options are always cheaper than arithmetic Asian options
    with identical parameters.

    A closed-form solution exists (see asian_geom_closed_form) because the
    product of lognormal random variables is itself lognormal. This makes
    the geometric Asian option useful as a control variate for pricing the
    arithmetic Asian option.

    Parameters
    ----------
    data : np.ndarray
        Either a 1D path or 2D array of shape (num_paths, num_steps).
    K : float
        Strike price.
    """
    if data.ndim == 1:
        return max(np.exp(np.mean(np.log(data))) - K, 0.0)
    geom_mean = np.exp(np.mean(np.log(data), axis=1))
    return np.maximum(geom_mean - K, 0.0)


def euro_call_closed_form(
    S_0: float, r: float, sigma: float, T: float, K: float
) -> float:
    """
    Black-Scholes closed-form price for a European call option.

    C = S_0 * N(d1) - K * exp(-rT) * N(d2)

    where:
        d1 = [log(S_0/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Used to validate the Monte Carlo estimators — the MC price should
    converge to this value as num_paths increases.

    Parameters
    ----------
    S_0 : float
        Initial asset price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility.
    T : float
        Time to maturity in years.
    K : float
        Strike price.
    """
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S_0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def asian_geom_closed_form(
    S_0: float, r: float, sigma: float, T: float, K: float
) -> float:
    """
    Closed-form price for a geometric Asian call option.

    Because the geometric mean of lognormal random variables is itself
    lognormal, a Black-Scholes-type formula applies with modified parameters:
        sigma_G = sigma / sqrt(3)   (effective volatility of the geometric mean)
        b = 0.5 * (r - 0.5 * sigma_G^2)  (adjusted drift)

    This is used as the control variate price when estimating the arithmetic
    Asian option price. Since arithmetic and geometric Asian options are
    highly correlated (both depend on the same paths), the control variate
    correction is highly effective.

    Parameters
    ----------
    S_0 : float
        Initial asset price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility.
    T : float
        Time to maturity in years.
    K : float
        Strike price.
    """
    sigma_G = sigma / np.sqrt(3)
    b = 0.5 * (r - 0.5 * sigma_G**2)
    d1 = (np.log(S_0 / K) + T * (b + 0.5 * sigma_G**2)) / (sigma_G * np.sqrt(T))
    d2 = d1 - sigma_G * np.sqrt(T)
    return S_0 * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
