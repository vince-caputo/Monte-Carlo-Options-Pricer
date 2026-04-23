import numpy as np
from scipy.stats import qmc, norm
from gbm import GBM


def plain_mc(
    S_0, r, sigma, T,
    num_paths, num_steps,
    payoff_func, K,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Plain Monte Carlo estimator.

    Estimates the option price as:
        price = exp(-rT) * mean(payoff(paths))

    By the law of large numbers, this converges to the true risk-neutral
    expected payoff as num_paths -> infinity. The convergence rate is
    O(1/sqrt(n)) — halving the error requires 4x as many paths.

    The 95% confidence interval is computed using the CLT:
        error = 1.96 * std / sqrt(num_paths)

    Parameters
    ----------
    S_0, r, sigma, T : float
        GBM parameters.
    num_paths, num_steps : int
        Simulation dimensions.
    payoff_func : callable
        Payoff function from payoffs.py.
    K : float
        Strike price.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    mean : float
        Estimated option price.
    error : float
        Half-width of 95% confidence interval.
    """
    gbm = GBM(num_steps, num_paths, S_0, T, r, sigma, rng)
    data = gbm.get_data()
    discounted_payoffs = np.exp(-r * T) * payoff_func(data, K)
    mean = discounted_payoffs.mean()
    std = discounted_payoffs.std(ddof=1)
    error = 1.96 * std / np.sqrt(num_paths)
    return mean, error


def anti_mc(
    S_0, r, sigma, T,
    num_paths, num_steps,
    payoff_func, K,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Antithetic variates Monte Carlo estimator.

    For each simulated path driven by Z ~ N(0, dt), a paired antithetic
    path is generated using -Z. Since exp(x) and exp(-x) are negatively
    correlated, the paired payoffs partially cancel each other's sampling
    error. Averaging each pair before discounting reduces variance without
    increasing the number of GBM paths.

    The variance reduction factor relative to plain MC is:
        VR = Var(anti) / Var(plain) = (error_anti / error_plain)^2

    Values below 1.0 indicate improvement. For European calls this is
    typically 0.4-0.6 (40-60% variance reduction).

    Parameters
    ----------
    Same as plain_mc.

    Returns
    -------
    mean : float
        Estimated option price.
    error : float
        Half-width of 95% confidence interval.
    """
    gbm = GBM(num_steps, num_paths, S_0, T, r, sigma, rng, anti=True)
    data, anti_data = gbm.get_data()
    discounted_payoffs = np.exp(-r * T) * 0.5 * (
        payoff_func(data, K) + payoff_func(anti_data, K)
    )
    mean = discounted_payoffs.mean()
    std = discounted_payoffs.std(ddof=1)
    error = 1.96 * std / np.sqrt(num_paths)
    return mean, error


def control_mc(
    S_0, r, sigma, T,
    num_paths, num_steps,
    payoff_func, K,
    control_payoff_func, control_price,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Control variates Monte Carlo estimator.

    Uses a correlated quantity with a known analytical price to correct
    the Monte Carlo estimate. Let X be the target payoff (e.g. arithmetic
    Asian call) and Y be the control payoff (e.g. geometric Asian call)
    with known true price E[Y] = control_price.

    The corrected estimator is:
        Z = X - b * (Y - control_price)

    where the optimal correction coefficient is:
        b = Cov(X, Y) / Var(Y)

    This b minimises Var(Z). The resulting variance reduction is:
        Var(Z) = Var(X) * (1 - Corr(X, Y)^2)

    For arithmetic vs geometric Asian options, Corr(X, Y) is very high
    (typically > 0.99), so variance reduction can exceed 95%.

    Parameters
    ----------
    Same as plain_mc, plus:
    control_payoff_func : callable
        Payoff function for the control variate (e.g. asian_geom_call).
    control_price : float
        Known analytical price of the control variate option.

    Returns
    -------
    mean : float
        Estimated option price.
    error : float
        Half-width of 95% confidence interval.
    """
    gbm = GBM(num_steps, num_paths, S_0, T, r, sigma, rng)
    data = gbm.get_data()
    discounted_X = np.exp(-r * T) * payoff_func(data, K)
    discounted_Y = np.exp(-r * T) * control_payoff_func(data, K)

    cov = np.cov(discounted_X, discounted_Y, ddof=1)
    cov_XY = cov[0, 1]
    var_Y = cov[1, 1]
    b = cov_XY / var_Y

    Z = discounted_X - b * (discounted_Y - control_price)
    mean = Z.mean()
    std = Z.std(ddof=1)
    error = 1.96 * std / np.sqrt(num_paths)
    return mean, error


def qmc_mc(
    S_0, r, sigma, T,
    num_paths, num_steps,
    payoff_func, K,
):
    """
    Quasi-Monte Carlo (QMC) estimator using Sobol sequences.

    Standard Monte Carlo uses pseudo-random numbers which can cluster
    unevenly across the sample space by chance. Sobol sequences are
    deterministically constructed low-discrepancy sequences that fill
    the space more uniformly, leading to faster convergence.

    Theoretical convergence rates:
        Plain MC:  O(1/sqrt(n))
        QMC:       O(log(n)^d / n)  -- approximately O(1/n) for low d

    In practice this means QMC requires far fewer paths to achieve the
    same accuracy as plain MC for low-dimensional problems.

    Implementation:
        1. Generate uniform samples in [0,1]^d using a scrambled Sobol sequence
        2. Transform to standard normals via the inverse CDF (ppf)
        3. Use these normals in place of the pseudo-random Z in GBM path generation

    Limitation: The QMC advantage diminishes as num_steps (dimensionality)
    increases. For num_steps >> 30, the log(n)^d factor grows large and the
    theoretical advantage over plain MC weakens. Brownian bridge construction
    can partially recover the advantage in high dimensions by ensuring the
    most important dimensions receive the best coverage.

    Note: QMC does not use an rng parameter since Sobol sequences are
    deterministic. Scrambling provides randomisation for error estimation.

    Parameters
    ----------
    S_0, r, sigma, T : float
        GBM parameters.
    num_paths, num_steps : int
        Simulation dimensions. num_steps is also the dimensionality of
        the Sobol sequence.
    payoff_func : callable
        Payoff function from payoffs.py.
    K : float
        Strike price.

    Returns
    -------
    mean : float
        Estimated option price.
    error : float
        Half-width of 95% confidence interval (estimated via scrambling).
    """
    # Sobol sequence: num_paths x num_steps uniform samples in [0,1]
    sampler = qmc.Sobol(d=num_steps, scramble=True)
    # Sobol requires power-of-2 sample counts for optimal properties
    # Round up to next power of 2 if needed
    m = int(np.ceil(np.log2(num_paths)))
    u = sampler.random_base2(m)[:num_paths]  # shape (num_paths, num_steps)

    # Transform uniform samples to standard normals via inverse CDF
    Z = norm.ppf(u)  # shape (num_paths, num_steps)

    # Build GBM paths using quasi-random normals
    dt = T / num_steps
    # Scale Z to have variance dt (matching the GBM noise term sigma * dW)
    increments = np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * Z)
    paths = S_0 * np.cumprod(increments, axis=1)

    discounted_payoffs = np.exp(-r * T) * payoff_func(paths, K)
    mean = discounted_payoffs.mean()
    std = discounted_payoffs.std(ddof=1)
    error = 1.96 * std / np.sqrt(num_paths)
    return mean, error
