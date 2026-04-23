import string
import numpy as np


class GBM:
    """
    Simulates asset price paths under Geometric Brownian Motion (GBM).

    The exact solution to the GBM SDE is used:
        S(t + dt) = S(t) * exp((r - sigma^2/2) * dt + sigma * dW)
    where dW ~ N(0, dt). This avoids discretisation error.

    Parameters
    ----------
    num_steps : int
        Number of time steps per path. Determines path granularity.
        Required for path-dependent options (e.g. Asian options).
    num_paths : int
        Number of simulated paths.
    init_price : float
        Initial asset price S_0.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility of the asset.
    rng : np.random.Generator
        NumPy random generator. Pass a seeded generator for reproducibility.
    anti : bool
        If True, also generates antithetic paths (using -Z instead of Z)
        for use in the antithetic variates estimator.
    """

    def __init__(
        self,
        num_steps: int = 52,
        num_paths: int = 10**5,
        init_price: float = 100.0,
        T: float = 1.0,
        r: float = 0.03,
        sigma: float = 0.2,
        rng: np.random.Generator = np.random.default_rng(),
        anti: bool = False,
    ):
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.init_price = init_price
        self.T = T
        self.r = r
        self.sigma = sigma
        self.rng = rng
        self.anti = anti

        self.data = None
        self.ticker = None
        self.anti_data = None
        self._gen_ticker()
        self._gen_data()

    def _gen_ticker(self):
        """Generate a random 4-letter ticker label for the simulated asset."""
        self.ticker = "".join(self.rng.choice(list(string.ascii_uppercase), size=4))

    def _gen_data(self):
        """
        Simulate GBM paths.

        Each path is constructed as a cumulative product of multiplicative
        increments. The increment over one step dt is:
            exp((r - sigma^2/2) * dt + sigma * Z)
        where Z ~ N(0, dt). The (r - sigma^2/2) term is the Ito correction
        that arises from applying Ito's lemma to log(S), ensuring the
        expected growth rate of S equals r rather than r + sigma^2/2.
        """
        dt = self.T / self.num_steps
        Z = self.rng.normal(0, np.sqrt(dt), size=(self.num_paths, self.num_steps))
        increments = np.exp((self.r - self.sigma**2 / 2) * dt + self.sigma * Z)
        self.data = self.init_price * np.cumprod(increments, axis=1)

        if self.anti:
            # Antithetic paths use -Z, negatively correlated with original paths.
            # Averaging paired payoffs reduces variance without additional simulation cost.
            anti_increments = np.exp(
                (self.r - self.sigma**2 / 2) * dt + self.sigma * (-Z)
            )
            self.anti_data = self.init_price * np.cumprod(anti_increments, axis=1)

    def get_data(self):
        """Return simulated paths. Returns (data, anti_data) if anti=True."""
        if self.anti:
            return self.data, self.anti_data
        return self.data
