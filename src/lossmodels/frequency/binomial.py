import numpy as np
from scipy.stats import binom

from .base import FrequencyModel
from ..utils.numeric import eval_dist


class Binomial(FrequencyModel):
    """
    Binomial frequency model.

    Parameters
    ----------
    n : int
        Number of trials
    p : float
        Probability of success
    """

    def __init__(self, n: int, p: float):
        if n <= 0:
            raise ValueError("n must be positive")
        if not (0 <= p <= 1):
            raise ValueError("p must be between 0 and 1")

        self.n = n
        self.p = p

    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.binomial(self.n, self.p, size=size)

    def mean(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)

    def pmf(self, k):
        """Probability mass function P(N = k)."""
        return eval_dist(lambda v: binom.pmf(v, self.n, self.p), k)

    def cdf(self, k):
        """Cumulative distribution function P(N <= k)."""
        return eval_dist(lambda v: binom.cdf(v, self.n, self.p), k)

    def __repr__(self):
        return f"Binomial(n={self.n}, p={self.p})"