import numpy as np
from scipy.stats import poisson

from .base import FrequencyModel
from ..utils.numeric import eval_dist


class Poisson(FrequencyModel):
    """
    Poisson frequency model.

    Parameterization
    ----------------
    N ~ Poisson(lam)

    Support: {0, 1, 2, ...}

    Parameters
    ----------
    lam : float
        Expected claim count, with lam >= 0.
    """

    def __init__(self, lam: float):
        if lam < 0:
            raise ValueError("lam must be nonnegative.")
        self.lam = lam

    def sample(self, size: int = 1) -> np.ndarray:
        """Generate random samples of claim counts."""
        if size <= 0:
            raise ValueError("size must be positive.")
        return np.random.poisson(self.lam, size=size)

    def mean(self) -> float:
        """E[N] = lam."""
        return self.lam

    def variance(self) -> float:
        """Var(N) = lam."""
        return self.lam

    def pmf(self, k):
        """Probability mass function P(N = k)."""
        return eval_dist(lambda v: poisson.pmf(v, self.lam), k)

    def cdf(self, k):
        """Cumulative distribution function P(N <= k)."""
        return eval_dist(lambda v: poisson.cdf(v, self.lam), k)

    def __repr__(self) -> str:
        return f"Poisson(lam={self.lam})"