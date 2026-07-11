import numpy as np
from scipy.stats import nbinom

from ..utils.numeric import eval_dist
from ..utils.random import RNGLike, resolve_rng
from .base import FrequencyModel


class NegativeBinomial(FrequencyModel):
    """
    Negative Binomial frequency model.

    Parameterization
    ----------------
    N = number of failures before the r-th success
    Support: {0, 1, 2, ...}

    Parameters
    ----------
    r : float
        Number of successes, with r > 0.
    p : float
        Probability of success, with 0 < p <= 1.

    Notes
    -----
    This matches SciPy's negative binomial parameterization:
    scipy.stats.nbinom(r, p)

    Under this convention:
        E[N] = r(1 - p) / p
        Var(N) = r(1 - p) / p^2
    """

    def __init__(self, r: float, p: float):
        if r <= 0:
            raise ValueError("r must be positive.")
        if not (0 < p <= 1):
            raise ValueError("p must be in (0, 1].")

        self.r = r
        self.p = p

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        """
        Generate random samples of claim counts.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        return resolve_rng(rng).negative_binomial(self.r, self.p, size=size)

    def mean(self) -> float:
        """
        E[N] = r(1 - p) / p
        """
        return self.r * (1 - self.p) / self.p

    def variance(self) -> float:
        """
        Var(N) = r(1 - p) / p^2
        """
        return self.r * (1 - self.p) / (self.p ** 2)

    def pmf(self, k):
        return eval_dist(lambda v: nbinom.pmf(v, self.r, self.p), k)

    def cdf(self, k):
        return eval_dist(lambda v: nbinom.cdf(v, self.r, self.p), k)

    def __repr__(self) -> str:
        return f"NegativeBinomial(r={self.r}, p={self.p})"
