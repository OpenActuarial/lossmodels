import numpy as np
from ..utils.random import RNGLike, resolve_rng
from scipy.stats import geom

from .base import FrequencyModel
from ..utils.numeric import eval_dist


class Geometric(FrequencyModel):
    """
    Geometric frequency model.

    Support starting at 0: {0, 1, 2, 3, ...}

    Parameters
    ----------
    p : float
        Probability of success

    Notes
    -----
    NumPy and SciPy define the geometric distribution on {1, 2, 3, ...}
    as the number of trials until first success. This implementation shifts
    that convention by 1 so the support starts at 0, which is more natural
    for claim counts.
    """

    def __init__(self, p: float):
        if not (0 < p <= 1):
            raise ValueError("p must be in (0, 1].")

        self.p = p

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        return resolve_rng(rng).geometric(self.p, size=size) - 1

    def mean(self) -> float:
        return (1 - self.p) / self.p

    def variance(self) -> float:
        return (1 - self.p) / (self.p ** 2)

    def pmf(self, k):
        """Probability mass function P(N = k)."""
        return eval_dist(lambda v: geom.pmf(v + 1, self.p), k)

    def cdf(self, k):
        """Cumulative distribution function P(N <= k)."""
        return eval_dist(lambda v: geom.cdf(v + 1, self.p), k)

    def __repr__(self):
        return f"Geometric(p={self.p})"
