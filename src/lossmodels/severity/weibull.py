import numpy as np
from scipy.special import gamma as gamma_func
from scipy.stats import weibull_min

from ..utils.numeric import eval_dist
from ..utils.random import RNGLike, resolve_rng
from .base import SeverityModel


class Weibull(SeverityModel):
    """
    Weibull severity model.

    Parameterization
    ----------------
    X ~ Weibull(shape=k, scale=lam)
    Support: x > 0

    Parameters
    ----------
    k : float
        Shape parameter, with k > 0.
    lam : float
        Scale parameter, with lam > 0.
    """

    def __init__(self, k: float, lam: float):
        if k <= 0:
            raise ValueError("k must be positive.")
        if lam <= 0:
            raise ValueError("lam must be positive.")

        self.k = k
        self.lam = lam

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")

        return self.lam * resolve_rng(rng).weibull(a=self.k, size=size)

    def mean(self) -> float:
        return float(self.lam * gamma_func(1 + 1 / self.k))

    def variance(self) -> float:
        m1 = gamma_func(1 + 1 / self.k)
        m2 = gamma_func(1 + 2 / self.k)
        return float(self.lam ** 2 * (m2 - m1 ** 2))

    def pdf(self, x):
        return eval_dist(lambda v: weibull_min.pdf(v, c=self.k, scale=self.lam), x)

    def cdf(self, x):
        return eval_dist(lambda v: weibull_min.cdf(v, c=self.k, scale=self.lam), x)

    def quantile(self, p):
        return eval_dist(lambda v: weibull_min.ppf(v, c=self.k, scale=self.lam), p)

    def __repr__(self) -> str:
        return f"Weibull(k={self.k}, lam={self.lam})"
