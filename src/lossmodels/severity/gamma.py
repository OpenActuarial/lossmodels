import numpy as np
from scipy.stats import gamma

from ..utils.numeric import eval_dist
from ..utils.random import RNGLike, resolve_rng
from .base import SeverityModel


class Gamma(SeverityModel):
    """
    Gamma severity model.

    Parameterization
    ----------------
    X ~ Gamma(shape=alpha, scale=theta)
    Support: x > 0

    Parameters
    ----------
    alpha : float
        Shape parameter, with alpha > 0.
    theta : float
        Scale parameter, with theta > 0.
    """

    def __init__(self, alpha: float, theta: float):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if theta <= 0:
            raise ValueError("theta must be positive.")

        self.alpha = alpha
        self.theta = theta

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")

        return resolve_rng(rng).gamma(shape=self.alpha, scale=self.theta, size=size)

    def mean(self) -> float:
        return self.alpha * self.theta

    def variance(self) -> float:
        return self.alpha * (self.theta ** 2)

    def pdf(self, x):
        return eval_dist(lambda v: gamma.pdf(v, a=self.alpha, scale=self.theta), x)

    def cdf(self, x):
        return eval_dist(lambda v: gamma.cdf(v, a=self.alpha, scale=self.theta), x)

    def quantile(self, p):
        return eval_dist(lambda v: gamma.ppf(v, a=self.alpha, scale=self.theta), p)

    def __repr__(self) -> str:
        return f"Gamma(alpha={self.alpha}, theta={self.theta})"
