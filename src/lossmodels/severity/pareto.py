import numpy as np
from ..utils.random import RNGLike, scipy_random_state
from scipy.stats import pareto

from .base import SeverityModel
from ..utils.numeric import eval_dist


class Pareto(SeverityModel):
    """
    Pareto Type I severity model.

    Parameterization
    ----------------
    X ~ Pareto(alpha, theta)
    Support: x >= theta

    Density:
        f(x) = alpha * theta^alpha / x^(alpha + 1),  x >= theta

    Parameters
    ----------
    alpha : float
        Shape parameter, with alpha > 0.
    theta : float
        Scale (minimum) parameter, with theta > 0.
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

        return pareto.rvs(b=self.alpha, scale=self.theta, size=size, random_state=scipy_random_state(rng))

    def mean(self) -> float:
        if self.alpha <= 1:
            raise ValueError("Mean does not exist for alpha <= 1.")
        return self.alpha * self.theta / (self.alpha - 1)

    def variance(self) -> float:
        if self.alpha <= 2:
            raise ValueError("Variance does not exist for alpha <= 2.")
        return (self.alpha * self.theta ** 2) / ((self.alpha - 1) ** 2 * (self.alpha - 2))

    def pdf(self, x):
        return eval_dist(lambda v: pareto.pdf(v, b=self.alpha, scale=self.theta), x)

    def cdf(self, x):
        return eval_dist(lambda v: pareto.cdf(v, b=self.alpha, scale=self.theta), x)

    def quantile(self, p):
        return eval_dist(lambda v: pareto.ppf(v, b=self.alpha, scale=self.theta), p)

    def __repr__(self) -> str:
        return f"Pareto(alpha={self.alpha}, theta={self.theta})"
