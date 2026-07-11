import numpy as np
from scipy.stats import expon

from ..utils.numeric import eval_dist
from ..utils.random import RNGLike, resolve_rng
from .base import SeverityModel


class Exponential(SeverityModel):
    """
    Exponential severity model.

    Parameterization
    ----------------
    X ~ Exponential(rate)

    Support: x >= 0

    Mean = 1 / rate
    Variance = 1 / rate^2

    Parameters
    ----------
    rate : float
        Rate parameter (lambda), with rate > 0.
    """

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError("rate must be positive.")

        self.rate = rate
        self.scale = 1.0 / rate  # SciPy uses scale = 1 / rate

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        """
        Generate random samples.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        return resolve_rng(rng).exponential(scale=self.scale, size=size)

    def mean(self) -> float:
        return 1.0 / self.rate

    def variance(self) -> float:
        return 1.0 / (self.rate ** 2)

    def pdf(self, x):
        return eval_dist(lambda v: expon.pdf(v, scale=self.scale), x)

    def cdf(self, x):
        return eval_dist(lambda v: expon.cdf(v, scale=self.scale), x)

    def quantile(self, p):
        return eval_dist(lambda v: expon.ppf(v, scale=self.scale), p)

    def excess_loss(self, d: float) -> float:
        """
        E[(X - d)+] = exp(-rate * d) / rate
        """
        if d < 0:
            raise ValueError("d must be nonnegative.")

        return float(np.exp(-self.rate * d) / self.rate)

    def limited_expected_value(self, d: float) -> float:
        """
        E[min(X, d)] = (1 - exp(-rate * d)) / rate
        """
        if d < 0:
            raise ValueError("d must be nonnegative.")

        return float((1.0 - np.exp(-self.rate * d)) / self.rate)

    def __repr__(self) -> str:
        return f"Exponential(rate={self.rate})"
