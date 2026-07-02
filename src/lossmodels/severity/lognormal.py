import numpy as np
from ..utils.random import RNGLike, resolve_rng
from scipy.stats import lognorm

from .base import SeverityModel
from ..utils.numeric import eval_dist


class Lognormal(SeverityModel):
    """
    Lognormal severity model.

    Parameterization
    ----------------
    If Y = log(X) ~ Normal(mu, sigma^2), then X is Lognormal(mu, sigma).
    Support: x > 0

    Parameters
    ----------
    mu : float
        Mean of log(X).
    sigma : float
        Standard deviation of log(X), with sigma > 0.
    """

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        self.mu = mu
        self.sigma = sigma

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")

        return resolve_rng(rng).lognormal(mean=self.mu, sigma=self.sigma, size=size)

    def mean(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma ** 2))

    def variance(self) -> float:
        sigma2 = self.sigma ** 2
        return float((np.exp(sigma2) - 1) * np.exp(2 * self.mu + sigma2))

    def pdf(self, x):
        return eval_dist(lambda v: lognorm.pdf(v, s=self.sigma, scale=np.exp(self.mu)), x)

    def cdf(self, x):
        return eval_dist(lambda v: lognorm.cdf(v, s=self.sigma, scale=np.exp(self.mu)), x)

    def quantile(self, p):
        return eval_dist(lambda v: lognorm.ppf(v, s=self.sigma, scale=np.exp(self.mu)), p)

    def __repr__(self) -> str:
        return f"Lognormal(mu={self.mu}, sigma={self.sigma})"
