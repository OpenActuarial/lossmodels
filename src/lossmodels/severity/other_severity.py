"""Other continuous severities (Loss Models, Appendix A.5).

    InverseGaussian(mu, theta)        invgauss(mu=mu/theta, scale=theta)
    LogT(r, mu, sigma)                X = exp(sigma * T_r + mu), T_r ~ Student-t
    SingleParameterPareto(alpha, theta)  pareto(b=alpha, scale=theta)  [Type I]

Lognormal already lives in ``lossmodels.Lognormal``.
"""

import numpy as np
from scipy.stats import invgauss, pareto
from scipy.stats import t as student_t

from ..utils.numeric import eval_dist
from ..utils.random import RNGLike, scipy_random_state
from .base import SeverityModel


class InverseGaussian(SeverityModel):
    """Inverse Gaussian (Wald) severity, Klugman ``(mu, theta)`` parameterization.

    X ~ InverseGaussian(mu, theta), support x > 0.
        E[X] = mu,  Var[X] = mu^3 / theta.

    Mapped to ``scipy.stats.invgauss(mu=mu/theta, scale=theta)``.
    """

    def __init__(self, mu: float, theta: float):
        if mu <= 0 or theta <= 0:
            raise ValueError("mu and theta must be positive.")
        self.mu = mu
        self.theta = theta
        self._d = invgauss(mu=mu / theta, scale=theta)

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.mu ** 3 / self.theta

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        return self._d.rvs(size=size, random_state=scipy_random_state(rng))

    def pdf(self, x):
        return eval_dist(lambda v: self._d.pdf(v), x)

    def cdf(self, x):
        return eval_dist(lambda v: self._d.cdf(v), x)

    def quantile(self, p):
        return eval_dist(lambda v: self._d.ppf(v), p)

    def __repr__(self) -> str:
        return f"InverseGaussian(mu={self.mu}, theta={self.theta})"


class LogT(SeverityModel):
    """Log-t severity.

    If Y has a Student-t distribution with r degrees of freedom, then
    X = exp(sigma * Y + mu) has the log-t distribution (support x > 0). It is
    heavier-tailed than the lognormal; **no positive moments exist**, so
    :meth:`mean` and :meth:`variance` raise. ``mu`` may be negative.

        F(x) = F_r((ln x - mu) / sigma),   F_r the Student-t cdf.
    """

    def __init__(self, r: float, mu: float, sigma: float):
        if r <= 0:
            raise ValueError("r (degrees of freedom) must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self._t = student_t(df=r)

    def mean(self) -> float:
        raise ValueError("Positive moments do not exist for the log-t distribution.")

    def variance(self) -> float:
        raise ValueError("Positive moments do not exist for the log-t distribution.")

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        return np.exp(self.sigma * self._t.rvs(size=size, random_state=scipy_random_state(rng)) + self.mu)

    def pdf(self, x):
        def f(v):
            v = np.asarray(v, dtype=float)
            pos = v > 0
            safe = np.where(pos, v, 1.0)
            z = (np.log(safe) - self.mu) / self.sigma
            return np.where(pos, self._t.pdf(z) / (safe * self.sigma), 0.0)

        return eval_dist(f, x)

    def cdf(self, x):
        def f(v):
            v = np.asarray(v, dtype=float)
            pos = v > 0
            safe = np.where(pos, v, 1.0)
            z = (np.log(safe) - self.mu) / self.sigma
            return np.where(pos, self._t.cdf(z), 0.0)

        return eval_dist(f, x)

    def quantile(self, p):
        return eval_dist(
            lambda v: np.exp(self.sigma * self._t.ppf(v) + self.mu), p
        )

    def __repr__(self) -> str:
        return f"LogT(r={self.r}, mu={self.mu}, sigma={self.sigma})"


class SingleParameterPareto(SeverityModel):
    """Single-Parameter Pareto severity (Pareto Type I; theta is a fixed lower bound).

    X ~ SingleParameterPareto(alpha, theta), support x > theta.
        f(x) = alpha theta^alpha / x^(alpha+1),  x > theta
        F(x) = 1 - (theta/x)^alpha,  x > theta
        E[X^k] = alpha theta^k / (alpha - k),  k < alpha.

    This is the same distribution as :class:`lossmodels.Pareto`; it is provided
    under its *Loss Models* Appendix A name, where only ``alpha`` is a true
    parameter -- ``theta`` is the known lower truncation point set in advance.
    """

    def __init__(self, alpha: float, theta: float):
        if alpha <= 0 or theta <= 0:
            raise ValueError("alpha and theta must be positive.")
        self.alpha = alpha
        self.theta = theta
        self._d = pareto(b=alpha, scale=theta)

    def _moment(self, k: float) -> float:
        if k >= self.alpha:
            raise ValueError(f"E[X^{k}] does not exist; need k < alpha.")
        return self.alpha * self.theta ** k / (self.alpha - k)

    def mean(self) -> float:
        if self.alpha <= 1:
            raise ValueError("Mean does not exist for alpha <= 1.")
        return self._moment(1)

    def variance(self) -> float:
        if self.alpha <= 2:
            raise ValueError("Variance does not exist for alpha <= 2.")
        return self._moment(2) - self._moment(1) ** 2

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        return self._d.rvs(size=size, random_state=scipy_random_state(rng))

    def pdf(self, x):
        return eval_dist(lambda v: self._d.pdf(v), x)

    def cdf(self, x):
        return eval_dist(lambda v: self._d.cdf(v), x)

    def quantile(self, p):
        return eval_dist(lambda v: self._d.ppf(v), p)

    def __repr__(self) -> str:
        return f"SingleParameterPareto(alpha={self.alpha}, theta={self.theta})"
