"""Transformed Gamma family of continuous severities (the inverse members).

Klugman (Loss Models, Appendix A.3) parameterizations. The forward members
(Gamma, Weibull, Exponential) already exist in ``lossmodels``; this module adds
their inverses. Parameter -> SciPy mapping (verified against the table moments):

    InverseGamma(alpha, theta)     invgamma(a=alpha, scale=theta)
    InverseWeibull(theta, tau)     invweibull(c=tau, scale=theta)
    InverseExponential(theta)      invweibull(c=1, scale=theta)
"""

from math import gamma as _G

import numpy as np
from scipy.stats import invgamma, invweibull

from ..utils.numeric import eval_dist
from ..utils.random import RNGLike, scipy_random_state
from .base import SeverityModel


class InverseGamma(SeverityModel):
    """Inverse gamma (Vinci) severity.

    X ~ InverseGamma(alpha, theta), support x > 0.
        F(x) = 1 - Gamma(alpha; theta/x)
        E[X^k] = theta^k Gamma(alpha - k) / Gamma(alpha),  k < alpha.
    """

    def __init__(self, alpha: float, theta: float):
        if alpha <= 0 or theta <= 0:
            raise ValueError("alpha and theta must be positive.")
        self.alpha = alpha
        self.theta = theta
        self._d = invgamma(a=alpha, scale=theta)

    def _moment(self, k: float) -> float:
        if k >= self.alpha:
            raise ValueError(f"E[X^{k}] does not exist; need k < alpha.")
        return self.theta ** k * _G(self.alpha - k) / _G(self.alpha)

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
        return f"InverseGamma(alpha={self.alpha}, theta={self.theta})"


class InverseWeibull(SeverityModel):
    """Inverse Weibull (log-Gompertz) severity.

    X ~ InverseWeibull(theta, tau), support x > 0.
        F(x) = exp[-(theta/x)^tau]
        E[X^k] = theta^k Gamma(1 - k/tau),  k < tau.
    """

    def __init__(self, theta: float, tau: float):
        if theta <= 0 or tau <= 0:
            raise ValueError("theta and tau must be positive.")
        self.theta = theta
        self.tau = tau
        self._d = invweibull(c=tau, scale=theta)

    def _moment(self, k: float) -> float:
        if k >= self.tau:
            raise ValueError(f"E[X^{k}] does not exist; need k < tau.")
        return self.theta ** k * _G(1 - k / self.tau)

    def mean(self) -> float:
        if self.tau <= 1:
            raise ValueError("Mean does not exist for tau <= 1.")
        return self._moment(1)

    def variance(self) -> float:
        if self.tau <= 2:
            raise ValueError("Variance does not exist for tau <= 2.")
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
        return f"InverseWeibull(theta={self.theta}, tau={self.tau})"


class InverseExponential(SeverityModel):
    """Inverse exponential severity.

    X ~ InverseExponential(theta), support x > 0.
        F(x) = exp(-theta/x)
        E[X^k] = theta^k Gamma(1 - k),  k < 1.

    The mean (k = 1) does not exist, so :meth:`mean` and :meth:`variance` raise.
    """

    def __init__(self, theta: float):
        if theta <= 0:
            raise ValueError("theta must be positive.")
        self.theta = theta
        self._d = invweibull(c=1, scale=theta)

    def _moment(self, k: float) -> float:
        if k >= 1:
            raise ValueError(f"E[X^{k}] does not exist; need k < 1.")
        return self.theta ** k * _G(1 - k)

    def mean(self) -> float:
        raise ValueError(
            "Mean does not exist for the inverse exponential distribution."
        )

    def variance(self) -> float:
        raise ValueError(
            "Variance does not exist for the inverse exponential distribution."
        )

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
        return f"InverseExponential(theta={self.theta})"
