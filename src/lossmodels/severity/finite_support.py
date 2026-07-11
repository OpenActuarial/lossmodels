"""Continuous severities with finite support (Loss Models, Appendix A.6).

Both are supported on (0, theta) with theta a known scale parameter.

    Beta(a, b, theta)                 beta(a, b, scale=theta)
    GeneralizedBeta(a, b, theta, tau) U = (X/theta)^tau ~ Beta(a, b)
"""

from math import gamma as _G

import numpy as np
from scipy.stats import beta as beta_dist

from ..utils.numeric import eval_dist
from ..utils.random import RNGLike, scipy_random_state
from .base import SeverityModel


class Beta(SeverityModel):
    """Beta severity on (0, theta).

    X ~ Beta(a, b, theta), support 0 < x < theta.
        E[X^k] = theta^k Gamma(a+b) Gamma(a+k) / (Gamma(a) Gamma(a+b+k)),  k > -a.
        E[X] = theta a / (a + b).
    """

    def __init__(self, a: float, b: float, theta: float):
        if a <= 0 or b <= 0 or theta <= 0:
            raise ValueError("a, b, theta must all be positive.")
        self.a = a
        self.b = b
        self.theta = theta
        self._d = beta_dist(a, b, scale=theta)

    def _moment(self, k: float) -> float:
        return (
            self.theta ** k
            * _G(self.a + self.b)
            * _G(self.a + k)
            / (_G(self.a) * _G(self.a + self.b + k))
        )

    def mean(self) -> float:
        return self.theta * self.a / (self.a + self.b)

    def variance(self) -> float:
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
        return f"Beta(a={self.a}, b={self.b}, theta={self.theta})"


class GeneralizedBeta(SeverityModel):
    """Generalized beta severity on (0, theta).

    X ~ GeneralizedBeta(a, b, theta, tau), support 0 < x < theta, defined by
    U = (X/theta)^tau ~ Beta(a, b). Equivalently X = theta * U^(1/tau)::

        F(x) = Beta(a, b; (x/theta)^tau)
        E[X^k] = theta^k Gamma(a+b) Gamma(a + k/tau) / (Gamma(a) Gamma(a + b + k/tau)),
                 k > -a*tau.
    """

    def __init__(self, a: float, b: float, theta: float, tau: float):
        if a <= 0 or b <= 0 or theta <= 0 or tau <= 0:
            raise ValueError("a, b, theta, tau must all be positive.")
        self.a = a
        self.b = b
        self.theta = theta
        self.tau = tau
        self._b = beta_dist(a, b)  # standard Beta(a, b) on (0, 1)

    def _moment(self, k: float) -> float:
        return (
            self.theta ** k
            * _G(self.a + self.b)
            * _G(self.a + k / self.tau)
            / (_G(self.a) * _G(self.a + self.b + k / self.tau))
        )

    def mean(self) -> float:
        return self._moment(1)

    def variance(self) -> float:
        return self._moment(2) - self._moment(1) ** 2

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        return self.theta * self._b.rvs(size=size, random_state=scipy_random_state(rng)) ** (1.0 / self.tau)

    def pdf(self, x):
        def f(v):
            v = np.asarray(v, dtype=float)
            pos = v > 0
            safe = np.where(pos, v, 1.0)
            u = (safe / self.theta) ** self.tau
            dens = (
                self._b.pdf(u)
                * self.tau
                * safe ** (self.tau - 1)
                / self.theta ** self.tau
            )
            return np.where(pos, dens, 0.0)

        return eval_dist(f, x)

    def cdf(self, x):
        def f(v):
            v = np.asarray(v, dtype=float)
            pos = v > 0
            safe = np.where(pos, v, 1.0)
            u = np.clip((safe / self.theta) ** self.tau, 0.0, 1.0)
            return np.where(v <= 0, 0.0, self._b.cdf(u))

        return eval_dist(f, x)

    def quantile(self, p):
        return eval_dist(
            lambda v: self.theta * self._b.ppf(v) ** (1.0 / self.tau), p
        )

    def __repr__(self) -> str:
        return (
            f"GeneralizedBeta(a={self.a}, b={self.b}, theta={self.theta}, "
            f"tau={self.tau})"
        )
