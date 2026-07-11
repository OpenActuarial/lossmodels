"""Transformed Beta family of continuous severities.

Parameterizations follow Appendix A.2 of *Loss Models: From Data to Decisions*
(Klugman, Panjer & Willmot). Every distribution carries a scale parameter
``theta`` and is backed by an exactly-equivalent SciPy distribution; the moment
formulas below are the Appendix A closed forms (so ``mean``/``variance`` match
the book exactly and raise when the moment does not exist).

Parameter -> SciPy mapping (verified against the Appendix A E[X^k] formulas):

    Burr(alpha, theta, gamma)            burr12(c=gamma, d=alpha, scale=theta)
    InverseBurr(tau, theta, gamma)       burr(c=gamma, d=tau, scale=theta)
    GeneralizedPareto(alpha, theta, tau) betaprime(a=tau, b=alpha, scale=theta)
    Pareto / ParetoII(alpha, theta)      lomax(c=alpha, scale=theta)
    InversePareto(tau, theta)            betaprime(a=tau, b=1, scale=theta)
    Loglogistic(gamma, theta)            fisk(c=gamma, scale=theta)
    Paralogistic(alpha, theta)           burr12(c=alpha, d=alpha, scale=theta)
    InverseParalogistic(tau, theta)      burr(c=tau, d=tau, scale=theta)
"""

from math import gamma as _G

import numpy as np
from scipy.stats import betaprime, burr, burr12, fisk, lomax

from ..utils.numeric import eval_dist
from ..utils.random import RNGLike, scipy_random_state
from .base import SeverityModel


class Burr(SeverityModel):
    """Burr (Type XII, Singh-Maddala) severity.

    X ~ Burr(alpha, theta, gamma), support x > 0::

        F(x) = 1 - [1 / (1 + (x/theta)^gamma)]^alpha
        E[X^k] = theta^k Gamma(1 + k/gamma) Gamma(alpha - k/gamma) / Gamma(alpha),
                 for -gamma < k < alpha * gamma.
    """

    def __init__(self, alpha: float, theta: float, gamma: float):
        if alpha <= 0 or theta <= 0 or gamma <= 0:
            raise ValueError("alpha, theta, gamma must all be positive.")
        self.alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self._d = burr12(c=gamma, d=alpha, scale=theta)

    def _moment(self, k: float) -> float:
        if not -self.gamma < k < self.alpha * self.gamma:
            raise ValueError(
                f"E[X^{k}] does not exist; need -gamma < k < alpha*gamma."
            )
        return (
            self.theta ** k
            * _G(1 + k / self.gamma)
            * _G(self.alpha - k / self.gamma)
            / _G(self.alpha)
        )

    def mean(self) -> float:
        if self.alpha * self.gamma <= 1:
            raise ValueError("Mean does not exist for alpha*gamma <= 1.")
        return self._moment(1)

    def variance(self) -> float:
        if self.alpha * self.gamma <= 2:
            raise ValueError("Variance does not exist for alpha*gamma <= 2.")
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
        return f"Burr(alpha={self.alpha}, theta={self.theta}, gamma={self.gamma})"


class InverseBurr(SeverityModel):
    """Inverse Burr (Dagum) severity.

    X ~ InverseBurr(tau, theta, gamma), support x > 0::

        F(x) = [ (x/theta)^gamma / (1 + (x/theta)^gamma) ]^tau
        E[X^k] = theta^k Gamma(tau + k/gamma) Gamma(1 - k/gamma) / Gamma(tau),
                 for -tau*gamma < k < gamma.
    """

    def __init__(self, tau: float, theta: float, gamma: float):
        if tau <= 0 or theta <= 0 or gamma <= 0:
            raise ValueError("tau, theta, gamma must all be positive.")
        self.tau = tau
        self.theta = theta
        self.gamma = gamma
        self._d = burr(c=gamma, d=tau, scale=theta)

    def _moment(self, k: float) -> float:
        if not -self.tau * self.gamma < k < self.gamma:
            raise ValueError(
                f"E[X^{k}] does not exist; need -tau*gamma < k < gamma."
            )
        return (
            self.theta ** k
            * _G(self.tau + k / self.gamma)
            * _G(1 - k / self.gamma)
            / _G(self.tau)
        )

    def mean(self) -> float:
        if self.gamma <= 1:
            raise ValueError("Mean does not exist for gamma <= 1.")
        return self._moment(1)

    def variance(self) -> float:
        if self.gamma <= 2:
            raise ValueError("Variance does not exist for gamma <= 2.")
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
        return f"InverseBurr(tau={self.tau}, theta={self.theta}, gamma={self.gamma})"


class GeneralizedPareto(SeverityModel):
    """Generalized Pareto severity (Klugman transformed-beta form; NOT the EVT GPD).

    X ~ GeneralizedPareto(alpha, theta, tau), support x > 0::

        F(x) = Beta(tau, alpha; x/(x+theta))   (regularized incomplete beta)
        E[X^k] = theta^k Gamma(tau + k) Gamma(alpha - k) / (Gamma(alpha) Gamma(tau)),
                 for -tau < k < alpha.

    The extreme-value GPD (peaks-over-threshold tail) lives in ``extremeloss``;
    this is the three-parameter loss-severity distribution from Loss Models.
    """

    def __init__(self, alpha: float, theta: float, tau: float):
        if alpha <= 0 or theta <= 0 or tau <= 0:
            raise ValueError("alpha, theta, tau must all be positive.")
        self.alpha = alpha
        self.theta = theta
        self.tau = tau
        self._d = betaprime(a=tau, b=alpha, scale=theta)

    def _moment(self, k: float) -> float:
        if not -self.tau < k < self.alpha:
            raise ValueError(f"E[X^{k}] does not exist; need -tau < k < alpha.")
        return (
            self.theta ** k
            * _G(self.tau + k)
            * _G(self.alpha - k)
            / (_G(self.alpha) * _G(self.tau))
        )

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
        return (
            f"GeneralizedPareto(alpha={self.alpha}, theta={self.theta}, "
            f"tau={self.tau})"
        )


class ParetoII(SeverityModel):
    """Pareto (Type II, Lomax) severity -- the *Loss Models* two-parameter "Pareto".

    X ~ Pareto(alpha, theta), support x > 0.
        f(x) = alpha theta^alpha / (x + theta)^(alpha+1)
        F(x) = 1 - [theta / (x + theta)]^alpha
        E[X^k] = theta^k k! / [(alpha-1)...(alpha-k)]  (positive integer k < alpha)
        E[X] = theta / (alpha - 1),  alpha > 1.

    Note: this is distinct from :class:`lossmodels.Pareto`, which is the Pareto
    Type I (single-parameter) distribution with support x >= theta. Appendix A
    lists this Type II / Lomax form under the name "Pareto"; the Type I appears
    under "Single-Parameter Pareto".
    """

    def __init__(self, alpha: float, theta: float):
        if alpha <= 0 or theta <= 0:
            raise ValueError("alpha and theta must be positive.")
        self.alpha = alpha
        self.theta = theta
        self._d = lomax(c=alpha, scale=theta)

    def _moment(self, k: float) -> float:
        if not -1 < k < self.alpha:
            raise ValueError(f"E[X^{k}] does not exist; need -1 < k < alpha.")
        return self.theta ** k * _G(k + 1) * _G(self.alpha - k) / _G(self.alpha)

    def mean(self) -> float:
        if self.alpha <= 1:
            raise ValueError("Mean does not exist for alpha <= 1.")
        return self.theta / (self.alpha - 1)

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
        return f"ParetoII(alpha={self.alpha}, theta={self.theta})"


class InversePareto(SeverityModel):
    """Inverse Pareto severity.

    X ~ InversePareto(tau, theta), support x > 0.
        f(x) = tau theta x^(tau-1) / (x + theta)^(tau+1)
        F(x) = [x / (x + theta)]^tau
        E[X^k] = theta^k Gamma(tau + k) Gamma(1 - k) / Gamma(tau),  -tau < k < 1.

    The mean (k = 1) lies on the boundary of the moment range and does not exist,
    so :meth:`mean` and :meth:`variance` always raise.
    """

    def __init__(self, tau: float, theta: float):
        if tau <= 0 or theta <= 0:
            raise ValueError("tau and theta must be positive.")
        self.tau = tau
        self.theta = theta
        self._d = betaprime(a=tau, b=1, scale=theta)

    def _moment(self, k: float) -> float:
        if not -self.tau < k < 1:
            raise ValueError(f"E[X^{k}] does not exist; need -tau < k < 1.")
        return self.theta ** k * _G(self.tau + k) * _G(1 - k) / _G(self.tau)

    def mean(self) -> float:
        raise ValueError("Mean does not exist for the inverse Pareto distribution.")

    def variance(self) -> float:
        raise ValueError(
            "Variance does not exist for the inverse Pareto distribution."
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
        return f"InversePareto(tau={self.tau}, theta={self.theta})"


class Loglogistic(SeverityModel):
    """Loglogistic (Fisk) severity.

    X ~ Loglogistic(gamma, theta), support x > 0.
        F(x) = (x/theta)^gamma / (1 + (x/theta)^gamma)
        E[X^k] = theta^k Gamma(1 + k/gamma) Gamma(1 - k/gamma),  -gamma < k < gamma.
    """

    def __init__(self, gamma: float, theta: float):
        if gamma <= 0 or theta <= 0:
            raise ValueError("gamma and theta must be positive.")
        self.gamma = gamma
        self.theta = theta
        self._d = fisk(c=gamma, scale=theta)

    def _moment(self, k: float) -> float:
        if not -self.gamma < k < self.gamma:
            raise ValueError(f"E[X^{k}] does not exist; need -gamma < k < gamma.")
        return self.theta ** k * _G(1 + k / self.gamma) * _G(1 - k / self.gamma)

    def mean(self) -> float:
        if self.gamma <= 1:
            raise ValueError("Mean does not exist for gamma <= 1.")
        return self._moment(1)

    def variance(self) -> float:
        if self.gamma <= 2:
            raise ValueError("Variance does not exist for gamma <= 2.")
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
        return f"Loglogistic(gamma={self.gamma}, theta={self.theta})"


class Paralogistic(SeverityModel):
    """Paralogistic severity (a Burr distribution with gamma = alpha).

    X ~ Paralogistic(alpha, theta), support x > 0.
        E[X^k] = theta^k Gamma(1 + k/alpha) Gamma(alpha - k/alpha) / Gamma(alpha),
                 for -alpha < k < alpha^2.
    """

    def __init__(self, alpha: float, theta: float):
        if alpha <= 0 or theta <= 0:
            raise ValueError("alpha and theta must be positive.")
        self.alpha = alpha
        self.theta = theta
        self._d = burr12(c=alpha, d=alpha, scale=theta)

    def _moment(self, k: float) -> float:
        if not -self.alpha < k < self.alpha ** 2:
            raise ValueError(
                f"E[X^{k}] does not exist; need -alpha < k < alpha^2."
            )
        return (
            self.theta ** k
            * _G(1 + k / self.alpha)
            * _G(self.alpha - k / self.alpha)
            / _G(self.alpha)
        )

    def mean(self) -> float:
        if self.alpha ** 2 <= 1:
            raise ValueError("Mean does not exist for alpha^2 <= 1.")
        return self._moment(1)

    def variance(self) -> float:
        if self.alpha ** 2 <= 2:
            raise ValueError("Variance does not exist for alpha^2 <= 2.")
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
        return f"Paralogistic(alpha={self.alpha}, theta={self.theta})"


class InverseParalogistic(SeverityModel):
    """Inverse paralogistic severity (an inverse Burr with gamma = tau).

    X ~ InverseParalogistic(tau, theta), support x > 0.
        E[X^k] = theta^k Gamma(tau + k/tau) Gamma(1 - k/tau) / Gamma(tau),
                 for -tau^2 < k < tau.
    """

    def __init__(self, tau: float, theta: float):
        if tau <= 0 or theta <= 0:
            raise ValueError("tau and theta must be positive.")
        self.tau = tau
        self.theta = theta
        self._d = burr(c=tau, d=tau, scale=theta)

    def _moment(self, k: float) -> float:
        if not -self.tau ** 2 < k < self.tau:
            raise ValueError(f"E[X^{k}] does not exist; need -tau^2 < k < tau.")
        return (
            self.theta ** k
            * _G(self.tau + k / self.tau)
            * _G(1 - k / self.tau)
            / _G(self.tau)
        )

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
        return f"InverseParalogistic(tau={self.tau}, theta={self.theta})"
