import numpy as np
from scipy.integrate import quad

from ..utils.random import RNGLike, resolve_rng

from ..severity.base import SeverityModel


class PolicyLimit(SeverityModel):
    """
    Severity model with a policy limit applied.

    If X is the ground-up loss, the payment per loss is:
        Y = min(X, u)

    Parameters
    ----------
    severity : SeverityModel
        Ground-up severity model.
    u : float
        Policy limit, with u >= 0.
    """

    def __init__(self, severity: SeverityModel, u: float):
        if u < 0:
            raise ValueError("u must be nonnegative.")
        self.severity = severity
        self.u = u

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        """Generate random samples of payment per loss after policy limit."""
        if size <= 0:
            raise ValueError("size must be positive.")
        if rng is None:
            ground_up = self.severity.sample(size=size)
        else:
            ground_up = self.severity.sample(size=size, rng=resolve_rng(rng))
        return np.minimum(ground_up, self.u)

    def mean(self) -> float:
        """
        Expected payment per loss after policy limit: E[min(X, u)].

        Uses the underlying severity's limited_expected_value method.
        """
        return self.severity.limited_expected_value(self.u)

    def variance(self, n_sim: int = 100_000) -> float:
        """
        Variance of payment per loss after policy limit, computed deterministically.

        For Y = min(X, u), the second moment is

            E[Y^2] = integral_0^u 2 y S_X(y) dy

        so the variance follows from E[Y^2] - E[Y]^2 without simulation.
        The ``n_sim`` argument is retained for backward compatibility but is
        no longer used.
        """
        if self.u == 0.0:
            return 0.0
        second_moment = 2.0 * quad(
            lambda y: y * (1.0 - float(self.severity.cdf(y))),
            0.0,
            self.u,
            limit=200,
        )[0]
        m1 = self.mean()
        return float(second_moment - m1 * m1)

    def cdf(self, x: float) -> float:
        """
        CDF of the payment distribution.

        For Y = min(X, u),
            F_Y(x) = 0,       x < 0
                   = F_X(x),  0 <= x < u
                   = 1,       x >= u
        """
        if x < 0:
            return 0.0
        if x >= self.u:
            return 1.0
        return float(self.severity.cdf(x))

    def probability_capped(self) -> float:
        """P(X > u): probability that the limit is binding."""
        return 1.0 - self.severity.cdf(self.u)

    def loss_elimination_ratio(self) -> float:
        """Loss Elimination Ratio (LER)."""
        ex = self.severity.mean()
        if ex == 0:
            return 0.0
        return (ex - self.mean()) / ex

    def __repr__(self) -> str:
        return f"PolicyLimit(severity={repr(self.severity)}, u={self.u})"
