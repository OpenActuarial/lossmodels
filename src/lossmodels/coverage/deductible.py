import numpy as np
from scipy.integrate import quad

from ..severity.base import SeverityModel
from ..utils.random import RNGLike, resolve_rng


class OrdinaryDeductible(SeverityModel):
    """
    Severity model with an ordinary deductible applied.

    If X is the ground-up loss, the payment per loss is:
        Y = max(X - d, 0)

    Parameters
    ----------
    severity : SeverityModel
        Ground-up severity model.
    d : float
        Deductible amount, with d >= 0.
    """

    def __init__(self, severity: SeverityModel, d: float):
        if d < 0:
            raise ValueError("d must be nonnegative.")
        self.severity = severity
        self.d = d

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        """Generate random samples of payment per loss after deductible."""
        if size <= 0:
            raise ValueError("size must be positive.")
        if rng is None:
            ground_up = self.severity.sample(size=size)
        else:
            ground_up = self.severity.sample(size=size, rng=resolve_rng(rng))
        return np.maximum(ground_up - self.d, 0.0)

    def mean(self) -> float:
        """
        Expected payment per loss after deductible: E[(X - d)+].

        Uses the underlying severity's excess_loss method if available.
        """
        return self.severity.excess_loss(self.d)

    def variance(self, n_sim: int = 100_000) -> float:
        """
        Variance of payment per loss after deductible, computed deterministically.

        For Y = (X - d)+, the second moment is

            E[Y^2] = integral_0^inf 2 y S_X(d + y) dy

        so the variance follows from E[Y^2] - E[Y]^2 without simulation.
        The integral diverges when the underlying severity has no second
        moment, matching the behavior of ``severity.variance()``.
        The ``n_sim`` argument is retained for backward compatibility but is
        no longer used.
        """
        second_moment = 2.0 * quad(
            lambda y: y * (1.0 - float(self.severity.cdf(self.d + y))),
            0.0,
            np.inf,
            limit=200,
        )[0]
        m1 = self.mean()
        return float(second_moment - m1 * m1)

    def cdf(self, x: float) -> float:
        """
        CDF of the payment distribution.

        For Y = (X - d)+,
            F_Y(x) = 0,            x < 0
                   = F_X(d + x),   x >= 0
        """
        if x < 0:
            return 0.0
        return float(self.severity.cdf(self.d + x))

    def payment_probability(self) -> float:
        """P(X > d): probability that a payment is made."""
        return 1.0 - self.severity.cdf(self.d)

    def loss_elimination_ratio(self) -> float:
        """Loss Elimination Ratio (LER)."""
        ex = self.severity.mean()
        if ex == 0:
            return 0.0
        return (ex - self.mean()) / ex

    def __repr__(self) -> str:
        return f"OrdinaryDeductible(severity={repr(self.severity)}, d={self.d})"
