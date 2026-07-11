import numpy as np
from scipy.integrate import quad

from ..severity.base import SeverityModel
from ..utils.random import RNGLike, resolve_rng


class Layer(SeverityModel):
    """
    Severity model with a deductible and a payment limit applied.

    If X is the ground-up loss, the payment per loss is:
        Y = min((X - d)+, u)

    where:
    - d is the deductible / attachment point
    - u is the maximum payment in the layer

    Parameters
    ----------
    severity : SeverityModel
        Ground-up severity model.
    d : float
        Deductible / attachment point, with d >= 0.
    u : float
        Layer width (maximum payment), with u >= 0.
    """

    def __init__(self, severity: SeverityModel, d: float, u: float):
        if d < 0:
            raise ValueError("d must be nonnegative.")
        if u < 0:
            raise ValueError("u must be nonnegative.")
        self.severity = severity
        self.d = d
        self.u = u

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        """Generate random samples of payment per loss in the layer."""
        if size <= 0:
            raise ValueError("size must be positive.")
        if rng is None:
            ground_up = self.severity.sample(size=size)
        else:
            ground_up = self.severity.sample(size=size, rng=resolve_rng(rng))
        return np.minimum(np.maximum(ground_up - self.d, 0.0), self.u)

    def mean(self) -> float:
        """
        Expected payment per loss in the layer: E[min((X-d)+, u)].

        This can be written as:
            E[(X-d)+] - E[(X-d-u)+]
        """
        return self.severity.excess_loss(self.d) - self.severity.excess_loss(self.d + self.u)

    def variance(self, n_sim: int = 100_000) -> float:
        """
        Variance of payment per loss in the layer, computed deterministically.

        For Y = min((X - d)+, u), the second moment is

            E[Y^2] = integral_0^u 2 y S_X(d + y) dy

        so the variance follows from E[Y^2] - E[Y]^2 without simulation.
        The ``n_sim`` argument is retained for backward compatibility but is
        no longer used.
        """
        if self.u == 0.0:
            return 0.0
        second_moment = 2.0 * quad(
            lambda y: y * (1.0 - float(self.severity.cdf(self.d + y))),
            0.0,
            self.u,
            limit=200,
        )[0]
        m1 = self.mean()
        return float(second_moment - m1 * m1)

    def cdf(self, x: float) -> float:
        """
        CDF of the payment distribution.

        For Y = min((X - d)+, u),
            F_Y(x) = 0,            x < 0
                   = F_X(d + x),   0 <= x < u
                   = 1,            x >= u
        """
        if x < 0:
            return 0.0
        if x >= self.u:
            return 1.0
        return float(self.severity.cdf(self.d + x))

    def payment_probability(self) -> float:
        """Probability that the layer pays anything: P(X > d)."""
        return 1.0 - self.severity.cdf(self.d)

    def exhaustion_probability(self) -> float:
        """Probability that the layer is fully exhausted: P(X > d + u)."""
        return 1.0 - self.severity.cdf(self.d + self.u)

    def __repr__(self) -> str:
        return f"Layer(severity={repr(self.severity)}, d={self.d}, u={self.u})"
