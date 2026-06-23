import numpy as np

from ..frequency.base import FrequencyModel
from ..severity.base import SeverityModel


class EmpiricalSeverity(SeverityModel):
    """
    Empirical severity model based on observed loss data.

    Parameters
    ----------
    data : array-like
        Observed severity values. Must be nonempty and nonnegative.
    """

    def __init__(self, data):
        data = np.asarray(data, dtype=float)

        if data.size == 0:
            raise ValueError("data must not be empty.")
        if np.any(data < 0):
            raise ValueError("severity data must be nonnegative.")

        self.data = data

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate bootstrap samples from the empirical severity distribution.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        return np.random.choice(self.data, size=size, replace=True)

    def mean(self) -> float:
        return float(np.mean(self.data))

    def variance(self) -> float:
        return float(np.var(self.data, ddof=0))

    def pdf(self, x: float) -> float:
        """
        Empirical severity is discrete, so this returns the empirical point mass at x.
        For continuous-looking data, this will often be 0 except at exact observed values.
        """
        if x < 0:
            return 0.0

        return float(np.mean(self.data == x))

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0

        return float(np.mean(self.data <= x))

    def quantile(self, p):
        """Empirical quantile (inverse CDF) from the observed data."""
        q = np.quantile(self.data, p)
        return float(q) if np.ndim(p) == 0 else np.asarray(q, dtype=float)

    def excess_loss(self, d: float) -> float:
        """
        E[(X - d)+] computed empirically.
        """
        if d < 0:
            raise ValueError("d must be nonnegative.")

        return float(np.mean(np.maximum(self.data - d, 0.0)))

    def limited_expected_value(self, d: float) -> float:
        """
        E[min(X, d)] computed empirically.
        """
        if d < 0:
            raise ValueError("d must be nonnegative.")

        return float(np.mean(np.minimum(self.data, d)))

    def __repr__(self) -> str:
        return f"EmpiricalSeverity(n={len(self.data)})"


class EmpiricalFrequency(FrequencyModel):
    """
    Empirical frequency model based on observed claim count data.

    Parameters
    ----------
    data : array-like
        Observed claim counts. Must be nonempty, nonnegative, and integer-valued.
    """

    def __init__(self, data):
        data = np.asarray(data)

        if data.size == 0:
            raise ValueError("data must not be empty.")
        if np.any(data < 0):
            raise ValueError("frequency data must be nonnegative.")
        if not np.all(np.equal(data, np.floor(data))):
            raise ValueError("frequency data must be integer-valued.")

        self.data = data.astype(int)

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate bootstrap samples from the empirical frequency distribution.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        return np.random.choice(self.data, size=size, replace=True)

    def mean(self) -> float:
        return float(np.mean(self.data))

    def variance(self) -> float:
        return float(np.var(self.data, ddof=0))

    def pmf(self, k: int) -> float:
        if k < 0:
            return 0.0

        return float(np.mean(self.data == k))

    def cdf(self, k: int) -> float:
        if k < 0:
            return 0.0

        return float(np.mean(self.data <= k))

    def __repr__(self) -> str:
        return f"EmpiricalFrequency(n={len(self.data)})"