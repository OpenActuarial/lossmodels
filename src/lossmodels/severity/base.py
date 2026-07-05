import numpy as np
from ..utils.random import RNGLike

from abc import ABC, abstractmethod
from scipy.integrate import quad
from scipy.optimize import brentq


class SeverityModel(ABC):
    """
    Base class for severity (loss size) distributions.

    All severity models must implement:

    - sample
    - mean
    - variance
    """

    @abstractmethod
    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        """Generate random loss samples"""
        pass

    @abstractmethod
    def mean(self) -> float:
        """Expected loss"""
        pass

    @abstractmethod
    def variance(self) -> float:
        """Variance of loss"""
        pass

    def std(self) -> float:
        return np.sqrt(self.variance())

    # --- Tail interface (the cross-package severity protocol) ---
    def sf(self, x):
        """Survival function ``P(X > x)``; the complement of :meth:`cdf`.

        Together with :meth:`mean_excess` this is the small protocol that
        downstream consumers (e.g. ``ratingmodels`` pooling charges) rely
        on -- any object with these two methods qualifies.
        """
        return 1.0 - self.cdf(x)

    def mean_excess(self, d):
        r"""Mean excess ``E[X - d | X > d]`` via the limited-expected-value identity.

        Uses :math:`E[(X-d)_+] = E[X] - E[\min(X, d)]` divided by ``S(d)`` --
        finite-interval integration only (and closed-form
        :meth:`limited_expected_value` where a subclass provides one), which
        is numerically far better behaved for heavy tails than integrating
        the survival function to infinity. An infinite mean means an
        infinite mean excess, returned as ``inf``; where ``S(d) = 0`` the
        conditional expectation is undefined and ``nan`` is returned.
        Subclasses with a fully closed form may still override.
        """
        arr = np.asarray(d, dtype=float)
        try:
            mean = float(self.mean())
        except (ValueError, OverflowError):
            mean = float("inf")  # models that raise where the mean diverges
        out = np.empty(np.atleast_1d(arr).shape, dtype=float)
        for i, di in enumerate(np.atleast_1d(arr)):
            surv = float(self.sf(float(di)))
            if surv <= 0.0:
                out[i] = np.nan
            elif not np.isfinite(mean):
                out[i] = np.inf
            else:
                out[i] = (mean - float(self.limited_expected_value(float(di)))) / surv
        return float(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)

    # --- Quantile / inverse CDF ---
    def quantile(self, p):
        """Inverse CDF (quantile / Value-at-Risk).

        Numerically inverts ``cdf``. Subclasses with a closed-form inverse
        override this for speed and accuracy. Returns a Python ``float`` for a
        scalar ``p`` and a ``numpy.ndarray`` for array-like ``p``.
        """
        arr = np.asarray(p, dtype=float)
        out = np.array([self._invert_cdf(float(pi)) for pi in np.atleast_1d(arr)], dtype=float)
        return float(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)

    def ppf(self, p):
        """Alias for :meth:`quantile` (inverse CDF)."""
        return self.quantile(p)

    def _invert_cdf(self, p: float) -> float:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        if p == 0.0:
            return 0.0
        if p == 1.0:
            return float("inf")
        if not hasattr(self, "cdf"):
            raise TypeError("Severity model must implement cdf(x) to invert.")
        hi = 1.0
        while self.cdf(hi) < p:
            hi *= 2.0
            if hi > 1e15:
                break
        return float(brentq(lambda x: self.cdf(x) - p, 0.0, hi, maxiter=200))

    # --- Actuarial-specific methods ---
    def limited_expected_value(self, d: float, n_sim: int = 100_000) -> float:
        """
        E[min(X, d)] computed deterministically from the survival function.

        For a nonnegative loss random variable X,

            E[min(X, d)] = integral_0^d S_X(x) dx

        where S_X(x) = 1 - F_X(x).

        The ``n_sim`` argument is retained for backward compatibility but is no
        longer used.
        """
        del n_sim  # backward-compatibility placeholder

        if d < 0:
            raise ValueError("d must be nonnegative.")
        if d == 0:
            return 0.0
        if not hasattr(self, "cdf"):
            raise TypeError("Severity model must implement cdf(x).")

        value, _ = quad(lambda x: 1.0 - self.cdf(x), 0.0, d, limit=200)
        return float(value)

    def excess_loss(self, d: float, n_sim: int = 100_000) -> float:
        """
        E[(X - d)+] computed deterministically from the survival function.

        For a nonnegative loss random variable X,

            E[(X - d)+] = integral_d^infinity S_X(x) dx

        where S_X(x) = 1 - F_X(x).

        This remains well defined even when E[X] does not exist, provided the
        tail integral from d to infinity converges. The ``n_sim`` argument is
        retained for backward compatibility but is no longer used.
        """
        del n_sim  # backward-compatibility placeholder

        if d < 0:
            raise ValueError("d must be nonnegative.")
        if not hasattr(self, "cdf"):
            raise TypeError("Severity model must implement cdf(x).")

        value, _ = quad(lambda x: 1.0 - self.cdf(x), d, np.inf, limit=200)
        return float(value)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
