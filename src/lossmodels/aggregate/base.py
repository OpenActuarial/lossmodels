from abc import ABC, abstractmethod

import numpy as np

from ..utils.random import RNGLike, resolve_rng
from .risk_measures import tvar as empirical_tvar
from .risk_measures import var as empirical_var


class AggregateModel(ABC):
    """
    Base class for aggregate loss models.

    All aggregate models must implement:
    - sample
    - mean
    - variance
    """

    @abstractmethod
    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        """Generate random samples of aggregate loss."""
        pass

    @abstractmethod
    def mean(self) -> float:
        """Expected aggregate loss."""
        pass

    @abstractmethod
    def variance(self) -> float:
        """Variance of aggregate loss."""
        pass

    def std(self) -> float:
        """Standard deviation of aggregate loss."""
        return np.sqrt(self.variance())

    def var(self, q: float, n_sim: int = 100_000, rng: RNGLike = None) -> float:
        """
        Value-at-Risk at probability level q using simulation.

        Pass ``rng`` (seed or ``numpy.random.Generator``) for a reproducible
        estimate; ``None`` uses the legacy global ``numpy.random`` state.
        """
        if not (0 < q < 1):
            raise ValueError("q must be between 0 and 1")
        samples = self.sample(n_sim) if rng is None else self.sample(n_sim, rng=resolve_rng(rng))
        return empirical_var(samples, q)

    def tvar(self, q: float, n_sim: int = 100_000, rng: RNGLike = None) -> float:
        """
        Tail Value-at-Risk at probability level q using simulation.

        Pass ``rng`` (seed or ``numpy.random.Generator``) for a reproducible
        estimate; ``None`` uses the legacy global ``numpy.random`` state.
        """
        if not (0 < q < 1):
            raise ValueError("q must be between 0 and 1")
        samples = self.sample(n_sim) if rng is None else self.sample(n_sim, rng=resolve_rng(rng))
        return empirical_tvar(samples, q)

    def stop_loss(self, d: float, n_sim: int = 100_000, rng: RNGLike = None) -> float:
        """
        Expected stop-loss premium E[(S - d)+] using simulation.

        Pass ``rng`` (seed or ``numpy.random.Generator``) for a reproducible
        estimate; ``None`` uses the legacy global ``numpy.random`` state.
        """
        if d < 0:
            raise ValueError("d must be nonnegative")
        samples = self.sample(n_sim) if rng is None else self.sample(n_sim, rng=resolve_rng(rng))
        return float(np.mean(np.maximum(samples - d, 0.0)))

    def limited_expected_value(self, d: float, n_sim: int = 100_000, rng: RNGLike = None) -> float:
        """
        Limited expected value E[min(S, d)] using simulation.

        Pass ``rng`` (seed or ``numpy.random.Generator``) for a reproducible
        estimate; ``None`` uses the legacy global ``numpy.random`` state.
        """
        if d < 0:
            raise ValueError("d must be nonnegative")
        samples = self.sample(n_sim) if rng is None else self.sample(n_sim, rng=resolve_rng(rng))
        return float(np.mean(np.minimum(samples, d)))

    def __repr__(self):
        return f"{self.__class__.__name__}()"
