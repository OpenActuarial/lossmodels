import numpy as np

from ..frequency.base import FrequencyModel
from ..severity.base import SeverityModel
from ..utils.random import RNGLike, resolve_rng
from .base import AggregateModel


class CollectiveRiskModel(AggregateModel):
    """
    Collective risk model for aggregate loss:

        S = X1 + X2 + ... + XN

    where:
    - N is the claim count random variable (frequency)
    - Xi are iid claim severities

    Assumes:
    - severities are iid
    - N is independent of severities
    """

    def __init__(self, frequency: FrequencyModel, severity: SeverityModel):
        self.frequency = frequency
        self.severity = severity

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        """
        Generate random samples of aggregate loss.

        ``rng`` may be ``None`` (legacy global ``numpy.random`` state), an
        ``int`` seed, or a ``numpy.random.Generator``; a seed or generator is
        threaded through both the frequency and severity draws so the whole
        aggregate simulation is reproducible.

        Vectorized: all claim counts are drawn at once, then ``counts.sum()``
        severities in a single call, and each simulation's aggregate is a
        segment sum. Because a ``numpy`` generator is a sequential stream,
        drawing the severities in one call yields the *same* sequence as the
        former per-simulation loop, so results are unchanged for a given seed --
        only much faster for large ``size``.
        """
        if size <= 0:
            raise ValueError("size must be positive")

        rng = None if rng is None else resolve_rng(rng)
        counts = (
            self.frequency.sample(size=size)
            if rng is None
            else self.frequency.sample(size=size, rng=rng)
        )
        totals = np.asarray(counts).astype(int)
        total = int(totals.sum())
        aggregate_losses = np.zeros(size, dtype=float)
        if total > 0:
            draws = (
                self.severity.sample(size=total)
                if rng is None
                else self.severity.sample(size=total, rng=rng)
            )
            draws = np.asarray(draws, dtype=float)
            # map each drawn severity to its simulation, then sum per simulation.
            # Simulations with zero claims contribute no draws and stay 0.
            sim_index = np.repeat(np.arange(size), totals)
            aggregate_losses = np.bincount(sim_index, weights=draws, minlength=size).astype(float)

        return aggregate_losses

    def mean(self) -> float:
        """
        E[S] = E[N] * E[X]
        """
        return self.frequency.mean() * self.severity.mean()

    def variance(self) -> float:
        """
        Var(S) = E[N] Var(X) + Var(N) (E[X])^2
        """
        en = self.frequency.mean()
        vn = self.frequency.variance()
        ex = self.severity.mean()
        vx = self.severity.variance()

        return en * vx + vn * (ex ** 2)

    def frequency_mean(self) -> float:
        return self.frequency.mean()

    def severity_mean(self) -> float:
        return self.severity.mean()

    def summary(self) -> dict:
        """
        Return a small summary of the model.
        """
        return {
            "frequency_model": repr(self.frequency),
            "severity_model": repr(self.severity),
            "frequency_mean": self.frequency.mean(),
            "severity_mean": self.severity.mean(),
            "aggregate_mean": self.mean(),
            "aggregate_variance": self.variance(),
            "aggregate_std": self.std(),
        }

    def __repr__(self):
        return (
            f"CollectiveRiskModel("
            f"frequency={repr(self.frequency)}, "
            f"severity={repr(self.severity)})"
        )
