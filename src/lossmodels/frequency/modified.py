"""The zero-modified subclass of the (a, b, 1) frequency class.

``ZeroModified`` wraps any (a, b, 0) frequency model and places an arbitrary
probability ``p0_modified`` at zero, rescaling the rest (Loss Models B.3.2):

    p_0^M = p0_modified
    p_k^M = (1 - p0_modified) * p_k / (1 - p_0),   k = 1, 2, ...
"""

import numpy as np
from ..utils.random import RNGLike, resolve_rng

from .base import FrequencyModel
from .truncated import ZeroTruncated
from ..utils.numeric import eval_dist


class ZeroModified(FrequencyModel):
    """Zero-modified version of an (a, b, 0) frequency model.

    Parameters
    ----------
    base : FrequencyModel
        Any frequency model exposing ``pmf``, ``cdf``, ``mean``, ``variance``,
        and ``sample``.
    p0_modified : float
        The (arbitrary) probability mass placed at zero, in [0, 1). Setting it to
        0 recovers the zero-truncated distribution.
    """

    def __init__(self, base: FrequencyModel, p0_modified: float):
        for attr in ("pmf", "cdf", "mean", "variance", "sample"):
            if not hasattr(base, attr):
                raise TypeError(f"base must implement {attr}().")
        if not 0.0 <= p0_modified < 1.0:
            raise ValueError("p0_modified must be in [0, 1).")
        self.base = base
        self.p0_modified = float(p0_modified)
        self._p0 = float(base.pmf(0))
        if self._p0 >= 1.0:
            raise ValueError("base places all mass at zero; cannot modify.")

    def pmf(self, k):
        scale = (1.0 - self.p0_modified) / (1.0 - self._p0)

        def f(v):
            v = np.asarray(v, dtype=float)
            return np.where(
                v == 0,
                self.p0_modified,
                np.where(v >= 1, scale * self.base.pmf(v), 0.0),
            )

        return eval_dist(f, k)

    def cdf(self, k):
        scale = (1.0 - self.p0_modified) / (1.0 - self._p0)

        def f(v):
            v = np.asarray(v, dtype=float)
            upper = self.p0_modified + scale * (self.base.cdf(v) - self._p0)
            return np.where(v < 0, 0.0, np.where(v < 1, self.p0_modified, upper))

        return eval_dist(f, k)

    def _zt_moments(self):
        ztm = self.base.mean() / (1.0 - self._p0)
        ztv = (self.base.variance() + self.base.mean() ** 2) / (1.0 - self._p0) - ztm ** 2
        return ztm, ztv

    def mean(self) -> float:
        ztm, _ = self._zt_moments()
        return (1.0 - self.p0_modified) * ztm

    def variance(self) -> float:
        ztm, ztv = self._zt_moments()
        return (
            (1.0 - self.p0_modified) * ztv
            + self.p0_modified * (1.0 - self.p0_modified) * ztm ** 2
        )

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = None if rng is None else resolve_rng(rng)
        keep = resolve_rng(rng).random(size) >= self.p0_modified
        n_keep = int(keep.sum())
        out = np.zeros(size, dtype=int)
        if n_keep:
            out[keep] = ZeroTruncated(self.base).sample(n_keep, rng=rng)
        return out

    def __repr__(self) -> str:
        return f"ZeroModified({self.base!r}, p0_modified={self.p0_modified})"
