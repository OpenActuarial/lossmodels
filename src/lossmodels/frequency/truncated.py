"""The zero-truncated members of the (a, b, 1) frequency class.

``ZeroTruncated`` is a generic wrapper: give it any (a, b, 0) frequency model
(``Poisson``, ``Geometric``, ``Binomial``, ``NegativeBinomial``) and it removes
the mass at zero and renormalizes, exactly as in Loss Models Appendix B.3.1:

    p_k^T = p_k / (1 - p_0),  k = 1, 2, ...

``Logarithmic`` is the one zero-truncated distribution with no corresponding
(a, b, 0) member, so it is implemented directly.
"""

from math import log

import numpy as np
from ..utils.random import RNGLike, resolve_rng

from .base import FrequencyModel
from ..utils.numeric import eval_dist


class ZeroTruncated(FrequencyModel):
    """Zero-truncated version of an (a, b, 0) frequency model.

    Parameters
    ----------
    base : FrequencyModel
        Any frequency model exposing ``pmf``, ``cdf``, ``mean``, ``variance``,
        and ``sample`` (e.g. ``Poisson(2.0)``). Its mass at zero is removed and
        the remaining probabilities are rescaled by ``1 / (1 - p_0)``.
    """

    def __init__(self, base: FrequencyModel):
        for attr in ("pmf", "cdf", "mean", "variance", "sample"):
            if not hasattr(base, attr):
                raise TypeError(f"base must implement {attr}().")
        self.base = base
        self._p0 = float(base.pmf(0))
        if self._p0 >= 1.0:
            raise ValueError("base places all mass at zero; cannot truncate.")

    def pmf(self, k):
        def f(v):
            v = np.asarray(v, dtype=float)
            return np.where(v >= 1, self.base.pmf(v) / (1.0 - self._p0), 0.0)

        return eval_dist(f, k)

    def cdf(self, k):
        def f(v):
            v = np.asarray(v, dtype=float)
            return np.where(
                v >= 1, (self.base.cdf(v) - self._p0) / (1.0 - self._p0), 0.0
            )

        return eval_dist(f, k)

    def mean(self) -> float:
        return self.base.mean() / (1.0 - self._p0)

    def variance(self) -> float:
        m = self.mean()
        e2 = (self.base.variance() + self.base.mean() ** 2) / (1.0 - self._p0)
        return e2 - m ** 2

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = None if rng is None else resolve_rng(rng)
        pieces = []
        n = 0
        while n < size:
            n_draw = max(2 * (size - n), 16)
            raw = self.base.sample(n_draw) if rng is None else self.base.sample(n_draw, rng=rng)
            draw = np.asarray(raw)
            nz = draw[draw > 0].astype(int)
            pieces.append(nz)
            n += nz.size
        return np.concatenate(pieces)[:size]

    def __repr__(self) -> str:
        return f"ZeroTruncated({self.base!r})"


class Logarithmic(FrequencyModel):
    """Logarithmic frequency distribution (Loss Models B.3.1.3).

    N ~ Logarithmic(beta), support {1, 2, ...}.
        p_k = (beta / (1+beta))^k / (k ln(1+beta)),  k = 1, 2, ...
        E[N] = beta / ln(1+beta)
        Var[N] = beta [1 + beta - beta/ln(1+beta)] / ln(1+beta)

    It is the r -> 0 limit of the zero-truncated negative binomial.
    """

    def __init__(self, beta: float):
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.beta = beta
        self._c = log(1.0 + beta)
        self._r = beta / (1.0 + beta)

    def pmf(self, k):
        def f(v):
            v = np.asarray(v, dtype=float)
            is_pos_int = (v >= 1) & (np.mod(v, 1) == 0)
            safe = np.where(is_pos_int, v, 1.0)
            return np.where(is_pos_int, self._r ** safe / (safe * self._c), 0.0)

        return eval_dist(f, k)

    def cdf(self, k):
        def f(v):
            arr = np.asarray(v, dtype=float)
            flat = np.atleast_1d(arr)
            idx = np.floor(flat).astype(int)
            kmax = int(idx.max()) if idx.size and idx.max() >= 1 else 0
            if kmax >= 1:
                js = np.arange(1, kmax + 1, dtype=float)
                cum = np.concatenate([[0.0], np.cumsum(self._r ** js / (js * self._c))])
            else:
                cum = np.array([0.0])
            out = np.where(idx < 1, 0.0, cum[np.clip(idx, 0, kmax)])
            return out.reshape(arr.shape)

        return eval_dist(f, k)

    def mean(self) -> float:
        return self.beta / self._c

    def variance(self) -> float:
        return self.beta * (1.0 + self.beta - self.beta / self._c) / self._c

    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        return resolve_rng(rng).logseries(self._r, size=size)

    def __repr__(self) -> str:
        return f"Logarithmic(beta={self.beta})"
