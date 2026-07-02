import numpy as np
from ..utils.random import RNGLike, resolve_rng
from scipy.integrate import quad

from .base import SeverityModel


def _to_array(func, x: np.ndarray) -> np.ndarray:
    """Evaluate ``func`` on a 1D array, vectorized if possible else per-element."""
    try:
        out = np.asarray(func(x), dtype=float)
        if out.shape == x.shape:
            return out
    except Exception:
        pass
    return np.array([float(func(float(v))) for v in x], dtype=float)


class SplicedSeverity(SeverityModel):
    """Two-piece spliced severity: a body below a threshold, a tail above it.

    The density is

        f(x) = w * f_body(x) / F_body(u),    0 < x <= u
        f(x) = (1 - w) * f_tail(x),          x > u

    where ``u`` is the threshold and ``w = P(X <= u)`` is the body mass. The
    body is renormalized onto ``(0, u]`` and the tail must be a distribution
    supported on ``[u, inf)`` with ``F_tail(u) = 0`` (e.g. a GPD with location
    ``u`` from ``extremeloss``, or a :class:`~lossmodels.Pareto` with
    ``theta = u``). The CDF is

        F(x) = w * F_body(x) / F_body(u),    x <= u
        F(x) = w + (1 - w) * F_tail(x),      x > u

    which is continuous at ``u`` (both pieces equal ``w`` there). The density
    may jump at ``u``; this is the ordinary (unconstrained) spliced model, not a
    smooth splice.

    The result is an ordinary severity model -- it exposes ``sample(size)`` and
    ``mean()``, so it drops straight back into ``risksim`` and ``extremeloss``
    as a tail-corrected severity.

    Parameters
    ----------
    body : severity model
        Body distribution. Must implement ``cdf``, ``pdf``, ``quantile`` and
        ``limited_expected_value`` (every ``lossmodels`` severity qualifies).
    tail : tail distribution on ``[u, inf)``
        Must implement ``cdf`` (with ``cdf(u) == 0``), ``pdf`` and
        ``sample(size)``. ``quantile`` enables an exact spliced quantile;
        ``mean`` / ``variance`` enable the spliced moments.
    threshold : float
        Split point ``u > 0``.
    weight : float
        Body mass ``w = P(X <= u)`` in ``(0, 1)``.
    """

    def __init__(self, body, tail, threshold: float, weight: float):
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        if not 0.0 < weight < 1.0:
            raise ValueError("weight must be in (0, 1).")
        for label, obj, required in (
            ("body", body, ("cdf", "pdf", "quantile")),
            ("tail", tail, ("cdf", "pdf", "sample")),
        ):
            missing = [m for m in required if not hasattr(obj, m)]
            if missing:
                raise TypeError(f"{label} is missing required method(s): {', '.join(missing)}")

        self.body = body
        self.tail = tail
        self.threshold = float(threshold)
        self.weight = float(weight)

        self._Fb_u = float(body.cdf(self.threshold))
        if self._Fb_u <= 0.0:
            raise ValueError("body must place positive probability below the threshold.")
        tail_at_u = float(tail.cdf(self.threshold))
        if abs(tail_at_u) > 1e-6:
            raise ValueError(
                "tail must be supported on [threshold, inf) with cdf(threshold)=0; "
                f"got cdf(threshold)={tail_at_u:.3g}."
            )

    # --- distribution functions ---
    def cdf(self, x):
        arr = np.asarray(x, dtype=float)
        xx = np.atleast_1d(arr)
        below = xx <= self.threshold
        fb = _to_array(self.body.cdf, xx)
        ft = _to_array(self.tail.cdf, xx)
        out = np.where(
            below,
            self.weight * fb / self._Fb_u,
            self.weight + (1.0 - self.weight) * ft,
        )
        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)

    def pdf(self, x):
        arr = np.asarray(x, dtype=float)
        xx = np.atleast_1d(arr)
        below = xx <= self.threshold
        fb = _to_array(self.body.pdf, xx)
        ft = _to_array(self.tail.pdf, xx)
        out = np.where(
            below,
            self.weight * fb / self._Fb_u,
            (1.0 - self.weight) * ft,
        )
        return float(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)

    def quantile(self, p):
        arr = np.asarray(p, dtype=float)
        ps = np.atleast_1d(arr)
        out = np.empty_like(ps)
        lo = ps <= self.weight
        if lo.any():
            qb = self.body.quantile(ps[lo] * self._Fb_u / self.weight)
            out[lo] = np.atleast_1d(np.asarray(qb, dtype=float))
        hi = ~lo
        if hi.any():
            pt = (ps[hi] - self.weight) / (1.0 - self.weight)
            if hasattr(self.tail, "quantile"):
                out[hi] = np.atleast_1d(np.asarray(self.tail.quantile(pt), dtype=float))
            else:
                out[hi] = [self._invert_cdf(float(pp)) for pp in ps[hi]]
        return float(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)

    # --- sampling and moments ---
    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = None if rng is None else resolve_rng(rng)
        gen = resolve_rng(rng)
        n_body = int(gen.binomial(size, self.weight))
        n_tail = size - n_body
        parts = []
        if n_body > 0:
            u = gen.uniform(0.0, self._Fb_u, size=n_body)
            parts.append(np.asarray(self.body.quantile(u), dtype=float).reshape(-1))
        if n_tail > 0:
            tail_draws = (
                self.tail.sample(n_tail) if rng is None else self.tail.sample(n_tail, rng=rng)
            )
            parts.append(np.asarray(tail_draws, dtype=float).reshape(-1))
        out = np.concatenate(parts) if parts else np.empty(0)
        gen.shuffle(out)
        return out

    def mean(self) -> float:
        if not hasattr(self.tail, "mean"):
            raise TypeError("tail must implement mean() for the spliced mean.")
        u, w, fb_u = self.threshold, self.weight, self._Fb_u
        lev = self.body.limited_expected_value(u)      # E[min(body, u)]
        body_integral = lev - u * (1.0 - fb_u)         # int_0^u t f_body(t) dt
        body_term = w * body_integral / fb_u
        tail_term = (1.0 - w) * float(self.tail.mean())
        return float(body_term + tail_term)

    def variance(self) -> float:
        if not (hasattr(self.tail, "mean") and hasattr(self.tail, "variance")):
            raise TypeError("tail must implement mean() and variance() for the spliced variance.")
        u, w, fb_u = self.threshold, self.weight, self._Fb_u
        body_second, _ = quad(lambda t: t * t * self.body.pdf(t), 0.0, u, limit=200)
        e_x2_body = w * body_second / fb_u
        tail_mean = float(self.tail.mean())
        e_x2_tail = (1.0 - w) * (float(self.tail.variance()) + tail_mean ** 2)
        m = self.mean()
        return float(e_x2_body + e_x2_tail - m * m)

    def __repr__(self) -> str:
        return (
            f"SplicedSeverity(body={self.body!r}, tail={self.tail!r}, "
            f"threshold={self.threshold}, weight={self.weight})"
        )
