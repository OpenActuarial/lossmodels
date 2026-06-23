"""Numeric helpers shared across distribution models."""

from __future__ import annotations

import numpy as np


def eval_dist(func, x):
    """Evaluate a vectorized distribution function on scalar or array input.

    Applies ``func`` (typically a vectorized SciPy method such as
    ``lognorm.pdf``) to ``x`` and returns a Python ``float`` when ``x`` is a
    scalar and a ``numpy.ndarray`` when ``x`` is array-like. This lets the
    distribution methods accept either a single value or a whole vector while
    preserving their original scalar-in / scalar-out behavior (which the
    ``scipy.integrate.quad``-based actuarial methods rely on).
    """
    arr = np.asarray(x, dtype=float)
    out = func(arr)
    if arr.ndim == 0:
        return float(out)
    return np.asarray(out, dtype=float)
