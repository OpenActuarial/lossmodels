"""Random-generation plumbing shared by every sampler in lossmodels.

Every ``sample`` method accepts ``rng``:

- ``None`` (default): draws come from the legacy global ``numpy.random``
  state, bit-for-bit compatible with existing code that seeds via
  ``np.random.seed``.
- ``int``: a fresh ``np.random.default_rng(seed)`` -- reproducible.
- ``np.random.Generator``: used as-is (and advanced), so one generator can
  be threaded through a whole frequency/severity/aggregate composition.
"""
from __future__ import annotations

import numpy as np

RNGLike = np.random.Generator | int | None


def resolve_rng(rng: RNGLike):
    """Return an object exposing Generator-style distribution methods.

    ``None`` resolves to the ``numpy.random`` module itself (the legacy
    global stream), an ``int`` to ``np.random.default_rng(seed)``, and a
    ``Generator`` to itself. The module and ``Generator`` expose the
    distribution methods used in this package (``poisson``, ``gamma``,
    ``choice``, ``shuffle``, ...) under identical names and signatures.
    """
    if rng is None:
        return np.random
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


def scipy_random_state(rng: RNGLike):
    """``random_state`` argument for ``scipy.stats`` ``rvs`` calls.

    ``None`` keeps SciPy on the legacy global stream (backward compatible);
    anything else is normalized to a ``Generator``.
    """
    if rng is None:
        return None
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng
