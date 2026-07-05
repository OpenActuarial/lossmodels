r"""Increased limits factors and loss elimination ratios, with error bars.

Both tables are ratios of limited expected values from a fitted severity
model:

.. math::
    \mathrm{ILF}(x; b) = \frac{E[\min(X, x)]}{E[\min(X, b)]},
    \qquad
    \mathrm{LER}(d) = \frac{E[\min(X, d)]}{E[X]}.

The point values need only :meth:`~lossmodels.severity.base.SeverityModel.limited_expected_value`;
the *error bars* need the parameter covariance from
:func:`~lossmodels.estimation.uncertainty.fit_uncertainty` -- the delta
method carries fit uncertainty through to the factor actually filed, which
is where it matters: an ILF quoted to four decimals off 300 claims is
false precision, and now the table says so.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from ..estimation.uncertainty import FitUncertainty, model_parameters

__all__ = ["increased_limits_table", "loss_elimination_table"]


def _delta_se(model, uncertainty: FitUncertainty, func, step: float = 1e-5):
    """Delta-method standard error of ``func(model)`` under the fit covariance."""
    params = model_parameters(model)
    names = list(params)
    if names != list(uncertainty.param_names):
        raise ValueError(
            "uncertainty was computed for different parameters "
            f"({list(uncertainty.param_names)!r}) than this model exposes "
            f"({names!r}); pass the FitUncertainty of the same fitted model"
        )
    theta = np.array([params[n] for n in names], dtype=float)
    cls = type(model)
    grad = np.empty(len(theta))
    for i in range(len(theta)):
        # step scaled to the parameter's own magnitude: a fixed floor of 1.0
        # would make the step enormous relative to small parameters (an
        # exponential *rate* is ~1e-5) and the truncation error dominates
        h = step * abs(theta[i]) if theta[i] != 0.0 else step
        up, dn = theta.copy(), theta.copy()
        up[i] += h
        dn[i] -= h
        grad[i] = (
            func(cls(**dict(zip(names, up)))) - func(cls(**dict(zip(names, dn))))
        ) / (2.0 * h)
    var = float(grad @ np.asarray(uncertainty.covariance, dtype=float) @ grad)
    return float(np.sqrt(max(var, 0.0)))


def _with_bands(rows, values, ses, confidence_level, value_col):
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    out = pd.DataFrame(rows)
    out[f"{value_col}_se"] = ses
    out["ci_low"] = np.asarray(values) - z * np.asarray(ses)
    out["ci_high"] = np.asarray(values) + z * np.asarray(ses)
    return out


def increased_limits_table(
    model,
    limits,
    base_limit: float,
    uncertainty: FitUncertainty | None = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """ILFs at each limit relative to a base limit, optionally with bands.

    Parameters
    ----------
    model
        A fitted severity model (anything with ``limited_expected_value``).
    limits : array-like
        Policy limits to tabulate, each positive.
    base_limit : float
        The limit the factors are relative to; its own row has
        ``ilf = 1.0`` (and, when bands are requested, ``se = 0`` exactly --
        the ratio's gradient vanishes at the base).
    uncertainty : FitUncertainty, optional
        Output of :func:`fit_uncertainty` for *this* model; enables
        delta-method ``ilf_se`` / ``ci_low`` / ``ci_high`` columns.
    confidence_level : float
        Wald level for the bands.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``limit``: ``lev``, ``ilf`` (+ bands when requested).
    """
    lims = np.atleast_1d(np.asarray(limits, dtype=float))
    if np.any(lims <= 0) or base_limit <= 0:
        raise ValueError("limits and base_limit must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    base_lev = float(model.limited_expected_value(float(base_limit)))
    if base_lev <= 0:
        raise ValueError("limited expected value at the base limit is zero")
    levs = np.array(
        [float(model.limited_expected_value(float(x))) for x in lims]
    )
    rows = {"limit": lims, "lev": levs, "ilf": levs / base_lev}
    if uncertainty is None:
        return pd.DataFrame(rows).set_index("limit")
    ses = [
        _delta_se(
            model,
            uncertainty,
            lambda m, _x=float(x): m.limited_expected_value(_x)
            / m.limited_expected_value(float(base_limit)),
        )
        for x in lims
    ]
    return _with_bands(
        rows, rows["ilf"], ses, confidence_level, "ilf"
    ).set_index("limit")


def loss_elimination_table(
    model,
    deductibles,
    uncertainty: FitUncertainty | None = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Loss elimination ratios at each deductible, optionally with bands.

    ``LER(d) = E[min(X, d)] / E[X]`` -- the share of ground-up loss a
    deductible removes. Requires a finite mean: a severity with an
    infinite mean has no loss to eliminate a share *of*, and the function
    raises rather than tabulating ratios of infinity.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``deductible``: ``lev``, ``ler`` (+ bands when
        requested, as in :func:`increased_limits_table`).
    """
    ded = np.atleast_1d(np.asarray(deductibles, dtype=float))
    if np.any(ded < 0):
        raise ValueError("deductibles must be nonnegative")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    mean = float(model.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError(
            "loss elimination ratios require a positive finite mean; this "
            f"severity's mean is {mean!r}"
        )
    levs = np.array([float(model.limited_expected_value(float(d))) for d in ded])
    rows = {"deductible": ded, "lev": levs, "ler": levs / mean}
    if uncertainty is None:
        return pd.DataFrame(rows).set_index("deductible")
    ses = [
        _delta_se(
            model,
            uncertainty,
            lambda m, _d=float(d): m.limited_expected_value(_d) / m.mean(),
        )
        for d in ded
    ]
    return _with_bands(
        rows, rows["ler"], ses, confidence_level, "ler"
    ).set_index("deductible")
