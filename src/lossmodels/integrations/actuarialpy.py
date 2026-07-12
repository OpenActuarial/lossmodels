"""Consume the canonical ``actuarialpy.Experience`` at claim grain.

Import this module explicitly; it is the ecosystem seam and requires
``actuarialpy`` (``pip install actuarialpy`` or ``pip install openactuarial``).
The core package stays array-level and does not depend on it.
"""
from __future__ import annotations

import warnings

import numpy as np

try:
    from actuarialpy import (
        Experience,
        ExperienceSet,
        resolve_amount,
        resolve_date,
        single_role_or_none,
    )
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "this integration consumes actuarialpy.Experience; install it with "
        "pip install actuarialpy (or pip install openactuarial)"
    ) from _err


def _resolve_listing(exp):
    if isinstance(exp, ExperienceSet):
        if len(exp.listings) != 1:
            raise ValueError(
                "pass the listing member explicitly (book['claims']); the set "
                f"has {sorted(exp.listings) or 'no'} named listings"
            )
        (exp,) = exp.listings.values()
    return exp


def _claim_grain_guard(exp: Experience) -> None:
    if exp.pivots:
        raise ValueError(
            "this Experience carries recorded wide_by pivots -- it is an "
            "aggregated experience tab, and severity/tail fits need one row "
            "per claim. Bind the claims listing instead, e.g. "
            "Experience(claim_lines, expense='paid_amount', date='incurred_date')."
        )
    if exp.exposure:
        warnings.warn(
            "an exposure role is bound, which usually marks an aggregated "
            "experience tab; severity/tail fits expect claim-level amounts "
            "(one row per claim).",
            stacklevel=3,
        )


def claim_amounts(exp: Experience, *, amount_col: str | None = None) -> np.ndarray:
    """Claim-level amounts resolved from the bound expense role."""
    exp = _resolve_listing(exp)
    _claim_grain_guard(exp)
    frame, col = resolve_amount(exp, amount_col)
    values = frame[col].to_numpy(dtype=float)
    return values[~np.isnan(values)]


def frequency_counts(exp: Experience, *, freq: str = "M") -> np.ndarray:
    """Counts per period: the bound count role summed, or rows per period
    for a claims listing. Requires a bound date role."""
    import pandas as pd
    from pandas.tseries.frequencies import get_period_alias

    exp = _resolve_listing(exp)
    date_col = resolve_date(exp)
    periods = pd.to_datetime(exp.data[date_col]).dt.to_period(get_period_alias(freq) or freq)
    count_col = single_role_or_none(exp.count)
    if count_col is not None:
        counts = exp.data.groupby(periods)[count_col].sum()
    else:
        counts = exp.data.groupby(periods).size()
    return counts.sort_index().to_numpy(dtype=float)


def fit_severity_from_experience(
    exp: Experience, *, amount_col: str | None = None,
    by: str | list[str] | None = None, **kwargs,
):
    """``fit_best_severity`` over the claim amounts of a claims-listing Experience.

    ``by`` fits one severity model per group (e.g. ``by="claim_type"``) and
    returns a dict keyed by group value -- the per-service-category fit loop
    as one call. For a single group, ``exp.filter(query=...)`` first.
    """
    from lossmodels.estimation import fit_best_severity

    exp = _resolve_listing(exp)
    if by is None:
        return fit_best_severity(claim_amounts(exp, amount_col=amount_col), **kwargs)
    _claim_grain_guard(exp)
    frame, col = resolve_amount(exp, amount_col)
    fits = {}
    for key, group in frame.groupby(by):
        values = group[col].to_numpy(dtype=float)
        fits[key] = fit_best_severity(values[~np.isnan(values)], **kwargs)
    return fits


def fit_frequency_from_experience(exp: Experience, *, freq: str = "M", **kwargs):
    """``fit_best_frequency`` over per-period counts from the bound roles."""
    from lossmodels.estimation import fit_best_frequency

    return fit_best_frequency(frequency_counts(exp, freq=freq), **kwargs)
