import numpy as np
import pandas as pd

from .censoring import _prepare, censored_log_likelihood, kaplan_meier


def log_likelihood(model, data, truncation=None, censored=None) -> float:
    """
    Compute the log-likelihood of observed data under a fitted model.

    Parameters
    ----------
    model : object
        Model instance with a pdf(x) method for continuous models
        or pmf(x) method for discrete models.
    data : array-like
        Observed data.
    truncation, censored : array-like, optional
        Per-observation left-truncation points and right-censoring flags.
        When either is given the individual-data likelihood of
        :func:`lossmodels.estimation.censored_log_likelihood` is used
        (requires ``model.cdf``).

    Returns
    -------
    float
        Log-likelihood value.
    """
    if truncation is not None or censored is not None:
        return censored_log_likelihood(model, data, truncation, censored)
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("data must not be empty.")

    if hasattr(model, "pdf"):
        evaluate = model.pdf
    elif hasattr(model, "pmf"):
        evaluate = model.pmf
    else:
        raise TypeError("model must implement either pdf(x) or pmf(x).")

    # Fast vectorized path; fall back to per-element for non-vectorized models.
    try:
        vals = np.asarray(evaluate(data), dtype=float)
        if vals.shape != data.shape:
            raise ValueError("non-vectorized result")
    except Exception:
        vals = np.array([evaluate(float(x)) for x in data], dtype=float)

    if np.any(~np.isfinite(vals)) or np.any(vals <= 0):
        return float(-np.inf)

    return float(np.sum(np.log(vals)))


def aic(model, data, k: int, truncation=None, censored=None) -> float:
    """
    Compute Akaike Information Criterion.

    Parameters
    ----------
    model : object
        Fitted model.
    data : array-like
        Observed data.
    k : int
        Number of estimated parameters.

    Returns
    -------
    float
        AIC value.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    ll = log_likelihood(model, data, truncation=truncation, censored=censored)
    if not np.isfinite(ll):
        return float(np.inf)

    return float(2 * k - 2 * ll)


def bic(model, data, k: int, truncation=None, censored=None) -> float:
    """
    Compute Bayesian Information Criterion.

    Parameters
    ----------
    model : object
        Fitted model.
    data : array-like
        Observed data.
    k : int
        Number of estimated parameters.

    Returns
    -------
    float
        BIC value.
    """
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("data must not be empty.")
    if k <= 0:
        raise ValueError("k must be positive.")

    ll = log_likelihood(model, data, truncation=truncation, censored=censored)
    if not np.isfinite(ll):
        return float(np.inf)

    n = data.size
    return float(np.log(n) * k - 2 * ll)

# ---------------------------------------------------------------------------
# Absolute goodness-of-fit (distance-to-empirical) statistics and tail checks.
#
# AIC/BIC above are *relative* -- they rank candidates but never say whether the
# winner is acceptable. The statistics below are *absolute*: smaller is better.
# Because parameters are estimated from the same data, none of these carry the
# standard textbook p-values -- use a parametric bootstrap if you need a formal
# test. KS is most sensitive in the body; Anderson-Darling weights the tails and
# is the better whole-distribution statistic for heavy-tailed losses; the tail
# quantile table is the most decision-relevant check for tail risk.
# ---------------------------------------------------------------------------


def _sorted_data(data) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("data must not be empty.")
    return np.sort(data)


def _eval_cdf(model, x: np.ndarray) -> np.ndarray:
    """Evaluate model.cdf on an array, falling back to per-element if needed."""
    if not hasattr(model, "cdf"):
        raise TypeError("model must implement cdf(x).")
    try:
        out = np.asarray(model.cdf(x), dtype=float)
        if out.shape != x.shape:
            raise ValueError("non-vectorized result")
        return out
    except Exception:
        return np.array([model.cdf(float(v)) for v in x], dtype=float)


def pit_values(model, data, truncation=None, censored=None) -> np.ndarray:
    r"""Probability-integral-transform values for the *uncensored* observations.

    Each uncensored ground-up value :math:`x_i` with truncation point
    :math:`t_i` maps to

    .. math:: u_i = \frac{F(x_i) - F(t_i)}{1 - F(t_i)},

    which is Uniform(0, 1) under a correctly specified model, whatever the
    (possibly heterogeneous) truncation points. Requires fully uncensored
    data: dropping censored observations would leave the remaining PIT values
    uniform only on a sub-interval of (0, 1), so for censored data use
    :func:`ks_statistic`, which compares against the Kaplan-Meier estimate
    instead.
    """
    values, trunc, cens = _prepare(data, truncation, censored)
    if np.any(cens):
        raise ValueError(
            "pit_values requires fully uncensored data: censored observations "
            "carry no exact value, and dropping them leaves the remainder "
            "uniform only on a sub-interval of (0, 1). For censored data use "
            "ks_statistic, which compares against the Kaplan-Meier estimate."
        )
    if values.size == 0:
        raise ValueError("no observations to transform.")
    f_x = _eval_cdf(model, values)
    f_t = np.where(trunc > 0, _eval_cdf(model, trunc), 0.0)
    denom = np.maximum(1.0 - f_t, 1e-300)
    u = (f_x - f_t) / denom
    return np.clip(u, 0.0, 1.0)


def ks_statistic(model, data, truncation=None, censored=None) -> float:
    """Kolmogorov-Smirnov distance ``sup_x |F_n(x) - F(x)|`` (smaller is better).

    Most sensitive in the body of the distribution. See the module note on
    estimated-parameter p-values. With ``truncation`` only, the statistic is
    computed on the PIT sample of :func:`pit_values` against the Uniform(0, 1)
    cdf -- exact under heterogeneous truncation. With censoring present, it is
    the sup-distance between the Kaplan-Meier estimate and the model cdf, both
    conditional on exceeding the smallest truncation point.
    """
    if censored is not None and np.any(np.asarray(censored, dtype=bool)):
        values, trunc, cens = _prepare(data, truncation, censored)
        times, surv = kaplan_meier(values, trunc, cens)
        t_min = float(np.min(trunc))
        f_t = float(_eval_cdf(model, np.array([t_min]))[0]) if t_min > 0 else 0.0
        denom = max(1.0 - f_t, 1e-300)
        f_model = (_eval_cdf(model, times) - f_t) / denom  # conditional on X > t_min
        f_km = 1.0 - surv
        f_km_left = np.concatenate([[0.0], f_km[:-1]])
        return float(np.max(np.maximum(np.abs(f_km - f_model), np.abs(f_km_left - f_model))))
    if truncation is not None:
        f = np.sort(pit_values(model, data, truncation, None))
        n = f.size
        d_plus = np.max(np.arange(1, n + 1) / n - f)
        d_minus = np.max(f - np.arange(0, n) / n)
        return float(max(d_plus, d_minus))
    x = _sorted_data(data)
    n = x.size
    f = _eval_cdf(model, x)
    d_plus = np.max(np.arange(1, n + 1) / n - f)
    d_minus = np.max(f - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


def anderson_darling(model, data, truncation=None, censored=None) -> float:
    """Anderson-Darling statistic A^2 (smaller is better).

    Weights the tails more than KS, so it is the better whole-distribution
    statistic for heavy-tailed loss data. With ``truncation`` the statistic
    is computed on the (exactly uniform) PIT sample of :func:`pit_values`;
    censored data are not supported -- use :func:`ks_statistic`.
    """
    if censored is not None and np.any(np.asarray(censored, dtype=bool)):
        raise ValueError(
            "anderson_darling does not support censored data; use "
            "ks_statistic, which compares against the Kaplan-Meier estimate."
        )
    if truncation is not None:
        f = np.clip(np.sort(pit_values(model, data, truncation, None)), 1e-12, 1.0 - 1e-12)
        n = f.size
        i = np.arange(1, n + 1)
        srt = np.sum((2 * i - 1) * (np.log(f) + np.log(1.0 - f[::-1])))
        return float(-n - srt / n)
    x = _sorted_data(data)
    n = x.size
    f = np.clip(_eval_cdf(model, x), 1e-12, 1.0 - 1e-12)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(f) + np.log(1.0 - f[::-1])))
    return float(-n - s / n)


def cramer_von_mises(model, data, truncation=None, censored=None) -> float:
    """Cramer-von Mises statistic W^2 (smaller is better).

    With ``truncation`` the statistic is computed on the (exactly uniform)
    PIT sample of :func:`pit_values`; censored data are not supported -- use
    :func:`ks_statistic`.
    """
    if censored is not None and np.any(np.asarray(censored, dtype=bool)):
        raise ValueError(
            "cramer_von_mises does not support censored data; use "
            "ks_statistic, which compares against the Kaplan-Meier estimate."
        )
    if truncation is not None:
        f = np.sort(pit_values(model, data, truncation, None))
        n = f.size
        i = np.arange(1, n + 1)
        return float(1.0 / (12.0 * n) + np.sum((f - (2 * i - 1) / (2.0 * n)) ** 2))
    x = _sorted_data(data)
    n = x.size
    f = _eval_cdf(model, x)
    i = np.arange(1, n + 1)
    return float(1.0 / (12.0 * n) + np.sum((f - (2 * i - 1) / (2.0 * n)) ** 2))


def tail_quantile_table(model, data, probs=(0.90, 0.95, 0.99, 0.995)) -> list:
    """Compare fitted vs empirical high quantiles -- the tail-fit check.

    Returns a list of dicts with keys ``prob``, ``empirical``, ``fitted``,
    ``abs_error``, ``rel_error``. A model can win on AIC and still miss here;
    for tail risk (VaR / TVaR / stop-loss) this is the diagnostic that matters.
    Requires the model to implement ``quantile(p)``.
    """
    if not hasattr(model, "quantile"):
        raise TypeError("model must implement quantile(p) for the tail table.")
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("data must not be empty.")
    rows = []
    for p in probs:
        # Inverted-CDF quantile: the same estimator the ecosystem's var/tvar
        # use, so the tail diagnostic is measured against what will actually
        # be computed downstream.
        emp = float(np.quantile(data, p, method="inverted_cdf"))
        fit = float(model.quantile(p))
        rows.append({
            "prob": float(p),
            "empirical": emp,
            "fitted": fit,
            "abs_error": fit - emp,
            "rel_error": (fit - emp) / emp if emp != 0 else float("nan"),
        })
    return rows


def goodness_of_fit(model, data, k: int, truncation=None, censored=None) -> dict:
    """One-call fit report combining relative and absolute measures.

    Returns ``n``, ``log_likelihood``, ``aic``, ``bic`` (relative -- compare
    across candidates) and ``ks``, ``anderson_darling``, ``cramer_von_mises``
    (absolute distance-to-empirical -- smaller is better). See the module note
    on the estimated-parameter caveat for the distance statistics.

    With ``truncation``/``censored``, the likelihood-based measures use the
    individual-data likelihood; ``ks`` compares against the PIT sample
    (truncation only) or the Kaplan-Meier estimate (censoring present), and
    ``anderson_darling``/``cramer_von_mises`` are reported as NaN under
    censoring, where they are not defined here. ``n_uncensored`` is added to
    the report.
    """
    has_censoring = censored is not None and np.any(np.asarray(censored, dtype=bool))
    out = {
        "n": int(np.asarray(data).size),
        "log_likelihood": log_likelihood(model, data, truncation=truncation, censored=censored),
        "aic": aic(model, data, k=k, truncation=truncation, censored=censored),
        "bic": bic(model, data, k=k, truncation=truncation, censored=censored),
        "ks": ks_statistic(model, data, truncation=truncation, censored=censored),
        "anderson_darling": (
            float("nan") if has_censoring
            else anderson_darling(model, data, truncation=truncation, censored=censored)
        ),
        "cramer_von_mises": (
            float("nan") if has_censoring
            else cramer_von_mises(model, data, truncation=truncation, censored=censored)
        ),
    }
    if truncation is not None or censored is not None:
        cens = np.zeros(int(np.asarray(data).size), dtype=bool) if censored is None \
            else np.broadcast_to(np.asarray(censored, dtype=bool), np.asarray(data).shape)
        out["n_uncensored"] = int(np.sum(~cens))
    return out


def compare_fits(models, data, truncation=None, censored=None) -> "pd.DataFrame":
    """Side-by-side scorecard for fitted models on one dataset.

    The companion to :func:`fit_best_severity`: rather than returning a
    single winner by one criterion, every candidate is scored on every
    criterion so the trade-offs are visible -- a model can win AIC while
    losing the tail (Anderson-Darling weights the tails; Kolmogorov-Smirnov
    the body).

    Parameters
    ----------
    models : mapping or sequence
        ``name -> fitted model``, or a sequence (named by class, deduped).
    data : array-like
        Observations; the same data every model is scored on.
    truncation, censored : optional
        Passed through to every statistic, as in :func:`log_likelihood`.

    Returns
    -------
    pandas.DataFrame
        One row per model: ``n_params``, ``loglik``, ``aic``, ``bic``,
        ``ks``, ``ad``, ``cvm``. Lower is better for everything except
        ``loglik``.
    """
    from .uncertainty import model_parameters

    if hasattr(models, "items"):
        named = list(models.items())
    else:
        named, seen = [], {}
        for m in models:
            base = type(m).__name__
            seen[base] = seen.get(base, 0) + 1
            named.append((base if seen[base] == 1 else f"{base}_{seen[base]}", m))
    if not named:
        raise ValueError("no models given")

    rows = []
    for _, m in named:
        k = len(model_parameters(m))
        rows.append(
            {
                "n_params": k,
                "loglik": log_likelihood(m, data, truncation=truncation, censored=censored),
                "aic": aic(m, data, k, truncation=truncation, censored=censored),
                "bic": bic(m, data, k, truncation=truncation, censored=censored),
                "ks": ks_statistic(m, data, truncation=truncation, censored=censored),
                "ad": anderson_darling(m, data, truncation=truncation, censored=censored),
                "cvm": cramer_von_mises(m, data, truncation=truncation, censored=censored),
            }
        )
    return pd.DataFrame(rows, index=pd.Index([n for n, _ in named], name="model"))
