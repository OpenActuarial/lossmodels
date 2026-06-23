import numpy as np


def log_likelihood(model, data) -> float:
    """
    Compute the log-likelihood of observed data under a fitted model.

    Parameters
    ----------
    model : object
        Model instance with a pdf(x) method for continuous models
        or pmf(x) method for discrete models.
    data : array-like
        Observed data.

    Returns
    -------
    float
        Log-likelihood value.
    """
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


def aic(model, data, k: int) -> float:
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

    ll = log_likelihood(model, data)
    if not np.isfinite(ll):
        return float(np.inf)

    return float(2 * k - 2 * ll)


def bic(model, data, k: int) -> float:
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

    ll = log_likelihood(model, data)
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


def ks_statistic(model, data) -> float:
    """Kolmogorov-Smirnov distance sup_x |F_n(x) - F(x)| (smaller is better).

    Most sensitive in the body of the distribution. See the module note on
    estimated-parameter p-values.
    """
    x = _sorted_data(data)
    n = x.size
    f = _eval_cdf(model, x)
    d_plus = np.max(np.arange(1, n + 1) / n - f)
    d_minus = np.max(f - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


def anderson_darling(model, data) -> float:
    """Anderson-Darling statistic A^2 (smaller is better).

    Weights the tails more than KS, so it is the better whole-distribution
    statistic for heavy-tailed loss data.
    """
    x = _sorted_data(data)
    n = x.size
    f = np.clip(_eval_cdf(model, x), 1e-12, 1.0 - 1e-12)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(f) + np.log(1.0 - f[::-1])))
    return float(-n - s / n)


def cramer_von_mises(model, data) -> float:
    """Cramer-von Mises statistic W^2 (smaller is better)."""
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
        emp = float(np.quantile(data, p))
        fit = float(model.quantile(p))
        rows.append({
            "prob": float(p),
            "empirical": emp,
            "fitted": fit,
            "abs_error": fit - emp,
            "rel_error": (fit - emp) / emp if emp != 0 else float("nan"),
        })
    return rows


def goodness_of_fit(model, data, k: int) -> dict:
    """One-call fit report combining relative and absolute measures.

    Returns ``n``, ``log_likelihood``, ``aic``, ``bic`` (relative -- compare
    across candidates) and ``ks``, ``anderson_darling``, ``cramer_von_mises``
    (absolute distance-to-empirical -- smaller is better). See the module note
    on the estimated-parameter caveat for the distance statistics.
    """
    return {
        "n": int(np.asarray(data).size),
        "log_likelihood": log_likelihood(model, data),
        "aic": aic(model, data, k=k),
        "bic": bic(model, data, k=k),
        "ks": ks_statistic(model, data),
        "anderson_darling": anderson_darling(model, data),
        "cramer_von_mises": cramer_von_mises(model, data),
    }
