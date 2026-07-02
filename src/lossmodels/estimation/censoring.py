r"""Fitting under left truncation and right censoring (deductibles and limits).

Real claim data rarely arrives ground-up and complete. A deductible ``d``
left-truncates: losses below ``d`` never enter the data, and each observed
loss is a draw from ``X | X > d``. A policy limit right-censors: when the
payment hits its maximum we know only that the ground-up loss was *at least*
the censoring point.

For observation :math:`i` with ground-up value :math:`x_i`, truncation point
:math:`t_i` (0 when none) and censoring indicator :math:`\delta_i`
(:math:`\delta_i = 0` for a censored observation, where :math:`x_i` is the
censoring point), the individual-data log-likelihood (Loss Models, ch. 11) is

.. math::
    \ell(\theta) \;=\; \sum_i \Big[\, \delta_i \log f(x_i)
        + (1 - \delta_i) \log S(x_i) - \log S(t_i) \,\Big],

where :math:`S = 1 - F` is the survival function. Fitting a complete-data
likelihood to truncated or censored values is biased -- often badly -- which
is exactly why this module exists.

:func:`payments_to_ground_up` converts the common per-payment layout
(payments net of a deductible, capped at a maximum payment) into the
``(values, truncation, censored)`` triple every fitter in
:mod:`lossmodels.estimation` accepts.
"""
import numpy as np
from scipy.optimize import minimize

_TINY = 1e-300


def _prepare(values, truncation=None, censored=None):
    """Validate and broadcast (values, truncation, censored).

    Returns
    -------
    tuple of np.ndarray
        ``(values, truncation, censored)`` with ``values > 0`` ground-up
        amounts, ``truncation >= 0`` per-observation left-truncation points
        (0 = untruncated), and boolean ``censored`` flags (True = the value is
        a right-censoring point, i.e. a lower bound on the loss).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty 1D array.")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("values must be finite and strictly positive.")

    if truncation is None:
        truncation = np.zeros_like(values)
    else:
        truncation = np.broadcast_to(
            np.asarray(truncation, dtype=float), values.shape
        ).copy()
        if np.any(~np.isfinite(truncation)) or np.any(truncation < 0):
            raise ValueError("truncation must be finite and nonnegative.")

    if censored is None:
        censored = np.zeros(values.shape, dtype=bool)
    else:
        censored = np.broadcast_to(np.asarray(censored, dtype=bool), values.shape).copy()

    if np.any(values < truncation):
        raise ValueError(
            "each value must be at least its truncation point (values are "
            "ground-up amounts observed only when the loss exceeds the "
            "deductible)."
        )
    if np.any(~censored & (values <= truncation)):
        raise ValueError(
            "uncensored values must strictly exceed their truncation point."
        )
    return values, truncation, censored


def payments_to_ground_up(payments, deductible=0.0, max_payment=None, rtol=1e-9):
    """Convert per-payment data to ``(values, truncation, censored)``.

    The per-payment convention: with deductible ``d`` and maximum payment
    ``m`` (the policy limit net of the deductible), the recorded payment is
    ``Y = min(X - d, m)`` for a ground-up loss ``X > d``.

    Parameters
    ----------
    payments : array-like
        Positive payment amounts.
    deductible : float or array-like
        Deductible per observation (scalar broadcasts). Default 0.
    max_payment : float or array-like, optional
        Maximum payment per observation. Payments within relative tolerance
        ``rtol`` of the maximum are flagged as right-censored.
    rtol : float
        Relative tolerance for detecting capped payments.

    Returns
    -------
    tuple of np.ndarray
        ``(values, truncation, censored)`` on the ground-up scale, ready for
        any fitter in :mod:`lossmodels.estimation`.
    """
    payments = np.asarray(payments, dtype=float)
    if payments.ndim != 1 or payments.size == 0:
        raise ValueError("payments must be a nonempty 1D array.")
    if np.any(~np.isfinite(payments)) or np.any(payments <= 0):
        raise ValueError("payments must be finite and strictly positive.")

    deductible = np.broadcast_to(np.asarray(deductible, dtype=float), payments.shape)
    if np.any(deductible < 0):
        raise ValueError("deductible must be nonnegative.")

    values = payments + deductible
    truncation = deductible.copy()

    if max_payment is None:
        censored = np.zeros(payments.shape, dtype=bool)
    else:
        max_payment = np.broadcast_to(
            np.asarray(max_payment, dtype=float), payments.shape
        )
        if np.any(max_payment <= 0):
            raise ValueError("max_payment must be positive.")
        if np.any(payments > max_payment * (1.0 + rtol)):
            raise ValueError("payments cannot exceed max_payment.")
        censored = payments >= max_payment * (1.0 - rtol)

    return values, truncation, censored


def _survival(model, x):
    """S(x) = 1 - F(x), vectorized with a per-element fallback."""
    x = np.asarray(x, dtype=float)
    try:
        f = np.asarray(model.cdf(x), dtype=float)
        if f.shape != x.shape:
            raise ValueError("non-vectorized result")
    except Exception:
        f = np.array([model.cdf(float(v)) for v in x], dtype=float)
    return 1.0 - f


def _density(model, x):
    x = np.asarray(x, dtype=float)
    try:
        f = np.asarray(model.pdf(x), dtype=float)
        if f.shape != x.shape:
            raise ValueError("non-vectorized result")
    except Exception:
        f = np.array([model.pdf(float(v)) for v in x], dtype=float)
    return f


def censored_log_likelihood(model, values, truncation=None, censored=None) -> float:
    r"""Log-likelihood under left truncation and right censoring.

    Implements :math:`\sum_i [\delta_i \log f(x_i) + (1-\delta_i)\log S(x_i)
    - \log S(t_i)]`. With no truncation and no censoring this reduces to the
    ordinary log-likelihood. Requires ``model.pdf`` and ``model.cdf``.

    Returns ``-inf`` when the model assigns nonpositive density or survival
    where the data require it (e.g. a support violation).
    """
    values, truncation, censored = _prepare(values, truncation, censored)
    if not hasattr(model, "pdf") or not hasattr(model, "cdf"):
        raise TypeError("model must implement pdf(x) and cdf(x).")

    ll = 0.0
    unc = ~censored
    if np.any(unc):
        f = _density(model, values[unc])
        if np.any(~np.isfinite(f)) or np.any(f <= 0):
            return float(-np.inf)
        ll += float(np.sum(np.log(np.maximum(f, _TINY))))
    if np.any(censored):
        s = _survival(model, values[censored])
        if np.any(~np.isfinite(s)) or np.any(s <= 0):
            return float(-np.inf)
        ll += float(np.sum(np.log(np.maximum(s, _TINY))))

    positive_t = truncation > 0
    if np.any(positive_t):
        s_t = _survival(model, truncation[positive_t])
        if np.any(~np.isfinite(s_t)) or np.any(s_t <= 0):
            return float(-np.inf)
        ll -= float(np.sum(np.log(np.maximum(s_t, _TINY))))
    return float(ll)


def fit_mle_censored(
    model_class,
    values,
    initial_params,
    bounds=None,
    truncation=None,
    censored=None,
):
    """Generic numerical MLE under left truncation and right censoring.

    The censored-and-truncated analogue of
    :func:`lossmodels.estimation.fit_mle`: ``model_class(*params)`` must
    provide ``pdf`` and ``cdf``. Uses L-BFGS-B (with ``bounds``) refined by
    Nelder-Mead if the gradient-based step fails to improve.

    Returns
    -------
    object
        Fitted ``model_class`` instance.
    """
    values, truncation, censored = _prepare(values, truncation, censored)
    initial_params = np.asarray(initial_params, dtype=float)
    if initial_params.size == 0:
        raise ValueError("initial_params must not be empty.")

    def neg_ll(params):
        try:
            model = model_class(*params)
        except Exception:
            return 1e300
        ll = censored_log_likelihood(model, values, truncation, censored)
        return 1e300 if not np.isfinite(ll) else -ll

    result = minimize(
        neg_ll,
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B" if bounds is not None else "BFGS",
    )
    best_x, best_f = result.x, float(result.fun)

    # polish: L-BFGS-B with numeric gradients can stall on flat likelihoods
    refine = minimize(neg_ll, x0=best_x, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10})
    if np.isfinite(refine.fun) and refine.fun < best_f:
        best_x, best_f = refine.x, float(refine.fun)

    if not np.isfinite(best_f):
        raise RuntimeError("censored MLE optimization failed to find a finite likelihood.")
    return model_class(*best_x)


def kaplan_meier(values, truncation=None, censored=None):
    r"""Product-limit (Kaplan-Meier) survival estimate under left truncation
    and right censoring.

    With event (uncensored) times :math:`y_1 < y_2 < \dots`, :math:`d_j`
    events at :math:`y_j`, and risk set
    :math:`R_j = \#\{i : t_i < y_j \le x_i\}`,

    .. math:: \hat S(y) = \prod_{y_j \le y} \Big(1 - \frac{d_j}{R_j}\Big).

    Under left truncation the estimator is *conditional on survival to the
    smallest truncation point* :math:`t_{\min}`: it estimates
    :math:`S(y)/S(t_{\min})`. With a single common deductible this is exactly
    the conditional severity the data can identify. Risk sets near
    :math:`t_{\min}` can be small when truncation points are heterogeneous;
    interpret the left end with care.

    Returns
    -------
    tuple of np.ndarray
        ``(times, survival)``: distinct event times and the estimated
        survival immediately after each.
    """
    values, trunc, cens = _prepare(values, truncation, censored)
    event_vals = values[~cens]
    if event_vals.size == 0:
        raise ValueError("at least one uncensored observation is required.")
    times, d = np.unique(event_vals, return_counts=True)
    sorted_vals = np.sort(values)
    sorted_trunc = np.sort(trunc)
    # R_j = #{t_i < y_j} - #{x_i < y_j}; censored ties at y_j stay in the risk set
    entered = np.searchsorted(sorted_trunc, times, side="left")
    exited = np.searchsorted(sorted_vals, times, side="left")
    risk = entered - exited
    if np.any(risk <= 0):
        raise ValueError("empty risk set encountered; truncation pattern too sparse.")
    surv = np.cumprod(1.0 - d / risk)
    return times, surv
