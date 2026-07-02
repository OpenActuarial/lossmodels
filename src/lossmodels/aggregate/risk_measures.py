import numpy as np


def _validate_losses(losses: np.ndarray) -> np.ndarray:
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1:
        raise ValueError("losses must be 1D.")
    if len(losses) == 0:
        raise ValueError("losses must not be empty.")
    return losses


def _validate_q(q) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    if q_arr.size == 0 or not np.all((q_arr > 0) & (q_arr < 1)):
        raise ValueError("q must be between 0 and 1.")
    return q_arr


def _var_rank(n: int, q: np.ndarray) -> np.ndarray:
    """Rank k of the VaR order statistic: k = ceil(n*q), guarded against
    floating-point error when n*q is an exact integer."""
    nq = n * q
    k = np.where(np.abs(nq - np.round(nq)) < 1e-8, np.round(nq), np.ceil(nq))
    return np.clip(k.astype(np.intp), 1, n)


def var(losses: np.ndarray, q):
    """
    Value-at-Risk at level q.

    Implements the lower empirical quantile

        VaR_q = inf{x : F(x) >= q},

    i.e. the order statistic ``x_(ceil(n*q))`` (equal to
    ``np.quantile(losses, q, method="inverted_cdf")``). Ecosystem-standard
    estimator shared with ``risksim`` and ``extremeloss``.

    Parameters
    ----------
    losses : array-like
        Loss samples.
    q : float or array-like
        Quantile level(s), with 0 < q < 1.

    Returns
    -------
    float or np.ndarray
        Scalar for scalar ``q``; array of the same length for array ``q``.
    """
    q_arr = _validate_q(q)
    losses = np.sort(_validate_losses(losses))
    k = _var_rank(losses.size, q_arr)
    out = losses[k - 1]
    return float(out) if q_arr.ndim == 0 else np.asarray(out, dtype=float)


def tvar(losses: np.ndarray, q):
    """
    Tail Value-at-Risk at level q.

    Implements the average-quantile (coherent) definition

        TVaR_q = (1 / (1 - q)) * integral_q^1 VaR_u du,

    via the Acerbi-Tasche empirical plug-in: with sorted losses
    ``x_(1) <= ... <= x_(n)`` and ``k = ceil(n*q)``,

        TVaR_q = [ sum_{i>k} x_(i) + x_(k) * (k - n*q) ] / (n * (1 - q)).

    Exact for the empirical distribution (handles ties/atoms correctly),
    always ``>= var(losses, q)``, and equal to the mean of the largest
    ``n*(1-q)`` observations when ``n*q`` is an integer. Ecosystem-standard
    estimator shared with ``risksim`` and ``extremeloss``.

    Parameters
    ----------
    losses : array-like
        Loss samples.
    q : float or array-like
        Quantile level(s), with 0 < q < 1.

    Returns
    -------
    float or np.ndarray
        Scalar for scalar ``q``; array of the same length for array ``q``.
    """
    q_arr = _validate_q(q)
    losses = np.sort(_validate_losses(losses))
    n = losses.size
    k = _var_rank(n, q_arr)
    csum = np.concatenate(([0.0], np.cumsum(losses)))
    tail_sum = csum[n] - csum[k]
    var_vals = losses[k - 1]
    nq = n * q_arr
    weight = np.where(np.abs(nq - np.round(nq)) < 1e-8, 0.0, k - nq)
    out = (tail_sum + var_vals * weight) / (n * (1.0 - q_arr))
    # TVaR >= VaR holds as a theorem in exact arithmetic; enforce it so
    # floating-point noise (e.g. a constant tail at a layer limit) can never
    # produce tvar infinitesimally below var.
    out = np.maximum(out, var_vals)
    return float(out) if q_arr.ndim == 0 else np.asarray(out, dtype=float)


def stop_loss(losses: np.ndarray, d: float) -> float:
    """
    Expected stop-loss premium E[(S - d)+].

    Parameters
    ----------
    losses : np.ndarray
        Array of loss samples.
    d : float
        Deductible / attachment point, with d >= 0.

    Returns
    -------
    float
        Expected stop-loss value.
    """
    if d < 0:
        raise ValueError("d must be nonnegative.")
    losses = _validate_losses(losses)
    return float(np.mean(np.maximum(losses - d, 0.0)))


def lev(losses: np.ndarray, d: float) -> float:
    """
    Limited expected value E[min(S, d)].

    Parameters
    ----------
    losses : np.ndarray
        Array of loss samples.
    d : float
        Limit, with d >= 0.

    Returns
    -------
    float
        Limited expected value.
    """
    if d < 0:
        raise ValueError("d must be nonnegative.")
    losses = _validate_losses(losses)
    return float(np.mean(np.minimum(losses, d)))


def exceedance_probability(losses: np.ndarray, d: float) -> float:
    """
    Probability P(S > d).

    Parameters
    ----------
    losses : np.ndarray
        Array of loss samples.
    d : float
        Threshold.

    Returns
    -------
    float
        Estimated exceedance probability.
    """
    losses = _validate_losses(losses)
    return float(np.mean(losses > d))
