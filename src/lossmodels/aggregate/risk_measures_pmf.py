import numpy as np


def _validate_pmf(pmf):
    pmf = np.asarray(pmf, dtype=float)

    if pmf.ndim != 1:
        raise ValueError("pmf must be 1D.")
    if len(pmf) == 0:
        raise ValueError("pmf must not be empty.")
    if np.any(pmf < 0):
        raise ValueError("pmf must be nonnegative.")

    total = pmf.sum()
    if total <= 0:
        raise ValueError("pmf must sum to a positive value.")

    return pmf / total


def var_from_pmf(pmf, h: float, q: float):
    """
    Compute VaR from a lattice pmf.
    """
    if not (0 < q < 1):
        raise ValueError("q must be in (0,1).")
    if h <= 0:
        raise ValueError("h must be positive.")

    pmf = _validate_pmf(pmf)
    cdf = np.cumsum(pmf)

    idx = np.searchsorted(cdf, q)
    return float(idx * h)


def tvar_from_pmf(pmf, h: float, q: float):
    """
    Compute TVaR from a lattice pmf.
    """
    if not (0 < q < 1):
        raise ValueError("q must be in (0,1).")
    if h <= 0:
        raise ValueError("h must be positive.")

    pmf = _validate_pmf(pmf)
    x = h * np.arange(len(pmf))
    cdf = np.cumsum(pmf)

    idx = np.searchsorted(cdf, q)
    v = float(x[idx])
    cdf_at_var = float(cdf[idx])

    # Average-quantile definition on a lattice with an atom at VaR:
    #   TVaR_q = [ sum_{x > v} x * p(x) + v * (F(v) - q) ] / (1 - q)
    # The atom at v contributes only the probability mass F(v) - q that lies
    # above level q, which is what makes TVaR coherent on discrete supports.
    above = x > v
    partial_tail_ev = float(np.sum(x[above] * pmf[above]))
    return (partial_tail_ev + v * (cdf_at_var - q)) / (1.0 - q)


def stop_loss_from_pmf(pmf, h: float, d: float):
    """
    Compute stop-loss premium E[(S - d)^+].
    """
    if h <= 0:
        raise ValueError("h must be positive.")

    pmf = _validate_pmf(pmf)
    x = h * np.arange(len(pmf))

    excess = np.maximum(x - d, 0.0)
    return float(np.sum(excess * pmf))


def mean_from_pmf(pmf, h: float):
    pmf = _validate_pmf(pmf)
    x = h * np.arange(len(pmf))
    return float(np.sum(x * pmf))
