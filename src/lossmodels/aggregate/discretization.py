import numpy as np


def discretize_severity(severity, h: float, max_loss: float, method: str = "midpoint"):
    """
    Discretize a severity model onto a lattice with spacing h.

    Parameters
    ----------
    severity : object
        Severity model with a cdf(x) method.
    h : float
        Lattice step size.
    max_loss : float
        Maximum loss level for discretization.
        The final bucket absorbs all remaining tail probability.
    method : {"upper", "lower", "midpoint"}
        Discretization scheme. Defaults to ``"midpoint"`` (changed from
        ``"upper"`` in 0.6.0, since midpoint is the standard, nearly unbiased
        choice for Panjer/FFT input; the bound methods systematically shift
        the mean by about h/2). ``"upper"`` places the mass of ``[jh, (j+1)h)``
        at ``jh`` (losses rounded *down*; the discretized cdf is an upper bound
        on the true cdf, so the discretized mean is biased low). ``"lower"``
        places the mass of ``((j-1)h, jh]`` at ``jh`` (losses rounded *up*; cdf
        lower bound, mean biased high) -- together the two bracket the true
        distribution. ``"midpoint"`` is the rounding / mass-dispersal method
        (mass of ``(jh - h/2, jh + h/2]`` at ``jh``), the usual choice for
        Panjer/FFT input.

    Returns
    -------
    np.ndarray
        Probability mass vector on the lattice.
    """
    if h <= 0:
        raise ValueError("h must be positive.")
    if max_loss <= 0:
        raise ValueError("max_loss must be positive.")
    if not hasattr(severity, "cdf"):
        raise TypeError("severity must implement cdf(x).")
    if method not in {"upper", "lower", "midpoint"}:
        raise ValueError("method must be 'upper', 'lower', or 'midpoint'.")

    m = int(np.floor(max_loss / h))
    if m < 1:
        raise ValueError("max_loss must be at least as large as h.")

    probs = np.zeros(m + 1, dtype=float)

    if method == "upper":
        # Bucket 0 represents losses in [0, h] and must include any atom at 0.
        probs[0] = float(severity.cdf(h))
        for j in range(1, m):
            left = j * h
            right = (j + 1) * h
            probs[j] = float(severity.cdf(right) - severity.cdf(left))
        probs[m] = float(1.0 - severity.cdf(m * h))

    elif method == "lower":
        # Backward difference: bucket j carries the mass of ((j-1)h, jh], so
        # bucket 0 carries only the atom at zero (F(0), typically 0 for a
        # continuous severity). Using cdf(h) here would double-count [0, h]
        # against bucket 1 and, after normalization, bias the whole pmf low --
        # the rounded-up scheme must bound the true mean from above.
        probs[0] = float(severity.cdf(0.0))
        for j in range(1, m):
            left = (j - 1) * h
            right = j * h
            probs[j] = float(severity.cdf(right) - severity.cdf(left))
        probs[m] = float(1.0 - severity.cdf((m - 1) * h))

    elif method == "midpoint":
        probs[0] = float(severity.cdf(h / 2.0))
        for j in range(1, m):
            left = (j - 0.5) * h
            right = (j + 0.5) * h
            probs[j] = float(severity.cdf(right) - severity.cdf(left))
        probs[m] = float(1.0 - severity.cdf((m - 0.5) * h))

    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0:
        raise ValueError("Discretization produced zero total probability.")

    probs /= total
    return probs


def bucket_representatives(h: float, size: int) -> np.ndarray:
    if h <= 0:
        raise ValueError("h must be positive.")
    if size <= 0:
        raise ValueError("size must be positive.")
    return h * np.arange(size, dtype=float)


def mean_from_discretized_pmf(pmf: np.ndarray, h: float) -> float:
    pmf = np.asarray(pmf, dtype=float)
    if pmf.ndim != 1:
        raise ValueError("pmf must be a 1D array.")
    if len(pmf) == 0:
        raise ValueError("pmf must not be empty.")
    if h <= 0:
        raise ValueError("h must be positive.")
    if np.any(pmf < 0):
        raise ValueError("pmf must be nonnegative.")

    total = pmf.sum()
    if total <= 0:
        raise ValueError("pmf must sum to a positive value.")

    pmf = pmf / total
    x = bucket_representatives(h, len(pmf))
    return float(np.sum(x * pmf))
