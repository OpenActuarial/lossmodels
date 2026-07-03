import numpy as np

from ..frequency import Binomial, Geometric, NegativeBinomial, Poisson


def _panjer_ab0_parameters(frequency):
    """
    Return (a, b, p0_func) for Panjer recursion.

    p0_func(f0) returns P(S = 0) = G_N(f0), where G_N is the pgf of N.
    """
    if isinstance(frequency, Poisson):
        lam = frequency.lam

        def p0_func(f0):
            return np.exp(lam * (f0 - 1.0))

        return 0.0, lam, p0_func

    if isinstance(frequency, Binomial):
        n = frequency.n
        p = frequency.p
        q = 1.0 - p

        if q <= 0:
            raise ValueError("Binomial Panjer recursion requires p < 1.")

        a = -p / q
        b = (n + 1.0) * p / q

        def p0_func(f0):
            return (q + p * f0) ** n

        return a, b, p0_func

    if isinstance(frequency, Geometric):
        p = frequency.p
        q = 1.0 - p

        a = q
        b = 0.0

        def p0_func(f0):
            return p / (1.0 - q * f0)

        return a, b, p0_func

    if isinstance(frequency, NegativeBinomial):
        r = frequency.r
        p = frequency.p
        q = 1.0 - p

        a = q
        b = (r - 1.0) * q

        def p0_func(f0):
            return (p / (1.0 - q * f0)) ** r

        return a, b, p0_func

    raise TypeError(
        "Panjer recursion currently supports Poisson, Binomial, Geometric, "
        "and NegativeBinomial frequency models."
    )


def panjer_recursion(frequency, severity_pmf: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Compute the aggregate loss pmf using Panjer recursion.

    Parameters
    ----------
    frequency : frequency model
        One of Poisson, Binomial, Geometric, or NegativeBinomial.
    severity_pmf : np.ndarray
        Discretized severity pmf f, where f[j] = P(X = j*h) on a lattice.
    n_steps : int
        Number of aggregate steps to compute.

    Returns
    -------
    np.ndarray
        Aggregate loss pmf g of length n_steps + 1.

    Notes
    -----
    The recursion is:

        g(0) = G_N(f0)

        g(k) = [1 / (1 - a f0)] *
               sum_{j=1}^k (a + b j / k) f(j) g(k-j)

    for k >= 1.
    """
    severity_pmf = np.asarray(severity_pmf, dtype=float)

    if severity_pmf.ndim != 1:
        raise ValueError("severity_pmf must be a 1D array.")
    if len(severity_pmf) == 0:
        raise ValueError("severity_pmf must not be empty.")
    if np.any(severity_pmf < 0):
        raise ValueError("severity_pmf must be nonnegative.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    total = severity_pmf.sum()
    if total <= 0:
        raise ValueError("severity_pmf must sum to a positive value.")

    f = severity_pmf / total
    f0 = float(f[0])

    a, b, p0_func = _panjer_ab0_parameters(frequency)

    denom = 1.0 - a * f0
    if denom <= 0:
        raise ValueError("Invalid Panjer denominator: 1 - a*f0 must be positive.")

    g = np.zeros(n_steps + 1, dtype=float)
    g[0] = float(p0_func(f0))
    if float(g[0]) == 0.0:
        raise ValueError(
            "Panjer recursion underflowed: the aggregate mass at zero is below "
            "float precision (large expected claim count). Use "
            "fft_aggregate_poisson for high-frequency aggregates."
        )

    max_j = len(f) - 1

    for k in range(1, n_steps + 1):
        s = 0.0
        upper = min(k, max_j)

        for j in range(1, upper + 1):
            s += (a + b * j / k) * f[j] * g[k - j]

        g[k] = s / denom

    # Numerical cleanup
    g = np.maximum(g, 0.0)
    total_g = g.sum()
    if total_g > 0:
        g /= total_g

    return g


def cdf_from_pmf(pmf: np.ndarray) -> np.ndarray:
    """
    Compute cdf from pmf.
    """
    pmf = np.asarray(pmf, dtype=float)

    if pmf.ndim != 1:
        raise ValueError("pmf must be a 1D array.")
    if len(pmf) == 0:
        raise ValueError("pmf must not be empty.")
    if np.any(pmf < 0):
        raise ValueError("pmf must be nonnegative.")

    total = pmf.sum()
    if total <= 0:
        raise ValueError("pmf must sum to a positive value.")

    pmf = pmf / total
    return np.cumsum(pmf)


def mean_from_aggregate_pmf(pmf: np.ndarray, h: float) -> float:
    """
    Compute the mean aggregate loss from a lattice pmf.
    """
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
    x = h * np.arange(len(pmf), dtype=float)
    return float(np.sum(x * pmf))
