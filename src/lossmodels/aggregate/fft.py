import numpy as np

from ..frequency import Poisson


def fft_aggregate_poisson(frequency, severity_pmf: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Compute aggregate loss pmf for a compound Poisson model using Fast Fourier Transform (FFT).

    Parameters
    ----------
    frequency : Poisson
        Poisson frequency model.
    severity_pmf : np.ndarray
        Discretized severity pmf on a lattice.
    n_steps : int
        Number of aggregate points to return minus 1. The returned array has
        length n_steps + 1.

    Returns
    -------
    np.ndarray
        Aggregate loss pmf of length n_steps + 1.

    Notes
    -----
    If f is the severity pmf and N ~ Poisson(lam), then the aggregate pgf/transform
    is:

        G_S(t) = exp(lam * (G_X(t) - 1))

    On a lattice, we approximate this using the discrete Fourier transform.
    """
    if not isinstance(frequency, Poisson):
        raise TypeError("fft_aggregate_poisson currently supports only Poisson frequency.")

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
    lam = frequency.lam

    # Zero-pad / truncate to target length
    m = n_steps + 1
    f_padded = np.zeros(m, dtype=float)
    copy_len = min(len(f), m)
    f_padded[:copy_len] = f[:copy_len]

    fft_f = np.fft.fft(f_padded)
    fft_g = np.exp(lam * (fft_f - 1.0))
    g = np.fft.ifft(fft_g).real

    # Numerical cleanup
    g = np.maximum(g, 0.0)
    total_g = g.sum()
    if total_g <= 0:
        raise ValueError("FFT aggregate pmf has zero total probability.")

    g /= total_g
    return g


def cdf_from_pmf_fft(pmf: np.ndarray) -> np.ndarray:
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


def mean_from_aggregate_pmf_fft(pmf: np.ndarray, h: float) -> float:
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
