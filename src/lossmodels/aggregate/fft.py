import numpy as np

from ..frequency import Poisson


class AggregateMassLossWarning(UserWarning):
    """Emitted when FFT cleanup absorbs a material amount of probability mass.

    The compound FFT can lose or wrap mass through aliasing, an under-long
    lattice (the aggregate extends past ``n_steps``), or numerical negatives.
    Clipping to zero and renormalizing hides all three: the returned pmf looks
    complete even when the raw transform did not sum to one. A material
    correction means the lattice/``n_steps`` is too short -- widen it. Silence
    with ``on_mass_loss="ignore"`` or fail with ``on_mass_loss="raise"``.
    """


def fft_aggregate_poisson(
    frequency,
    severity_pmf: np.ndarray,
    n_steps: int,
    *,
    on_mass_loss: str = "warn",
    mass_tol: float = 1e-6,
    return_diagnostics: bool = False,
):
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
    on_mass_loss : {"warn", "raise", "ignore"}
        What to do when the clip-and-renormalize cleanup moves more than
        ``mass_tol`` of probability -- a signal of aliasing or a too-short
        lattice. ``"warn"`` (default) emits :class:`AggregateMassLossWarning`.
    mass_tol : float
        Tolerance for the cleanup correction (negative mass, or the
        pre-normalization total's distance from one).
    return_diagnostics : bool
        If ``True``, return ``(pmf, diagnostics)`` where ``diagnostics`` reports
        the negative mass clipped, the raw (pre-normalization) total, and the
        normalization factor applied.

    Returns
    -------
    np.ndarray
        Aggregate loss pmf of length n_steps + 1 (or ``(pmf, dict)`` when
        ``return_diagnostics``).

    Notes
    -----
    If f is the severity pmf and N ~ Poisson(lam), then the aggregate pgf/transform
    is:

        G_S(t) = exp(lam * (G_X(t) - 1))

    On a lattice, we approximate this using the discrete Fourier transform.
    """
    if not isinstance(frequency, Poisson):
        raise TypeError("fft_aggregate_poisson currently supports only Poisson frequency.")
    if on_mass_loss not in ("warn", "raise", "ignore"):
        raise ValueError("on_mass_loss must be 'warn', 'raise', or 'ignore'")

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

    # Numerical cleanup -- record what it absorbs before hiding it.
    negative_mass = float(-g[g < 0.0].sum())
    g = np.maximum(g, 0.0)
    raw_total = float(g.sum())
    if raw_total <= 0:
        raise ValueError("FFT aggregate pmf has zero total probability.")

    correction = max(negative_mass, abs(raw_total - 1.0))
    if correction > mass_tol and on_mass_loss != "ignore":
        msg = (
            f"FFT aggregate cleanup moved {correction:.3g} of probability mass "
            f"(negative mass {negative_mass:.3g}, pre-normalization total "
            f"{raw_total:.6g}); the lattice or n_steps is likely too short and "
            f"tail mass has been truncated or aliased. Widen the support."
        )
        if on_mass_loss == "raise":
            raise ValueError(msg)
        import warnings

        warnings.warn(msg, AggregateMassLossWarning, stacklevel=2)

    g = g / raw_total
    if return_diagnostics:
        return g, {
            "negative_mass": negative_mass,
            "raw_mass": raw_total,
            "normalization_factor": 1.0 / raw_total,
            "normalized": True,
        }
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
