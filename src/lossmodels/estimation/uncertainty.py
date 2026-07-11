"""Parameter uncertainty for fitted models: observed information at the MLE.

A fitted distribution without standard errors is a point estimate pretending
to be a model. Every fitter in this package returns a plain distribution
object; :func:`fit_uncertainty` takes that object *back*, evaluates the
numerical Hessian of the log-likelihood at the fitted parameters, and
returns the observed-information covariance -- generically, for any model
whose ``__init__`` parameters are stored as same-named attributes (all
in-package distributions qualify). Truncation and censoring flow through
:func:`~lossmodels.estimation.diagnostics.log_likelihood` unchanged, so the
uncertainty matches however the model was actually fit.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from .diagnostics import log_likelihood

__all__ = ["FitUncertainty", "fit_uncertainty", "model_parameters"]


def model_parameters(model) -> dict[str, float]:
    """The model's ``__init__`` parameters read back from its attributes."""
    sig = inspect.signature(type(model).__init__)
    names = [n for n in sig.parameters if n != "self"]
    out = {}
    for name in names:
        if not hasattr(model, name):
            raise TypeError(
                f"{type(model).__name__} stores no attribute {name!r} matching "
                "its __init__ parameter; fit_uncertainty needs same-named "
                "attributes to rebuild the model"
            )
        out[name] = float(getattr(model, name))
    if not out:
        raise TypeError(f"{type(model).__name__} exposes no parameters")
    return out


@dataclass
class FitUncertainty:
    """Observed-information uncertainty for a fitted model."""

    param_names: list[str]
    estimates: np.ndarray
    covariance: np.ndarray
    n_obs: int

    @property
    def se(self) -> np.ndarray:
        """Standard errors: square roots of the covariance diagonal."""
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    def summary(self, confidence_level: float = 0.95) -> pd.DataFrame:
        """Estimates, standard errors, and Wald confidence intervals.

        Intervals are on the natural parameter scale,
        :math:`\\hat\\theta \\pm z\\,\\mathrm{se}`; for strictly positive
        parameters near zero a log-scale interval may be preferable.
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        se = self.se
        return pd.DataFrame(
            {
                "estimate": self.estimates,
                "se": se,
                "ci_low": self.estimates - z * se,
                "ci_high": self.estimates + z * se,
            },
            index=pd.Index(self.param_names, name="parameter"),
        )


def fit_uncertainty(
    model,
    data,
    truncation=None,
    censored=None,
    step: float = 1e-5,
) -> FitUncertainty:
    """Covariance of a fitted model's parameters from the observed information.

    Parameters
    ----------
    model
        A fitted distribution (any object whose ``__init__`` parameters are
        stored as same-named attributes).
    data : array-like
        The observations the model was fit to.
    truncation, censored : optional
        Passed to :func:`log_likelihood`; supply exactly what the fit used,
        or the curvature will not describe the fit that was performed.
    step : float
        Relative finite-difference step for the Hessian.

    Returns
    -------
    FitUncertainty
        With ``covariance``, ``se``, and a ``summary()`` table.

    Raises
    ------
    ValueError
        If the Hessian is not negative definite at the supplied parameters --
        typically the MLE is on a parameter boundary or the model was not
        actually fit to this data.
    """
    params = model_parameters(model)
    names = list(params)
    theta = np.array([params[n] for n in names], dtype=float)
    cls = type(model)

    def loglik(vec):
        try:
            candidate = cls(**dict(zip(names, (float(v) for v in vec), strict=True)))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"could not evaluate the likelihood at {dict(zip(names, vec, strict=True))!r} "
                f"while differentiating: {exc}"
            ) from exc
        return float(log_likelihood(candidate, data, truncation=truncation,
                                     censored=censored))

    k = len(theta)
    # per-parameter steps scaled to each parameter's own magnitude; a fixed
    # floor of 1.0 would swamp small-scale parameters (e.g. exponential rates
    # of order 1e-5) with truncation error
    h = np.where(theta != 0.0, step * np.abs(theta), step)
    f0 = loglik(theta)
    hess = np.empty((k, k), dtype=float)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        hess[i, i] = (loglik(theta + ei) - 2.0 * f0 + loglik(theta - ei)) / h[i] ** 2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            mixed = (
                loglik(theta + ei + ej)
                - loglik(theta + ei - ej)
                - loglik(theta - ei + ej)
                + loglik(theta - ei - ej)
            ) / (4.0 * h[i] * h[j])
            hess[i, j] = hess[j, i] = mixed
    if not np.all(np.isfinite(hess)):
        raise ValueError("log-likelihood Hessian is not finite at the estimates")
    info = -hess
    try:
        np.linalg.cholesky(info)
    except np.linalg.LinAlgError:
        raise ValueError(
            "observed information is not positive definite: the parameters "
            "are not an interior maximum of this data's likelihood (boundary "
            "solution, or model/data mismatch)"
        ) from None
    covariance = np.linalg.inv(info)
    return FitUncertainty(
        param_names=names,
        estimates=theta,
        covariance=covariance,
        n_obs=int(np.asarray(data).size),
    )
