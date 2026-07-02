import numpy as np

from .diagnostics import aic, bic
from .mle import (
    fit_burr,
    fit_exponential,
    fit_gamma,
    fit_inverse_gamma,
    fit_loglogistic,
    fit_lognormal,
    fit_pareto,
    fit_paretoII,
    fit_weibull,
)
from .moments import (
    fit_exponential_moments,
    fit_gamma_moments,
    fit_lognormal_moments,
    fit_pareto_moments,
    fit_weibull_moments,
)


SEVERITY_MLE_FITTERS = {
    "exponential": (fit_exponential, 1),
    "gamma": (fit_gamma, 2),
    "lognormal": (fit_lognormal, 2),
    "pareto": (fit_pareto, 2),
    "weibull": (fit_weibull, 2),
    # FAM / ASTAM families (0.6.0); all support truncation/censoring
    "paretoII": (fit_paretoII, 2),
    "loglogistic": (fit_loglogistic, 2),
    "inverse_gamma": (fit_inverse_gamma, 2),
    "burr": (fit_burr, 3),
}

SEVERITY_MOMENT_FITTERS = {
    "exponential": (fit_exponential_moments, 1),
    "gamma": (fit_gamma_moments, 2),
    "lognormal": (fit_lognormal_moments, 2),
    "pareto": (fit_pareto_moments, 2),
    "weibull": (fit_weibull_moments, 2),
}


def fit_best_severity(data, candidates=None, method="mle", criterion="aic",
                      truncation=None, censored=None):
    """
    Fit a set of severity models and return the best one by AIC or BIC.

    Parameters
    ----------
    data : array-like
        Severity observations.
    candidates : list of str, optional
        Candidate model names. Defaults to all supported severity fitters.
    method : {"mle", "moments"}
        Fitting method.
    criterion : {"aic", "bic"}
        Selection criterion.

    Returns
    -------
    dict
        Dictionary with keys:
        - "best_name"
        - "best_model"
        - "criterion"
        - "method"
        - "results"

    Notes
    -----
    Models that fail to fit are skipped.
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("data must not be empty.")

    if method not in {"mle", "moments"}:
        raise ValueError("method must be 'mle' or 'moments'.")
    if (truncation is not None or censored is not None) and method != "mle":
        raise ValueError("truncation/censoring is only supported with method='mle'.")
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'.")

    fitters = SEVERITY_MLE_FITTERS if method == "mle" else SEVERITY_MOMENT_FITTERS

    if candidates is None:
        candidates = list(fitters.keys())

    results = []

    for name in candidates:
        if name not in fitters:
            raise ValueError(f"Unsupported candidate: {name}")

        fitter, k = fitters[name]

        try:
            if truncation is not None or censored is not None:
                model = fitter(data, truncation=truncation, censored=censored)
            else:
                model = fitter(data)
            score = (
                aic(model, data, k=k, truncation=truncation, censored=censored)
                if criterion == "aic"
                else bic(model, data, k=k, truncation=truncation, censored=censored)
            )

            results.append({
                "name": name,
                "model": model,
                "k": k,
                "score": float(score),
            })
        except Exception as exc:
            results.append({
                "name": name,
                "model": None,
                "k": k,
                "score": float("inf"),
                "error": str(exc),
            })

    valid = [r for r in results if np.isfinite(r["score"])]
    if not valid:
        error_details = "; ".join(
            f'{r["name"]}: {r.get("error", "unknown error")}'
            for r in results
        )
        raise RuntimeError(
            f"No candidate models could be fit successfully. Errors: {error_details}"
        )

    best = min(valid, key=lambda r: r["score"])

    return {
        "best_name": best["name"],
        "best_model": best["model"],
        "criterion": criterion,
        "method": method,
        "results": results,
    }
