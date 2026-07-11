import numpy as np

from .diagnostics import aic, bic
from .mle import fit_negbinomial, fit_poisson
from .moments import fit_negbinomial_moments, fit_poisson_moments

FREQUENCY_MLE_FITTERS = {
    "poisson": (fit_poisson, 1),
    "negbinomial": (fit_negbinomial, 2),
}

FREQUENCY_MOMENT_FITTERS = {
    "poisson": (fit_poisson_moments, 1),
    "negbinomial": (fit_negbinomial_moments, 2),
}


def fit_best_frequency(data, candidates=None, method="mle", criterion="aic"):
    """
    Fit a set of frequency models and return the best one by AIC or BIC.
    """
    data = np.asarray(data)

    if data.size == 0:
        raise ValueError("data must not be empty.")
    if np.any(data < 0):
        raise ValueError("frequency data must be nonnegative.")

    if method not in {"mle", "moments"}:
        raise ValueError("method must be 'mle' or 'moments'.")
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'.")

    fitters = (
        FREQUENCY_MLE_FITTERS if method == "mle" else FREQUENCY_MOMENT_FITTERS
    )

    if candidates is None:
        candidates = list(fitters.keys())

    results = []

    for name in candidates:
        if name not in fitters:
            raise ValueError(f"Unsupported candidate: {name}")

        fitter, k = fitters[name]

        try:
            model = fitter(data)
            score = aic(model, data, k=k) if criterion == "aic" else bic(model, data, k=k)

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
        raise RuntimeError("No candidate frequency models could be fit successfully.")

    best = min(valid, key=lambda r: r["score"])

    return {
        "best_name": best["name"],
        "best_model": best["model"],
        "criterion": criterion,
        "method": method,
        "results": results,
    }
