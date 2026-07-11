import numpy as np

from lossmodels.estimation import (
    aic,
    bic,
    fit_best_severity,
    fit_exponential,
    fit_gamma,
    fit_lognormal,
    fit_weibull,
    log_likelihood,
)

np.random.seed(123)

# Example severity data
severity_data = np.random.lognormal(mean=1.0, sigma=0.5, size=5000)

print("=== Fit and Compare Severity Models ===")
print(f"Number of observations: {len(severity_data)}")
print(f"Sample mean: {severity_data.mean():.6f}")
print(f"Sample variance: {severity_data.var():.6f}")
print()

# Fit individual models
models = {
    "exponential": (fit_exponential(severity_data), 1),
    "gamma": (fit_gamma(severity_data), 2),
    "lognormal": (fit_lognormal(severity_data), 2),
    "weibull": (fit_weibull(severity_data), 2),
}

print("=== Individual Model Fits ===")
for name, (model, k) in models.items():
    ll = log_likelihood(model, severity_data)
    aic_val = aic(model, severity_data, k=k)
    bic_val = bic(model, severity_data, k=k)

    print(f"{name:12s} {model!r}")
    print(f"  log-likelihood: {ll:.6f}")
    print(f"  AIC:            {aic_val:.6f}")
    print(f"  BIC:            {bic_val:.6f}")
print()

# Automatic selection
result = fit_best_severity(
    severity_data,
    candidates=["exponential", "gamma", "lognormal", "weibull"],
    method="mle",
    criterion="aic",
)

print("=== Best Model by AIC ===")
print("Best name:", result["best_name"])
print("Best model:", result["best_model"])
print()

print("=== Ranked Results ===")
ranked = sorted(result["results"], key=lambda x: x["score"])
for row in ranked:
    if row["model"] is None:
        print(f"{row['name']:12s} failed: {row.get('error', 'unknown error')}")
    else:
        print(f"{row['name']:12s} score={row['score']:.6f} model={row['model']!r}")