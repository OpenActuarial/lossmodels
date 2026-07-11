import numpy as np

from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.empirical import EmpiricalFrequency, EmpiricalSeverity
from lossmodels.estimation import fit_exponential, fit_lognormal, fit_poisson
from lossmodels.estimation.diagnostics import aic, bic, log_likelihood

np.random.seed(123)

# -------------------------------------------------
# Example observed data
# -------------------------------------------------
# Frequency observations: annual claim counts
freq_data = np.random.poisson(lam=2.0, size=500)

# Severity observations: claim sizes
sev_data = np.random.lognormal(mean=10.0, sigma=0.8, size=500)

# -------------------------------------------------
# Empirical models
# -------------------------------------------------
empirical_freq = EmpiricalFrequency(freq_data)
empirical_sev = EmpiricalSeverity(sev_data)

print("=== Empirical Models ===")
print("Empirical frequency mean:", empirical_freq.mean())
print("Empirical severity mean:", empirical_sev.mean())
print()

# -------------------------------------------------
# Fit parametric models
# -------------------------------------------------
poisson_model = fit_poisson(freq_data)
lognormal_model = fit_lognormal(sev_data)
exponential_model = fit_exponential(sev_data)

print("=== Fitted Models ===")
print("Poisson:", poisson_model)
print("Lognormal:", lognormal_model)
print("Exponential:", exponential_model)
print()

# -------------------------------------------------
# Compare severity fits
# -------------------------------------------------
ll_lognormal = log_likelihood(lognormal_model, sev_data)
ll_exponential = log_likelihood(exponential_model, sev_data)

aic_lognormal = aic(lognormal_model, sev_data, k=2)
aic_exponential = aic(exponential_model, sev_data, k=1)

bic_lognormal = bic(lognormal_model, sev_data, k=2)
bic_exponential = bic(exponential_model, sev_data, k=1)

print("=== Severity Fit Comparison ===")
print("Lognormal log-likelihood:", ll_lognormal)
print("Exponential log-likelihood:", ll_exponential)
print("Lognormal AIC:", aic_lognormal)
print("Exponential AIC:", aic_exponential)
print("Lognormal BIC:", bic_lognormal)
print("Exponential BIC:", bic_exponential)
print()

# -------------------------------------------------
# Build aggregate model using fitted distributions
# -------------------------------------------------
model = CollectiveRiskModel(poisson_model, lognormal_model)

print("=== Aggregate Model ===")
print("Theoretical mean:", model.mean())
print("Theoretical variance:", model.variance())
print("VaR 95%:", model.var(0.95))
print("TVaR 95%:", model.tvar(0.95))
print("Stop-loss at 100,000:", model.stop_loss(100_000))
print()

samples = model.sample(50_000)

print("=== Simulation Check ===")
print("Sample mean:", samples.mean())
print("Sample variance:", samples.var())