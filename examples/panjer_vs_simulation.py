import numpy as np

from lossmodels.aggregate import (
    CollectiveRiskModel,
    discretize_severity,
    mean_from_aggregate_pmf,
    panjer_recursion,
)
from lossmodels.frequency import Poisson
from lossmodels.severity import Exponential

np.random.seed(123)

# -------------------------------------------------
# Model setup
# -------------------------------------------------
freq = Poisson(lam=2.0)
sev = Exponential(rate=1.0)

model = CollectiveRiskModel(freq, sev)

print("=== Model Setup ===")
print("Frequency model:", freq)
print("Severity model:", sev)
print("Theoretical aggregate mean:", model.mean())
print()

# -------------------------------------------------
# Monte Carlo simulation
# -------------------------------------------------
n_sim = 100_000
sim_samples = model.sample(n_sim)

sim_mean = np.mean(sim_samples)
sim_var_95 = model.var(0.95, n_sim=n_sim)
sim_tvar_95 = model.tvar(0.95, n_sim=n_sim)

print("=== Monte Carlo Results ===")
print("Sample mean:", sim_mean)
print("VaR 95%:", sim_var_95)
print("TVaR 95%:", sim_tvar_95)
print()

# -------------------------------------------------
# Panjer recursion setup
# -------------------------------------------------
h = 0.01
max_loss = 20.0
# The aggregate lives on the same lattice as the discretized severity. Here
# max_loss (20) is already ~10 aggregate means out (E[S] = 2), so spanning
# [0, max_loss] captures essentially all of the aggregate probability and the
# recursion needs no further steps. Sizing n_steps to the severity support
# keeps this exact -- the previous value of 50_000 pushed the grid out to
# S = 500 (~250 aggregate means) at a large and pointless cost.
n_steps = int(round(max_loss / h))

severity_pmf = discretize_severity(sev, h=h, max_loss=max_loss)
aggregate_pmf = panjer_recursion(freq, severity_pmf, n_steps=n_steps)

panjer_mean = mean_from_aggregate_pmf(aggregate_pmf, h=h)

x_values = h * np.arange(len(aggregate_pmf))

print("=== Panjer Recursion Results ===")
print("Step size h:", h)
print("Max loss used for discretization:", max_loss)
print("Number of aggregate steps:", n_steps)
print("Panjer mean:", panjer_mean)
print()

# -------------------------------------------------
# Compare simulation vs Panjer
# -------------------------------------------------
print("=== Comparison ===")
print("Theoretical mean:", model.mean())
print("Simulation mean:", sim_mean)
print("Panjer mean:", panjer_mean)
print()

print("First 15 aggregate pmf values from Panjer:")
for x, p in zip(x_values[:15], aggregate_pmf[:15], strict=True):
    print(f"S = {x:5.2f} -> {p:.6f}")