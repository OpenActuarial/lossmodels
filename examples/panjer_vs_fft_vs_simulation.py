import time

import numpy as np

from lossmodels.aggregate import (
    CollectiveRiskModel,
    discretize_severity,
    fft_aggregate_poisson,
    mean_from_aggregate_pmf,
    mean_from_aggregate_pmf_fft,
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
# Simulation
# -------------------------------------------------
n_sim = 100_000

t0 = time.perf_counter()
sim_samples = model.sample(n_sim)
t1 = time.perf_counter()

sim_mean = float(np.mean(sim_samples))
sim_var_95 = model.var(0.95, n_sim=n_sim)
sim_tvar_95 = model.tvar(0.95, n_sim=n_sim)

print("=== Monte Carlo Simulation ===")
print("Runtime (sec):", round(t1 - t0, 4))
print("Sample mean:", sim_mean)
print("VaR 95%:", sim_var_95)
print("TVaR 95%:", sim_tvar_95)
print()

# -------------------------------------------------
# Discretization setup
# -------------------------------------------------
h = 0.01
max_loss = 20.0
n_steps = 5_000

severity_pmf = discretize_severity(sev, h=h, max_loss=max_loss)

# -------------------------------------------------
# Panjer recursion
# -------------------------------------------------
t0 = time.perf_counter()
g_panjer = panjer_recursion(freq, severity_pmf, n_steps=n_steps)
t1 = time.perf_counter()

panjer_mean = mean_from_aggregate_pmf(g_panjer, h=h)

print("=== Panjer Recursion ===")
print("Runtime (sec):", round(t1 - t0, 4))
print("Panjer mean:", panjer_mean)
print()

# -------------------------------------------------
# FFT
# -------------------------------------------------
t0 = time.perf_counter()
g_fft = fft_aggregate_poisson(freq, severity_pmf, n_steps=n_steps)
t1 = time.perf_counter()

fft_mean = mean_from_aggregate_pmf_fft(g_fft, h=h)

print("=== FFT Aggregate Method ===")
print("Runtime (sec):", round(t1 - t0, 4))
print("FFT mean:", fft_mean)
print()

# -------------------------------------------------
# Comparison
# -------------------------------------------------
print("=== Comparison ===")
print("Theoretical mean:", model.mean())
print("Simulation mean: ", sim_mean)
print("Panjer mean:     ", panjer_mean)
print("FFT mean:        ", fft_mean)
print()

print("Absolute errors:")
print("Simulation:", abs(sim_mean - model.mean()))
print("Panjer:    ", abs(panjer_mean - model.mean()))
print("FFT:       ", abs(fft_mean - model.mean()))
print()

print("First 10 Panjer pmf values:")
for i, p in enumerate(g_panjer[:10]):
    print(f"S = {i*h:5.2f} -> {p:.6f}")

print()
print("First 10 FFT pmf values:")
for i, p in enumerate(g_fft[:10]):
    print(f"S = {i*h:5.2f} -> {p:.6f}")