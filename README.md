# lossmodels

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

A Python library for actuarial loss modeling using frequency–severity methods.

---

## Overview

`lossmodels` is a Python library for frequency-severity modeling, aggregate loss analysis, actuarial coverage modifications, credibility, and model fitting. It is designed for actuarial students, analysts, insurance data scientists, and quantitative developers who want a lightweight, readable implementation of core loss modeling techniques.

## Highlights

- Frequency models: Poisson, Negative Binomial, Binomial, Geometric
- Severity models: Exponential, Gamma, Lognormal, Pareto, Weibull
- Empirical distributions for frequency and severity data
- Aggregate loss modeling with:
  - Monte Carlo simulation
  - Panjer recursion
  - FFT aggregation for Poisson frequency
- Risk measures:
  - VaR
  - TVaR
  - Stop-loss
  - Limited expected value
  - PMF-based risk metrics
- Coverage modifications:
  - Ordinary deductibles
  - Policy limits
  - Layers
- Estimation tools:
  - Maximum likelihood estimation
  - Method of moments
  - AIC / BIC diagnostics
  - Best-fit selection for supported severity and frequency models
- Credibility models:
  - Bühlmann
  - Bühlmann–Straub

## Installation

Install from PyPI:

```bash
pip install lossmodels
```

Install in development mode from source:

```bash
pip install -e .
```

Current package metadata:

- Version: `0.1.4`
- Python: `>=3.10`
- Core dependencies: `numpy`, `scipy`

## Package Structure

```text
lossmodels/
├── aggregate/     # aggregate loss models, discretization, Panjer, FFT, risk metrics
├── coverage/      # deductibles, limits, and layers
├── credibility/   # Bühlmann and Bühlmann–Straub credibility
├── empirical/     # empirical frequency and severity distributions
├── estimation/    # MLE, method of moments, diagnostics, model selection
├── frequency/     # discrete claim count models
├── severity/      # continuous claim severity models
└── utils/         # helper utilities
```

## Quick Start

### Frequency-Severity Aggregate Model

```python
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal
from lossmodels.aggregate import CollectiveRiskModel

freq = Poisson(lam=2.0)
sev = Lognormal(mu=10.0, sigma=0.8)

model = CollectiveRiskModel(freq, sev)

print("Mean:", model.mean())
print("Variance:", model.variance())
print("VaR 95%:", model.var(0.95))
print("TVaR 95%:", model.tvar(0.95))

samples = model.sample(50_000)
print("Simulated mean:", samples.mean())
```

### Fit Models to Data

```python
from lossmodels.estimation import (
    fit_lognormal,
    fit_poisson,
    fit_best_severity,
    fit_best_frequency,
)

freq_model = fit_poisson(freq_data)
sev_model = fit_lognormal(sev_data)

best_severity = fit_best_severity(sev_data, criterion="aic")
best_frequency = fit_best_frequency(freq_data, criterion="bic")

print(best_severity["best_name"])
print(best_frequency["best_name"])
```

### Coverage Modifications

```python
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal
from lossmodels.coverage import OrdinaryDeductible, PolicyLimit, Layer
from lossmodels.aggregate import CollectiveRiskModel

base_sev = Lognormal(mu=10.0, sigma=0.8)

with_deductible = OrdinaryDeductible(base_sev, d=10_000)
with_limit = PolicyLimit(base_sev, u=50_000)
layer = Layer(base_sev, d=10_000, u=40_000)

model = CollectiveRiskModel(Poisson(lam=2.0), layer)
print(model.mean())
```

## Available Models

### Frequency

- `Poisson`
- `NegativeBinomial`
- `Binomial`
- `Geometric`

### Severity

- `Exponential`
- `Gamma`
- `Lognormal`
- `Pareto`
- `Weibull`

### Empirical

- `EmpiricalFrequency`
- `EmpiricalSeverity`

### Coverage

- `OrdinaryDeductible`
- `PolicyLimit`
- `Layer`

### Credibility

- `Buhlmann`
- `BuhlmannStraub`

## Aggregate Loss Methods

The `aggregate` module supports multiple ways to analyze total loss.

### Simulation

```python
samples = model.sample(100_000)
```

### Panjer Recursion

```python
from lossmodels.aggregate import discretize_severity, panjer_recursion

pmf = discretize_severity(sev, h=100.0, max_loss=200_000.0)
agg_pmf = panjer_recursion(freq, pmf, n_steps=5000)
```

### FFT Aggregation

```python
from lossmodels.aggregate import fft_aggregate_poisson

agg_pmf = fft_aggregate_poisson(freq, pmf, n_steps=5000)
```

### PMF-Based Risk Measures

```python
from lossmodels.aggregate import var_from_pmf, tvar_from_pmf, stop_loss_from_pmf

var95 = var_from_pmf(agg_pmf, h=100.0, q=0.95)
tvar95 = tvar_from_pmf(agg_pmf, h=100.0, q=0.95)
sl = stop_loss_from_pmf(agg_pmf, h=100.0, d=50_000.0)
```

## Estimation and Model Selection

The `estimation` module includes:

- MLE fitters for supported frequency and severity distributions
- method-of-moments estimators
- log-likelihood, AIC, and BIC diagnostics
- `fit_best_severity(...)`
- `fit_best_frequency(...)`

Current automated model selection support includes:

- Severity candidates: `exponential`, `gamma`, `lognormal`, `weibull`
- Frequency candidates: `poisson`, `negbinomial`

## Examples

The repository currently includes the following example scripts:

- `credibility_example.py`
- `deductible_example.py`
- `fit_and_compare_models.py`
- `fit_and_simulate.py`
- `layer_example.py`
- `limit_example.py`
- `panjer_vs_fft_vs_simulation.py`
- `panjer_vs_simulation.py`
- `stop_loss_example.py`

## Testing

Run the test suite with:

```bash
pytest -v
```

Run only non-slow tests with:

```bash
pytest -v -m "not slow"
```

## Project Scope

`lossmodels` currently focuses on core actuarial material around frequency-severity modeling and aggregate loss methods, with readable implementations that are useful for:

- learning actuarial loss models
- prototyping insurance analytics workflows
- validating calculations against textbook-style examples
- building small actuarial tools on top of a reusable package

## License

MIT License
