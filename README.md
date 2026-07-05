# lossmodels

A Python library for actuarial loss modeling using frequency–severity methods.

---

## Overview

**`lossmodels`** is a readable, NumPy/SciPy-backed implementation of the core
loss-modeling toolkit: a large catalog of severity and frequency distributions,
aggregate (collective-risk) loss models, coverage modifications, and fitting and
model-selection utilities. It is aimed at actuarial students, analysts, insurance
data scientists, and quantitative developers who want a lightweight library whose
parameterizations line up with the standard references.

The continuous-severity and discrete-frequency inventories use the standard
scale-parameterized forms common in actuarial loss modeling, so means, moments,
and quantiles line up with the formulas you would compute by hand.

## Highlights

- **Severity models** — a comprehensive continuous inventory: lognormal, gamma,
  Weibull, exponential and their inverses; the transformed-beta family (Burr,
  inverse Burr, generalized Pareto, Pareto/Lomax, inverse Pareto, loglogistic,
  paralogistic, inverse paralogistic); inverse Gaussian, log-t,
  single-parameter Pareto; and finite-support beta / generalized beta.
- **Frequency models** — the `(a, b, 0)` class (Poisson, binomial, geometric,
  negative binomial) and the `(a, b, 1)` class (zero-truncated, zero-modified,
  logarithmic).
- **Spliced / composite severities** — a body distribution below a threshold and a
  heavier tail above it, behind a single severity interface.
- **Empirical distributions** for raw frequency and severity data.
- **Aggregate loss modeling** — Monte Carlo simulation, Panjer recursion, and FFT
  aggregation, plus distribution-level and PMF-level risk measures (VaR, TVaR,
  stop-loss, limited expected value).
- **Coverage modifications** — ordinary deductibles, policy limits, and layers,
  each wrapping any severity and re-usable everywhere a severity is accepted.
- **Estimation** — maximum likelihood and method-of-moments fitters,
  log-likelihood / AIC / BIC diagnostics, goodness-of-fit statistics (KS,
  Anderson–Darling, Cramér–von Mises), and automatic best-fit selection.

## Installation

From PyPI:

```bash
pip install lossmodels
```

In development mode from a source checkout:

```bash
pip install -e .
```

Requires Python `>=3.10`; the only runtime dependencies are `numpy` and `scipy`.

## Package structure

```text
lossmodels/
├── severity/      # continuous claim-severity distributions
├── frequency/     # discrete claim-count distributions
├── aggregate/     # collective-risk models, discretization, Panjer, FFT, risk metrics
├── coverage/      # deductibles, limits, and layers
├── empirical/     # empirical frequency and severity distributions
├── estimation/    # MLE, method of moments, diagnostics, model selection
└── utils/         # shared numeric / validation helpers
```

## Quick start

### A frequency–severity aggregate model

```python
from lossmodels import Poisson, Lognormal, CollectiveRiskModel

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

Every model name is importable directly from the top level
(`from lossmodels import Burr, ZeroTruncated, ...`) or from its submodule
(`from lossmodels.severity import Burr`).

### Using the extended distribution catalog

```python
from lossmodels import Burr, ParetoII, ZeroTruncated, Poisson, CollectiveRiskModel

sev = ParetoII(alpha=3.0, theta=1000.0)        # the two-parameter "Pareto" (Type II / Lomax)
freq = ZeroTruncated(Poisson(2.0))             # an (a, b, 1) frequency

model = CollectiveRiskModel(freq, sev)
print(model.mean(), model.var(0.99))

burr = Burr(alpha=3.0, theta=1000.0, gamma=2.0)
print(burr.mean(), burr.cdf(2500.0), burr.quantile(0.95))
print(burr.limited_expected_value(1000.0))     # E[min(X, 1000)]
```

## Severity models

### The shared severity interface

Every severity exposes the same interface, so they are interchangeable across the
aggregate, coverage, and estimation tooling:

| Method | Meaning |
| --- | --- |
| `pdf(x)`, `cdf(x)` | density and distribution function (scalar or array) |
| `quantile(p)` / `ppf(p)` | inverse CDF / Value-at-Risk |
| `sample(size)` | random variates |
| `mean()`, `variance()`, `std()` | moments (raise where they do not exist) |
| `limited_expected_value(d)` | `E[min(X, d)]` |
| `excess_loss(d)` | `E[(X − d)₊]`, the expected cost per loss above `d` |

Moments are returned from closed-form expressions and **raise a
`ValueError` outside their existence range** (e.g. a Pareto mean with `alpha ≤ 1`),
rather than silently returning a wrong or infinite number. A few distributions
(inverse Pareto, inverse exponential, log-t) have no finite positive moments, so
their `mean`/`variance` always raise.

### Base and classic distributions

| Distribution | Constructor | Support |
| --- | --- | --- |
| Exponential | `Exponential(rate)` | `x > 0` |
| Gamma | `Gamma(alpha, theta)` | `x > 0` |
| Lognormal | `Lognormal(mu, sigma)` | `x > 0` |
| Weibull | `Weibull(k, lam)` | `x > 0` |
| Pareto (Type I) | `Pareto(alpha, theta)` | `x ≥ theta` |

### Transformed beta family

| Distribution | Constructor |
| --- | --- |
| Burr (Type XII / Singh–Maddala) | `Burr(alpha, theta, gamma)` |
| Inverse Burr (Dagum) | `InverseBurr(tau, theta, gamma)` |
| Generalized Pareto (transformed-beta) | `GeneralizedPareto(alpha, theta, tau)` |
| Pareto (Type II / Lomax) | `ParetoII(alpha, theta)` |
| Inverse Pareto | `InversePareto(tau, theta)` |
| Loglogistic (Fisk) | `Loglogistic(gamma, theta)` |
| Paralogistic | `Paralogistic(alpha, theta)` |
| Inverse paralogistic | `InverseParalogistic(tau, theta)` |

### Transformed gamma family

| Distribution | Constructor |
| --- | --- |
| Inverse gamma (Vinci) | `InverseGamma(alpha, theta)` |
| Inverse Weibull (log-Gompertz) | `InverseWeibull(theta, tau)` |
| Inverse exponential | `InverseExponential(theta)` |

### Other distributions

| Distribution | Constructor |
| --- | --- |
| Inverse Gaussian (Wald) | `InverseGaussian(mu, theta)` |
| Log-t | `LogT(r, mu, sigma)` |
| Single-parameter Pareto (Type I) | `SingleParameterPareto(alpha, theta)` |

### Finite-support distributions

| Distribution | Constructor | Support |
| --- | --- | --- |
| Beta | `Beta(a, b, theta)` | `0 < x < theta` |
| Generalized beta | `GeneralizedBeta(a, b, theta, tau)` | `0 < x < theta` |

### Parameterization notes

A few naming points matter:

- **Two Paretos.** `ParetoII(alpha, theta)` is the two-parameter
  "Pareto" (Type II / Lomax, support `x > 0`). The classic `Pareto(alpha, theta)`
  is the **Type I** distribution (support `x ≥ theta`), commonly called the
  single-parameter Pareto; it is also exposed under that name as
  `SingleParameterPareto`.
- **`GeneralizedPareto` here is the three-parameter transformed-beta
  distribution**, not the extreme-value GPD used in peaks-over-threshold tail
  fitting. Extreme-value distributions (GEV, Gumbel, Fréchet, GPD, Hill) are out
  of scope for this severity catalog.
- `Exponential` is parameterized by `rate` (= `1/theta`) and `Weibull` by shape
  `k` and scale `lam`; the transformed-beta and transformed-gamma families follow
  the scale-parameter (`theta`) convention.

## Frequency models

### The shared frequency interface

Every frequency model exposes `pmf(k)`, `cdf(k)`, `sample(size)`, `mean()`,
`variance()`, and `std()`.

### The (a, b, 0) class

| Distribution | Constructor |
| --- | --- |
| Poisson | `Poisson(lam)` |
| Binomial | `Binomial(n, p)` |
| Geometric | `Geometric(p)` |
| Negative binomial | `NegativeBinomial(r, p)` |

### The (a, b, 1) class

The `(a, b, 1)` distributions are built from a base `(a, b, 0)` model:

```python
from lossmodels import Poisson, NegativeBinomial, ZeroTruncated, ZeroModified, Logarithmic

zt = ZeroTruncated(Poisson(2.0))                 # zero is impossible
zm = ZeroModified(NegativeBinomial(3.0, 1/3), p0_modified=0.4)   # custom mass at zero
lg = Logarithmic(beta=3.0)                       # no (a, b, 0) parent

print(zt.mean(), zt.pmf(0))   # 0.0 mass at zero
print(zm.mean(), zm.pmf(0))   # 0.4 at zero
print(lg.mean())
```

| Distribution | Constructor | Notes |
| --- | --- | --- |
| Zero-truncated | `ZeroTruncated(base)` | removes the mass at zero of any `(a, b, 0)` model and renormalizes |
| Zero-modified | `ZeroModified(base, p0_modified)` | places an arbitrary probability `p0_modified ∈ [0, 1)` at zero |
| Logarithmic | `Logarithmic(beta)` | the limiting member with no `(a, b, 0)` parent |

The wrappers respect the base model's probability parameterization; only
`Logarithmic` uses `beta`. For a `beta`-parameterized geometric or negative
binomial, set `p = 1 / (1 + beta)`.

## Spliced severities

`SplicedSeverity` joins a body distribution (renormalized below the threshold) to a
tail distribution supported above it. The tail must satisfy `cdf(threshold) = 0`
(e.g. a Type I `Pareto` with `theta` equal to the threshold), and the mixing
`weight` is the probability mass assigned to the body.

```python
from lossmodels import Lognormal, Pareto, SplicedSeverity

u = 50_000
body = Lognormal(mu=10.0, sigma=0.8)
tail = Pareto(alpha=2.5, theta=u)     # Type I tail, cdf(u) = 0

spliced = SplicedSeverity(body=body, tail=tail, threshold=u, weight=body.cdf(u))
print(spliced.mean(), spliced.quantile(0.99))
```

The result is itself a severity (`pdf`, `cdf`, `quantile`, `sample`, `mean`,
`variance`), so a tail-corrected severity drops straight back into the aggregate
and coverage tooling.

## Empirical distributions

```python
from lossmodels import EmpiricalSeverity, EmpiricalFrequency

sev = EmpiricalSeverity(claim_amounts)
freq = EmpiricalFrequency(claim_counts)
```

Both expose the standard severity / frequency interface built from the raw data.

## Coverage modifications

Deductibles, limits, and layers each wrap a severity and are themselves severities,
so they compose and feed any aggregate model.

```python
from lossmodels import Lognormal, Poisson, CollectiveRiskModel
from lossmodels.coverage import OrdinaryDeductible, PolicyLimit, Layer

base_sev = Lognormal(mu=10.0, sigma=0.8)

with_deductible = OrdinaryDeductible(base_sev, d=10_000)
with_limit = PolicyLimit(base_sev, u=50_000)
layer = Layer(base_sev, d=10_000, u=40_000)     # pays the loss in the layer above d up to u

model = CollectiveRiskModel(Poisson(lam=2.0), layer)
print(model.mean())
```

| Modification | Constructor |
| --- | --- |
| Ordinary deductible | `OrdinaryDeductible(severity, d)` |
| Policy limit | `PolicyLimit(severity, u)` |
| Layer | `Layer(severity, d, u)` |

## Aggregate loss modeling

### The collective-risk model

`CollectiveRiskModel(frequency, severity)` is the main entry point. It exposes:

| Method | Meaning |
| --- | --- |
| `mean()`, `variance()`, `std()` | aggregate moments |
| `var(q)`, `tvar(q)` | Value-at-Risk and Tail-VaR at level `q` |
| `stop_loss(d)` | stop-loss premium `E[(S − d)₊]` |
| `limited_expected_value(d)` | `E[min(S, d)]` |
| `frequency_mean()`, `severity_mean()` | component means |
| `sample(size)` | simulate aggregate losses |
| `summary()` | a one-call overview of the model |

### Panjer recursion and FFT

For exact (discretized) aggregate distributions, discretize the severity and then
recurse or transform:

```python
from lossmodels import Poisson, Lognormal
from lossmodels.aggregate import (
    discretize_severity, panjer_recursion, fft_aggregate_poisson,
)

freq = Poisson(lam=3.0)
sev = Lognormal(mu=8.0, sigma=0.7)

sev_pmf = discretize_severity(sev, h=100.0, max_loss=500_000.0)
agg_pmf = panjer_recursion(freq, sev_pmf, n_steps=5000)
# FFT aggregation (Poisson frequency):
agg_pmf_fft = fft_aggregate_poisson(freq, sev_pmf, n_steps=5000)
```

### Risk measures

Distribution-level helpers operate on a simulated loss vector, and PMF-level
helpers operate on a discretized aggregate PMF:

```python
from lossmodels.aggregate import (
    var, tvar, stop_loss, lev,                       # on a sample
    var_from_pmf, tvar_from_pmf, stop_loss_from_pmf,  # on an aggregate PMF
)

losses = model.sample(100_000)
print(var(losses, 0.95), tvar(losses, 0.95), stop_loss(losses, 50_000), lev(losses, 50_000))

print(var_from_pmf(agg_pmf, h=100.0, q=0.95))
print(tvar_from_pmf(agg_pmf, h=100.0, q=0.95))
print(stop_loss_from_pmf(agg_pmf, h=100.0, d=50_000.0))
```

## Estimation and model selection

The `estimation` module provides MLE and method-of-moments fitters, fit
diagnostics, goodness-of-fit statistics, and automatic best-fit selection.

```python
from lossmodels import (
    fit_lognormal, fit_poisson,
    fit_best_severity, fit_best_frequency,
    goodness_of_fit, aic, bic,
)

sev_model = fit_lognormal(severity_data)
freq_model = fit_poisson(count_data)

best_sev = fit_best_severity(severity_data, criterion="aic")
best_freq = fit_best_frequency(count_data, criterion="bic")

print(best_sev["best_name"], best_sev["best_model"])
print(best_freq["best_name"])
```

`fit_best_severity` / `fit_best_frequency` return a dict with `best_name`,
`best_model`, `criterion`, `method`, and the full per-candidate `results`.

- **Severity fitters and selection candidates:** `exponential`, `gamma`,
  `lognormal`, `pareto`, `weibull` (MLE or method of moments). Also available
  individually: `fit_exponential`, `fit_gamma`, `fit_lognormal`, `fit_pareto`,
  `fit_weibull`, plus the generic `fit_mle`.
- **Frequency fitters and selection candidates:** `poisson`, `negbinomial`
  (`fit_poisson`, `fit_negbinomial`).
- **Diagnostics and goodness of fit:** `log_likelihood`, `aic`, `bic`,
  `ks_statistic`, `anderson_darling`, `cramer_von_mises`, `tail_quantile_table`,
  and `goodness_of_fit`.

## Examples

The repository includes runnable example scripts, including:

- `fit_and_simulate.py`, `fit_and_compare_models.py`
- `panjer_vs_simulation.py`, `panjer_vs_fft_vs_simulation.py`
- `deductible_example.py`, `limit_example.py`, `layer_example.py`
- `stop_loss_example.py`

## Testing

Run the test suite with:

```bash
pytest -q
```

Run only the fast tests with:

```bash
pytest -q -m "not slow"
```

## License

MIT License