# lossmodels

Severity and frequency fitting and aggregate loss distributions.

[![CI](https://github.com/OpenActuarial/lossmodels/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenActuarial/lossmodels/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lossmodels)](https://pypi.org/project/lossmodels/)
[![Python](https://img.shields.io/pypi/pyversions/lossmodels)](https://pypi.org/project/lossmodels/)

## Overview

`lossmodels` covers the loss-distribution workflow from data to aggregate:
a severity catalog spanning the classic distributions and the transformed
beta and transformed gamma families, frequency models, maximum likelihood
that is truncation- and censoring-aware, fit diagnostics and model selection,
and aggregate distributions via simulation, FFT, and Panjer recursion.

Every severity model shares one interface — moments, quantiles, limited
expected values, increased-limits tables — so fitted models drop directly
into coverage modifications, aggregates, and the wider ecosystem.

## Installation

```bash
pip install lossmodels
```

Requires Python 3.10 or newer.

## Quick start

```python
from lossmodels import Poisson, Lognormal, CollectiveRiskModel

freq = Poisson(lam=2.0)
sev = Lognormal(mu=10.0, sigma=0.8)
model = CollectiveRiskModel(freq, sev)

print("Mean     :", model.mean())
print("Variance :", model.variance())
print("VaR 95%  :", model.var(0.95))
print("TVaR 95% :", model.tvar(0.95))

samples = model.sample(50_000, rng=0)
print("Simulated mean:", samples.mean())
```

## What's inside

- **Severity** — classic and extended catalogs (including the transformed
  beta and transformed gamma families) behind one shared interface, plus
  increased-limits and loss-elimination tables.
- **Frequency** — Poisson, negative binomial, and related counting models.
- **Estimation** — MLE with truncation and censoring support, parameter
  uncertainty, goodness-of-fit diagnostics, and model comparison.
- **Aggregate** — `CollectiveRiskModel` simulation, FFT and Panjer
  recursion, stop-loss and risk measures.
- **Coverage** — deductibles, limits, and coinsurance applied analytically.
- **Empirical** — nonparametric counterparts for validation.

The full API reference and end-to-end worked examples live at
**[openactuarial.org/lossmodels.html](https://openactuarial.org/lossmodels.html)**.

## The OpenActuarial ecosystem

`lossmodels` is one of seven packages that share conventions — tidy tables,
explicit distribution parameterizations, reproducible random-number handling —
and compose across package seams:

| Package | Role |
|---|---|
| [actuarialpy](https://github.com/OpenActuarial/actuarialpy) | Calculation primitives the workflow packages build on |
| [experiencestudies](https://github.com/OpenActuarial/experiencestudies) | Experience reporting, actual-vs-expected, claimant and concentration analysis |
| [projectionmodels](https://github.com/OpenActuarial/projectionmodels) | Claim, premium, and expense projection over a renewal horizon |
| [ratingmodels](https://github.com/OpenActuarial/ratingmodels) | Manual and experience rating, credibility, indication, GLM relativities |
| **[lossmodels](https://github.com/OpenActuarial/lossmodels)** | Severity and frequency fitting, aggregate loss distributions |
| [extremeloss](https://github.com/OpenActuarial/extremeloss) | Extreme-value tails: POT/GPD, GEV, return levels, splicing |
| [risksim](https://github.com/OpenActuarial/risksim) | Portfolio Monte Carlo, dependence, reinsurance contracts, risk measures |

Install everything at once with `pip install openactuarial`.

## Development

```bash
git clone https://github.com/OpenActuarial/lossmodels
cd lossmodels
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

CI runs the same gate on Python 3.10–3.14 across Linux and Windows.

## Versioning and stability

All ecosystem packages are pre-1.0: minor releases may change APIs, and every
release is documented in [CHANGELOG.md](CHANGELOG.md). Current per-package API
stability is tracked at
[openactuarial.org/stability.html](https://openactuarial.org/stability.html).

## License

MIT — see [LICENSE](LICENSE).
