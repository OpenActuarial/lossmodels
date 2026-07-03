# Coverage Modifications

## Overview

`lossmodels.coverage` provides wrappers that transform a ground-up severity model into an insured payment distribution.

Available classes:

- `OrdinaryDeductible`
- `PolicyLimit`
- `Layer`

These classes behave like severity models, so they can be used anywhere a severity model is accepted, including aggregate loss models.

## Ordinary Deductible

If `X` is the ground-up loss and `d` is the deductible, then payment per loss is:

\[
Y = (X - d)^+ = \max(X - d, 0)
\]

```python
from lossmodels.severity import Lognormal
from lossmodels.coverage import OrdinaryDeductible

ground_up = Lognormal(mu=10.0, sigma=0.8)
paid = OrdinaryDeductible(ground_up, d=1000.0)

print("Mean payment:", paid.mean())
print("CDF at 500:", paid.cdf(500.0))
print("Simulated payments:", paid.sample(5))
```

## Policy Limit

If `u` is the policy limit, then payment per loss is:

\[
Y = \min(X, u)
\]

```python
from lossmodels.severity import Gamma
from lossmodels.coverage import PolicyLimit

ground_up = Gamma(alpha=2.0, theta=500.0)
paid = PolicyLimit(ground_up, u=2000.0)

print("Mean payment:", paid.mean())
print("CDF at 1500:", paid.cdf(1500.0))
```

## Layer

A layer with attachment `d` and width `u` pays:

\[
Y = \min((X - d)^+, u)
\]

This is useful for excess layers and limited layers.

```python
from lossmodels.severity import Pareto
from lossmodels.coverage import Layer

ground_up = Pareto(alpha=2.5, theta=1000.0)
layer = Layer(ground_up, d=5000.0, u=10000.0)

print("Mean layer payment:", layer.mean())
print("Payment probability:", layer.payment_probability())
print("CDF at 2500:", layer.cdf(2500.0))
```

## Using Coverage with Aggregate Models

Because coverage modifications return severity-like objects, they can be used directly in a collective risk model.

```python
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal
from lossmodels.coverage import Layer
from lossmodels.aggregate import CollectiveRiskModel

freq = Poisson(lam=2.0)
ground_up = Lognormal(mu=9.5, sigma=1.0)
sev = Layer(ground_up, d=1000.0, u=5000.0)

model = CollectiveRiskModel(freq, sev)

print("Aggregate mean:", model.mean())
print("Aggregate VaR 95%:", model.var(0.95))
```