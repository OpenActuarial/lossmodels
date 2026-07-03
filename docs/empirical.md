# Empirical Distributions

## Overview

`lossmodels.empirical` provides nonparametric frequency and severity models based directly on observed data.

Available classes:

- `EmpiricalSeverity`
- `EmpiricalFrequency`

These are useful when you want to model losses directly from observed experience without fitting a parametric distribution.

## Empirical Severity

```python
import numpy as np
from lossmodels.empirical import EmpiricalSeverity

data = np.array([120.0, 250.0, 400.0, 400.0, 900.0])

sev = EmpiricalSeverity(data)

print("Mean:", sev.mean())
print("Variance:", sev.variance())
print("CDF at 400:", sev.cdf(400.0))
print("Point mass at 400:", sev.pdf(400.0))
print("Excess loss above 300:", sev.excess_loss(300.0))
print("LEV at 500:", sev.limited_expected_value(500.0))
print("Bootstrap sample:", sev.sample(5))
```

Notes:

- `pdf(x)` returns the empirical point mass at `x`
- for continuous-looking data, `pdf(x)` will often be 0 except at observed values

## Empirical Frequency

```python
import numpy as np
from lossmodels.empirical import EmpiricalFrequency

counts = np.array([0, 1, 1, 2, 3, 1, 0, 2])

freq = EmpiricalFrequency(counts)

print("Mean:", freq.mean())
print("Variance:", freq.variance())
print("PMF at 1:", freq.pmf(1))
print("CDF at 2:", freq.cdf(2))
print("Bootstrap sample:", freq.sample(10))
```

## Use in Aggregate Modeling

You can combine empirical frequency and empirical severity in a collective risk model.

```python
import numpy as np
from lossmodels.empirical import EmpiricalFrequency, EmpiricalSeverity
from lossmodels.aggregate import CollectiveRiskModel

counts = np.array([0, 1, 2, 1, 3, 0, 2])
losses = np.array([100.0, 250.0, 400.0, 900.0, 1200.0])

freq = EmpiricalFrequency(counts)
sev = EmpiricalSeverity(losses)

model = CollectiveRiskModel(freq, sev)

print("Aggregate mean:", model.mean())
print("Aggregate TVaR 95%:", model.tvar(0.95))
```