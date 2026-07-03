# Severity Models

## Overview

Severity models describe the size of individual losses.

## Available Models

- Exponential
- Gamma
- Lognormal
- Pareto
- Weibull
- EmpiricalSeverity

## Example

```python
from lossmodels.severity import Lognormal

model = Lognormal(mu=10.0, sigma=0.8)
print(model.mean())
print(model.var())
```