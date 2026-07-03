# Frequency Models

## Overview

Frequency models describe the number of loss events.

## Available Models

- Poisson
- Negative Binomial
- Binomial
- Geometric
- EmpiricalFrequency

## Example

```python
from lossmodels.frequency import Poisson

model = Poisson(lam=2.0)
print(model.mean())
print(model.var())
```