# Aggregate Models

## Overview

Aggregate models combine frequency and severity to model total losses.

## Methods

- Simulation
- Panjer recursion
- FFT

## Example

```python
from lossmodels.frequency import Poisson
from lossmodels.severity import Exponential
from lossmodels.aggregate import CollectiveRiskModel

freq = Poisson(lam=2.0)
sev = Exponential(rate=1.0)

model = CollectiveRiskModel(freq, sev)

print(model.mean())
print(model.var(0.95))
```