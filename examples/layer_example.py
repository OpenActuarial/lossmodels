from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.coverage import Layer
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal

freq = Poisson(lam=2.0)
sev = Lognormal(mu=10.0, sigma=0.8)

layer = Layer(sev, d=10_000, u=40_000)

model = CollectiveRiskModel(freq, layer)

print("=== Layer Example ===")
print("Layer mean:", layer.mean())
print("Aggregate mean:", model.mean())
print("VaR 95%:", model.var(0.95))
print("TVaR 95%:", model.tvar(0.95))