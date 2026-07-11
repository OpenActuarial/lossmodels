from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal

freq = Poisson(lam=2.5)
sev = Lognormal(mu=10.5, sigma=0.9)

model = CollectiveRiskModel(freq, sev)

print("=== Stop-Loss Example ===")
print("Mean:", model.mean())
print("VaR 95%:", model.var(0.95))
print("TVaR 95%:", model.tvar(0.95))
print("Stop-loss at 50k:", model.stop_loss(50_000))