import numpy as np

from lossmodels.aggregate.base import AggregateModel


class DummyAggregateModel(AggregateModel):
    def __init__(self, samples):
        self._samples = np.asarray(samples, dtype=float)

    def sample(self, size: int = 1) -> np.ndarray:
        if size != len(self._samples):
            raise ValueError("size must equal the number of stored samples")
        return self._samples.copy()

    def mean(self) -> float:
        return float(np.mean(self._samples))

    def variance(self) -> float:
        return float(np.var(self._samples))


def test_aggregate_model_var_uses_empirical_definition():
    model = DummyAggregateModel([0, 1, 2, 3, 4, 5])
    assert np.isclose(model.var(0.5, n_sim=6), 2.0)


def test_aggregate_model_tvar_weights_atom_at_var_correctly():
    model = DummyAggregateModel([0, 1, 2, 2, 100])
    # Average-quantile TVaR: the atom at VaR = 2 contributes only the mass
    # above level q, giving (2 + 100) / (5 * 0.4) = 51.
    assert np.isclose(model.tvar(0.6, n_sim=5), 51.0)