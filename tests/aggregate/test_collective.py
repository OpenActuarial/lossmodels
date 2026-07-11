import numpy as np
import pytest

from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.coverage import Layer
from lossmodels.frequency import Poisson
from lossmodels.severity import Exponential, Lognormal


def test_collective_init_valid():
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)

    model = CollectiveRiskModel(freq, sev)

    assert model.frequency is freq
    assert model.severity is sev


def test_collective_sample_shape():
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_collective_sample_invalid_size():
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    with pytest.raises(ValueError):
        model.sample(0)

    with pytest.raises(ValueError):
        model.sample(-1)


def test_collective_mean_formula():
    freq = Poisson(lam=2.5)
    sev = Exponential(rate=0.5)
    model = CollectiveRiskModel(freq, sev)

    expected = freq.mean() * sev.mean()
    assert np.isclose(model.mean(), expected)


def test_collective_variance_formula():
    freq = Poisson(lam=2.5)
    sev = Exponential(rate=0.5)
    model = CollectiveRiskModel(freq, sev)

    expected = freq.mean() * sev.variance() + freq.variance() * (sev.mean() ** 2)
    assert np.isclose(model.variance(), expected)


def test_collective_sample_mean_close_to_theoretical():
    np.random.seed(123)

    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    samples = model.sample(200_000)

    assert np.isclose(np.mean(samples), model.mean(), rtol=0.03)


def test_collective_sample_variance_close_to_theoretical():
    np.random.seed(123)

    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    samples = model.sample(200_000)

    assert np.isclose(np.var(samples), model.variance(), rtol=0.06)


def test_collective_std():
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    assert np.isclose(model.std(), np.sqrt(model.variance()))


def test_collective_var_and_tvar_ordering():
    np.random.seed(123)

    freq = Poisson(lam=3.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    var_95 = model.var(0.95, n_sim=100_000)
    tvar_95 = model.tvar(0.95, n_sim=100_000)

    assert tvar_95 >= var_95


def test_collective_stop_loss_nonnegative():
    np.random.seed(123)

    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    value = model.stop_loss(2.0, n_sim=100_000)
    assert value >= 0.0


def test_collective_lev_between_zero_and_mean():
    np.random.seed(123)

    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    lev_value = model.limited_expected_value(2.0, n_sim=100_000)

    assert lev_value >= 0.0
    assert lev_value <= model.mean()


def test_collective_frequency_and_severity_mean_helpers():
    freq = Poisson(lam=2.5)
    sev = Exponential(rate=0.5)
    model = CollectiveRiskModel(freq, sev)

    assert np.isclose(model.frequency_mean(), freq.mean())
    assert np.isclose(model.severity_mean(), sev.mean())


def test_collective_summary_keys():
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    summary = model.summary()

    expected_keys = {
        "frequency_model",
        "severity_model",
        "frequency_mean",
        "severity_mean",
        "aggregate_mean",
        "aggregate_variance",
        "aggregate_std",
    }

    assert set(summary.keys()) == expected_keys


def test_collective_summary_values():
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    summary = model.summary()

    assert np.isclose(summary["frequency_mean"], freq.mean())
    assert np.isclose(summary["severity_mean"], sev.mean())
    assert np.isclose(summary["aggregate_mean"], model.mean())
    assert np.isclose(summary["aggregate_variance"], model.variance())
    assert np.isclose(summary["aggregate_std"], model.std())


def test_collective_repr():
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    text = repr(model)

    assert "CollectiveRiskModel" in text
    assert "Poisson" in text
    assert "Exponential" in text


def test_collective_zero_frequency_case():
    """
    Use a Poisson with very small lambda to make sure zero-claim cases are handled.
    """
    np.random.seed(123)

    freq = Poisson(lam=0.01)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)

    samples = model.sample(50_000)

    assert np.all(samples >= 0.0)
    assert np.any(samples == 0.0)


def test_collective_with_layer_severity_runs():
    """
    Integration-style test: severity wrapper should work inside collective model.
    """
    np.random.seed(123)

    freq = Poisson(lam=2.0)
    sev = Lognormal(mu=10.0, sigma=0.8)
    layer = Layer(sev, d=10_000, u=40_000)

    model = CollectiveRiskModel(freq, layer)
    samples = model.sample(20_000)

    assert samples.shape == (20_000,)
    assert np.all(samples >= 0.0)
    assert model.mean() >= 0.0