import numpy as np
import pytest

from lossmodels.estimation import (
    fit_exponential_moments,
    fit_gamma_moments,
    fit_lognormal_moments,
    fit_negbinomial_moments,
    fit_poisson_moments,
    fit_weibull_moments,
)
from lossmodels.frequency import NegativeBinomial, Poisson
from lossmodels.severity import Exponential, Gamma, Lognormal, Weibull

# ---------------------------
# Negative Binomial
# ---------------------------

def test_fit_negbinomial_moments_returns_model():
    np.random.seed(123)
    data = np.random.negative_binomial(n=3, p=0.5, size=1000)

    model = fit_negbinomial_moments(data)

    assert isinstance(model, NegativeBinomial)


def test_fit_negbinomial_moments_recovers_mean_reasonably():
    np.random.seed(123)
    data = np.random.negative_binomial(n=3, p=0.5, size=2000)

    model = fit_negbinomial_moments(data)

    assert np.isclose(model.mean(), np.mean(data), rtol=0.05)


def test_fit_negbinomial_moments_requires_overdispersion():
    data = np.array([0, 1, 2, 1, 2, 1, 0, 1])

    with pytest.raises(ValueError):
        fit_negbinomial_moments(data)

# ---------------------------
# Poisson
# ---------------------------

def test_fit_poisson_moments_returns_model():
    data = np.array([0, 1, 2, 3, 4])
    model = fit_poisson_moments(data)

    assert isinstance(model, Poisson)


def test_fit_poisson_moments_matches_sample_mean():
    data = np.array([0, 1, 2, 3, 4])
    model = fit_poisson_moments(data)

    assert np.isclose(model.lam, np.mean(data))


def test_fit_poisson_moments_invalid_data():
    with pytest.raises(ValueError):
        fit_poisson_moments([])

    with pytest.raises(ValueError):
        fit_poisson_moments([0, 1, -1, 2])

    with pytest.raises(ValueError):
        fit_poisson_moments([0, 1.5, 2])


def test_fit_poisson_moments_zero_mean_returns_zero_lambda():
    model = fit_poisson_moments([0, 0, 0])
    assert model.lam == 0.0


# ---------------------------
# Exponential
# ---------------------------

def test_fit_exponential_moments_returns_model():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    model = fit_exponential_moments(data)

    assert isinstance(model, Exponential)


def test_fit_exponential_moments_matches_formula():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    model = fit_exponential_moments(data)

    expected_rate = 1.0 / np.mean(data)
    assert np.isclose(model.rate, expected_rate)


def test_fit_exponential_moments_invalid_data():
    with pytest.raises(ValueError):
        fit_exponential_moments([])

    with pytest.raises(ValueError):
        fit_exponential_moments([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        fit_exponential_moments([1.0, -1.0, 2.0])


# ---------------------------
# Gamma
# ---------------------------

def test_fit_gamma_moments_returns_model():
    np.random.seed(123)
    data = np.random.gamma(shape=2.0, scale=3.0, size=5000)

    model = fit_gamma_moments(data)

    assert isinstance(model, Gamma)


def test_fit_gamma_moments_recovers_mean_reasonably():
    np.random.seed(123)
    data = np.random.gamma(shape=2.0, scale=3.0, size=2000)

    model = fit_gamma_moments(data)

    assert np.isclose(model.mean(), np.mean(data), rtol=0.05)


def test_fit_gamma_moments_recovers_variance_reasonably():
    np.random.seed(123)
    data = np.random.gamma(shape=2.0, scale=3.0, size=2000)

    model = fit_gamma_moments(data)

    assert np.isclose(model.variance(), np.var(data), rtol=0.10)


def test_fit_gamma_moments_invalid_data():
    with pytest.raises(ValueError):
        fit_gamma_moments([])

    with pytest.raises(ValueError):
        fit_gamma_moments([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        fit_gamma_moments([1.0, -1.0, 2.0])


def test_fit_gamma_moments_zero_variance_rejected():
    with pytest.raises(ValueError):
        fit_gamma_moments([2.0, 2.0, 2.0])


# ---------------------------
# Lognormal
# ---------------------------

def test_fit_lognormal_moments_returns_model():
    np.random.seed(123)
    data = np.random.lognormal(mean=1.0, sigma=0.5, size=1000)

    model = fit_lognormal_moments(data)

    assert isinstance(model, Lognormal)


def test_fit_lognormal_moments_recovers_mean_reasonably():
    np.random.seed(123)
    data = np.random.lognormal(mean=1.0, sigma=0.5, size=2000)

    model = fit_lognormal_moments(data)

    assert np.isclose(model.mean(), np.mean(data), rtol=0.05)


def test_fit_lognormal_moments_recovers_variance_reasonably():
    np.random.seed(123)
    data = np.random.lognormal(mean=1.0, sigma=0.5, size=2000)

    model = fit_lognormal_moments(data)

    assert np.isclose(model.variance(), np.var(data), rtol=0.10)


def test_fit_lognormal_moments_invalid_data():
    with pytest.raises(ValueError):
        fit_lognormal_moments([])

    with pytest.raises(ValueError):
        fit_lognormal_moments([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        fit_lognormal_moments([1.0, -1.0, 2.0])


def test_fit_lognormal_moments_zero_variance_case():
    with pytest.raises(ValueError):
        fit_lognormal_moments([2.0, 2.0, 2.0])


# ---------------------------
# Weibull
# ---------------------------

def test_fit_weibull_moments_returns_model():
    np.random.seed(123)
    data = 2.0 * np.random.weibull(a=1.5, size=1000)

    model = fit_weibull_moments(data)

    assert isinstance(model, Weibull)


def test_fit_weibull_moments_recovers_mean_reasonably():
    np.random.seed(123)
    data = 2.0 * np.random.weibull(a=1.5, size=2000)

    model = fit_weibull_moments(data)

    assert np.isclose(model.mean(), np.mean(data), rtol=0.05)


def test_fit_weibull_moments_recovers_variance_reasonably():
    np.random.seed(123)
    data = 2.0 * np.random.weibull(a=1.5, size=2000)

    model = fit_weibull_moments(data)

    assert np.isclose(model.variance(), np.var(data), rtol=0.15)


def test_fit_weibull_moments_invalid_data():
    with pytest.raises(ValueError):
        fit_weibull_moments([])

    with pytest.raises(ValueError):
        fit_weibull_moments([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        fit_weibull_moments([1.0, -1.0, 2.0])


def test_fit_weibull_moments_zero_variance_rejected():
    with pytest.raises(ValueError):
        fit_weibull_moments([2.0, 2.0, 2.0])