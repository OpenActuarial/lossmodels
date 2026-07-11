import warnings

import numpy as np
import pytest

from lossmodels.estimation import (
    fit_exponential,
    fit_gamma,
    fit_lognormal,
    fit_mle,
    fit_negbinomial,
    fit_pareto,
    fit_poisson,
    fit_weibull,
)
from lossmodels.frequency import NegativeBinomial, Poisson
from lossmodels.severity import Exponential, Gamma, Lognormal, Pareto, Weibull

# ---------------------------
# fit_exponential
# ---------------------------

def test_fit_exponential_returns_model():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    model = fit_exponential(data)

    assert isinstance(model, Exponential)


def test_fit_exponential_matches_closed_form():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    model = fit_exponential(data)

    expected_rate = 1.0 / np.mean(data)
    assert np.isclose(model.rate, expected_rate)


def test_fit_exponential_invalid_data():
    with pytest.raises(ValueError):
        fit_exponential([])

    with pytest.raises(ValueError):
        fit_exponential([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        fit_exponential([1.0, -1.0, 2.0])


# ---------------------------
# fit_lognormal
# ---------------------------

def test_fit_lognormal_returns_model():
    data = np.array([1.0, 2.0, 4.0, 8.0])
    model = fit_lognormal(data)

    assert isinstance(model, Lognormal)


def test_fit_lognormal_matches_closed_form():
    data = np.array([1.0, 2.0, 4.0, 8.0])
    model = fit_lognormal(data)

    log_data = np.log(data)
    expected_mu = np.mean(log_data)
    expected_sigma = np.sqrt(np.mean((log_data - expected_mu) ** 2))

    assert np.isclose(model.mu, expected_mu)
    assert np.isclose(model.sigma, expected_sigma)


def test_fit_lognormal_invalid_data():
    with pytest.raises(ValueError):
        fit_lognormal([])

    with pytest.raises(ValueError):
        fit_lognormal([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        fit_lognormal([1.0, -1.0, 2.0])


# ---------------------------
# fit_negbinomial
# ---------------------------

def test_fit_negbinomial_returns_model():
    np.random.seed(123)
    data = np.random.negative_binomial(n=3, p=0.5, size=1000)

    model = fit_negbinomial(data)

    assert isinstance(model, NegativeBinomial)


def test_fit_negbinomial_recovers_mean_reasonably():
    np.random.seed(123)
    data = np.random.negative_binomial(n=3, p=0.5, size=2000)

    model = fit_negbinomial(data)

    assert np.isclose(model.mean(), np.mean(data), rtol=0.05)


# ---------------------------
# fit_poisson
# ---------------------------

def test_fit_poisson_returns_model():
    data = np.array([0, 1, 2, 3, 4])
    model = fit_poisson(data)

    assert isinstance(model, Poisson)


def test_fit_poisson_matches_closed_form():
    data = np.array([0, 1, 2, 3, 4])
    model = fit_poisson(data)

    expected_lam = np.mean(data)
    assert np.isclose(model.lam, expected_lam)


def test_fit_poisson_invalid_data():
    with pytest.raises(ValueError):
        fit_poisson([])

    with pytest.raises(ValueError):
        fit_poisson([0, 1, -1, 2])

    with pytest.raises(ValueError):
        fit_poisson([0, 1.5, 2])


def test_fit_poisson_zero_mean_returns_zero_lambda():
    model = fit_poisson([0, 0, 0])
    assert model.lam == 0.0


# ---------------------------
# fit_gamma
# ---------------------------

def test_fit_gamma_returns_model():
    np.random.seed(123)
    data = np.random.gamma(shape=2.0, scale=3.0, size=1000)

    model = fit_gamma(data)

    assert isinstance(model, Gamma)


def test_fit_gamma_recovers_mean_reasonably():
    np.random.seed(123)
    data = np.random.gamma(shape=2.0, scale=3.0, size=2000)

    model = fit_gamma(data)

    assert np.isclose(model.mean(), np.mean(data), rtol=0.05)


def test_fit_gamma_invalid_data():
    with pytest.raises(ValueError):
        fit_gamma([])

    with pytest.raises(ValueError):
        fit_gamma([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        fit_gamma([1.0, -1.0, 2.0])


# ---------------------------
# fit_pareto
# ---------------------------

def test_fit_pareto_returns_model():
    np.random.seed(123)
    data = 2.0 * (1.0 + np.random.pareto(a=3.0, size=1000))

    model = fit_pareto(data)

    assert isinstance(model, Pareto)


def test_fit_pareto_matches_closed_form_mle():
    data = np.array([2.0, 3.0, 4.0, 8.0], dtype=float)

    model = fit_pareto(data)

    expected_theta = np.min(data)
    expected_alpha = len(data) / np.sum(np.log(data / expected_theta))

    assert np.isclose(model.theta, expected_theta)
    assert np.isclose(model.alpha, expected_alpha)


def test_fit_pareto_invalid_data():
    with pytest.raises(ValueError):
        fit_pareto([])

    with pytest.raises(ValueError):
        fit_pareto([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        fit_pareto([1.0, -1.0, 2.0])


def test_fit_pareto_all_equal_data_raises():
    with pytest.raises(ValueError):
        fit_pareto([2.0, 2.0, 2.0, 2.0])

# ---------------------------
# fit_weibull
# ---------------------------

def test_fit_weibull_returns_model():
    np.random.seed(123)
    data = 2.0 * np.random.weibull(a=1.5, size=1000)

    model = fit_weibull(data)

    assert isinstance(model, Weibull)


def test_fit_weibull_recovers_mean_reasonably():
    np.random.seed(123)
    data = 2.0 * np.random.weibull(a=1.5, size=2000)

    model = fit_weibull(data)

    assert np.isclose(model.mean(), np.mean(data), rtol=0.05)


def test_fit_weibull_invalid_data():
    with pytest.raises(ValueError):
        fit_weibull([])

    with pytest.raises(ValueError):
        fit_weibull([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        fit_weibull([1.0, -1.0, 2.0])


# ---------------------------
# fit_mle (generic numerical MLE)
# ---------------------------

def test_fit_mle_exponential_returns_model():
    np.random.seed(123)
    data = np.random.exponential(scale=2.0, size=5000)  # true rate = 0.5

    model = fit_mle(
        Exponential,
        data,
        initial_params=[1.0],
        bounds=[(1e-8, None)],
    )

    assert isinstance(model, Exponential)


def test_fit_mle_exponential_recovers_rate_reasonably():
    np.random.seed(123)
    data = np.random.exponential(scale=2.0, size=2000)  # true rate = 0.5

    model = fit_mle(
        Exponential,
        data,
        initial_params=[1.0],
        bounds=[(1e-8, None)],
    )

    assert np.isclose(model.rate, 0.5, rtol=0.1)


def test_fit_mle_lognormal_returns_model():
    np.random.seed(123)
    data = np.random.lognormal(mean=1.0, sigma=0.5, size=1000)

    model = fit_mle(
        Lognormal,
        data,
        initial_params=[0.8, 0.8],
        bounds=[(None, None), (1e-8, None)],
    )

    assert isinstance(model, Lognormal)


def test_fit_mle_lognormal_recovers_parameters_reasonably():
    np.random.seed(123)
    true_mu = 1.0
    true_sigma = 0.5
    data = np.random.lognormal(mean=true_mu, sigma=true_sigma, size=2000)

    model = fit_mle(
        Lognormal,
        data,
        initial_params=[0.8, 0.8],
        bounds=[(None, None), (1e-8, None)],
    )

    assert np.isclose(model.mu, true_mu, rtol=0.1)
    assert np.isclose(model.sigma, true_sigma, rtol=0.15)


def test_fit_mle_invalid_data():
    with pytest.raises(ValueError):
        fit_mle(Exponential, [], initial_params=[1.0])

    with pytest.raises(ValueError):
        fit_mle(Exponential, [1.0, 0.0, 2.0], initial_params=[1.0])


def test_fit_mle_invalid_initial_params():
    data = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        fit_mle(Exponential, data, initial_params=[])


def test_fit_mle_bad_model_fails():
    class BadModel:
        def __init__(self, x):
            self.x = x

        def pdf(self, data):
            return np.zeros_like(data, dtype=float)

    data = np.array([1.0, 2.0, 3.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError):
            fit_mle(BadModel, data, initial_params=[1.0])