import numpy as np
import pytest

from lossmodels.estimation import fit_exponential, fit_lognormal, fit_poisson
from lossmodels.estimation.diagnostics import aic, bic, log_likelihood
from lossmodels.frequency import Poisson
from lossmodels.severity import Exponential

# ---------------------------
# log_likelihood
# ---------------------------

def test_log_likelihood_exponential_finite():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    model = Exponential(rate=1.0)

    ll = log_likelihood(model, data)

    assert np.isfinite(ll)
    assert isinstance(ll, float)


def test_log_likelihood_poisson_finite():
    data = np.array([0, 1, 2, 3, 1, 0])
    model = Poisson(lam=1.5)

    ll = log_likelihood(model, data)

    assert np.isfinite(ll)
    assert isinstance(ll, float)


def test_log_likelihood_empty_data():
    model = Exponential(rate=1.0)

    with pytest.raises(ValueError):
        log_likelihood(model, [])


def test_log_likelihood_bad_model_type():
    class BadModel:
        pass

    with pytest.raises(TypeError):
        log_likelihood(BadModel(), [1.0, 2.0, 3.0])


def test_log_likelihood_returns_negative_inf_for_zero_probability():
    data = np.array([1.0, 2.0, 3.0])

    class BadPdfModel:
        def pdf(self, x):
            return 0.0

    ll = log_likelihood(BadPdfModel(), data)
    assert ll == float("-inf")


def test_log_likelihood_prefers_better_exponential_fit():
    np.random.seed(123)
    data = np.random.exponential(scale=2.0, size=5000)  # true rate = 0.5

    fitted = fit_exponential(data)
    bad = Exponential(rate=1.5)

    ll_fitted = log_likelihood(fitted, data)
    ll_bad = log_likelihood(bad, data)

    assert ll_fitted > ll_bad


def test_log_likelihood_prefers_better_poisson_fit():
    np.random.seed(123)
    data = np.random.poisson(lam=2.0, size=5000)

    fitted = fit_poisson(data)
    bad = Poisson(lam=5.0)

    ll_fitted = log_likelihood(fitted, data)
    ll_bad = log_likelihood(bad, data)

    assert ll_fitted > ll_bad


# ---------------------------
# AIC
# ---------------------------

def test_aic_basic():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    model = Exponential(rate=1.0)

    ll = log_likelihood(model, data)
    expected = 2 * 1 - 2 * ll

    assert np.isclose(aic(model, data, k=1), expected)


def test_aic_invalid_k():
    data = np.array([1.0, 2.0, 3.0])
    model = Exponential(rate=1.0)

    with pytest.raises(ValueError):
        aic(model, data, k=0)

    with pytest.raises(ValueError):
        aic(model, data, k=-1)


def test_aic_infinite_when_loglikelihood_is_negative_infinity():
    data = np.array([1.0, 2.0, 3.0])

    class BadPdfModel:
        def pdf(self, x):
            return 0.0

    value = aic(BadPdfModel(), data, k=1)
    assert value == float("inf")


def test_aic_prefers_better_exponential_fit():
    np.random.seed(123)
    data = np.random.exponential(scale=2.0, size=5000)

    fitted = fit_exponential(data)
    bad = Exponential(rate=1.5)

    aic_fitted = aic(fitted, data, k=1)
    aic_bad = aic(bad, data, k=1)

    assert aic_fitted < aic_bad


# ---------------------------
# BIC
# ---------------------------

def test_bic_basic():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    model = Exponential(rate=1.0)

    ll = log_likelihood(model, data)
    expected = np.log(len(data)) * 1 - 2 * ll

    assert np.isclose(bic(model, data, k=1), expected)


def test_bic_invalid_inputs():
    model = Exponential(rate=1.0)

    with pytest.raises(ValueError):
        bic(model, [], k=1)

    with pytest.raises(ValueError):
        bic(model, [1.0, 2.0, 3.0], k=0)

    with pytest.raises(ValueError):
        bic(model, [1.0, 2.0, 3.0], k=-1)


def test_bic_infinite_when_loglikelihood_is_negative_infinity():
    data = np.array([1.0, 2.0, 3.0])

    class BadPdfModel:
        def pdf(self, x):
            return 0.0

    value = bic(BadPdfModel(), data, k=1)
    assert value == float("inf")


def test_bic_prefers_better_lognormal_fit():
    np.random.seed(123)
    data = np.random.lognormal(mean=1.0, sigma=0.5, size=5000)

    fitted = fit_lognormal(data)

    # Poor comparison model
    bad = fit_exponential(data)

    bic_fitted = bic(fitted, data, k=2)
    bic_bad = bic(bad, data, k=1)

    assert bic_fitted < bic_bad