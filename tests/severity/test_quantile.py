import numpy as np
import pytest
from scipy.stats import lognorm

from lossmodels.severity import Lognormal, Gamma, Pareto, Weibull, Exponential
from lossmodels.severity.base import SeverityModel
from lossmodels.empirical import EmpiricalSeverity

MODELS = [
    Lognormal(mu=9.0, sigma=1.0),
    Gamma(alpha=2.0, theta=5000.0),
    Pareto(alpha=2.5, theta=10000.0),
    Weibull(k=1.5, lam=8000.0),
    Exponential(rate=1.0 / 5000.0),
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("p", [0.05, 0.5, 0.9, 0.99])
def test_quantile_roundtrip(model, p):
    q = model.quantile(p)
    assert model.cdf(q) == pytest.approx(p, abs=1e-6)


@pytest.mark.parametrize("model", MODELS)
def test_ppf_is_quantile_alias(model):
    assert model.ppf(0.5) == pytest.approx(model.quantile(0.5))


@pytest.mark.parametrize("model", MODELS)
def test_quantile_array_monotone(model):
    ps = np.array([0.1, 0.5, 0.9, 0.99])
    qs = model.quantile(ps)
    assert isinstance(qs, np.ndarray)
    assert qs.shape == ps.shape
    assert np.all(np.diff(qs) > 0)


@pytest.mark.parametrize("model", MODELS)
def test_pdf_cdf_scalar_returns_float(model):
    assert isinstance(model.pdf(1000.0), float)
    assert isinstance(model.cdf(1000.0), float)


@pytest.mark.parametrize("model", MODELS)
def test_pdf_cdf_array_vectorize(model):
    x = np.array([500.0, 1000.0, 5000.0, 20000.0])
    assert model.pdf(x).shape == x.shape
    assert model.cdf(x).shape == x.shape
    # array path matches per-element evaluation
    assert model.cdf(x) == pytest.approx(np.array([model.cdf(float(v)) for v in x]))


def test_empirical_quantile_matches_numpy():
    data = np.random.default_rng(0).lognormal(9, 1.0, 2000)
    es = EmpiricalSeverity(data)
    assert es.quantile(0.5) == pytest.approx(float(np.quantile(data, 0.5)))


def test_base_numerical_quantile_fallback():
    # a custom severity that does NOT override quantile -> exercises the base
    # numerical CDF-inversion fallback
    class _Custom(SeverityModel):
        def sample(self, size=1):
            return lognorm.rvs(s=1.0, scale=np.exp(9.0), size=size)

        def mean(self):
            return float(np.exp(9.5))

        def variance(self):
            return 1.0

        def cdf(self, x):
            return float(lognorm.cdf(x, s=1.0, scale=np.exp(9.0)))

    c = _Custom()
    q = c.quantile(0.9)
    assert c.cdf(q) == pytest.approx(0.9, abs=1e-5)
    assert q == pytest.approx(lognorm.ppf(0.9, s=1.0, scale=np.exp(9.0)), rel=1e-4)
