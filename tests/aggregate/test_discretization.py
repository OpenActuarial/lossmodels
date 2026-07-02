import numpy as np
import pytest

from lossmodels.aggregate.discretization import (
    bucket_representatives,
    discretize_severity,
    mean_from_discretized_pmf,
)
from lossmodels.coverage import Layer, OrdinaryDeductible, PolicyLimit
from lossmodels.severity import Exponential, Lognormal


def test_bucket_representatives():
    reps = bucket_representatives(h=2.0, size=4)
    assert np.allclose(reps, [0.0, 2.0, 4.0, 6.0])


def test_bucket_representatives_invalid():
    with pytest.raises(ValueError):
        bucket_representatives(h=0.0, size=4)
    with pytest.raises(ValueError):
        bucket_representatives(h=1.0, size=0)


def test_discretize_severity_basic_properties():
    sev = Exponential(rate=1.0)
    pmf = discretize_severity(sev, h=0.5, max_loss=10.0)
    assert isinstance(pmf, np.ndarray)
    assert pmf.ndim == 1
    assert len(pmf) == 21
    assert np.all(pmf >= 0.0)
    assert np.isclose(pmf.sum(), 1.0)


def test_discretize_severity_all_methods_return_valid_pmf():
    sev = Exponential(rate=1.0)
    for method in ["upper", "lower", "midpoint"]:
        pmf = discretize_severity(sev, h=0.5, max_loss=10.0, method=method)
        assert np.all(pmf >= 0.0)
        assert np.isclose(pmf.sum(), 1.0)


def test_discretize_severity_invalid_inputs():
    sev = Exponential(rate=1.0)
    with pytest.raises(ValueError):
        discretize_severity(sev, h=0.0, max_loss=10.0)
    with pytest.raises(ValueError):
        discretize_severity(sev, h=1.0, max_loss=0.0)
    with pytest.raises(ValueError):
        discretize_severity(sev, h=2.0, max_loss=1.0)
    with pytest.raises(ValueError):
        discretize_severity(sev, h=1.0, max_loss=10.0, method="bad")


def test_discretize_severity_requires_cdf():
    class BadSeverity:
        pass

    with pytest.raises(TypeError):
        discretize_severity(BadSeverity(), h=1.0, max_loss=10.0)


def test_discretize_severity_upper_includes_mass_at_zero():
    sev = OrdinaryDeductible(Exponential(rate=1.0), d=1.0)
    pmf = discretize_severity(sev, h=0.1, max_loss=5.0, method="upper")

    expected_mass_at_zero = sev.cdf(0.0)
    assert np.isclose(pmf[0], expected_mass_at_zero + (sev.cdf(0.1) - sev.cdf(0.0)))
    assert pmf[0] >= expected_mass_at_zero
    assert np.isclose(pmf.sum(), 1.0)


def test_discretize_severity_works_for_coverage_transforms():
    models = [
        OrdinaryDeductible(Exponential(rate=1.0), d=1.0),
        PolicyLimit(Exponential(rate=1.0), u=2.0),
        Layer(Exponential(rate=1.0), d=1.0, u=2.0),
    ]
    for sev in models:
        pmf = discretize_severity(sev, h=0.1, max_loss=5.0)
        assert np.all(pmf >= 0.0)
        assert np.isclose(pmf.sum(), 1.0)


def test_mean_from_discretized_pmf_close_for_exponential_midpoint():
    sev = Exponential(rate=1.0)
    pmf = discretize_severity(sev, h=0.1, max_loss=10.0, method="midpoint")
    approx_mean = mean_from_discretized_pmf(pmf, h=0.1)
    assert np.isclose(approx_mean, sev.mean(), rtol=0.05)


def test_mean_from_discretized_pmf_close_for_lognormal_midpoint():
    sev = Lognormal(mu=1.0, sigma=0.5)
    pmf = discretize_severity(sev, h=0.25, max_loss=20.0, method="midpoint")
    approx_mean = mean_from_discretized_pmf(pmf, h=0.25)
    assert np.isclose(approx_mean, sev.mean(), rtol=0.10)


def test_mean_from_discretized_pmf_invalid():
    with pytest.raises(ValueError):
        mean_from_discretized_pmf([], h=1.0)
    with pytest.raises(ValueError):
        mean_from_discretized_pmf([0.5, 0.5], h=0.0)
    with pytest.raises(ValueError):
        mean_from_discretized_pmf([-0.1, 1.1], h=1.0)

def test_default_method_is_midpoint():
    from lossmodels.severity import Exponential

    model = Exponential(rate=1.0)
    default = discretize_severity(model, h=0.05, max_loss=40.0)
    midpoint = discretize_severity(model, h=0.05, max_loss=40.0, method="midpoint")
    np.testing.assert_allclose(default, midpoint)
    # midpoint mean is nearly unbiased
    assert abs(mean_from_discretized_pmf(default, 0.05) - 1.0) < 5e-4


def test_method_means_bracket_true_mean():
    from lossmodels.severity import Exponential

    model = Exponential(rate=1.0)
    h, m = 0.05, 40.0
    mu = {
        meth: mean_from_discretized_pmf(
            discretize_severity(model, h=h, max_loss=m, method=meth), h
        )
        for meth in ("upper", "midpoint", "lower")
    }
    assert mu["upper"] <= mu["midpoint"] <= mu["lower"]
    assert mu["upper"] <= 1.0 <= mu["lower"]
