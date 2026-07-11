"""Tests for the FAM / ASTAM continuous severity inventory (added in 0.5.0).

Moment expectations are computed independently from the Loss Models closed-form
E[X^k] tables (via math.gamma), so they check the public ``mean``/``variance``
against the exam tables rather than against the implementation's internals.
"""

from math import gamma as G

import numpy as np
import pytest

from lossmodels.severity import (
    Beta,
    Burr,
    GeneralizedBeta,
    GeneralizedPareto,
    InverseBurr,
    InverseExponential,
    InverseGamma,
    InverseGaussian,
    InverseParalogistic,
    InversePareto,
    InverseWeibull,
    Loglogistic,
    LogT,
    Paralogistic,
    ParetoII,
    SingleParameterPareto,
)

# (model, x0, mean, E[X^2])  -- mean/E2 None where the moment does not exist.
# Supports are (0, inf) unless noted in SUPPORT_LO.
MOMENT_CASES = [
    ("Burr", Burr(3, 1000, 2), 1234.0,
     1000 * G(1.5) * G(2.5) / G(3), 1000**2 * G(2) * G(2) / G(3)),
    ("InverseBurr", InverseBurr(4, 1000, 3), 1234.0,
     1000 * G(4 + 1 / 3) * G(1 - 1 / 3) / G(4),
     1000**2 * G(4 + 2 / 3) * G(1 - 2 / 3) / G(4)),
    ("GeneralizedPareto", GeneralizedPareto(4, 1000, 3), 1234.0,
     1000 * G(4) * G(3) / (G(4) * G(3)), 1000**2 * G(5) * G(2) / (G(4) * G(3))),
    ("ParetoII", ParetoII(3, 1000), 1234.0, 1000 / 2.0, 1000**2 * 2 / (2 * 1)),
    ("Loglogistic", Loglogistic(3, 1000), 1234.0,
     1000 * G(1 + 1 / 3) * G(1 - 1 / 3), 1000**2 * G(1 + 2 / 3) * G(1 - 2 / 3)),
    ("Paralogistic", Paralogistic(3, 1000), 1234.0,
     1000 * G(1 + 1 / 3) * G(3 - 1 / 3) / G(3),
     1000**2 * G(1 + 2 / 3) * G(3 - 2 / 3) / G(3)),
    ("InverseParalogistic", InverseParalogistic(3, 1000), 1234.0,
     1000 * G(3 + 1 / 3) * G(1 - 1 / 3) / G(3),
     1000**2 * G(3 + 2 / 3) * G(1 - 2 / 3) / G(3)),
    ("InverseGamma", InverseGamma(4, 1000), 1234.0,
     1000 * G(3) / G(4), 1000**2 * G(2) / G(4)),
    ("InverseWeibull", InverseWeibull(1000, 4), 1234.0,
     1000 * G(1 - 1 / 4), 1000**2 * G(1 - 2 / 4)),
    ("InverseGaussian", InverseGaussian(500, 2000), 450.0,
     500.0, 500.0**3 / 2000 + 500.0**2),
    ("SingleParameterPareto", SingleParameterPareto(3, 1000), 2500.0,
     3 * 1000 / 2.0, 3 * 1000**2 / 1.0),
    ("Beta", Beta(2, 3, 1000), 400.0,
     1000 * 2 / 5.0, 1000**2 * G(5) * G(4) / (G(2) * G(7))),
    ("GeneralizedBeta", GeneralizedBeta(2, 3, 1000, 2), 500.0,
     1000 * G(5) * G(2.5) / (G(2) * G(5.5)),
     1000**2 * G(5) * G(3) / (G(2) * G(6))),
    # no-moment distributions: mean/E2 = None
    ("InversePareto", InversePareto(3, 1000), 1234.0, None, None),
    ("InverseExponential", InverseExponential(1000), 1234.0, None, None),
    ("LogT", LogT(5, 0, 1), 3.0, None, None),
]

SUPPORT_LO = {"SingleParameterPareto": 1000.0}  # Type I lower bound


@pytest.mark.parametrize("name,model,x0,mean,e2", MOMENT_CASES)
def test_mean_matches_table(name, model, x0, mean, e2):
    if mean is None:
        with pytest.raises(ValueError):
            model.mean()
    else:
        assert np.isclose(model.mean(), mean, rtol=1e-6)


@pytest.mark.parametrize("name,model,x0,mean,e2", MOMENT_CASES)
def test_variance_matches_table(name, model, x0, mean, e2):
    if e2 is None:
        with pytest.raises(ValueError):
            model.variance()
    else:
        assert np.isclose(model.variance(), e2 - mean**2, rtol=1e-6)


@pytest.mark.parametrize("name,model,x0,mean,e2", MOMENT_CASES)
def test_cdf_quantile_roundtrip(name, model, x0, mean, e2):
    assert np.isclose(model.quantile(model.cdf(x0)), x0, rtol=1e-4)


@pytest.mark.parametrize("name,model,x0,mean,e2", MOMENT_CASES)
def test_cdf_spans_zero_to_one(name, model, x0, mean, e2):
    lo = SUPPORT_LO.get(name, 0.0)
    assert float(model.cdf(lo)) < 1e-6 or lo > 0  # at/below support cdf ~ 0
    assert np.isclose(float(model.cdf(1e13)), 1.0, atol=1e-6)


@pytest.mark.parametrize("name,model,x0,mean,e2", MOMENT_CASES)
def test_scalar_matches_array(name, model, x0, mean, e2):
    pair = model.pdf(np.array([x0, x0 * 1.5]))
    assert isinstance(pair, np.ndarray) and pair.shape == (2,)
    assert np.isclose(float(model.pdf(x0)), pair[0], rtol=1e-9)
    assert isinstance(model.pdf(x0), float)


@pytest.mark.parametrize("name,model,x0,mean,e2", MOMENT_CASES)
def test_sample_shape_and_support(name, model, x0, mean, e2):
    lo = SUPPORT_LO.get(name, 0.0)
    s = model.sample(3000)
    assert isinstance(s, np.ndarray) and s.shape == (3000,)
    assert np.all(s >= lo - 1e-9)


def test_sample_mean_close_for_lighttailed():
    np.random.seed(7)
    s = ParetoII(3, 1000).sample(300_000)
    assert np.isclose(s.mean(), 500.0, rtol=0.05)


@pytest.mark.parametrize(
    "ctor,args",
    [
        (Burr, (3, 1000, 2)),
        (InverseBurr, (4, 1000, 3)),
        (GeneralizedPareto, (4, 1000, 3)),
        (ParetoII, (3, 1000)),
        (InversePareto, (3, 1000)),
        (Loglogistic, (3, 1000)),
        (Paralogistic, (3, 1000)),
        (InverseParalogistic, (3, 1000)),
        (InverseGamma, (4, 1000)),
        (InverseWeibull, (1000, 4)),
        (InverseExponential, (1000,)),
        (InverseGaussian, (500, 2000)),
        (SingleParameterPareto, (3, 1000)),
        (Beta, (2, 3, 1000)),
        (GeneralizedBeta, (2, 3, 1000, 2)),
    ],
)
def test_rejects_nonpositive_params(ctor, args):
    for i in range(len(args)):
        bad = list(args)
        bad[i] = -1.0
        with pytest.raises(ValueError):
            ctor(*bad)


def test_logt_rejects_bad_params():
    with pytest.raises(ValueError):
        LogT(-1, 0, 1)
    with pytest.raises(ValueError):
        LogT(5, 0, -1)
    LogT(5, -2.0, 1)  # negative mu is allowed


def test_paretoII_is_not_type_one_pareto():
    """FAM 'Pareto' (Type II/Lomax) has support x > 0 with positive density at 0;
    the existing lossmodels.Pareto (Type I) has zero density below theta."""
    from lossmodels import Pareto  # Type I

    p2 = ParetoII(3, 1000)
    p1 = Pareto(3, 1000)
    assert p2.pdf(500.0) > 0.0  # Type II supported on (0, inf)
    assert p1.pdf(500.0) == 0.0  # Type I supported on [theta, inf)
    assert np.isclose(p1.pdf(2000.0), SingleParameterPareto(3, 1000).pdf(2000.0))


def test_repr_contains_params():
    assert "Burr" in repr(Burr(3, 1000, 2))
    assert "ParetoII" in repr(ParetoII(3, 1000))
    assert "tau=" in repr(InverseWeibull(1000, 4))
