"""Distribution-level identities every severity must satisfy."""
import numpy as np
import pytest

import lossmodels as lm

ZOO = [
    lm.Exponential(0.002),
    lm.Gamma(2.5, 1000.0),
    lm.Lognormal(8.0, 1.2),
    lm.Weibull(1.7, 900.0),
    lm.Pareto(3.0, 100.0),
    lm.ParetoII(3.0, 100.0),
    lm.Loglogistic(2.2, 800.0),
    lm.Burr(2.0, 1000.0, 1.5),
    lm.GeneralizedPareto(3.0, 1000.0, 2.0),
    lm.InverseGamma(2.5, 300.0),
    lm.SingleParameterPareto(3.0, 100.0),
]
IDS = [type(m).__name__ for m in ZOO]
QG = [0.05, 0.25, 0.5, 0.75, 0.95, 0.995]


@pytest.mark.parametrize("m", ZOO, ids=IDS)
def test_cdf_quantile_round_trip(m):
    for q in QG:
        assert np.isclose(m.cdf(m.quantile(q)), q, atol=1e-9)


@pytest.mark.parametrize("m", ZOO, ids=IDS)
def test_cdf_monotone_and_bounded(m):
    grid = np.sort(np.concatenate([[m.quantile(q) for q in QG], [1e-6, 1e7]]))
    c = m.cdf(grid)
    assert np.all(np.diff(c) >= -1e-12)
    assert np.all((c >= 0) & (c <= 1))


@pytest.mark.parametrize("m", ZOO, ids=IDS)
def test_pdf_is_cdf_derivative(m):
    for q in (0.25, 0.5, 0.9):
        x = m.quantile(q)
        h = 1e-4 * x
        numeric = (m.cdf(x + h) - m.cdf(x - h)) / (2 * h)
        assert np.isclose(m.pdf(x), numeric, rtol=5e-4)


@pytest.mark.parametrize("m", ZOO, ids=IDS)
def test_lev_properties(m):
    grid = [m.quantile(q) for q in (0.25, 0.5, 0.9, 0.99)]
    levs = [m.limited_expected_value(d) for d in grid]
    assert all(b >= a - 1e-9 for a, b in zip(levs, levs[1:]))  # nondecreasing
    assert m.limited_expected_value(m.quantile(0.999999) * 50) == pytest.approx(m.mean(), rel=1e-3)


@pytest.mark.parametrize("m", ZOO, ids=IDS)
def test_sampling_support_and_reproducibility(m):
    a = m.sample(400, rng=7)
    assert np.array_equal(a, m.sample(400, rng=7))
    assert a.min() >= 0.0
    assert len(a) == 400


@pytest.mark.parametrize("model,attr", [
    (lm.Pareto(0.8, 100.0), "mean"),
    (lm.ParetoII(1.5, 100.0), "variance"),
    (lm.InverseGamma(1.5, 300.0), "variance"),
    (lm.Loglogistic(0.9, 800.0), "mean"),
    (lm.Burr(1.0, 1000.0, 0.8), "mean"),
], ids=["ParetoI-mean", "Lomax-var", "InvGamma-var", "Loglogistic-mean", "Burr-mean"])
def test_moments_raise_outside_existence(model, attr):
    with pytest.raises(ValueError):
        getattr(model, attr)()
