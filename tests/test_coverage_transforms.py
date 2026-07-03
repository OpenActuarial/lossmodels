"""Per-claim coverage transforms against closed-form LEV identities."""
import numpy as np
import pytest

import lossmodels as lm
from lossmodels.coverage import Layer, OrdinaryDeductible, PolicyLimit

E = lm.Exponential(0.001)  # theta = 1000, LEV(d) = 1000 (1 - e^{-d/1000})


def _lev(d):
    return 1000.0 * (1.0 - np.exp(-d / 1000.0))


def test_layer_u_is_width():
    lay = Layer(E, 100.0, 200.0)
    assert lay.mean() == pytest.approx(_lev(300.0) - _lev(100.0), rel=1e-9)
    assert lay.exhaustion_probability() == pytest.approx(np.exp(-0.3), rel=1e-9)
    assert lay.payment_probability() == pytest.approx(np.exp(-0.1), rel=1e-9)


@pytest.mark.parametrize("d", [50.0, 250.0, 1200.0])
def test_ordinary_deductible_identities(d):
    ded = OrdinaryDeductible(E, d)
    assert ded.mean() == pytest.approx(E.mean() - _lev(d), rel=1e-9)
    assert ded.loss_elimination_ratio() == pytest.approx(_lev(d) / E.mean(), rel=1e-9)
    assert ded.payment_probability() == pytest.approx(np.exp(-d / 1000.0), rel=1e-9)


@pytest.mark.parametrize("u", [300.0, 1500.0])
def test_policy_limit_identities(u):
    cap = PolicyLimit(E, u)
    assert cap.mean() == pytest.approx(_lev(u), rel=1e-9)
    assert cap.probability_capped() == pytest.approx(np.exp(-u / 1000.0), rel=1e-9)


def test_transforms_self_consistent_for_lognormal():
    m = lm.Lognormal(8.0, 1.2)
    d, w = 2000.0, 6000.0
    lay = Layer(m, d, w)
    assert lay.mean() == pytest.approx(
        m.limited_expected_value(d + w) - m.limited_expected_value(d), rel=1e-6
    )
    assert OrdinaryDeductible(m, d).mean() == pytest.approx(m.excess_loss(d), rel=1e-6)
    assert PolicyLimit(m, w).mean() == pytest.approx(m.limited_expected_value(w), rel=1e-6)


def test_mean_decomposes_into_lev_plus_excess():
    for m in (E, lm.Lognormal(8.0, 1.2), lm.ParetoII(3.0, 900.0)):
        for q in (0.3, 0.7, 0.95):
            d = m.quantile(q)
            assert m.limited_expected_value(d) + m.excess_loss(d) == pytest.approx(m.mean(), rel=1e-6)
