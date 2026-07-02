import numpy as np
import pytest

from lossmodels.aggregate.risk_measures_pmf import (
    mean_from_pmf,
    stop_loss_from_pmf,
    tvar_from_pmf,
    var_from_pmf,
)


def test_mean_from_pmf():
    pmf = np.array([0.5, 0.5])
    mean = mean_from_pmf(pmf, h=2.0)
    assert mean == 1.0


def test_var_from_pmf():
    pmf = np.array([0.2, 0.3, 0.5])
    var = var_from_pmf(pmf, h=1.0, q=0.6)
    assert var == 2.0


def test_tvar_from_pmf_uses_tail_at_or_above_var():
    pmf = np.array([0.2, 0.3, 0.5])
    tvar = tvar_from_pmf(pmf, h=1.0, q=0.6)
    expected = (2.0 * 0.5) / 0.5
    assert np.isclose(tvar, expected)


def test_tvar_from_pmf_matches_average_quantile_definition_with_mass_at_var():
    pmf = np.array([0.2, 0.3, 0.4, 0.1])
    q = 0.5
    # VaR = 1 with F(1) = 0.5 = q, so the atom at VaR carries zero weight:
    # TVaR = [sum_{x>1} x p(x) + 1 * (F(1) - q)] / (1 - q)
    #      = (2*0.4 + 3*0.1) / 0.5 = 2.2.
    expected = (2.0 * 0.4 + 3.0 * 0.1) / 0.5
    assert np.isclose(tvar_from_pmf(pmf, h=1.0, q=q), expected)


def test_stop_loss_from_pmf():
    pmf = np.array([0.2, 0.3, 0.5])
    sl = stop_loss_from_pmf(pmf, h=1.0, d=1.0)
    assert sl > 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        var_from_pmf([], h=1.0, q=0.5)
    with pytest.raises(ValueError):
        var_from_pmf([0.5, 0.5], h=0.0, q=0.5)
    with pytest.raises(ValueError):
        var_from_pmf([0.5, 0.5], h=1.0, q=1.5)