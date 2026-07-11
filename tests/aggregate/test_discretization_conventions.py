"""Regression tests for the discretization conventions.

The "upper" method rounds losses down (cdf upper bound), "lower" rounds losses
up (cdf lower bound): their means must bracket the true mean, with "midpoint"
in between. A prior bug set bucket 0 of the "lower" method to F(h) instead of
F(0), double-counting [0, h] and biasing the rounded-up scheme *below* the true
mean.
"""
import numpy as np
import pytest

from lossmodels.aggregate import discretize_severity, mean_from_pmf
from lossmodels.severity import Exponential

H, MAX_LOSS = 10.0, 4000.0
SEV = Exponential(0.01)  # mean 100; tail mass beyond MAX_LOSS is ~4e-18


def _mean(method: str) -> float:
    pmf = discretize_severity(SEV, h=H, max_loss=MAX_LOSS, method=method)
    assert pmf.sum() == pytest.approx(1.0)
    return mean_from_pmf(pmf, H)


def test_lower_bucket_zero_carries_only_the_atom_at_zero():
    pmf = discretize_severity(SEV, h=H, max_loss=MAX_LOSS, method="lower")
    assert pmf[0] == pytest.approx(float(SEV.cdf(0.0)), abs=1e-15)
    # bucket 1 carries the full mass of (0, h]
    assert pmf[1] == pytest.approx(float(SEV.cdf(H)), rel=1e-12)


def test_upper_and_lower_bracket_the_true_mean():
    m_up, m_mid, m_low = _mean("upper"), _mean("midpoint"), _mean("lower")
    true_mean = SEV.mean()
    assert m_up < true_mean < m_low
    assert m_up < m_mid < m_low
    # each scheme is off by at most ~h/2 vs the exact rounded means
    assert m_low - m_up == pytest.approx(H, rel=1e-6)


def test_exact_reference_means_exponential_lattice():
    # Closed forms for Exponential(rate) on lattice h (tail beyond MAX_LOSS
    # negligible): rounded-down mean = h/(e^{rh}-1); rounded-up = h/(1-e^{-rh}).
    r = SEV.rate
    assert _mean("upper") == pytest.approx(H / (np.exp(r * H) - 1.0), rel=1e-9)
    assert _mean("lower") == pytest.approx(H / (1.0 - np.exp(-r * H)), rel=1e-9)
