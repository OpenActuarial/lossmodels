"""Deterministic coverage variances against exponential closed forms.

For X ~ Exponential(theta) (mean theta):
    Layer Y = min((X-d)+, u):
        E[Y]   = theta e^{-d/theta} (1 - e^{-u/theta})
        E[Y^2] = 2 theta e^{-d/theta} (theta - e^{-u/theta} (theta + u))
    Deductible Y = (X-d)+:  E[Y] = theta e^{-d/theta}, E[Y^2] = 2 theta^2 e^{-d/theta}
    Limit Y = min(X, u):    E[Y] = theta (1 - e^{-u/theta}),
                            E[Y^2] = 2 theta (theta - e^{-u/theta} (theta + u))
"""

import numpy as np

import lossmodels.severity as sev
from lossmodels.coverage import Layer, OrdinaryDeductible, PolicyLimit

THETA, D, U = 300.0, 200.0, 500.0


def _exp_model():
    return sev.Exponential(rate=1.0 / THETA)


def test_layer_variance_closed_form():
    lay = Layer(_exp_model(), d=D, u=U)
    m1 = THETA * np.exp(-D / THETA) * (1 - np.exp(-U / THETA))
    m2 = 2 * THETA * np.exp(-D / THETA) * (THETA - np.exp(-U / THETA) * (THETA + U))
    assert np.isclose(lay.variance(), m2 - m1**2, rtol=1e-7)


def test_deductible_variance_closed_form():
    ded = OrdinaryDeductible(_exp_model(), d=D)
    m1 = THETA * np.exp(-D / THETA)
    m2 = 2 * THETA**2 * np.exp(-D / THETA)
    assert np.isclose(ded.variance(), m2 - m1**2, rtol=1e-7)


def test_limit_variance_closed_form():
    cap = PolicyLimit(_exp_model(), u=U)
    m1 = THETA * (1 - np.exp(-U / THETA))
    m2 = 2 * THETA * (THETA - np.exp(-U / THETA) * (THETA + U))
    assert np.isclose(cap.variance(), m2 - m1**2, rtol=1e-7)


def test_variance_is_deterministic():
    lay = Layer(_exp_model(), d=D, u=U)
    assert lay.variance() == lay.variance()


def test_degenerate_zero_width():
    assert Layer(_exp_model(), d=D, u=0.0).variance() == 0.0
    assert PolicyLimit(_exp_model(), u=0.0).variance() == 0.0
