"""Kaplan-Meier hand cases and MLE recovery under censoring and truncation."""
import numpy as np
import pytest

import lossmodels as lm


def test_kaplan_meier_hand_case():
    times, surv = lm.kaplan_meier([2.0, 4.0, 6.0, 9.0], censored=[False, True, False, False])
    assert np.allclose(times, [2.0, 6.0, 9.0])
    assert np.allclose(surv, [0.75, 0.375, 0.0])


def test_kaplan_meier_uncensored_matches_empirical_survival():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    times, surv = lm.kaplan_meier(vals)
    assert np.allclose(times, vals)
    assert np.allclose(surv, [0.8, 0.6, 0.4, 0.2, 0.0])


def test_censored_mle_recovers_exponential_rate():
    true = lm.Exponential(0.002)
    x = true.sample(4000, rng=3)
    cap = 800.0
    values = np.minimum(x, cap)
    censored = x > cap
    fit = lm.fit_mle_censored(lm.Exponential, values, initial_params=[0.001],
                              bounds=[(1e-6, 1.0)], censored=censored)
    assert fit.rate == pytest.approx(0.002, rel=0.07)


def test_truncated_mle_recovers_exponential_rate():
    true = lm.Exponential(0.002)
    x = true.sample(8000, rng=5)
    keep = x > 200.0
    fit = lm.fit_mle_censored(lm.Exponential, x[keep], initial_params=[0.001],
                              bounds=[(1e-6, 1.0)],
                              truncation=np.full(keep.sum(), 200.0))
    assert fit.rate == pytest.approx(0.002, rel=0.07)
