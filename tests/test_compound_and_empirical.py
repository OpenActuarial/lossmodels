"""Compound-distribution identities and the empirical risk-measure estimators."""
import numpy as np
import pytest

import lossmodels as lm
from lossmodels.aggregate import tvar, var

PAIRS = [
    (lm.Poisson(6.0), lm.Gamma(2.0, 400.0)),
    (lm.NegativeBinomial(4.0, 0.4), lm.Lognormal(6.0, 0.8)),
    (lm.Binomial(20, 0.25), lm.Exponential(0.002)),
]
IDS = ["Poi-Gamma", "NB-Lognormal", "Bin-Exponential"]


@pytest.mark.parametrize("freq,sev", PAIRS, ids=IDS)
def test_compound_mean_is_walds_identity(freq, sev):
    crm = lm.CollectiveRiskModel(freq, sev)
    assert crm.mean() == pytest.approx(freq.mean() * sev.mean(), rel=1e-12)


@pytest.mark.parametrize("freq,sev", PAIRS, ids=IDS)
def test_compound_variance_identity(freq, sev):
    crm = lm.CollectiveRiskModel(freq, sev)
    expected = freq.mean() * sev.variance() + freq.variance() * sev.mean() ** 2
    assert crm.variance() == pytest.approx(expected, rel=1e-12)


def test_crm_tail_metrics_reproducible_and_ordered():
    crm = lm.CollectiveRiskModel(lm.Poisson(6.0), lm.Gamma(2.0, 400.0))
    v1 = crm.var(0.99, n_sim=50_000, rng=9)
    v2 = crm.var(0.99, n_sim=50_000, rng=9)
    assert v1 == v2
    assert crm.tvar(0.99, n_sim=50_000, rng=9) >= v1


def test_crm_stop_loss_decreasing_under_common_random_numbers():
    crm = lm.CollectiveRiskModel(lm.Poisson(6.0), lm.Gamma(2.0, 400.0))
    sl = [crm.stop_loss(d, n_sim=50_000, rng=4) for d in (0.0, 2000.0, 6000.0, 12_000.0)]
    assert sl[0] == pytest.approx(crm.mean(), rel=0.02)
    assert all(b <= a for a, b in zip(sl, sl[1:]))


def test_empirical_var_is_inverted_cdf_quantile():
    rng = np.random.default_rng(0)
    losses = rng.lognormal(7.0, 1.0, size=997)
    for q in (0.5, 0.9, 0.99):
        assert var(losses, q) == np.quantile(losses, q, method="inverted_cdf")


def test_empirical_var_tvar_exact_hand_case():
    losses = np.arange(1.0, 101.0)  # 1..100, n q = 90 exactly
    assert var(losses, 0.9) == 90.0
    assert tvar(losses, 0.9) == pytest.approx(95.5, rel=1e-12)  # mean of the top 10
    assert tvar(losses, 0.01) == pytest.approx(51.0, rel=1e-12)  # (5050 - 1) / 99


def test_q_strictly_between_zero_and_one():
    losses = np.arange(1.0, 101.0)
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            var(losses, bad)
        with pytest.raises(ValueError):
            tvar(losses, bad)


def test_empirical_severity_and_frequency():
    data = np.array([100.0, 100.0, 300.0, 900.0])
    es = lm.EmpiricalSeverity(data)
    assert es.mean() == pytest.approx(data.mean(), rel=1e-12)
    s = es.sample(200, rng=2)
    assert set(np.unique(s)) <= set(data.tolist())
    assert np.array_equal(s, es.sample(200, rng=2))
    counts = np.array([0, 1, 1, 2, 3, 3, 3])
    ef = lm.EmpiricalFrequency(counts)
    assert ef.mean() == pytest.approx(counts.mean(), rel=1e-12)
    assert ef.pmf(3) == pytest.approx(3 / 7, rel=1e-12)
