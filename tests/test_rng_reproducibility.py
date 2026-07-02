"""Reproducibility contract for the rng parameter added in 0.6.0."""

import numpy as np
import pytest

import lossmodels.frequency as fr
import lossmodels.severity as sev
from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.coverage import Layer, OrdinaryDeductible, PolicyLimit

SEVERITIES = [
    sev.Exponential(rate=1 / 300.0),
    sev.Gamma(alpha=2.0, theta=500.0),
    sev.Lognormal(mu=7.0, sigma=1.2),
    sev.Weibull(k=1.5, lam=800.0),
    sev.Pareto(alpha=3.0, theta=1000.0),
    sev.ParetoII(alpha=3.0, theta=1000.0),
    sev.Burr(alpha=3.0, theta=1000.0, gamma=2.0),
    sev.InverseGamma(alpha=3.0, theta=500.0),
    sev.Beta(a=2.0, b=3.0, theta=1000.0),
    sev.SplicedSeverity(
        body=sev.Exponential(rate=1 / 300.0),
        tail=sev.Pareto(alpha=3.0, theta=1000.0),
        threshold=800.0,
        weight=0.8,
    ),
]

FREQUENCIES = [
    fr.Poisson(lam=2.5),
    fr.Binomial(n=12, p=0.3),
    fr.Geometric(p=0.25),
    fr.NegativeBinomial(r=2.5, p=0.4),
    fr.Logarithmic(beta=1.5),
    fr.ZeroTruncated(fr.Poisson(lam=2.0)),
    fr.ZeroModified(fr.Poisson(lam=2.0), p0_modified=0.35),
]


@pytest.mark.parametrize("model", SEVERITIES, ids=lambda m: type(m).__name__)
def test_severity_seed_reproducible(model):
    a = model.sample(200, rng=123)
    b = model.sample(200, rng=123)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("model", FREQUENCIES, ids=lambda m: type(m).__name__)
def test_frequency_seed_reproducible(model):
    a = model.sample(200, rng=123)
    b = model.sample(200, rng=123)
    np.testing.assert_array_equal(a, b)


def test_generator_advances_between_calls():
    model = sev.Gamma(alpha=2.0, theta=500.0)
    gen = np.random.default_rng(7)
    a = model.sample(50, rng=gen)
    b = model.sample(50, rng=gen)
    assert not np.array_equal(a, b)


def test_shared_generator_threads_through_composition():
    crm = CollectiveRiskModel(fr.Poisson(lam=2.0), sev.Gamma(alpha=2.0, theta=500.0))
    a = crm.sample(200, rng=42)
    b = crm.sample(200, rng=42)
    np.testing.assert_array_equal(a, b)
    # int seed must be normalized to ONE shared generator: per-claim severities
    # within a simulation must differ (a fresh generator per draw would repeat).
    heavy = CollectiveRiskModel(fr.Poisson(lam=30.0), sev.Exponential(rate=1.0))
    s = heavy.sample(20, rng=5)
    assert np.unique(np.round(s, 12)).size > 1


def test_aggregate_mc_methods_accept_rng():
    crm = CollectiveRiskModel(fr.Poisson(lam=2.0), sev.Gamma(alpha=2.0, theta=500.0))
    assert crm.var(0.99, n_sim=5_000, rng=11) == crm.var(0.99, n_sim=5_000, rng=11)
    assert crm.tvar(0.99, n_sim=5_000, rng=11) == crm.tvar(0.99, n_sim=5_000, rng=11)
    assert crm.stop_loss(1_000.0, n_sim=5_000, rng=11) == crm.stop_loss(
        1_000.0, n_sim=5_000, rng=11
    )


def test_coverage_sample_accepts_rng():
    x = sev.Gamma(alpha=2.0, theta=500.0)
    for wrapper in (
        Layer(x, d=200.0, u=1_000.0),
        OrdinaryDeductible(x, d=200.0),
        PolicyLimit(x, u=1_000.0),
    ):
        np.testing.assert_array_equal(
            wrapper.sample(100, rng=3), wrapper.sample(100, rng=3)
        )


def test_legacy_global_seed_path_unchanged():
    model = sev.Gamma(alpha=2.0, theta=500.0)
    np.random.seed(99)
    a = model.sample(50)
    np.random.seed(99)
    b = model.sample(50)
    np.testing.assert_array_equal(a, b)
