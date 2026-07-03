"""Conformance: every constructor matches its scipy.stats equivalent.

Pins the parameterization table on the docs lossmodels page: CDF/pmf,
quantiles, and moments agree with the verified scipy correspondence.
"""
import numpy as np
import pytest
from scipy import stats

import lossmodels as lm

XG = np.array([50.0, 200.0, 800.0, 2500.0, 9000.0, 40_000.0])
QG = np.array([0.05, 0.25, 0.5, 0.75, 0.95, 0.995])
KG = np.arange(0, 15)

CONTINUOUS = [
    (lm.Exponential(0.002), stats.expon(scale=500.0)),
    (lm.Gamma(2.5, 1000.0), stats.gamma(2.5, scale=1000.0)),
    (lm.Lognormal(8.0, 1.2), stats.lognorm(1.2, scale=np.exp(8.0))),
    (lm.Weibull(1.7, 900.0), stats.weibull_min(1.7, scale=900.0)),
    (lm.Pareto(3.0, 100.0), stats.pareto(3.0, scale=100.0)),
    (lm.ParetoII(3.0, 100.0), stats.lomax(3.0, scale=100.0)),
    (lm.Loglogistic(2.2, 800.0), stats.fisk(2.2, scale=800.0)),
    (lm.Burr(2.0, 1000.0, 1.5), stats.burr12(1.5, 2.0, scale=1000.0)),
    (lm.GeneralizedPareto(3.0, 1000.0, 2.0), stats.betaprime(2.0, 3.0, scale=1000.0)),
    (lm.InverseGamma(2.5, 300.0), stats.invgamma(2.5, scale=300.0)),
]
CIDS = [type(o).__name__ for o, _ in CONTINUOUS]

DISCRETE = [
    (lm.Poisson(3.2), stats.poisson(3.2)),
    (lm.NegativeBinomial(4.0, 0.3), stats.nbinom(4, 0.3)),
    (lm.Geometric(0.35), stats.nbinom(1, 0.35)),
    (lm.Binomial(10, 0.3), stats.binom(10, 0.3)),
]
DIDS = [type(o).__name__ for o, _ in DISCRETE]


@pytest.mark.parametrize("ours,ref", CONTINUOUS, ids=CIDS)
def test_continuous_cdf_matches_scipy(ours, ref):
    assert np.allclose(ours.cdf(XG), ref.cdf(XG), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("ours,ref", CONTINUOUS, ids=CIDS)
def test_continuous_pdf_matches_scipy(ours, ref):
    assert np.allclose(ours.pdf(XG), ref.pdf(XG), rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("ours,ref", CONTINUOUS, ids=CIDS)
def test_continuous_quantile_matches_scipy(ours, ref):
    assert np.allclose([ours.quantile(q) for q in QG], ref.ppf(QG), rtol=1e-9)


@pytest.mark.parametrize("ours,ref", CONTINUOUS, ids=CIDS)
def test_continuous_moments_match_scipy(ours, ref):
    assert np.isclose(ours.mean(), ref.mean(), rtol=1e-9)
    assert np.isclose(ours.variance(), ref.var(), rtol=1e-9)


@pytest.mark.parametrize("ours,ref", DISCRETE, ids=DIDS)
def test_discrete_pmf_matches_scipy(ours, ref):
    assert np.allclose([ours.pmf(k) for k in KG], ref.pmf(KG), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("ours,ref", DISCRETE, ids=DIDS)
def test_discrete_cdf_matches_scipy(ours, ref):
    assert np.allclose([ours.cdf(k) for k in KG], ref.cdf(KG), rtol=1e-10)


@pytest.mark.parametrize("ours,ref", DISCRETE, ids=DIDS)
def test_discrete_moments_match_scipy(ours, ref):
    assert np.isclose(ours.mean(), ref.mean(), rtol=1e-12)
    assert np.isclose(ours.variance(), ref.var(), rtol=1e-12)
