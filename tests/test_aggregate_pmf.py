"""Panjer / FFT aggregate machinery and the PMF-based risk measures."""
from itertools import pairwise

import numpy as np
import pytest

import lossmodels as lm
from lossmodels.aggregate import (
    cdf_from_pmf,
    discretize_severity,
    fft_aggregate_poisson,
    mean_from_pmf,
    panjer_recursion,
    stop_loss_from_pmf,
    tvar_from_pmf,
    var_from_pmf,
)

H = 25.0
SEV = lm.Gamma(2.0, 400.0)  # mean 800
SEV_PMF = discretize_severity(SEV, H, max_loss=20_000.0)


def test_discretized_severity_is_a_pmf_matching_the_mean():
    assert np.all(SEV_PMF >= 0)
    assert np.isclose(SEV_PMF.sum(), 1.0, atol=1e-6)
    assert mean_from_pmf(SEV_PMF, H) == pytest.approx(SEV.mean(), rel=2e-3)


def test_panjer_and_fft_agree_for_poisson():
    freq = lm.Poisson(6.0)
    a = panjer_recursion(freq, SEV_PMF, n_steps=4096)
    b = fft_aggregate_poisson(freq, SEV_PMF, n_steps=4096)
    assert np.allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("freq", [lm.Poisson(6.0), lm.NegativeBinomial(4.0, 0.4)],
                         ids=["Poisson", "NegBin"])
def test_aggregate_mean_is_walds_identity(freq):
    agg = panjer_recursion(freq, SEV_PMF, n_steps=8192)
    assert mean_from_pmf(agg, H) == pytest.approx(freq.mean() * mean_from_pmf(SEV_PMF, H), rel=1e-6)


def test_pmf_var_is_the_inverted_cdf_order_statistic():
    agg = panjer_recursion(lm.Poisson(6.0), SEV_PMF, n_steps=8192)
    cdf = cdf_from_pmf(agg)
    for q in (0.5, 0.9, 0.99):
        manual = H * int(np.searchsorted(cdf, q, side="left"))
        assert var_from_pmf(agg, H, q) == pytest.approx(manual, abs=H / 2)
        assert tvar_from_pmf(agg, H, q) >= var_from_pmf(agg, H, q) - 1e-9


def test_stop_loss_from_pmf_properties():
    agg = panjer_recursion(lm.Poisson(6.0), SEV_PMF, n_steps=8192)
    mean = mean_from_pmf(agg, H)
    assert stop_loss_from_pmf(agg, H, 0.0) == pytest.approx(mean, rel=1e-9)
    grid = [0.0, 2000.0, 5000.0, 10_000.0]
    sl = [stop_loss_from_pmf(agg, H, d) for d in grid]
    assert all(b <= a + 1e-9 for a, b in pairwise(sl))


def test_panjer_matches_collective_monte_carlo():
    freq, q = lm.Poisson(6.0), 0.99
    agg = panjer_recursion(freq, SEV_PMF, n_steps=8192)
    crm = lm.CollectiveRiskModel(freq, SEV)
    assert crm.var(q, n_sim=200_000, rng=1) == pytest.approx(var_from_pmf(agg, H, q), rel=0.02)


def test_panjer_underflow_raises_with_guidance():
    sev_pmf = discretize_severity(lm.Lognormal(7.37, 1.11), H, max_loss=25_000.0)
    with pytest.raises(ValueError, match="fft_aggregate_poisson"):
        panjer_recursion(lm.Poisson(3000.0), sev_pmf, n_steps=1000)
