"""Regression: the censored-payments coverage worked-example page numbers stay true."""
import numpy as np
import pytest

import lossmodels as lm
from lossmodels.aggregate import (
    discretize_severity,
    fft_aggregate_poisson,
    mean_from_pmf,
    stop_loss_from_pmf,
    var_from_pmf,
)
from lossmodels.coverage import Layer, OrdinaryDeductible


def test_coverage_page_numbers():
    rng = np.random.default_rng(7)
    true = lm.Lognormal(7.4, 1.1)
    d0, u0 = 500.0, 10_000.0
    x = true.sample(6000, rng=rng)
    payments = np.clip(x - d0, 0.0, u0)
    obs = payments[payments > 0]
    assert len(obs) == 5140 and int((obs == u0).sum()) == 278

    values, trunc, cens = lm.payments_to_ground_up(obs, deductible=d0, max_payment=u0)
    times, surv = lm.kaplan_meier(values, truncation=trunc, censored=cens)
    i = np.searchsorted(times, 2000.0)
    assert surv[min(i, len(surv) - 1)] == pytest.approx(0.4854, abs=5e-4)

    naive = lm.fit_lognormal(obs + d0)
    fitc = lm.fit_mle_censored(lm.Lognormal, values, initial_params=[7.0, 1.0],
                               truncation=trunc, censored=cens)
    assert (round(naive.mu, 3), round(naive.sigma, 3)) == (7.649, 0.833)
    assert (round(fitc.mu, 3), round(fitc.sigma, 3)) == (7.370, 1.112)

    sev = fitc
    cur, prop = Layer(sev, 500.0, 10_000.0), Layer(sev, 1000.0, 20_000.0)
    assert cur.mean() == pytest.approx(2122.75, abs=0.01)
    assert prop.mean() == pytest.approx(1978.08, abs=0.01)
    assert OrdinaryDeductible(sev, 500.0).loss_elimination_ratio() == pytest.approx(0.1600, abs=5e-4)
    assert OrdinaryDeductible(sev, 1000.0).loss_elimination_ratio() == pytest.approx(0.2876, abs=5e-4)

    results = {}
    for name, lay in (("current", cur), ("proposed", prop)):
        pmf = discretize_severity(lay, h=250.0, max_loss=25_000.0)
        agg = fft_aggregate_poisson(lm.Poisson(3000.0), pmf, n_steps=65_536)
        results[name] = (mean_from_pmf(agg, 250.0), var_from_pmf(agg, 250.0, 0.99),
                         stop_loss_from_pmf(agg, 250.0, 6_600_000.0))
        # Wald sanity: discretization bias stays inside a tenth of a percent
        assert results[name][0] == pytest.approx(3000.0 * lay.mean(), rel=1e-3)
    assert results["current"][1] == 6_803_250.0
    assert results["proposed"][1] == 6_448_250.0
    assert results["current"][2] > results["proposed"][2] > 0.0
