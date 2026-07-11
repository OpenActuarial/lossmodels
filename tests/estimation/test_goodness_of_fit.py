import numpy as np
import pytest

from lossmodels.estimation import (
    anderson_darling,
    cramer_von_mises,
    fit_best_severity,
    fit_lognormal,
    goodness_of_fit,
    ks_statistic,
    tail_quantile_table,
)
from lossmodels.severity import Lognormal


@pytest.fixture
def lognormal_data():
    return np.random.default_rng(1).lognormal(9.0, 1.0, 5000)


def test_gof_stats_finite_and_small_for_good_fit(lognormal_data):
    m = fit_lognormal(lognormal_data)
    ks = ks_statistic(m, lognormal_data)
    ad = anderson_darling(m, lognormal_data)
    cvm = cramer_von_mises(m, lognormal_data)
    assert np.isfinite(ks) and 0.0 <= ks < 0.1   # well-specified -> small KS
    assert np.isfinite(ad) and ad > 0.0
    assert np.isfinite(cvm) and cvm > 0.0


def test_ks_larger_for_wrong_model(lognormal_data):
    good = fit_lognormal(lognormal_data)
    bad = Lognormal(mu=5.0, sigma=2.0)  # deliberately mis-specified
    assert ks_statistic(bad, lognormal_data) > ks_statistic(good, lognormal_data)


def test_tail_quantile_table_structure_and_accuracy(lognormal_data):
    m = fit_lognormal(lognormal_data)
    tbl = tail_quantile_table(m, lognormal_data, probs=(0.9, 0.99))
    assert [r["prob"] for r in tbl] == [0.9, 0.99]
    for row in tbl:
        assert set(row) == {"prob", "empirical", "fitted", "abs_error", "rel_error"}
        assert abs(row["rel_error"]) < 0.15   # good fit tracks the upper quantiles


def test_goodness_of_fit_report_keys(lognormal_data):
    m = fit_lognormal(lognormal_data)
    rep = goodness_of_fit(m, lognormal_data, k=2)
    assert set(rep) == {
        "n", "log_likelihood", "aic", "bic",
        "ks", "anderson_darling", "cramer_von_mises",
    }
    assert rep["n"] == lognormal_data.size


def test_log_likelihood_vectorized_matches_per_element(lognormal_data):
    m = fit_lognormal(lognormal_data)
    sub = lognormal_data[:200]
    ll_loop = float(np.sum(np.log([m.pdf(float(x)) for x in sub])))
    ll_vec = float(np.sum(np.log(m.pdf(sub))))
    assert ll_vec == pytest.approx(ll_loop, rel=1e-9)


def test_fit_best_severity_recovers_lognormal(lognormal_data):
    best = fit_best_severity(lognormal_data)
    assert best["best_name"] == "lognormal"
