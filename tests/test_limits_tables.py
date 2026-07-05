"""ILF / LER tables: exponential closed forms, delta-method bands, guards."""
import numpy as np
import pytest

import lossmodels as lm


@pytest.fixture(scope="module")
def exp_model():
    return lm.Exponential(rate=1.0 / 40_000.0)  # mean 40k


def test_ilf_exact_for_exponential(exp_model):
    # LEV(x) = mu (1 - e^{-x/mu}); ILF = ratio of those
    mu = 40_000.0
    tab = lm.increased_limits_table(
        exp_model, limits=[100_000, 250_000, 500_000, 1_000_000],
        base_limit=100_000,
    )
    for x in tab.index:
        lev = mu * (1 - np.exp(-x / mu))
        base = mu * (1 - np.exp(-100_000 / mu))
        assert tab.loc[x, "lev"] == pytest.approx(lev, rel=1e-9)
        assert tab.loc[x, "ilf"] == pytest.approx(lev / base, rel=1e-9)
    assert tab.loc[100_000, "ilf"] == pytest.approx(1.0, rel=1e-12)
    assert tab["ilf"].is_monotonic_increasing


def test_ler_exact_and_se_matches_analytic_derivative():
    # exponential parameterized by rate r: LER(d) = 1 - e^{-r d};
    # d(LER)/dr = d e^{-r d}, so the delta-method se has a closed form
    rng = np.random.default_rng(3)
    data = rng.exponential(40_000.0, 2_500)
    fit = lm.fit_exponential(data)
    unc = lm.fit_uncertainty(fit, data)
    d = 60_000.0
    tab = lm.loss_elimination_table(fit, [d], uncertainty=unc)
    r = fit.rate
    assert tab.loc[d, "ler"] == pytest.approx(1 - np.exp(-r * d), rel=1e-9)
    se_exact = abs(d * np.exp(-r * d)) * unc.se[0]
    assert tab.loc[d, "ler_se"] == pytest.approx(se_exact, rel=1e-5)
    assert tab.loc[d, "ci_low"] < tab.loc[d, "ler"] < tab.loc[d, "ci_high"]


def test_base_limit_row_has_zero_se():
    rng = np.random.default_rng(9)
    data = rng.lognormal(9.0, 1.0, 3_000)
    fit = lm.fit_lognormal(data)
    unc = lm.fit_uncertainty(fit, data)
    tab = lm.increased_limits_table(
        fit, limits=[50_000, 100_000, 500_000], base_limit=100_000,
        uncertainty=unc,
    )
    assert tab.loc[100_000, "ilf_se"] == pytest.approx(0.0, abs=1e-8)
    off_base = tab.drop(index=100_000)
    assert (off_base["ilf_se"] > 0).all()
    # truth check: table CI covers the true-parameter ILF
    truth = lm.Lognormal(9.0, 1.0)
    true_ilf = truth.limited_expected_value(500_000.0) / truth.limited_expected_value(100_000.0)
    assert tab.loc[500_000, "ci_low"] <= true_ilf <= tab.loc[500_000, "ci_high"]


def test_infinite_mean_refuses():
    heavy = lm.Pareto(alpha=0.8, theta=1_000.0)  # mean infinite for alpha <= 1
    # the model's own "does not exist" error or the table's "finite mean"
    # guard both refuse -- either way, no ratios of infinity get tabulated
    with pytest.raises(ValueError, match="does not exist|finite mean"):
        lm.loss_elimination_table(heavy, [5_000.0])


def test_mismatched_uncertainty_refuses(exp_model):
    rng = np.random.default_rng(1)
    data = rng.lognormal(9.0, 1.0, 500)
    ln_fit = lm.fit_lognormal(data)
    ln_unc = lm.fit_uncertainty(ln_fit, data)
    with pytest.raises(ValueError, match="different parameters"):
        lm.increased_limits_table(exp_model, [10_000.0], base_limit=5_000.0,
                                  uncertainty=ln_unc)


def test_validation_guards(exp_model):
    with pytest.raises(ValueError, match="positive"):
        lm.increased_limits_table(exp_model, [-1.0], base_limit=100.0)
    with pytest.raises(ValueError, match="nonnegative"):
        lm.loss_elimination_table(exp_model, [-5.0])


def test_lognormal_lev_closed_form():
    from scipy.stats import norm

    mu, sigma = 9.0, 1.2
    m = lm.Lognormal(mu=mu, sigma=sigma)
    for d in (20_000.0, 100_000.0, 500_000.0):
        exact = (np.exp(mu + sigma**2 / 2)
                 * norm.cdf((np.log(d) - mu - sigma**2) / sigma)
                 + d * (1 - norm.cdf((np.log(d) - mu) / sigma)))
        assert m.limited_expected_value(d) == pytest.approx(exact, rel=1e-8)
