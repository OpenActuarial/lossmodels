"""Tests for fitting under left truncation and right censoring.

Closed forms are checked exactly; numeric fits are checked by parameter
recovery on simulated data (seeded), including the demonstration that the
naive complete-data fit is biased where the censored fit is not.
"""
import numpy as np
import pytest

import lossmodels as lm
from lossmodels.estimation import (
    censored_log_likelihood,
    fit_mle_censored,
    kaplan_meier,
    payments_to_ground_up,
    pit_values,
)


# --------------------------------------------------------------------------- #
# payments_to_ground_up
# --------------------------------------------------------------------------- #
def test_payments_to_ground_up_round_trip():
    payments = np.array([100.0, 500.0, 2000.0, 2000.0])
    values, trunc, cens = payments_to_ground_up(
        payments, deductible=250.0, max_payment=2000.0
    )
    assert np.allclose(values, payments + 250.0)
    assert np.allclose(trunc, 250.0)
    assert np.array_equal(cens, np.array([False, False, True, True]))


def test_payments_to_ground_up_no_limit_means_no_censoring():
    values, trunc, cens = payments_to_ground_up([10.0, 20.0], deductible=5.0)
    assert not cens.any()
    assert np.allclose(trunc, 5.0)


def test_payments_to_ground_up_validation():
    with pytest.raises(ValueError):
        payments_to_ground_up([-1.0], deductible=0.0)
    with pytest.raises(ValueError):
        payments_to_ground_up([300.0], deductible=0.0, max_payment=200.0)
    with pytest.raises(ValueError):
        payments_to_ground_up([100.0], deductible=-5.0)


# --------------------------------------------------------------------------- #
# closed forms
# --------------------------------------------------------------------------- #
def test_exponential_censored_truncated_closed_form():
    theta = 300.0
    x = lm.Exponential(rate=1.0 / theta).sample(60_000, rng=11)
    obs = x[x > 200.0]
    pay = np.minimum(obs - 200.0, 900.0)
    values, trunc, cens = payments_to_ground_up(pay, deductible=200.0, max_payment=900.0)

    fit = lm.fit_exponential(values, truncation=trunc, censored=cens)
    manual = (~cens).sum() / np.sum(values - trunc)
    assert fit.rate == pytest.approx(manual, rel=1e-12)
    assert 1.0 / fit.rate == pytest.approx(theta, rel=0.03)


def test_pareto_type1_truncated_closed_form():
    x = lm.Pareto(alpha=3.0, theta=1000.0).sample(60_000, rng=12)
    obs = x[x > 2000.0]
    fit = lm.fit_pareto(obs, truncation=2000.0)
    manual_alpha = obs.size / np.sum(np.log(obs / 2000.0))
    assert fit.theta == pytest.approx(2000.0)
    assert fit.alpha == pytest.approx(manual_alpha, rel=1e-12)
    assert fit.alpha == pytest.approx(3.0, rel=0.03)


def test_complete_data_paths_unchanged():
    x = lm.Lognormal(7.0, 1.2).sample(5_000, rng=13)
    fit = lm.fit_lognormal(x)
    logs = np.log(x)
    assert fit.mu == pytest.approx(float(np.mean(logs)), rel=1e-14)
    assert fit.sigma == pytest.approx(
        float(np.sqrt(np.mean((logs - np.mean(logs)) ** 2))), rel=1e-14
    )


# --------------------------------------------------------------------------- #
# numeric recovery under deductible + limit (and the naive bias it corrects)
# --------------------------------------------------------------------------- #
def test_lognormal_censored_recovery_beats_naive():
    mu_t, sigma_t = 7.0, 1.2
    x = lm.Lognormal(mu_t, sigma_t).sample(30_000, rng=14)
    d, max_pay = 800.0, 20_000.0
    obs = x[x > d]
    pay = np.minimum(obs - d, max_pay)
    values, trunc, cens = payments_to_ground_up(pay, deductible=d, max_payment=max_pay)

    fit = lm.fit_lognormal(values, truncation=trunc, censored=cens)
    naive = lm.fit_lognormal(values[~cens])

    assert fit.mu == pytest.approx(mu_t, abs=0.05)
    assert fit.sigma == pytest.approx(sigma_t, abs=0.05)
    # the naive complete-data fit on the same observations is visibly biased
    assert abs(naive.mu - mu_t) > 0.10
    assert abs(fit.mu - mu_t) < abs(naive.mu - mu_t)


def test_gamma_censored_recovery():
    x = lm.Gamma(alpha=2.0, theta=500.0).sample(30_000, rng=15)
    obs = x[x > 400.0]
    pay = np.minimum(obs - 400.0, 3_000.0)
    values, trunc, cens = payments_to_ground_up(pay, deductible=400.0, max_payment=3_000.0)
    fit = lm.fit_gamma(values, truncation=trunc, censored=cens)
    assert fit.alpha == pytest.approx(2.0, abs=0.12)
    assert fit.theta == pytest.approx(500.0, rel=0.08)


def test_paretoII_truncated_recovery():
    x = lm.ParetoII(alpha=2.5, theta=1500.0).sample(60_000, rng=16)
    obs = x[x > 1000.0]
    fit = lm.fit_paretoII(obs, truncation=1000.0)
    assert fit.alpha == pytest.approx(2.5, rel=0.10)
    assert fit.theta == pytest.approx(1500.0, rel=0.15)


def test_fit_mle_censored_generic_gamma():
    x = lm.Gamma(alpha=2.0, theta=500.0).sample(20_000, rng=17)
    obs = x[x > 300.0]
    fit = fit_mle_censored(
        lm.Gamma, obs, [1.5, 400.0],
        bounds=[(1e-6, None), (1e-6, None)],
        truncation=300.0,
    )
    assert fit.alpha == pytest.approx(2.0, abs=0.15)
    assert fit.theta == pytest.approx(500.0, rel=0.10)


# --------------------------------------------------------------------------- #
# new complete-data family fitters
# --------------------------------------------------------------------------- #
def test_fit_loglogistic_recovery():
    x = lm.Loglogistic(gamma=3.0, theta=1000.0).sample(20_000, rng=18)
    fit = lm.fit_loglogistic(x)
    assert fit.gamma == pytest.approx(3.0, rel=0.05)
    assert fit.theta == pytest.approx(1000.0, rel=0.05)


def test_fit_inverse_gamma_recovery():
    x = lm.InverseGamma(alpha=3.0, theta=500.0).sample(20_000, rng=19)
    fit = lm.fit_inverse_gamma(x)
    assert fit.alpha == pytest.approx(3.0, rel=0.08)
    assert fit.theta == pytest.approx(500.0, rel=0.10)


def test_fit_burr_achieves_true_likelihood():
    true = lm.Burr(alpha=2.0, theta=1000.0, gamma=1.8)
    x = true.sample(15_000, rng=20)
    fit = lm.fit_burr(x)
    # 3-parameter surface is flat; require the fit to be at least as good as truth
    assert lm.log_likelihood(fit, x) >= lm.log_likelihood(true, x) - 3.0


# --------------------------------------------------------------------------- #
# censored likelihood, selection, and diagnostics
# --------------------------------------------------------------------------- #
def test_censored_log_likelihood_reduces_to_complete():
    x = lm.Lognormal(7.0, 1.0).sample(2_000, rng=21)
    model = lm.Lognormal(7.0, 1.0)
    assert censored_log_likelihood(model, x) == pytest.approx(
        lm.log_likelihood(model, x), rel=1e-12
    )


def test_fit_best_severity_with_truncation_censoring():
    x = lm.Lognormal(7.0, 1.2).sample(25_000, rng=22)
    obs = x[x > 800.0]
    pay = np.minimum(obs - 800.0, 20_000.0)
    values, trunc, cens = payments_to_ground_up(pay, deductible=800.0, max_payment=20_000.0)
    best = lm.fit_best_severity(
        values,
        candidates=["exponential", "gamma", "lognormal", "paretoII", "weibull"],
        truncation=trunc,
        censored=cens,
    )
    assert best["best_name"] == "lognormal"


def test_fit_best_severity_moments_rejects_censoring():
    with pytest.raises(ValueError):
        lm.fit_best_severity([1.0, 2.0, 3.0], method="moments", truncation=0.5)


def test_pit_values_uniform_under_truncation():
    x = lm.Lognormal(7.0, 1.2).sample(30_000, rng=23)
    obs = x[x > 800.0]
    u = pit_values(lm.Lognormal(7.0, 1.2), obs, truncation=800.0)
    assert np.mean(u) == pytest.approx(0.5, abs=0.01)
    assert np.var(u) == pytest.approx(1.0 / 12.0, abs=0.005)


def test_pit_values_rejects_censoring():
    with pytest.raises(ValueError):
        pit_values(lm.Lognormal(7.0, 1.0), [1.0, 2.0], censored=[False, True])


def test_ks_truncation_only_true_vs_wrong_model():
    x = lm.Lognormal(7.0, 1.2).sample(30_000, rng=24)
    obs = x[x > 800.0]
    ks_true = lm.ks_statistic(lm.Lognormal(7.0, 1.2), obs, truncation=800.0)
    ks_wrong = lm.ks_statistic(
        lm.Exponential(rate=1.0 / np.mean(obs)), obs, truncation=800.0
    )
    assert ks_true < 0.02
    assert ks_wrong > 0.10


def test_ks_kaplan_meier_under_censoring():
    x = lm.Lognormal(7.0, 1.2).sample(30_000, rng=25)
    obs = x[x > 800.0]
    pay = np.minimum(obs - 800.0, 20_000.0)
    values, trunc, cens = payments_to_ground_up(pay, deductible=800.0, max_payment=20_000.0)
    ks_true = lm.ks_statistic(lm.Lognormal(7.0, 1.2), values, truncation=trunc, censored=cens)
    ks_wrong = lm.ks_statistic(
        lm.Exponential(rate=1.0 / np.mean(values)), values, truncation=trunc, censored=cens
    )
    assert ks_true < 0.02
    assert ks_wrong > 0.10


def test_kaplan_meier_matches_true_conditional_survival():
    model = lm.Lognormal(7.0, 1.2)
    x = model.sample(30_000, rng=26)
    obs = x[x > 800.0]
    pay = np.minimum(obs - 800.0, 20_000.0)
    values, trunc, cens = payments_to_ground_up(pay, deductible=800.0, max_payment=20_000.0)
    times, surv = kaplan_meier(values, trunc, cens)
    s_true = (1.0 - model.cdf(times)) / (1.0 - model.cdf(800.0))
    assert float(np.max(np.abs(surv - s_true))) < 0.02


def test_anderson_darling_and_cvm_reject_censoring():
    with pytest.raises(ValueError):
        lm.anderson_darling(lm.Lognormal(7.0, 1.0), [1.0, 2.0], censored=[False, True])
    with pytest.raises(ValueError):
        lm.cramer_von_mises(lm.Lognormal(7.0, 1.0), [1.0, 2.0], censored=[False, True])


def test_goodness_of_fit_censored_report():
    x = lm.Lognormal(7.0, 1.2).sample(10_000, rng=27)
    obs = x[x > 800.0]
    pay = np.minimum(obs - 800.0, 20_000.0)
    values, trunc, cens = payments_to_ground_up(pay, deductible=800.0, max_payment=20_000.0)
    gof = lm.goodness_of_fit(
        lm.Lognormal(7.0, 1.2), values, k=2, truncation=trunc, censored=cens
    )
    assert gof["n_uncensored"] == int((~cens).sum())
    assert np.isfinite(gof["aic"]) and np.isfinite(gof["ks"])
    assert np.isnan(gof["anderson_darling"]) and np.isnan(gof["cramer_von_mises"])


def test_goodness_of_fit_complete_data_keys_unchanged():
    x = lm.Lognormal(7.0, 1.0).sample(2_000, rng=28)
    gof = lm.goodness_of_fit(lm.Lognormal(7.0, 1.0), x, k=2)
    assert set(gof.keys()) == {
        "n", "log_likelihood", "aic", "bic", "ks",
        "anderson_darling", "cramer_von_mises",
    }


def test_prepare_validation():
    with pytest.raises(ValueError):
        censored_log_likelihood(lm.Lognormal(7.0, 1.0), [500.0], truncation=[800.0])
    with pytest.raises(ValueError):
        censored_log_likelihood(
            lm.Lognormal(7.0, 1.0), [800.0], truncation=[800.0], censored=[False]
        )
