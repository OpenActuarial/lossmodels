"""fit_uncertainty (observed information), compare_fits, and the tail protocol."""
import numpy as np
import pytest

import lossmodels as lm


@pytest.fixture(scope="module")
def lognormal_fit():
    rng = np.random.default_rng(7)
    data = rng.lognormal(mean=1.5, sigma=0.6, size=4000)
    return data, lm.fit_lognormal(data)


def test_lognormal_se_matches_closed_form(lognormal_fit):
    # observed information for (mu, sigma) is exactly diag(n/s^2, 2n/s^2)
    # at the MLE, so the numeric Hessian must reproduce s/sqrt(n) and
    # s/sqrt(2n) to differentiation accuracy
    data, fit = lognormal_fit
    unc = lm.fit_uncertainty(fit, data)
    n = len(data)
    assert unc.param_names == ["mu", "sigma"]
    np.testing.assert_allclose(unc.se[0], fit.sigma / np.sqrt(n), rtol=1e-3)
    np.testing.assert_allclose(unc.se[1], fit.sigma / np.sqrt(2 * n), rtol=1e-3)
    # off-diagonal is zero for the normal family
    corr = unc.covariance[0, 1] / (unc.se[0] * unc.se[1])
    assert abs(corr) < 0.05


def test_summary_covers_truth(lognormal_fit):
    data, fit = lognormal_fit
    tab = lm.fit_uncertainty(fit, data).summary(confidence_level=0.99)
    assert list(tab.columns) == ["estimate", "se", "ci_low", "ci_high"]
    assert tab.loc["mu", "ci_low"] <= 1.5 <= tab.loc["mu", "ci_high"]
    assert tab.loc["sigma", "ci_low"] <= 0.6 <= tab.loc["sigma", "ci_high"]


def test_generic_over_models():
    rng = np.random.default_rng(11)
    data = rng.gamma(2.5, 400.0, 3000)
    for fit in (lm.fit_gamma(data), lm.fit_lognormal(data), lm.fit_exponential(data)):
        unc = lm.fit_uncertainty(fit, data)
        assert np.all(np.isfinite(unc.se)) and np.all(unc.se > 0)
        assert unc.covariance.shape == (len(unc.param_names),) * 2


def test_truncation_flows_through():
    rng = np.random.default_rng(13)
    raw = rng.lognormal(1.0, 0.5, 8000)
    trunc = 2.0
    data = raw[raw > trunc]
    fit = lm.fit_lognormal(data, truncation=trunc)
    unc = lm.fit_uncertainty(fit, data, truncation=trunc)
    assert np.all(np.isfinite(unc.se))
    # ignoring the truncation is a different likelihood -> different curvature
    unc_wrong = lm.fit_uncertainty(fit, data)
    assert not np.allclose(unc.covariance, unc_wrong.covariance, rtol=1e-3)


def test_parameter_introspection_guard():
    class Odd:
        def __init__(self, a):
            self.b = a  # attribute name mismatch

    with pytest.raises(TypeError, match="same-named"):
        lm.model_parameters(Odd(1.0))


def test_compare_fits_ranks_the_generator(lognormal_fit):
    data, fit = lognormal_fit
    tab = lm.compare_fits(
        {
            "lognormal": fit,
            "gamma": lm.fit_gamma(data),
            "exponential": lm.fit_exponential(data),
        },
        data,
    )
    assert list(tab.columns) == ["n_params", "loglik", "aic", "bic", "ks", "ad", "cvm"]
    assert tab["aic"].idxmin() == "lognormal"
    assert tab["ks"].idxmin() == "lognormal"
    assert tab.loc["exponential", "n_params"] == 1
    assert ((tab["ks"] >= 0) & (tab["ks"] <= 1)).all()


def test_compare_fits_sequence_autonames(lognormal_fit):
    data, fit = lognormal_fit
    tab = lm.compare_fits([fit, lm.fit_lognormal(data)], data)
    assert list(tab.index) == ["Lognormal", "Lognormal_2"]


def test_exponential_tail_protocol_is_memoryless():
    exp = lm.Exponential(rate=0.01)
    # closed forms: sf = e^{-rate d}, mean excess = 1/rate at every d
    for d in (0.0, 50.0, 300.0):
        assert exp.sf(d) == pytest.approx(np.exp(-0.01 * d), rel=1e-12)
        assert exp.mean_excess(d) == pytest.approx(100.0, rel=1e-7)
    arr = exp.mean_excess(np.array([0.0, 50.0]))
    assert arr.shape == (2,) and np.allclose(arr, 100.0, rtol=1e-7)


def test_mean_excess_at_zero_is_the_mean():
    logn = lm.Lognormal(mu=1.0, sigma=0.5)
    assert logn.mean_excess(0.0) == pytest.approx(logn.mean(), rel=1e-6)
    assert logn.sf(0.0) == pytest.approx(1.0)


def test_layer_inherits_tail_protocol():
    from lossmodels.coverage import Layer

    layer = Layer(lm.Exponential(rate=0.01), d=50.0, u=500.0)
    assert 0.0 <= layer.sf(10.0) <= 1.0


def test_small_scale_parameter_se_exact():
    """Regression: exponential *rates* are ~1e-5; a fixed step floor of 1.0
    once swamped them with truncation error. Observed information for the
    exponential is exactly n/r^2, so se = r/sqrt(n) to differentiation
    accuracy."""
    rng = np.random.default_rng(17)
    data = rng.exponential(40_000.0, 3_000)
    fit = lm.fit_exponential(data)
    unc = lm.fit_uncertainty(fit, data)
    np.testing.assert_allclose(unc.se[0], fit.rate / np.sqrt(3_000), rtol=1e-4)


def test_censoring_flows_through_uncertainty():
    rng = np.random.default_rng(29)
    raw = rng.lognormal(1.0, 0.6, 4000)
    cap = np.quantile(raw, 0.8)
    data = np.minimum(raw, cap)
    censored = raw > cap
    fit = lm.fit_lognormal(data, censored=censored)
    unc = lm.fit_uncertainty(fit, data, censored=censored)
    assert np.all(np.isfinite(unc.se)) and np.all(unc.se > 0)
    # dropping the censoring flag is a different likelihood
    unc_wrong = lm.fit_uncertainty(fit, data)
    assert not np.allclose(unc.covariance, unc_wrong.covariance, rtol=1e-3)


def test_compare_fits_truncation_and_empty(lognormal_fit):
    data, fit = lognormal_fit
    trunc = float(np.quantile(data, 0.1))
    kept = data[data > trunc]
    refit = lm.fit_lognormal(kept, truncation=trunc)
    tab = lm.compare_fits({"ln": refit}, kept, truncation=trunc)
    assert tab.loc["ln", "loglik"] == pytest.approx(
        lm.log_likelihood(refit, kept, truncation=trunc), rel=1e-12)
    with pytest.raises(ValueError, match="no models"):
        lm.compare_fits({}, data)


def test_mean_excess_lev_identity_across_families():
    """Regression pin: e(d) * S(d) + LEV(d) == E[X] for every family --
    exact by construction today, a real check against any future
    reimplementation of mean_excess."""
    models = [
        lm.Lognormal(mu=8.5, sigma=1.0),
        lm.Gamma(alpha=2.0, theta=4_000.0),
        lm.Weibull(k=1.4, lam=9_000.0),
        lm.Exponential(rate=1 / 7_500.0),
    ]
    for m in models:
        mean = m.mean()
        for d in (1_000.0, 10_000.0, 40_000.0):
            surv = m.sf(d)
            lhs = m.mean_excess(d) * surv + m.limited_expected_value(d)
            np.testing.assert_allclose(lhs, mean, rtol=1e-9)


def test_fit_uncertainty_ci_coverage():
    """Empirical coverage of the Wald parameter intervals: simulate from
    known truth, count containment. The lognormal likelihood is exactly
    quadratic in (mu, log-ish sigma), so nominal 0.95 should be hit
    nearly on the nose."""
    rng = np.random.default_rng(5)
    mu_t, sigma_t = 1.5, 0.6
    reps, n = 250, 400
    mu_hits = sigma_hits = 0
    for _ in range(reps):
        data = rng.lognormal(mu_t, sigma_t, n)
        fit = lm.fit_lognormal(data)
        tab = lm.fit_uncertainty(fit, data).summary(confidence_level=0.95)
        mu_hits += tab.loc["mu", "ci_low"] <= mu_t <= tab.loc["mu", "ci_high"]
        sigma_hits += (tab.loc["sigma", "ci_low"] <= sigma_t
                       <= tab.loc["sigma", "ci_high"])
    assert 0.91 <= mu_hits / reps <= 0.985, mu_hits / reps
    assert 0.90 <= sigma_hits / reps <= 0.985, sigma_hits / reps
