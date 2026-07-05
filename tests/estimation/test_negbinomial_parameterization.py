"""Guard tests for the Negative Binomial MLE.

``fit_negbinomial`` scores its likelihood with ``scipy.stats.nbinom.logpmf``
for speed and numerical stability, while ``NegativeBinomial`` exposes ``pmf``
as its own public surface. Those two are only interchangeable because
``NegativeBinomial`` deliberately wraps ``scipy.stats.nbinom`` under the same
``(r, p)`` convention. The tests below pin that equivalence so a future change
to the model's parameterization cannot silently desynchronize the fitted
likelihood, guard the fit against regressing to a per-observation scalar loop,
and pin the refusal to fit non-overdispersed data (for which no finite MLE
exists -- the likelihood supremum is the Poisson limit).
"""

import time
import warnings

import numpy as np
import pytest
from scipy.stats import nbinom

from lossmodels import NegativeBinomial, Poisson, fit_best_frequency, fit_negbinomial


@pytest.mark.parametrize("r, p", [(2.0, 0.3), (5.0, 0.6), (0.7, 0.15), (10.0, 0.9)])
def test_model_pmf_matches_scipy_nbinom_convention(r, p):
    """The model's own PMF must equal ``scipy.nbinom(r, p)`` -- the exact
    parameterization the MLE likelihood assumes -- and must *not* coincide with
    the flipped ``(r, 1 - p)`` convention."""
    k = np.arange(0, 60)
    model_pmf = np.asarray(NegativeBinomial(r=r, p=p).pmf(k), dtype=float)
    np.testing.assert_allclose(model_pmf, nbinom.pmf(k, r, p), rtol=1e-12, atol=0.0)
    assert not np.allclose(model_pmf, nbinom.pmf(k, r, 1.0 - p))


@pytest.mark.parametrize("r, p", [(2.0, 0.3), (5.0, 0.6), (0.7, 0.15)])
def test_fit_likelihood_agrees_with_model_pmf(r, p):
    """The ``nbinom.logpmf`` the optimizer sums must equal the log of the
    model's own PMF wherever that PMF is positive, so the fitted parameters
    maximize the likelihood the model itself defines."""
    k = np.arange(0, 60)
    model_pmf = np.asarray(NegativeBinomial(r=r, p=p).pmf(k), dtype=float)
    positive = model_pmf > 0
    np.testing.assert_allclose(
        nbinom.logpmf(k[positive], r, p),
        np.log(model_pmf[positive]),
        rtol=1e-10,
        atol=1e-10,
    )


def test_fit_recovers_parameters_from_negbinomial_data():
    """End-to-end sanity: on genuine over-dispersed count data the fit recovers
    the mean tightly and the ``(r, p)`` pair within a loose band (they trade
    off against each other along the likelihood ridge)."""
    rng = np.random.default_rng(20260705)
    true = NegativeBinomial(r=4.0, p=0.35)
    data = rng.negative_binomial(true.r, true.p, size=20_000)
    fitted = fit_negbinomial(data)
    assert fitted.mean() == pytest.approx(float(data.mean()), rel=0.02)
    assert fitted.r == pytest.approx(true.r, rel=0.25)
    assert fitted.p == pytest.approx(true.p, rel=0.25)


def test_fit_is_frequency_aggregated_and_fast():
    """A modest overdispersed fit must be fast. The pre-fix implementation
    evaluated a scalar PMF once per observation per optimizer step (~20s+ on
    the official model-selection tests); the vectorized, frequency-aggregated
    version is orders of magnitude faster and clears this budget with margin."""
    data = np.random.default_rng(7).negative_binomial(6.0, 0.4, size=20_000)
    start = time.perf_counter()
    fit_negbinomial(data)
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize(
    "data",
    [
        np.random.default_rng(1).binomial(4, 0.5, size=1000),  # underdispersed
        np.zeros(50, dtype=int),                               # degenerate: all zeros
        np.full(50, 5),                                        # degenerate: constant
    ],
    ids=["underdispersed_binomial", "all_zeros", "constant"],
)
def test_fit_raises_on_non_overdispersed_data(data):
    """No finite negative-binomial MLE exists when variance <= mean -- the
    likelihood supremum is the Poisson limit (r -> inf, p -> 1). The fitter must
    refuse rather than return an arbitrary, non-optimal finite (r, p), matching
    the long-standing contract of fit_negbinomial_moments."""
    with pytest.raises(ValueError, match="overdispersed"):
        fit_negbinomial(data)


def test_fit_on_overdispersed_data_is_warning_free():
    """With non-overdispersed data rejected up front, the optimizer only ever
    runs on well-conditioned interior problems, so fits no longer emit SciPy's
    finite-difference 'invalid value' RuntimeWarning (the guard is the root-cause
    fix; no warning suppression is needed)."""
    for r, p in [(50.0, 0.8), (8.0, 0.35), (2.0, 0.2)]:
        data = np.random.default_rng(3).negative_binomial(r, p, size=3000)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fit_negbinomial(data)  # must not raise


def test_model_selection_falls_back_to_poisson_on_underdispersed_data():
    """fit_best_frequency must survive underdispersed data: the NB candidate
    raises, is dropped from the comparison, and Poisson is selected."""
    data = np.random.default_rng(11).binomial(4, 0.5, size=1000)  # underdispersed
    result = fit_best_frequency(data, candidates=["poisson", "negbinomial"])
    assert result["best_name"] == "poisson"
    assert isinstance(result["best_model"], Poisson)
    nb = next(r for r in result["results"] if r["name"] == "negbinomial")
    assert nb["model"] is None and "error" in nb
