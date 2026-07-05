"""Guard tests for the Negative Binomial MLE.

``fit_negbinomial`` scores its likelihood with ``scipy.stats.nbinom.logpmf``
for speed and numerical stability, while ``NegativeBinomial`` exposes ``pmf``
as its own public surface. Those two are only interchangeable because
``NegativeBinomial`` deliberately wraps ``scipy.stats.nbinom`` under the same
``(r, p)`` convention. The tests below pin that equivalence so a future change
to the model's parameterization cannot silently desynchronize the fitted
likelihood, and they guard the fit against regressing to a per-observation
scalar loop or re-emitting the benign finite-difference warning.
"""

import time
import warnings

import numpy as np
import pytest
from scipy.stats import nbinom

from lossmodels import NegativeBinomial, fit_negbinomial


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
    """A modest fit must be fast. The pre-fix implementation evaluated a scalar
    PMF once per observation per optimizer step (~20s+ on the official
    model-selection tests); the vectorized, frequency-aggregated version is
    orders of magnitude faster and clears this budget with wide margin."""
    rng = np.random.default_rng(7)
    data = rng.poisson(2.0, 20_000)
    start = time.perf_counter()
    fit_negbinomial(data)
    assert time.perf_counter() - start < 2.0


def test_fit_does_not_emit_invalid_value_warning():
    """The optimizer's finite-difference gradient probes the edge of the
    feasible box, where the likelihood is ``+inf``; the fit must swallow the
    resulting benign numpy ``invalid value`` warning rather than surface it.
    (This dataset provably triggers that warning without the suppression.)"""
    rng = np.random.default_rng(7)
    data = rng.poisson(2.0, 200)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="invalid value encountered")
        fit_negbinomial(data)  # must not raise
