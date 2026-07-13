"""Vectorized collective-risk sampling and observable FFT mass loss."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lossmodels.aggregate.collective import CollectiveRiskModel
from lossmodels.aggregate.fft import AggregateMassLossWarning, fft_aggregate_poisson
from lossmodels.frequency import Poisson
from lossmodels.frequency.poisson import Poisson as PoissonFreq
from lossmodels.severity.exponential import Exponential
from lossmodels.utils.random import resolve_rng


# --------------------------------------------------------------------------- #
# collective-risk vectorization
# --------------------------------------------------------------------------- #
def _model():
    return CollectiveRiskModel(PoissonFreq(3.0), Exponential(1000.0))


def test_sample_is_reproducible_for_a_seed():
    m = _model()
    a = m.sample(size=2000, rng=123)
    b = m.sample(size=2000, rng=123)
    np.testing.assert_array_equal(a, b)


def test_vectorized_matches_reference_loop():
    """The RNG stream is preserved, so the aggregate equals the per-sim loop."""
    freq, sev = PoissonFreq(4.0), Exponential(500.0)
    m = CollectiveRiskModel(freq, sev)

    def reference(size, seed):
        rng = resolve_rng(seed)
        counts = freq.sample(size=size, rng=rng)
        agg = np.zeros(size)
        for i, n in enumerate(counts):
            if n > 0:
                agg[i] = np.sum(sev.sample(size=int(n), rng=rng))
        return agg

    for seed in (0, 7, 99):
        np.testing.assert_allclose(m.sample(size=3000, rng=seed), reference(3000, seed), rtol=1e-9, atol=1e-6)


def test_zero_count_simulations_are_zero():
    # low frequency guarantees some zero-claim simulations, which must be exactly 0
    m = CollectiveRiskModel(PoissonFreq(0.3), Exponential(1000.0))
    agg = m.sample(size=5000, rng=1)
    assert (agg == 0.0).any()
    assert (agg >= 0.0).all()


def test_mean_in_line_with_theory():
    m = _model()
    expected = m.mean()  # E[S] = E[N] * E[X], whatever the severity parameterization
    agg = m.sample(size=50000, rng=5)
    assert abs(agg.mean() - expected) / expected < 0.03


# --------------------------------------------------------------------------- #
# FFT mass-loss diagnostics
# --------------------------------------------------------------------------- #
def test_fft_well_resolved_is_silent_and_normalized():
    sev = np.ones(40) / 40
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g, diag = fft_aggregate_poisson(Poisson(2.0), sev, n_steps=4000, return_diagnostics=True)
    assert diag["raw_mass"] == pytest.approx(1.0, abs=1e-6)
    assert g.sum() == pytest.approx(1.0, abs=1e-9)


def test_fft_truncated_lattice_warns():
    sev = np.ones(40) / 40
    with pytest.warns(AggregateMassLossWarning):
        fft_aggregate_poisson(Poisson(20.0), sev, n_steps=30)


def test_fft_truncated_lattice_can_raise():
    sev = np.ones(40) / 40
    with pytest.raises(ValueError, match="truncated or aliased"):
        fft_aggregate_poisson(Poisson(20.0), sev, n_steps=30, on_mass_loss="raise")


def test_fft_diagnostics_shape():
    sev = np.ones(10) / 10
    g, diag = fft_aggregate_poisson(Poisson(1.0), sev, n_steps=500, return_diagnostics=True)
    assert set(diag) == {"negative_mass", "raw_mass", "normalization_factor", "normalized"}
    assert diag["negative_mass"] >= 0.0
