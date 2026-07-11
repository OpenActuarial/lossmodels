import numpy as np
import pytest
from scipy.stats import poisson as poisson_dist

from lossmodels.aggregate.discretization import discretize_severity
from lossmodels.aggregate.fft import (
    cdf_from_pmf_fft,
    fft_aggregate_poisson,
    mean_from_aggregate_pmf_fft,
)
from lossmodels.aggregate.panjer import mean_from_aggregate_pmf, panjer_recursion
from lossmodels.frequency import Poisson
from lossmodels.severity import Exponential


def test_fft_poisson_with_unit_severity_matches_poisson():
    """
    If severity is degenerate at 1 lattice unit, aggregate losses should follow
    the Poisson frequency distribution directly.
    """
    freq = Poisson(lam=2.0)

    # X = 1 with probability 1 on lattice h = 1
    severity_pmf = np.array([0.0, 1.0])

    g = fft_aggregate_poisson(freq, severity_pmf, n_steps=10)

    expected = np.array([poisson_dist.pmf(k, 2.0) for k in range(11)], dtype=float)
    expected /= expected.sum()

    assert np.allclose(g, expected, atol=1e-5)


def test_fft_returns_valid_pmf():
    freq = Poisson(lam=2.0)
    severity_pmf = np.array([0.2, 0.5, 0.3])

    g = fft_aggregate_poisson(freq, severity_pmf, n_steps=20)

    assert isinstance(g, np.ndarray)
    assert g.ndim == 1
    assert len(g) == 21
    assert np.all(g >= 0.0)
    assert np.isclose(g.sum(), 1.0)


def test_fft_invalid_inputs():
    freq = Poisson(lam=2.0)

    with pytest.raises(ValueError):
        fft_aggregate_poisson(freq, [], n_steps=10)

    with pytest.raises(ValueError):
        fft_aggregate_poisson(freq, [-0.1, 1.1], n_steps=10)

    with pytest.raises(ValueError):
        fft_aggregate_poisson(freq, [0.5, 0.5], n_steps=0)


def test_fft_requires_poisson_frequency():
    class BadFrequency:
        pass

    with pytest.raises(TypeError):
        fft_aggregate_poisson(BadFrequency(), [0.5, 0.5], n_steps=10)


def test_cdf_from_pmf_fft_basic():
    pmf = np.array([0.2, 0.3, 0.5])
    cdf = cdf_from_pmf_fft(pmf)

    assert np.allclose(cdf, [0.2, 0.5, 1.0])


def test_cdf_from_pmf_fft_invalid():
    with pytest.raises(ValueError):
        cdf_from_pmf_fft([])

    with pytest.raises(ValueError):
        cdf_from_pmf_fft([-0.1, 1.1])


def test_mean_from_aggregate_pmf_fft_basic():
    pmf = np.array([0.2, 0.3, 0.5])
    mean = mean_from_aggregate_pmf_fft(pmf, h=2.0)

    expected = 0.2 * 0.0 + 0.3 * 2.0 + 0.5 * 4.0
    assert np.isclose(mean, expected)


def test_mean_from_aggregate_pmf_fft_invalid():
    with pytest.raises(ValueError):
        mean_from_aggregate_pmf_fft([], h=1.0)

    with pytest.raises(ValueError):
        mean_from_aggregate_pmf_fft([0.5, 0.5], h=0.0)

    with pytest.raises(ValueError):
        mean_from_aggregate_pmf_fft([-0.1, 1.1], h=1.0)


def test_fft_mean_reasonable_for_poisson_exponential():
    """
    Compare FFT mean to theoretical aggregate mean for a discretized
    Poisson-Exponential model.
    """
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)

    h = 0.01
    severity_pmf = discretize_severity(sev, h=h, max_loss=20.0)

    g = fft_aggregate_poisson(freq, severity_pmf, n_steps=5000)
    approx_mean = mean_from_aggregate_pmf_fft(g, h=h)

    theoretical_mean = freq.mean() * sev.mean()
    assert np.isclose(approx_mean, theoretical_mean, rtol=0.02)


def test_fft_cdf_is_monotone():
    freq = Poisson(lam=2.0)
    severity_pmf = np.array([0.2, 0.5, 0.3])

    g = fft_aggregate_poisson(freq, severity_pmf, n_steps=20)
    cdf = cdf_from_pmf_fft(g)

    assert np.all(np.diff(cdf) >= -1e-12)
    assert 0.0 <= cdf[0] <= 1.0
    assert np.isclose(cdf[-1], 1.0)


def test_fft_and_panjer_means_are_close():
    """
    For the same discretized Poisson-Exponential model, FFT and Panjer should
    produce similar aggregate means.
    """
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)

    h = 0.01
    severity_pmf = discretize_severity(sev, h=h, max_loss=20.0)

    g_fft = fft_aggregate_poisson(freq, severity_pmf, n_steps=5000)
    g_panjer = panjer_recursion(freq, severity_pmf, n_steps=5000)

    mean_fft = mean_from_aggregate_pmf_fft(g_fft, h=h)
    mean_panjer = mean_from_aggregate_pmf(g_panjer, h=h)

    assert np.isclose(mean_fft, mean_panjer, rtol=0.02)