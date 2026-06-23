import numpy as np
import pytest
from scipy.integrate import quad

from lossmodels.severity import Lognormal, Pareto, Exponential, SplicedSeverity


U = 50_000.0
W = 0.9


def make_splice(alpha=3.0, weight=W, u=U):
    body = Lognormal(mu=9.5, sigma=1.0)
    tail = Pareto(alpha=alpha, theta=u)   # theta=u -> cdf(u)=0, support [u, inf)
    return SplicedSeverity(body=body, tail=tail, threshold=u, weight=weight)


# --- construction / validation ---
def test_rejects_bad_threshold_and_weight():
    body, tail = Lognormal(mu=9.5, sigma=1.0), Pareto(alpha=3.0, theta=U)
    with pytest.raises(ValueError):
        SplicedSeverity(body, tail, threshold=-1.0, weight=W)
    with pytest.raises(ValueError):
        SplicedSeverity(body, tail, threshold=U, weight=1.5)


def test_rejects_tail_not_supported_above_threshold():
    body = Lognormal(mu=9.5, sigma=1.0)
    bad_tail = Lognormal(mu=9.5, sigma=1.0)   # cdf(U) != 0
    with pytest.raises(ValueError):
        SplicedSeverity(body, bad_tail, threshold=U, weight=W)


def test_rejects_tail_missing_sample():
    class NoSample:
        def cdf(self, x):
            return 0.0

        def pdf(self, x):
            return 0.0

    with pytest.raises(TypeError):
        SplicedSeverity(Lognormal(mu=9.5, sigma=1.0), NoSample(), threshold=U, weight=W)


# --- distribution behaviour ---
def test_cdf_continuous_at_threshold_equals_weight():
    sp = make_splice()
    assert sp.cdf(U) == pytest.approx(W, abs=1e-9)
    assert sp.cdf(U - 1e-3) == pytest.approx(W, abs=1e-3)


def test_cdf_monotone():
    sp = make_splice()
    xs = np.array([1e3, 1e4, U, 1e5, 1e6])
    assert np.all(np.diff(sp.cdf(xs)) > 0)


@pytest.mark.parametrize("p", [0.05, 0.5, W, 0.95, 0.999])
def test_quantile_roundtrip_both_regions(p):
    sp = make_splice()
    assert sp.cdf(sp.quantile(p)) == pytest.approx(p, abs=1e-6)


def test_pdf_piece_masses_sum_to_one():
    sp = make_splice()
    m_body, _ = quad(sp.pdf, 0, U)
    m_tail, _ = quad(sp.pdf, U, np.inf)
    assert m_body == pytest.approx(W, abs=1e-3)
    assert m_tail == pytest.approx(1.0 - W, abs=1e-3)


def test_scalar_vs_array_outputs():
    sp = make_splice()
    assert isinstance(sp.cdf(1000.0), float)
    assert isinstance(sp.pdf(1000.0), float)
    x = np.array([1e3, 1e5, 1e6])
    assert sp.cdf(x).shape == x.shape
    assert sp.pdf(x).shape == x.shape


# --- moments (deterministic, via numerical integration of the spliced pdf) ---
def test_mean_matches_integral():
    sp = make_splice(alpha=3.0)
    e_x = quad(lambda t: t * sp.pdf(t), 0, U)[0] + quad(lambda t: t * sp.pdf(t), U, np.inf)[0]
    assert sp.mean() == pytest.approx(e_x, rel=1e-3)


def test_variance_matches_integral():
    sp = make_splice(alpha=3.0)
    e_x2 = quad(lambda t: t * t * sp.pdf(t), 0, U)[0] + quad(lambda t: t * t * sp.pdf(t), U, np.inf)[0]
    assert sp.variance() == pytest.approx(e_x2 - sp.mean() ** 2, rel=1e-3)


def test_mean_raises_when_tail_mean_undefined():
    sp = make_splice(alpha=0.8)   # Pareto mean does not exist for alpha <= 1
    with pytest.raises(ValueError):
        sp.mean()


# --- behaves as a severity (sample/mean contract for risksim/extremeloss) ---
def test_sample_shape_and_body_fraction():
    np.random.seed(1)
    sp = make_splice()
    draws = sp.sample(200_000)
    assert draws.shape == (200_000,)
    assert (draws <= U).mean() == pytest.approx(W, abs=0.01)
