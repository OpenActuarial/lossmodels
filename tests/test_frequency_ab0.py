"""The (a, b, 0) recursion and the zero-truncated / zero-modified algebra."""
import numpy as np
import pytest

import lossmodels as lm

AB0 = [
    (lm.Poisson(3.2), 0.0, 3.2),
    (lm.NegativeBinomial(4.0, 0.3), 0.7, 3.0 * 0.7),
    (lm.Binomial(10, 0.3), -0.3 / 0.7, 11 * 0.3 / 0.7),
    (lm.Geometric(0.35), 0.65, 0.0),
]
IDS = [type(m).__name__ for m, _, _ in AB0]


@pytest.mark.parametrize("model,a,b", AB0, ids=IDS)
def test_ab0_recursion(model, a, b):
    # p_k / p_{k-1} == a + b / k for every member of the (a, b, 0) class
    for k in range(1, 11):
        prev, curr = model.pmf(k - 1), model.pmf(k)
        if prev < 1e-300:
            break
        assert np.isclose(curr / prev, a + b / k, rtol=1e-10), f"k={k}"


@pytest.mark.parametrize("model,a,b", AB0, ids=IDS)
def test_pmf_sums_to_one(model, a, b):
    ks = np.arange(0, 400)
    assert np.isclose(sum(model.pmf(int(k)) for k in ks), 1.0, atol=1e-10)


def test_closed_form_moments():
    lam, r, p, n, pb = 3.2, 4.0, 0.3, 10, 0.3
    po, nb, bi, ge = lm.Poisson(lam), lm.NegativeBinomial(r, p), lm.Binomial(n, pb), lm.Geometric(p)
    assert np.isclose(po.mean(), lam) and np.isclose(po.variance(), lam)
    assert np.isclose(nb.mean(), r * (1 - p) / p)
    assert np.isclose(nb.variance(), r * (1 - p) / p**2)
    assert np.isclose(bi.mean(), n * pb) and np.isclose(bi.variance(), n * pb * (1 - pb))
    assert np.isclose(ge.mean(), (1 - p) / p) and np.isclose(ge.variance(), (1 - p) / p**2)


@pytest.mark.parametrize("base", [lm.Poisson(2.0), lm.NegativeBinomial(3.0, 0.4)],
                         ids=["Poisson", "NegativeBinomial"])
def test_zero_truncated_algebra(base):
    zt = lm.ZeroTruncated(base)
    p0 = base.pmf(0)
    assert zt.pmf(0) == 0.0
    for k in range(1, 8):
        assert np.isclose(zt.pmf(k), base.pmf(k) / (1 - p0), rtol=1e-12)
    assert np.isclose(sum(zt.pmf(k) for k in range(0, 300)), 1.0, atol=1e-10)
    assert np.isclose(zt.mean(), base.mean() / (1 - p0), rtol=1e-10)


@pytest.mark.parametrize("base", [lm.Poisson(2.0), lm.NegativeBinomial(3.0, 0.4)],
                         ids=["Poisson", "NegativeBinomial"])
def test_zero_modified_algebra(base):
    p0m = 0.35
    zm = lm.ZeroModified(base, p0m)
    p0 = base.pmf(0)
    assert np.isclose(zm.pmf(0), p0m, rtol=1e-12)
    scale = (1 - p0m) / (1 - p0)
    for k in range(1, 8):
        assert np.isclose(zm.pmf(k), scale * base.pmf(k), rtol=1e-12)
    assert np.isclose(sum(zm.pmf(k) for k in range(0, 300)), 1.0, atol=1e-10)
    assert np.isclose(zm.mean(), scale * base.mean(), rtol=1e-10)


def test_frequency_sampling_reproducible():
    for model in (lm.Poisson(3.0), lm.NegativeBinomial(4.0, 0.3),
                  lm.ZeroModified(lm.Poisson(2.0), 0.3)):
        a = model.sample(500, rng=11)
        b = model.sample(500, rng=11)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, model.sample(500, rng=12))
