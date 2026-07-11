import numpy as np
import pytest

from lossmodels.empirical import EmpiricalFrequency, EmpiricalSeverity

# ---------------------------
# EmpiricalSeverity tests
# ---------------------------

def test_empirical_severity_init_valid():
    data = [100.0, 200.0, 500.0]
    model = EmpiricalSeverity(data)

    assert isinstance(model.data, np.ndarray)
    assert np.array_equal(model.data, np.array(data, dtype=float))


def test_empirical_severity_init_empty():
    with pytest.raises(ValueError):
        EmpiricalSeverity([])


def test_empirical_severity_init_negative_values():
    with pytest.raises(ValueError):
        EmpiricalSeverity([100.0, -50.0, 200.0])


def test_empirical_severity_mean():
    data = [100.0, 200.0, 500.0]
    model = EmpiricalSeverity(data)

    assert np.isclose(model.mean(), np.mean(data))


def test_empirical_severity_variance():
    data = [100.0, 200.0, 500.0]
    model = EmpiricalSeverity(data)

    assert np.isclose(model.variance(), np.var(data))


def test_empirical_severity_sample_shape():
    data = [100.0, 200.0, 500.0]
    model = EmpiricalSeverity(data)

    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_empirical_severity_sample_invalid_size():
    model = EmpiricalSeverity([100.0, 200.0])

    with pytest.raises(ValueError):
        model.sample(0)

    with pytest.raises(ValueError):
        model.sample(-1)


def test_empirical_severity_sample_values_come_from_data():
    np.random.seed(123)

    data = [100.0, 200.0, 500.0]
    model = EmpiricalSeverity(data)

    samples = model.sample(1000)

    assert set(np.unique(samples)).issubset(set(data))


def test_empirical_severity_pdf():
    data = [100.0, 100.0, 200.0, 500.0]
    model = EmpiricalSeverity(data)

    assert np.isclose(model.pdf(100.0), 2 / 4)
    assert np.isclose(model.pdf(200.0), 1 / 4)
    assert np.isclose(model.pdf(300.0), 0.0)
    assert np.isclose(model.pdf(-1.0), 0.0)


def test_empirical_severity_cdf():
    data = [100.0, 200.0, 500.0]
    model = EmpiricalSeverity(data)

    assert np.isclose(model.cdf(50.0), 0.0)
    assert np.isclose(model.cdf(100.0), 1 / 3)
    assert np.isclose(model.cdf(200.0), 2 / 3)
    assert np.isclose(model.cdf(1000.0), 1.0)
    assert np.isclose(model.cdf(-1.0), 0.0)


def test_empirical_severity_excess_loss():
    data = [100.0, 200.0, 500.0]
    model = EmpiricalSeverity(data)

    d = 150.0
    expected = np.mean(np.maximum(np.array(data) - d, 0.0))

    assert np.isclose(model.excess_loss(d), expected)


def test_empirical_severity_excess_loss_invalid_d():
    model = EmpiricalSeverity([100.0, 200.0])

    with pytest.raises(ValueError):
        model.excess_loss(-1.0)


def test_empirical_severity_limited_expected_value():
    data = [100.0, 200.0, 500.0]
    model = EmpiricalSeverity(data)

    d = 150.0
    expected = np.mean(np.minimum(np.array(data), d))

    assert np.isclose(model.limited_expected_value(d), expected)


def test_empirical_severity_limited_expected_value_invalid_d():
    model = EmpiricalSeverity([100.0, 200.0])

    with pytest.raises(ValueError):
        model.limited_expected_value(-1.0)


def test_empirical_severity_repr():
    model = EmpiricalSeverity([100.0, 200.0, 500.0])
    text = repr(model)

    assert "EmpiricalSeverity" in text
    assert "n=3" in text


# ---------------------------
# EmpiricalFrequency tests
# ---------------------------

def test_empirical_frequency_init_valid():
    data = [0, 1, 2, 1, 0]
    model = EmpiricalFrequency(data)

    assert isinstance(model.data, np.ndarray)
    assert np.array_equal(model.data, np.array(data, dtype=int))


def test_empirical_frequency_init_empty():
    with pytest.raises(ValueError):
        EmpiricalFrequency([])


def test_empirical_frequency_init_negative_values():
    with pytest.raises(ValueError):
        EmpiricalFrequency([0, 1, -1, 2])


def test_empirical_frequency_init_noninteger_values():
    with pytest.raises(ValueError):
        EmpiricalFrequency([0, 1.5, 2])


def test_empirical_frequency_mean():
    data = [0, 1, 2, 1, 0]
    model = EmpiricalFrequency(data)

    assert np.isclose(model.mean(), np.mean(data))


def test_empirical_frequency_variance():
    data = [0, 1, 2, 1, 0]
    model = EmpiricalFrequency(data)

    assert np.isclose(model.variance(), np.var(data))


def test_empirical_frequency_sample_shape():
    data = [0, 1, 2, 1, 0]
    model = EmpiricalFrequency(data)

    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_empirical_frequency_sample_invalid_size():
    model = EmpiricalFrequency([0, 1, 2])

    with pytest.raises(ValueError):
        model.sample(0)

    with pytest.raises(ValueError):
        model.sample(-1)


def test_empirical_frequency_sample_values_come_from_data():
    np.random.seed(123)

    data = [0, 1, 2, 1, 0]
    model = EmpiricalFrequency(data)

    samples = model.sample(1000)

    assert set(np.unique(samples)).issubset(set(data))


def test_empirical_frequency_pmf():
    data = [0, 1, 1, 2]
    model = EmpiricalFrequency(data)

    assert np.isclose(model.pmf(0), 1 / 4)
    assert np.isclose(model.pmf(1), 2 / 4)
    assert np.isclose(model.pmf(2), 1 / 4)
    assert np.isclose(model.pmf(3), 0.0)
    assert np.isclose(model.pmf(-1), 0.0)


def test_empirical_frequency_cdf():
    data = [0, 1, 1, 2]
    model = EmpiricalFrequency(data)

    assert np.isclose(model.cdf(-1), 0.0)
    assert np.isclose(model.cdf(0), 1 / 4)
    assert np.isclose(model.cdf(1), 3 / 4)
    assert np.isclose(model.cdf(2), 1.0)
    assert np.isclose(model.cdf(10), 1.0)


def test_empirical_frequency_repr():
    model = EmpiricalFrequency([0, 1, 2, 1, 0])
    text = repr(model)

    assert "EmpiricalFrequency" in text
    assert "n=5" in text