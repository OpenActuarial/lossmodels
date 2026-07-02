import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma as gamma_func, gammaln

from ..frequency import NegativeBinomial, Poisson
from ..severity import Exponential, Gamma, Lognormal, Pareto, Weibull


def _validate_positive_data(data, name: str = "data") -> np.ndarray:
    """Validate that input data are nonempty and strictly positive."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(data <= 0):
        raise ValueError(f"{name} must contain only positive values.")
    return data


def _validate_count_data(data, name: str = "data") -> np.ndarray:
    """Validate that input data are nonempty, nonnegative, and integer-valued."""
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(data < 0):
        raise ValueError(f"{name} must contain only nonnegative values.")
    if not np.all(np.equal(data, np.floor(data))):
        raise ValueError(f"{name} must contain only integer-valued counts.")
    return data.astype(int)


def fit_negbinomial_moments(data) -> NegativeBinomial:
    """
    Fit a Negative Binomial frequency model by the method of moments.

    Parameterization
    ----------------
    Mean = r(1-p)/p
    Variance = r(1-p)/p^2

    Solving:
        p_hat = mean / variance
        r_hat = mean^2 / (variance - mean)

    Notes
    -----
    This requires variance > mean. If variance <= mean, the method-of-moments
    Negative Binomial fit is not valid.
    """
    data = _validate_count_data(data)
    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))
    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")
    if var_x <= mean_x:
        raise ValueError(
            "Negative Binomial method-of-moments requires variance > mean."
        )
    p_hat = mean_x / var_x
    r_hat = mean_x**2 / (var_x - mean_x)
    return NegativeBinomial(r=float(r_hat), p=float(p_hat))


def fit_poisson_moments(data) -> Poisson:
    """
    Fit a Poisson frequency model by the method of moments.

    For Poisson(lambda):
        E[N] = lambda

    So:
        lambda_hat = sample mean

    Notes
    -----
    An all-zero dataset is valid and yields lambda_hat = 0.
    """
    data = _validate_count_data(data)
    lam_hat = float(np.mean(data))
    if lam_hat < 0:
        raise ValueError("Estimated lambda must be nonnegative.")
    return Poisson(lam=lam_hat)


def fit_exponential_moments(data) -> Exponential:
    """
    Fit an Exponential severity model by the method of moments.

    For Exponential(rate):
        E[X] = 1 / rate

    So:
        rate_hat = 1 / sample mean
    """
    data = _validate_positive_data(data)
    mean_x = float(np.mean(data))
    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")
    rate_hat = 1.0 / mean_x
    return Exponential(rate=rate_hat)


def fit_gamma_moments(data) -> Gamma:
    """
    Fit a Gamma severity model by the method of moments.

    For Gamma(alpha, theta):
        E[X] = alpha * theta
        Var(X) = alpha * theta^2

    So:
        alpha_hat = mean^2 / var
        theta_hat = var / mean
    """
    data = _validate_positive_data(data)
    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))
    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")
    if var_x <= 0:
        raise ValueError("Sample variance must be positive.")
    alpha_hat = mean_x**2 / var_x
    theta_hat = var_x / mean_x
    return Gamma(alpha=float(alpha_hat), theta=float(theta_hat))


def fit_lognormal_moments(data) -> Lognormal:
    """
    Fit a Lognormal severity model by the method of moments.

    If X ~ Lognormal(mu, sigma^2), then:
        E[X] = exp(mu + sigma^2 / 2)
        Var(X) = (exp(sigma^2) - 1) * exp(2mu + sigma^2)

    Solving gives:
        sigma2_hat = log(1 + var / mean^2)
        mu_hat = log(mean) - sigma2_hat / 2
    """
    data = _validate_positive_data(data)
    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))
    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")
    if var_x <= 0:
        raise ValueError("Sample variance must be positive.")
    sigma2_hat = np.log(1.0 + var_x / mean_x**2)
    mu_hat = np.log(mean_x) - 0.5 * sigma2_hat
    sigma_hat = np.sqrt(sigma2_hat)
    return Lognormal(mu=float(mu_hat), sigma=float(sigma_hat))

def fit_pareto_moments(data) -> Pareto:
    """
    Fit a Pareto Type I severity model by the method of moments.

    For Pareto(alpha, theta):
        E[X] = alpha * theta / (alpha - 1),   alpha > 1
        Var(X) = alpha * theta^2 / ((alpha - 1)^2 * (alpha - 2)),   alpha > 2

    Let m = sample mean and s2 = sample variance. Then:

        s2 / m^2 = 1 / (alpha * (alpha - 2))

    which gives:

        alpha_hat = 1 + sqrt(1 + m^2 / s2)
        theta_hat = m * (alpha_hat - 1) / alpha_hat

    Notes
    -----
    This requires positive sample variance.
    """
    data = _validate_positive_data(data)
    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))

    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")
    if var_x <= 0:
        raise ValueError("Sample variance must be positive.")

    alpha_hat = float(1.0 + np.sqrt(1.0 + (mean_x ** 2) / var_x))
    theta_hat = float(mean_x * (alpha_hat - 1.0) / alpha_hat)

    return Pareto(alpha=alpha_hat, theta=theta_hat)


def fit_weibull_moments(data) -> Weibull:
    """
    Fit a Weibull severity model by the method of moments using numerical
    matching of the first two moments.
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 1 or len(data) == 0:
        raise ValueError("data must be a non-empty 1D array.")
    if np.any(data <= 0):
        raise ValueError("Weibull fitting requires positive data.")

    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))

    if mean_x <= 0:
        raise ValueError("Sample mean must be positive.")

    cv2_target = var_x / (mean_x ** 2)

    def cv2_weibull(k):
        log_g1 = gammaln(1.0 + 1.0 / k)
        log_g2 = gammaln(1.0 + 2.0 / k)
        return float(np.exp(log_g2 - 2.0 * log_g1) - 1.0)

    def objective(k):
        return cv2_weibull(k) - cv2_target

    k_hat = float(brentq(objective, 0.1, 100.0))
    lam_hat = float(mean_x / gamma_func(1.0 + 1.0 / k_hat))

    return Weibull(k=k_hat, lam=lam_hat)
