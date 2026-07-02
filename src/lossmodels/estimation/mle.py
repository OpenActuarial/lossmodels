import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from scipy.stats import weibull_min

from ..frequency import NegativeBinomial, Poisson
from ..severity import (
    Burr,
    Exponential,
    Gamma,
    InverseGamma,
    Loglogistic,
    Lognormal,
    Pareto,
    ParetoII,
    Weibull,
)
from .censoring import _prepare, fit_mle_censored


def _validate_positive_data(data, name: str = "data") -> np.ndarray:
    """
    Validate that input data are nonempty and strictly positive.
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(data <= 0):
        raise ValueError(f"{name} must contain only positive values.")
    return data


def _validate_count_data(data, name: str = "data") -> np.ndarray:
    """
    Validate that input data are nonempty, nonnegative, and integer-valued.
    """
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(data < 0):
        raise ValueError(f"{name} must contain only nonnegative values.")
    if not np.all(np.equal(data, np.floor(data))):
        raise ValueError(f"{name} must contain only integer-valued counts.")
    return data.astype(int)


def fit_exponential(data, truncation=None, censored=None) -> Exponential:
    """
    Fit an Exponential severity model by maximum likelihood.

    For complete data the MLE is ``rate_hat = 1 / mean(data)``. Under left
    truncation and right censoring the MLE remains closed form
    (Loss Models, ch. 11): with ``n_u`` uncensored observations,

        rate_hat = n_u / sum_i (x_i - t_i)

    summed over *all* observations (censored ones contribute their censoring
    point). ``truncation``/``censored`` follow the conventions of
    :mod:`lossmodels.estimation.censoring`.
    """
    if truncation is None and censored is None:
        data = _validate_positive_data(data)
        mean_x = float(np.mean(data))
        if mean_x <= 0:
            raise ValueError("Mean of data must be positive.")
        return Exponential(rate=float(1.0 / mean_x))

    values, trunc, cens = _prepare(data, truncation, censored)
    n_u = int(np.sum(~cens))
    if n_u == 0:
        raise ValueError("at least one uncensored observation is required.")
    total_excess = float(np.sum(values - trunc))
    if total_excess <= 0:
        raise ValueError("total observed excess over truncation must be positive.")
    return Exponential(rate=float(n_u / total_excess))


def fit_lognormal(data, truncation=None, censored=None) -> Lognormal:
    """
    Fit a Lognormal severity model by maximum likelihood.

    If log(X) ~ Normal(mu, sigma^2), the MLEs are:
        mu_hat = mean(log(data))
        sigma_hat = sqrt(mean((log(data) - mu_hat)^2))

    Notes
    -----
    This uses the MLE version of the variance (ddof=0).
    """
    if truncation is not None or censored is not None:
        values, trunc, cens = _prepare(data, truncation, censored)
        unc = values[~cens]
        if unc.size < 2:
            raise ValueError("at least two uncensored observations are required.")
        mu0 = float(np.mean(np.log(unc)))
        sigma0 = float(max(np.std(np.log(unc)), 1e-3))
        return fit_mle_censored(
            Lognormal, values, [mu0, sigma0],
            bounds=[(None, None), (1e-8, None)],
            truncation=trunc, censored=cens,
        )
    data = _validate_positive_data(data)
    log_data = np.log(data)
    mu_hat = float(np.mean(log_data))
    sigma_hat = float(np.sqrt(np.mean((log_data - mu_hat) ** 2)))
    return Lognormal(mu=mu_hat, sigma=sigma_hat)


def fit_negbinomial(data) -> NegativeBinomial:
    """
    Fit a Negative Binomial frequency model by numerical maximum likelihood.

    Parameterization
    ----------------
    N = number of failures before the r-th success
    Support: {0, 1, 2, ...}
    Mean = r(1-p)/p
    Variance = r(1-p)/p^2
    """
    data = _validate_count_data(data)

    mean_x = float(np.mean(data))
    var_x = float(np.var(data, ddof=0))

    if var_x > mean_x and mean_x > 0:
        p0 = mean_x / var_x
        r0 = mean_x**2 / (var_x - mean_x)
        initial = np.array([r0, p0], dtype=float)
    else:
        initial = np.array([1.0, 0.5], dtype=float)

    bounds = [
        (1e-8, None),           # r > 0
        (1e-8, 1.0 - 1e-8),     # 0 < p < 1
    ]

    def neg_log_likelihood(params):
        r, p = params
        try:
            model = NegativeBinomial(r=r, p=p)
            pmf_vals = np.array([model.pmf(int(x)) for x in data], dtype=float)
            if np.any(~np.isfinite(pmf_vals)) or np.any(pmf_vals <= 0):
                return np.inf
            return float(-np.sum(np.log(pmf_vals)))
        except Exception:
            return np.inf

    result = minimize(
        neg_log_likelihood,
        x0=initial,
        bounds=bounds,
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(
            f"Negative Binomial MLE optimization failed: {result.message}"
        )

    r_hat, p_hat = result.x
    return NegativeBinomial(r=float(r_hat), p=float(p_hat))


def fit_poisson(data) -> Poisson:
    """
    Fit a Poisson frequency model by maximum likelihood.

    For N_i ~ Poisson(lam), the MLE is:
        lam_hat = mean(data)

    Notes
    -----
    An all-zero dataset is valid and yields lam_hat = 0.
    """
    data = _validate_count_data(data)
    lam_hat = float(np.mean(data))
    if lam_hat < 0:
        raise ValueError("Estimated lambda must be nonnegative.")
    return Poisson(lam=lam_hat)


def fit_gamma(data, truncation=None, censored=None) -> Gamma:
    """
    Fit a Gamma severity model by maximum likelihood using SciPy.

    Returns
    -------
    Gamma
        Fitted Gamma(alpha, theta) model.

    Notes
    -----
    This constrains loc = 0 so the support is x > 0, consistent with the
    severity model implementation.
    """
    if truncation is not None or censored is not None:
        values, trunc, cens = _prepare(data, truncation, censored)
        unc = values[~cens]
        if unc.size < 2:
            raise ValueError("at least two uncensored observations are required.")
        a0, _, t0 = gamma_dist.fit(unc, floc=0)
        return fit_mle_censored(
            Gamma, values, [max(a0, 1e-3), max(t0, 1e-6)],
            bounds=[(1e-8, None), (1e-8, None)],
            truncation=trunc, censored=cens,
        )
    data = _validate_positive_data(data)
    alpha_hat, loc_hat, theta_hat = gamma_dist.fit(data, floc=0)
    if loc_hat != 0:
        raise RuntimeError("Gamma fit returned nonzero location despite floc=0.")
    return Gamma(alpha=float(alpha_hat), theta=float(theta_hat))


def fit_pareto(data, truncation=None, censored=None) -> Pareto:
    """
    Fit a Pareto Type I severity model by maximum likelihood.

    For X_i ~ Pareto(alpha, theta) with support x >= theta, the MLEs are:

        theta_hat = min(data)
        alpha_hat = n / sum(log(data / theta_hat))

    Returns
    -------
    Pareto
        Fitted Pareto(alpha, theta) model.
    """
    if truncation is None and censored is None:
        data = _validate_positive_data(data)
        theta_hat = float(np.min(data))
        if theta_hat <= 0:
            raise ValueError("Estimated theta must be positive.")
        log_ratios = np.log(data / theta_hat)
        denom = float(np.sum(log_ratios))
        if denom <= 0:
            raise ValueError(
                "Pareto MLE requires data with at least one observation above the minimum."
            )
        alpha_hat = float(len(data) / denom)
        return Pareto(alpha=alpha_hat, theta=theta_hat)

    values, trunc, cens = _prepare(data, truncation, censored)
    n_u = int(np.sum(~cens))
    if n_u == 0:
        raise ValueError("at least one uncensored observation is required.")
    # Left truncation at t >= theta makes theta unidentifiable (the conditional
    # density alpha t^alpha x^-(alpha+1) does not involve theta), so theta is
    # pinned at the smallest observable point: the minimum untruncated value if
    # any observation is untruncated, else the smallest truncation point.
    untruncated = trunc <= 0
    if np.any(untruncated):
        theta_hat = float(np.min(values[untruncated]))
    else:
        theta_hat = float(np.min(trunc))
    lower = np.maximum(trunc, theta_hat)
    denom = float(np.sum(np.log(values / lower)))
    if denom <= 0:
        raise ValueError(
            "Pareto MLE requires at least one observation strictly above its "
            "effective lower bound."
        )
    return Pareto(alpha=float(n_u / denom), theta=theta_hat)


def fit_weibull(data, truncation=None, censored=None) -> Weibull:
    """
    Fit a Weibull severity model by maximum likelihood using SciPy.

    Returns
    -------
    Weibull
        Fitted Weibull(k, lam) model.

    Notes
    -----
    This constrains loc = 0 so the support is x > 0, consistent with the
    severity model implementation.
    """
    if truncation is not None or censored is not None:
        values, trunc, cens = _prepare(data, truncation, censored)
        unc = values[~cens]
        if unc.size < 2:
            raise ValueError("at least two uncensored observations are required.")
        k0, _, lam0 = weibull_min.fit(unc, floc=0)
        return fit_mle_censored(
            Weibull, values, [max(k0, 1e-3), max(lam0, 1e-6)],
            bounds=[(1e-8, None), (1e-8, None)],
            truncation=trunc, censored=cens,
        )
    data = _validate_positive_data(data)
    k_hat, loc_hat, lam_hat = weibull_min.fit(data, floc=0)
    if loc_hat != 0:
        raise RuntimeError("Weibull fit returned nonzero location despite floc=0.")
    return Weibull(k=float(k_hat), lam=float(lam_hat))


def fit_mle(model_class, data, initial_params, bounds=None, truncation=None, censored=None):
    """
    Generic numerical maximum likelihood estimation for models with a pdf method.

    Parameters
    ----------
    model_class : class
        A model class that can be instantiated as ``model_class(*params)``
        and provides a pdf(x) method.
    data : array-like
        Observed data.
    initial_params : array-like
        Initial parameter guess for the optimizer.
    bounds : list of tuple, optional
        Bounds passed to scipy.optimize.minimize.

    Returns
    -------
    object
        Fitted model instance of type model_class.
    """
    if truncation is not None or censored is not None:
        return fit_mle_censored(
            model_class, data, initial_params,
            bounds=bounds, truncation=truncation, censored=censored,
        )
    data = _validate_positive_data(data)
    initial_params = np.asarray(initial_params, dtype=float)

    if initial_params.size == 0:
        raise ValueError("initial_params must not be empty.")

    def neg_log_likelihood(params):
        try:
            model = model_class(*params)
            pdf_vals = np.array([model.pdf(x) for x in data], dtype=float)
            if np.any(~np.isfinite(pdf_vals)) or np.any(pdf_vals <= 0):
                return np.inf
            return float(-np.sum(np.log(pdf_vals)))
        except Exception:
            return np.inf

    result = minimize(
        neg_log_likelihood,
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B" if bounds is not None else "BFGS",
    )

    if not result.success:
        raise RuntimeError(f"MLE optimization failed: {result.message}")

    return model_class(*result.x)


def _uncensored_moments(values, censored):
    unc = values[~censored]
    if unc.size < 2:
        raise ValueError("at least two uncensored observations are required.")
    m1 = float(np.mean(unc))
    m2 = float(np.mean(unc**2))
    return unc, m1, m2


def fit_paretoII(data, truncation=None, censored=None) -> ParetoII:
    """Fit a Pareto Type II (Lomax) severity -- the FAM/ASTAM "Pareto" --
    by numerical maximum likelihood, with optional left truncation and right
    censoring (deductibles and limits).

    Initialization uses the method of moments on the uncensored values when
    the implied second moment exists, else a heavy-tail default.
    """
    values, trunc, cens = _prepare(data, truncation, censored)
    unc, m1, m2 = _uncensored_moments(values, cens)
    r = m2 / (m1 * m1)
    if r > 2.0 + 1e-9:
        alpha0 = 2.0 * (r - 1.0) / (r - 2.0)
        theta0 = m1 * (alpha0 - 1.0)
    else:
        alpha0, theta0 = 1.5, float(np.median(unc))
    return fit_mle_censored(
        ParetoII, values, [max(alpha0, 0.05), max(theta0, 1e-6)],
        bounds=[(1e-6, None), (1e-8, None)],
        truncation=trunc, censored=cens,
    )


def fit_loglogistic(data, truncation=None, censored=None) -> Loglogistic:
    """Fit a Loglogistic severity by numerical maximum likelihood, with
    optional left truncation and right censoring.

    Initialization: theta from the median (the loglogistic median is theta)
    and gamma from the interquartile ratio, q75/q25 = 9**(1/gamma).
    """
    values, trunc, cens = _prepare(data, truncation, censored)
    unc = values[~cens]
    if unc.size < 2:
        raise ValueError("at least two uncensored observations are required.")
    theta0 = float(np.median(unc))
    q25, q75 = np.percentile(unc, [25, 75])
    gamma0 = float(np.clip(np.log(9.0) / np.log(max(q75 / max(q25, 1e-300), 1.0 + 1e-6)), 0.3, 20.0))
    return fit_mle_censored(
        Loglogistic, values, [gamma0, max(theta0, 1e-6)],
        bounds=[(1e-3, None), (1e-8, None)],
        truncation=trunc, censored=cens,
    )


def fit_inverse_gamma(data, truncation=None, censored=None) -> InverseGamma:
    """Fit an Inverse Gamma severity by numerical maximum likelihood, with
    optional left truncation and right censoring.

    Initialization by moments on the uncensored values:
    alpha0 = 2 + m1^2/(m2 - m1^2), theta0 = m1 (alpha0 - 1).
    """
    values, trunc, cens = _prepare(data, truncation, censored)
    unc, m1, m2 = _uncensored_moments(values, cens)
    var = max(m2 - m1 * m1, 1e-12)
    alpha0 = 2.0 + m1 * m1 / var
    theta0 = m1 * (alpha0 - 1.0)
    return fit_mle_censored(
        InverseGamma, values, [max(alpha0, 0.1), max(theta0, 1e-6)],
        bounds=[(1e-6, None), (1e-8, None)],
        truncation=trunc, censored=cens,
    )


def fit_burr(data, truncation=None, censored=None) -> Burr:
    """Fit a Burr (Type XII) severity by numerical maximum likelihood, with
    optional left truncation and right censoring.

    A three-parameter fit: initialized at the loglogistic special case
    (alpha = 1, gamma from the interquartile ratio, theta at the median).
    """
    values, trunc, cens = _prepare(data, truncation, censored)
    unc = values[~cens]
    if unc.size < 3:
        raise ValueError("at least three uncensored observations are required.")
    theta0 = float(np.median(unc))
    q25, q75 = np.percentile(unc, [25, 75])
    gamma0 = float(np.clip(np.log(9.0) / np.log(max(q75 / max(q25, 1e-300), 1.0 + 1e-6)), 0.3, 20.0))
    return fit_mle_censored(
        Burr, values, [1.0, max(theta0, 1e-6), gamma0],
        bounds=[(1e-3, None), (1e-8, None), (1e-3, None)],
        truncation=trunc, censored=cens,
    )
