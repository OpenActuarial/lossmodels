"""lossmodels: actuarial loss distributions, aggregate modeling, and fit diagnostics.

The full public API is surfaced here for convenience, so common entry points are
available directly as ``lossmodels.Lognormal``, ``lossmodels.fit_best_severity``,
``lossmodels.goodness_of_fit``, etc. Submodule imports
(``from lossmodels.severity import Lognormal``) continue to work unchanged.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _version

try:
    __version__ = _version("lossmodels")
except _PackageNotFoundError:  # running from a source tree without an installed distribution
    __version__ = "0.0.0"

del _PackageNotFoundError, _version

from .severity import (
    SeverityModel,
    Exponential,
    Gamma,
    Lognormal,
    Pareto,
    SplicedSeverity,
    Weibull,
    # Loss Models continuous inventory (0.5.0)
    Burr,
    InverseBurr,
    GeneralizedPareto,
    ParetoII,
    InversePareto,
    Loglogistic,
    Paralogistic,
    InverseParalogistic,
    InverseGamma,
    InverseWeibull,
    InverseExponential,
    InverseGaussian,
    LogT,
    SingleParameterPareto,
    Beta,
    GeneralizedBeta,
)
from .frequency import (
    FrequencyModel,
    Binomial,
    Geometric,
    NegativeBinomial,
    Poisson,
    # (a, b, 1) class (0.5.0)
    ZeroTruncated,
    ZeroModified,
    Logarithmic,
)
from .aggregate import CollectiveRiskModel
from .empirical import EmpiricalSeverity, EmpiricalFrequency
from .estimation import (
    censored_log_likelihood,
    kaplan_meier,
    fit_mle_censored,
    payments_to_ground_up,
    fit_burr,
    fit_inverse_gamma,
    fit_loglogistic,
    fit_paretoII,
    pit_values,
    fit_exponential,
    fit_gamma,
    fit_lognormal,
    fit_pareto,
    fit_poisson,
    fit_weibull,
    fit_negbinomial,
    fit_mle,
    fit_best_severity,
    fit_best_frequency,
    log_likelihood,
    aic,
    bic,
    ks_statistic,
    anderson_darling,
    cramer_von_mises,
    tail_quantile_table,
    goodness_of_fit,
)

__all__ = [
    "__version__",
    "SeverityModel", "Exponential", "Gamma", "Lognormal", "Pareto",
    "SplicedSeverity", "Weibull",
    "Burr", "InverseBurr", "GeneralizedPareto", "ParetoII", "InversePareto",
    "Loglogistic", "Paralogistic", "InverseParalogistic",
    "InverseGamma", "InverseWeibull", "InverseExponential",
    "InverseGaussian", "LogT", "SingleParameterPareto",
    "Beta", "GeneralizedBeta",
    "FrequencyModel", "Binomial", "Geometric", "NegativeBinomial", "Poisson",
    "ZeroTruncated", "ZeroModified", "Logarithmic",
    "CollectiveRiskModel", "EmpiricalSeverity", "EmpiricalFrequency",
    "fit_exponential", "fit_gamma", "fit_lognormal", "fit_pareto", "fit_poisson",
    "fit_weibull", "fit_negbinomial", "fit_mle",
    "censored_log_likelihood",
    "kaplan_meier",
    "fit_mle_censored",
    "payments_to_ground_up",
    "fit_burr",
    "fit_inverse_gamma",
    "fit_loglogistic",
    "fit_paretoII",
    "pit_values",
    "fit_best_severity", "fit_best_frequency",
    "log_likelihood", "aic", "bic",
    "ks_statistic", "anderson_darling", "cramer_von_mises",
    "tail_quantile_table", "goodness_of_fit",
]
