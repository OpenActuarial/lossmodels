"""lossmodels: actuarial loss distributions, aggregate modeling, and fit diagnostics.

The full public API is surfaced here for convenience, so common entry points are
available directly as ``lossmodels.Lognormal``, ``lossmodels.fit_best_severity``,
``lossmodels.goodness_of_fit``, etc. Submodule imports
(``from lossmodels.severity import Lognormal``) continue to work unchanged.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("lossmodels")
except _PackageNotFoundError:  # running from a source tree without an installed distribution
    __version__ = "0.0.0"

del _PackageNotFoundError, _version

from .aggregate import CollectiveRiskModel
from .empirical import EmpiricalFrequency, EmpiricalSeverity
from .estimation import (
    FitUncertainty,
    aic,
    anderson_darling,
    bic,
    censored_log_likelihood,
    compare_fits,
    cramer_von_mises,
    fit_best_frequency,
    fit_best_severity,
    fit_burr,
    fit_exponential,
    fit_gamma,
    fit_inverse_gamma,
    fit_loglogistic,
    fit_lognormal,
    fit_mle,
    fit_mle_censored,
    fit_negbinomial,
    fit_pareto,
    fit_paretoII,
    fit_poisson,
    fit_uncertainty,
    fit_weibull,
    goodness_of_fit,
    kaplan_meier,
    ks_statistic,
    log_likelihood,
    model_parameters,
    payments_to_ground_up,
    pit_values,
    tail_quantile_table,
)
from .frequency import (
    Binomial,
    FrequencyModel,
    Geometric,
    Logarithmic,
    NegativeBinomial,
    Poisson,
    ZeroModified,
    # (a, b, 1) class (0.5.0)
    ZeroTruncated,
)
from .severity import (
    Beta,
    # Loss Models continuous inventory (0.5.0)
    Burr,
    Exponential,
    Gamma,
    GeneralizedBeta,
    GeneralizedPareto,
    InverseBurr,
    InverseExponential,
    InverseGamma,
    InverseGaussian,
    InverseParalogistic,
    InversePareto,
    InverseWeibull,
    Loglogistic,
    Lognormal,
    LogT,
    Paralogistic,
    Pareto,
    ParetoII,
    SeverityModel,
    SingleParameterPareto,
    SplicedSeverity,
    Weibull,
    increased_limits_table,
    loss_elimination_table,
)

__all__ = [
    "loss_elimination_table",
    "increased_limits_table",
    "compare_fits",
    "model_parameters",
    "fit_uncertainty",
    "FitUncertainty",
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
