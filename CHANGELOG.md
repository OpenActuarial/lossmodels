# Changelog

## 0.4.0

### Added
- `SplicedSeverity`: a generic two-piece spliced/composite severity holding a
  body distribution below a threshold and a tail distribution above it, with a
  mixing weight equal to the body mass. The body is renormalized onto `(0, u]`
  and the tail must be supported on `[u, inf)` with `cdf(u) = 0` (e.g. a
  `Pareto` with `theta = u`, or a generalized-Pareto tail supplied by
  `extremeloss`). Exposes the full severity interface (`pdf`, `cdf`, `quantile`,
  `sample`, `mean`, `variance`), so a tail-corrected severity drops straight
  back into `risksim` / `extremeloss`. The container is tail-agnostic and never
  imports `extremeloss`; it consumes any duck-typed tail object. Re-exported at
  the top level as `lossmodels.SplicedSeverity`.

## 0.3.0

### Added
- Severity models now expose `quantile(p)` / `ppf(p)` (inverse CDF / VaR). Closed
  form for Lognormal, Gamma, Pareto, Weibull, Exponential and the empirical
  severity; a numerical CDF-inversion fallback on the base class covers any other
  severity model.
- Goodness-of-fit statistics in `lossmodels.estimation`: `ks_statistic`,
  `anderson_darling`, `cramer_von_mises`, a `tail_quantile_table` (fitted vs
  empirical high quantiles), and a one-call `goodness_of_fit` report. These are
  *absolute* fit measures complementing the existing *relative* `aic`/`bic` and
  `fit_best_severity`/`fit_best_frequency` selectors.
- The full public API (distribution classes, fitters, selectors, diagnostics) is
  now re-exported at the top level, e.g. `lossmodels.Lognormal`,
  `lossmodels.fit_best_severity`, `lossmodels.goodness_of_fit`.

### Changed
- Severity `pdf`/`cdf` and frequency `pmf`/`cdf` now accept array input and
  vectorize (returning an array), while still returning a float for scalar input.
  This makes `log_likelihood` (and any whole-sample evaluation) fast on large
  datasets instead of looping in Python. The integration contract with `risksim`
  and `extremeloss` (which only call `.sample(size)` / `.mean()`) is unchanged.
- `Binomial` gained a `cdf` method for consistency with the other frequency
  models.

## 0.2.0

### Removed (breaking)

- **Credibility models (`Buhlmann`, `BuhlmannStraub`) were moved to the
  `actuarialpy` package** and are no longer part of `lossmodels`. Import them as
  `from actuarialpy import Buhlmann, BuhlmannStraub`; their behavior is
  unchanged.

  This reflects a deliberate scope decision: `lossmodels` is a Klugman-anchored
  loss-distribution toolkit (distributions, coverage modifications, estimation,
  model selection, and aggregate models), while credibility belongs next to the
  experience and ratemaking workflows in `actuarialpy` that consume it. The
  `.sample()` distribution protocol remains the contract that the rest of the
  ecosystem builds on.

### Fixed

- Package version is now consistent between `pyproject.toml` and
  `lossmodels.__version__` (both `0.2.0`).
