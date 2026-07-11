# Changelog

## 0.7.3

Bump version number to update PyPI with updated README.

## 0.7.2

### Changed
- **`fit_negbinomial` now refuses non-overdispersed data.** The negative
  binomial always satisfies variance/mean = 1/p > 1, so when the sample
  variance is not greater than the mean no finite `(r, p)` maximizes the
  likelihood -- the supremum is the Poisson limit (`r -> inf`, `p -> 1`). The
  fitter previously ran the optimizer anyway and returned an arbitrary,
  non-optimal finite result (with `success=True`) whose value was unstable
  across versions and platforms. It now raises `ValueError` pointing to
  `fit_poisson`, matching the long-standing contract of
  `fit_negbinomial_moments`. Because `fit_best_frequency` already drops
  candidates that fail to fit, model selection on underdispersed data now
  cleanly falls back to Poisson.

### Removed
- **Obsolete negbinomial fit warning suppression.** The `np.errstate` guard
  added in 0.7.1 silenced a SciPy finite-difference `RuntimeWarning` that only
  arose when the optimizer chased the `p -> 1` boundary on near-Poisson data.
  With such data now rejected up front, the optimizer only runs on
  well-conditioned interior problems and the warning no longer occurs, so the
  suppression has been removed rather than carried forward -- the guard is the
  root-cause fix.

### Tests
- Extended `tests/estimation/test_negbinomial_parameterization.py` to pin the
  new refusal (underdispersed, equidispersed, and degenerate inputs raise),
  confirm overdispersed fits are warning-free without suppression, and verify
  `fit_best_frequency` falls back to Poisson on underdispersed data.

## 0.7.1

### Changed
- **`fit_negbinomial` performance.** The Negative Binomial MLE now aggregates
  the data to distinct counts and scores each optimizer step with a single
  vectorized `scipy.stats.nbinom.logpmf` call weighted by the count
  multiplicities, instead of evaluating a scalar PMF once per observation per
  iteration. Fitted results are unchanged (the likelihood is identical up to
  floating point); only the runtime differs -- the two official
  `fit_best_frequency` model-selection tests drop from ~23.5s each to well
  under a second. Scoring in log space via `logpmf` also avoids the tail
  underflow of the previous `log(pmf(...))`.
- **`fit_negbinomial` warning hygiene.** The L-BFGS-B finite-difference
  gradient probes the edge of the feasible `(r, p)` box, where the negative
  log-likelihood is `+inf`; the resulting `inf - inf` inside SciPy's numerical
  derivative raised a benign numpy "invalid value encountered" `RuntimeWarning`
  during otherwise-successful fits. That specific floating-point warning is now
  suppressed for the duration of the optimization only.

### Fixed
- **`examples/panjer_vs_simulation.py` aggregate grid.** The Panjer recursion
  step count is now sized to the discretized severity support
  (`int(round(max_loss / h))`) rather than a fixed 50,000. The old value
  computed the aggregate PMF out to ~250 aggregate means at large cost with no
  effect on the reported mean; the example now runs an order of magnitude
  faster while reproducing the same result.

### Tests
- Added `tests/estimation/test_negbinomial_parameterization.py`, pinning that
  `NegativeBinomial(r, p).pmf` matches `scipy.stats.nbinom(r, p)` (and not the
  flipped `(r, 1 - p)` convention), that the fit's `logpmf` likelihood agrees
  with the model's own PMF, and that the fit stays frequency-aggregated,
  fast, and free of the finite-difference warning.

## 0.7.0

### Added
- **Parameter uncertainty.** `fit_uncertainty(model, data, truncation=,
  censored=)` evaluates the numerical Hessian of the log-likelihood at the
  fitted parameters and returns the observed-information covariance --
  generically, for any distribution whose `__init__` parameters are stored
  as same-named attributes (all in-package models qualify). The
  `FitUncertainty` result exposes `se`, `covariance`, and a `summary()`
  table with Wald confidence intervals. Truncation and censoring flow
  through `log_likelihood`, so the uncertainty matches the fit actually
  performed; a non-negative-definite Hessian raises rather than returning
  a covariance that means nothing.
- **`compare_fits`.** The companion to `fit_best_severity`: every candidate
  scored on every criterion (`loglik`, `aic`, `bic`, `ks`, `ad`, `cvm`) in
  one DataFrame, so the trade-offs are visible -- a model can win AIC while
  losing the tail.
- **Tail protocol.** `SeverityModel.sf(x)` and
  `SeverityModel.mean_excess(d)` (via the limited-expected-value identity
  `E[(X-d)+] = E[X] - E[min(X,d)]`, so heavy tails need no integration to
  infinity; infinite-mean models return `inf`; closed-form override point
  for subclasses) on every distribution and on
  `Layer`/`PolicyLimit` by inheritance. This is the small cross-package
  protocol that `ratingmodels.pooling_charge_from_severity` consumes --
  any object with these two methods qualifies, no package dependency
  required.
- **ILF and loss-elimination tables.** `increased_limits_table` and
  `loss_elimination_table` tabulate ratios of limited expected values from
  any fitted severity -- and, given the model's `FitUncertainty`, carry
  the fit covariance through the delta method to `se`/`ci` columns on the
  factor itself. The base-limit row has `se = 0` exactly (the ratio's
  gradient vanishes there); an infinite-mean severity refuses to produce
  elimination ratios rather than tabulating ratios of infinity.
- `pandas` is now a dependency (tabular outputs).

## 0.6.2

### Changed

- Documentation only: exam-track references (FAM/ASTAM, SOA tables) are
  replaced with the stable citation -- *Loss Models: From Data to Decisions*
  (Klugman, Panjer & Willmot), Appendix A. No behavior changes.

- More descriptive package `description` metadata.

### Added

- Conformance, identity, and integration test suites (scipy/closed-form
  conformance, mathematical identities, cross-package seams). Example
  scripts are now executed by the test suite.
- Worked-example regression test for the censored-payments coverage page.

### Fixed

- `panjer_recursion` raises with guidance when the aggregate mass at zero
  underflows (large expected claim counts) instead of silently returning
  zeros; `fft_aggregate_poisson` is the right tool there.

## 0.6.0

### Added
- Every `sample` method (severity, frequency, empirical, spliced, coverage
  wrappers, `CollectiveRiskModel`) accepts `rng`: `None` (legacy global
  `numpy.random` state, backward compatible with `np.random.seed`), an `int`
  seed, or a `numpy.random.Generator`. A seed or generator threads through
  frequency and severity draws so aggregate simulations are reproducible.
- `AggregateModel.var`, `tvar`, `stop_loss`, and `limited_expected_value`
  accept `rng` for reproducible Monte Carlo estimates.
- `lossmodels.utils.random` with `RNGLike`, `resolve_rng`, and
  `scipy_random_state`.
- **Fitting under left truncation and right censoring** (deductibles and
  policy limits), the Loss Models individual-data likelihood
  `sum[d_i log f(x_i) + (1-d_i) log S(x_i) - log S(t_i)]`:
  every severity fitter (`fit_exponential`, `fit_gamma`, `fit_lognormal`,
  `fit_pareto`, `fit_weibull`, and the new fitters below) accepts
  `truncation=` and `censored=`; complete-data code paths are unchanged.
  Exponential and Pareto Type I keep closed forms in the censored/truncated
  case (for Type I under truncation, theta is unidentifiable and pinned at
  the smallest observable point, documented in the docstring).
  `payments_to_ground_up` converts per-payment data (payments net of a
  deductible, capped at a maximum payment) into the `(values, truncation,
  censored)` triple; `censored_log_likelihood` and `fit_mle_censored` expose
  the general machinery; `kaplan_meier` gives the product-limit survival
  estimate under left truncation and right censoring.
- **Four new severity fitters** with the same truncation/censoring support:
  `fit_paretoII` (Lomax -- the FAM/ASTAM "Pareto"), `fit_loglogistic`,
  `fit_inverse_gamma`, and `fit_burr`, all registered with
  `fit_best_severity` (the default MLE candidate set grows from five to nine
  families; failed candidates are skipped as before).
- **Truncation/censoring-aware diagnostics**: `log_likelihood`, `aic`, `bic`,
  and `goodness_of_fit` accept `truncation=`/`censored=` and use the
  individual-data likelihood; new `pit_values` returns the
  probability-integral transform `(F(x)-F(t))/(1-F(t))`, exactly Uniform(0,1)
  under the true model for uncensored data with arbitrary truncation points;
  `ks_statistic` uses the PIT under truncation and a Kaplan-Meier comparison
  when censoring is present (`anderson_darling`/`cramer_von_mises` support
  truncation and explicitly reject censored data, where the PIT of the
  uncensored subsample is uniform only on a sub-interval of (0, 1)).
- `fit_best_severity` accepts `truncation=`/`censored=` (MLE method only)
  and scores candidates with the censored-data AIC/BIC.

### Changed
- **Breaking:** `discretize_severity` defaults to `method="midpoint"`
  (previously `"upper"`). Midpoint is the standard, nearly unbiased choice for
  Panjer/FFT input; the `"upper"` bound method biases the discretized mean low
  by roughly h/2 per claim. Pass `method="upper"` explicitly to reproduce old
  results.
- **Breaking (numeric):** `Layer.variance`, `OrdinaryDeductible.variance`, and
  `PolicyLimit.variance` are now computed deterministically from
  E[Y^2] = integral of 2 y S(y) dy instead of by simulation. Results are exact
  and repeatable; `n_sim` is retained for backward compatibility but ignored.
- **Breaking (numeric):** empirical `var` / `tvar` and the PMF-based
  `var_from_pmf` / `tvar_from_pmf` follow the ecosystem-wide convention:
  VaR is the inverted-CDF order statistic and TVaR the Acerbi-Tasche
  average-quantile estimator, so TVaR(q) >= VaR(q) always and PMF atoms at the
  VaR are weighted correctly.

### Fixed
- `tvar_from_pmf` previously overweighted the atom at the VaR when tail
  probability exceeded 1 - q.

## 0.5.0

### Added
- **FAM / ASTAM continuous severity inventory** (Loss Models Appendix A), with the
  exact Klugman exam parameterizations and analytic moments that raise outside
  their existence ranges:
  - Transformed beta family: `Burr`, `InverseBurr`, `GeneralizedPareto`,
    `ParetoII`, `InversePareto`, `Loglogistic`, `Paralogistic`,
    `InverseParalogistic`.
  - Transformed gamma family: `InverseGamma`, `InverseWeibull`,
    `InverseExponential`.
  - Other: `InverseGaussian`, `LogT` (no positive moments), `SingleParameterPareto`.
  - Finite support on `(0, theta)`: `Beta`, `GeneralizedBeta`.
  Each exposes the full severity interface (`pdf`, `cdf`, `quantile`/`ppf`,
  `sample`, `mean`, `variance`, plus the inherited `limited_expected_value` /
  `excess_loss`), so all of them drop straight into `risksim` and `extremeloss`
  via the `.sample()` / `.mean()` contract.
- **The (a, b, 1) frequency class** (Loss Models Appendix B.3):
  - `ZeroTruncated(base)` — a generic wrapper that removes the mass at zero of any
    `(a, b, 0)` model (`Poisson`, `Geometric`, `Binomial`, `NegativeBinomial`) and
    renormalizes.
  - `ZeroModified(base, p0_modified)` — places an arbitrary probability at zero.
  - `Logarithmic(beta)` — the one zero-truncated member with no `(a, b, 0)` parent,
    implemented directly (numerically stable pmf; `numpy.random.logseries` sampling).

### Notes
- `ParetoII` is the FAM/ASTAM two-parameter "Pareto" (Type II / Lomax, support
  `x > 0`). The pre-existing `Pareto` is the Pareto Type I (support `x >= theta`),
  which the FAM tables list as "Single-Parameter Pareto"; it is now also available
  under that name as `SingleParameterPareto`. `Pareto` is unchanged and the splice
  contract that depends on it (`cdf(theta) = 0`) is preserved.
- `GeneralizedPareto` here is the Klugman three-parameter transformed-beta
  distribution, **not** the extreme-value GPD. The EVT distributions (GEV, Gumbel,
  Frechet, GPD, Hill) live in `extremeloss` and are not duplicated here.
- The `(a, b, 1)` wrappers respect the library's existing probability
  parameterization of the base distributions; only `Logarithmic` is parameterized
  by `beta` (it has no `(a, b, 0)` parent). For a beta-parameterized geometric /
  negative binomial, use `p = 1 / (1 + beta)`.

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
