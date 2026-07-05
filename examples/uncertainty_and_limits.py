"""Severity fitting with error bars, all the way to the filed factor.

    claims -> candidate fits -> compare_fits (every criterion, one table)
    -> fit_uncertainty (observed information) -> parameter CIs
    -> increased_limits_table / loss_elimination_table with delta-method
       bands on the factor itself

Run with:  python examples/uncertainty_and_limits.py
"""
from __future__ import annotations

import numpy as np

import lossmodels as lm


def main() -> None:
    rng = np.random.default_rng(42)
    claims = rng.lognormal(mean=9.2, sigma=1.1, size=1_800)

    # ----- which distribution, and by how much ---------------------------- #
    fits = {
        "lognormal": lm.fit_lognormal(claims),
        "gamma": lm.fit_gamma(claims),
        "weibull": lm.fit_weibull(claims),
    }
    print("=== Model comparison (lower is better, except loglik) ===")
    print(lm.compare_fits(fits, claims).round(3).to_string())

    # ----- how well do we know the winner's parameters -------------------- #
    best = fits["lognormal"]
    unc = lm.fit_uncertainty(best, claims)
    print("\n=== Parameter uncertainty ===")
    print(unc.summary().round(4).to_string())

    # ----- carry it to the factors actually filed ------------------------- #
    print("\n=== Increased limits factors with bands ===")
    ilf = lm.increased_limits_table(
        best, limits=[100_000, 250_000, 500_000, 1_000_000],
        base_limit=100_000, uncertainty=unc,
    )
    print(ilf.round(4).to_string())

    print("\n=== Loss elimination ratios with bands ===")
    ler = lm.loss_elimination_table(
        best, deductibles=[5_000, 25_000, 100_000], uncertainty=unc,
    )
    print(ler.round(4).to_string())
    print("\nAn ILF quoted to four decimals off 1,800 claims is false")
    print("precision -- and now the table says exactly how false.")


if __name__ == "__main__":
    main()
