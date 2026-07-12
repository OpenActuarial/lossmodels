"""Experience seam: same fits from the object as from raw arrays."""
import numpy as np
import pandas as pd
import pytest
from actuarialpy import Experience, Source

from lossmodels.estimation import fit_best_frequency, fit_best_severity
from lossmodels.integrations.actuarialpy import (
    claim_amounts,
    fit_frequency_from_experience,
    fit_severity_from_experience,
    frequency_counts,
)


def _listing():
    rng = np.random.default_rng(7)
    dates = pd.to_datetime("2025-01-15") + pd.to_timedelta(rng.integers(0, 365, 300), "D")
    return pd.DataFrame({"claim_id": range(300), "incurred_date": dates,
                         "paid_amount": rng.gamma(2.0, 5_000.0, 300)})


def test_severity_fit_matches_array_path():
    df = _listing()
    exp = Experience(df, expense="paid_amount", date="incurred_date",
                     exposure_keys="claim_id")
    assert claim_amounts(exp) == pytest.approx(df["paid_amount"].to_numpy())
    via_exp = fit_severity_from_experience(exp)
    via_arr = fit_best_severity(df["paid_amount"].to_numpy())
    assert via_exp["best_name"] == via_arr["best_name"]


def test_frequency_counts_rows_per_period_and_count_role():
    df = _listing()
    exp = Experience(df, expense="paid_amount", date="incurred_date")
    counts = frequency_counts(exp, freq="M")
    assert counts.sum() == 300
    via_exp = fit_frequency_from_experience(exp)
    via_arr = fit_best_frequency(counts)
    assert via_exp["best_name"] == via_arr["best_name"]


def test_aggregated_tab_is_refused_or_warned():
    df = _listing()
    months = pd.date_range("2025-01-01", periods=3, freq="MS")
    membership = pd.DataFrame([{"claimant": c, "month": t, "mm": 1.0}
                               for c in ("a", "b") for t in months])
    lines = df.head(4).assign(claimant="a", ct=["x", "y", "x", "y"])
    tab = Experience.from_tables(
        membership, grain=["claimant", "month"], exposure="mm",
        sources=[Source(lines, expense="paid_amount", wide_by="ct",
                         date="incurred_date")],
        date="month", period="M", unmatched="warn")
    with pytest.raises(ValueError, match="aggregated experience tab"):
        claim_amounts(tab)
    exp2 = Experience(df.assign(mm=1.0), expense="paid_amount",
                      exposure="mm", date="incurred_date")
    with pytest.warns(UserWarning, match="claim-level"):
        claim_amounts(exp2)


def test_by_fits_one_model_per_claim_type():
    df = _listing().assign(claim_type=np.where(np.arange(300) % 3 == 0,
                                               "inpatient", "outpatient"))
    exp = Experience(df, expense="paid_amount", date="incurred_date",
                     dimensions="claim_type", exposure_keys="claim_id")
    fits = fit_severity_from_experience(exp, by="claim_type")
    assert set(fits) == {"inpatient", "outpatient"}
    one = fit_severity_from_experience(exp.filter(query="claim_type == 'inpatient'"))
    assert fits["inpatient"]["best_name"] == one["best_name"]

def test_experienceset_routes_to_the_named_listing():
    from actuarialpy import ExperienceSet, Source
    df = _listing()
    grain = pd.DataFrame({"claim_id": df["claim_id"], "n": 1.0})
    book = ExperienceSet.from_tables(
        grain, grain=["claim_id"], exposure="n",
        sources=[Source(df, expense="paid_amount", name="claims")])
    assert claim_amounts(book) == pytest.approx(claim_amounts(book["claims"]))
