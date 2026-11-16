"""Validation test suite for updated delivery run calculations."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.features import compute_delivery_runs as compute_delivery_runs_old
from staging.features_updated import compute_delivery_runs_updated

DATA_UPDATED_PATH = Path("Data Set/ball_by_ball updated.csv")


@pytest.fixture(scope="session")
def updated_df():
    return pd.read_csv(DATA_UPDATED_PATH, low_memory=False)


def test_normal_deliveries(updated_df):
    """Normal deliveries with no extras: total runs must equal runs off bat."""
    normal = updated_df[updated_df["extras_type"].isna() & (updated_df["extras1_runs"] == 0) & (updated_df["extras_2_runs"] == 0)]
    assert len(normal) == 189165
    res = compute_delivery_runs_updated(normal)
    assert (res == normal["runs"]).all()


def test_dot_balls(updated_df):
    """Dot balls with no extras: total runs must equal 0."""
    dots = updated_df[updated_df["extras_type"].isna() & (updated_df["runs"] == 0) & (updated_df["extras1_runs"] == 0) & (updated_df["extras_2_runs"] == 0)]
    res = compute_delivery_runs_updated(dots)
    assert (res == 0).all()


def test_singles_and_boundaries(updated_df):
    """Singles and boundaries off the bat with no extras."""
    fours = updated_df[updated_df["extras_type"].isna() & (updated_df["runs"] == 4) & (updated_df["extras1_runs"] == 0)]
    sixes = updated_df[updated_df["extras_type"].isna() & (updated_df["runs"] == 6) & (updated_df["extras1_runs"] == 0)]
    assert (compute_delivery_runs_updated(fours) == 4).all()
    assert (compute_delivery_runs_updated(sixes) == 6).all()


def test_standard_wides(updated_df):
    """Standard 1-run wides: extras1_runs == 1, runs == 0, total == 1."""
    std_wides = updated_df[(updated_df["extras_type"] == "wide") & (updated_df["extras1_runs"] == 1)]
    assert len(std_wides) == 7107
    res = compute_delivery_runs_updated(std_wides)
    assert (res == 1).all()


def test_multiple_run_wides(updated_df):
    """Multiple-run wides (2, 3, 5 runs): total runs must equal extras1_runs."""
    multi_wides = updated_df[(updated_df["extras_type"] == "wide") & (updated_df["extras1_runs"] > 1)]
    assert len(multi_wides) == 607
    res = compute_delivery_runs_updated(multi_wides)
    assert (res == multi_wides["extras1_runs"]).all()
    assert set(res.unique()) == {2, 3, 4, 5}


def test_no_balls_overstep(updated_df):
    """No-balls (Overstep): 1 penalty run + runs off bat + secondary extras."""
    noballs = updated_df[updated_df["extras_type"] == "Overstep"]
    assert len(noballs) == 576
    res = compute_delivery_runs_updated(noballs)
    expected = 1 + noballs["runs"] + noballs["extras_2_runs"]
    assert (res == expected).all()


def test_byes_and_boundary_byes(updated_df):
    """Byes: 1, 2, 3, 4 (boundary), 5 runs awarded as extras, batter runs 0."""
    byes = updated_df[updated_df["extras_type"] == "byes"]
    assert len(byes) == 529
    res = compute_delivery_runs_updated(byes)
    assert (res == byes["extras1_runs"]).all()
    assert (res[byes["extras1_runs"] == 4] == 4).all()


def test_legbyes_and_boundary_legbyes(updated_df):
    """Leg-byes: 1, 2, 3, 4 (boundary), 5 runs awarded as extras."""
    legbyes = updated_df[updated_df["extras_type"] == "legbyes"]
    assert len(legbyes) == 2748
    res = compute_delivery_runs_updated(legbyes)
    expected = legbyes["runs"] + legbyes["extras1_runs"] + legbyes["extras_2_runs"]
    assert (res == expected).all()


def test_penalty_runs(updated_df):
    """Penalty runs: 5 penalty runs awarded."""
    penalties = updated_df[updated_df["extras_type"] == "Penalty"]
    assert len(penalties) == 1
    res = compute_delivery_runs_updated(penalties)
    assert res.iloc[0] == 5


def test_wicket_deliveries(updated_df):
    """Wicket deliveries: runs can still be scored (e.g. runouts, caught off bat)."""
    wickets = updated_df[updated_df["is_wicket"] == 1]
    assert len(wickets) == 10881
    res = compute_delivery_runs_updated(wickets)
    assert (res >= 0).all()


def test_run_outs(updated_df):
    """Run outs: striker may have completed runs before dismissal."""
    runouts = updated_df[updated_df["dismissal_type"] == "RunOut"]
    assert len(runouts) == 600
    res = compute_delivery_runs_updated(runouts)
    assert (res >= 0).all()
    assert (res == runouts["runs"] + runouts["extras1_runs"] + runouts["extras_2_runs"]).all()


def test_super_overs(updated_df):
    """Super overs (innings 3 & 4): runs correctly calculated for all deliveries."""
    super_overs = updated_df[updated_df["innings"].isin([3, 4])]
    assert len(super_overs) == 38
    res = compute_delivery_runs_updated(super_overs)
    assert (res == super_overs["runs"] + super_overs["extras1_runs"] + super_overs["extras_2_runs"]).all()


def compute_delivery_runs_legacy(df: pd.DataFrame) -> pd.Series:
    """Legacy delivery runs implementation for diff reconciliation."""
    runs = df["runs"].fillna(0).astype(int).values
    extras_2 = df["extras_2_runs"].fillna(0).astype(int).values
    extras_type = df["extras_type"].values

    is_wide = extras_type == "wide"
    is_nb = extras_type == "Overstep"
    is_other_extra = np.isin(extras_type, ["legbyes", "byes", "other", "Penalty"])

    total_runs = runs.copy()
    total_runs = np.where(is_wide, 1 + extras_2, total_runs)
    total_runs = np.where(is_nb, 1 + runs + extras_2, total_runs)
    total_runs = np.where(
        is_other_extra,
        np.where(runs == 0, 1 + extras_2, runs + extras_2),
        total_runs,
    )
    return pd.Series(total_runs, index=df.index, dtype=int)


def test_total_reconciliation_exact_difference(updated_df):
    """Total runs difference between old and updated calculation must be exactly 2,929 runs."""
    old_calc = compute_delivery_runs_legacy(updated_df)
    new_calc = compute_delivery_runs_updated(updated_df)
    prod_calc = compute_delivery_runs_old(updated_df)
    assert old_calc.sum() == 281328
    assert new_calc.sum() == 284257
    assert prod_calc.sum() == 284257
    assert (new_calc.sum() - old_calc.sum()) == 2929
