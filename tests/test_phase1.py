"""Unit and regression tests for Phase 1 of PressureIQ."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src import config
from src.data_loader import load_raw_datasets, clean_and_merge_metadata
from src.features import compute_delivery_runs, engineer_ball_features


@pytest.fixture(scope="session")
def raw_data():
    return load_raw_datasets()


@pytest.fixture(scope="session")
def processed_data(raw_data):
    merged_df = clean_and_merge_metadata(
        ball_df=raw_data["ball_by_ball"],
        matches_df=raw_data["matches"],
        competitions_df=raw_data["competitions"],
    )
    return engineer_ball_features(merged_df)


def test_no_data_leakage_ball_one(processed_data):
    """Assert that ball 1 of every innings has zero prior runs and zero prior wickets."""
    ball_ones = processed_data.groupby(["match_id", "innings"]).head(1)
    assert (ball_ones["cum_runs_pre"] == 0).all(), "cum_runs_pre on ball 1 must be 0!"
    assert (ball_ones["wickets_down_pre"] == 0).all(), "wickets_down_pre on ball 1 must be 0!"
    assert (ball_ones["legal_balls_bowled_pre"] == 0).all(), "legal_balls_bowled_pre on ball 1 must be 0!"


def test_illegal_deliveries_not_counted_as_legal(processed_data):
    """Wides and No-Balls must not increment legal deliveries."""
    wides = processed_data[processed_data["extras_type"] == "wide"]
    oversteps = processed_data[processed_data["extras_type"] == "Overstep"]
    assert (~wides["is_legal_ball"]).all(), "Wides must not be marked legal!"
    assert (~oversteps["is_legal_ball"]).all(), "Oversteps must not be marked legal!"


def test_innings_two_target_and_rrr(processed_data):
    """Innings 2 must have valid non-negative targets, runs needed, and RRR."""
    inn2 = processed_data[processed_data["innings"] == 2]
    assert (inn2["runs_needed_pre"] >= 0).all(), "runs_needed_pre must be >= 0!"
    assert (inn2["rrr_pre"] >= 0).all(), "rrr_pre must be >= 0!"
    assert inn2["target_runs"].notna().mean() > 0.98, "target_runs must be populated for inn 2!"


def test_pressure_distribution(processed_data):
    """Pressure flag should partition ~25-45% of deliveries into high-pressure contexts."""
    pressure_pct = processed_data["is_pressure"].mean() * 100.0
    assert 20.0 <= pressure_pct <= 50.0, f"Unexpected pressure delivery percentage: {pressure_pct}%"

