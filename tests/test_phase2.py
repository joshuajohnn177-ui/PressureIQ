"""Automated test suite for Phase 2: Win Probability Modeling, Calibration & WPA Engine."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import json

from src import config
from src.win_prob_model import DualInningsWinProbModel, prepare_features
from src.wpa_engine import compute_ball_wpa, compute_leverage_index


@pytest.fixture(scope="session")
def valid_deliveries():
    df = pd.read_parquet(config.PROCESSED_DELIVERIES_PARQUET)
    valid = df[df["win_team_id"].notna() & df["innings"].isin([1, 2])].copy()
    valid["batting_team_won"] = (valid["batting_team_id"] == valid["win_team_id"]).astype(int)
    valid["is_ipl"] = (valid["competition_tier"] == "IPL").astype(int)
    return valid


@pytest.fixture(scope="session")
def trained_model():
    model_path = config.OUTPUTS_DIR / "win_prob_model.joblib"
    if model_path.exists():
        return DualInningsWinProbModel.load(model_path)
    df = pd.read_parquet(config.PROCESSED_DELIVERIES_PARQUET)
    valid = df[df["win_team_id"].notna() & df["innings"].isin([1, 2])].copy()
    valid["batting_team_won"] = (valid["batting_team_id"] == valid["win_team_id"]).astype(int)
    valid["is_ipl"] = (valid["competition_tier"] == "IPL").astype(int)
    m = DualInningsWinProbModel(model_type="ensemble")
    m.fit(valid.head(5000))
    return m


def test_target_variable_integrity(valid_deliveries):
    """Assert binary target exists, has no nulls, and is balanced."""
    assert "batting_team_won" in valid_deliveries.columns
    assert valid_deliveries["batting_team_won"].isna().sum() == 0
    assert set(valid_deliveries["batting_team_won"].unique()) == {0, 1}
    win_rate = valid_deliveries["batting_team_won"].mean()
    assert 0.40 <= win_rate <= 0.60, f"Win rate skewed unexpectedly: {win_rate}"


def test_chronological_match_splitting():
    """Assert zero match overlap between splits and chronological order."""
    split_path = config.OUTPUTS_DIR / "split_summary.json"
    if not split_path.exists():
        pytest.skip("split_summary.json not found yet")
        
    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
        
    assert split_info["train_matches"] > 0
    assert split_info["val_matches"] > 0
    assert split_info["test_matches"] > 0
    assert split_info["train_date_range"][1] <= split_info["val_date_range"][0]
    assert split_info["val_date_range"][1] <= split_info["test_date_range"][0]


def test_probability_predictions_range(valid_deliveries, trained_model):
    """Win probabilities must be bounded strictly in [0, 1]."""
    sample = valid_deliveries.head(1000)
    probs = trained_model.predict_proba(sample)
    assert len(probs) == len(sample)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]!"
    assert not np.isnan(probs).any(), "NaN found in predicted probabilities!"


def test_wides_and_extras_attribution(valid_deliveries, trained_model):
    """Assert striker receives 0.0 on wides and pure byes/legbyes."""
    sample = valid_deliveries.head(3000).copy()
    sample_wpa = compute_ball_wpa(sample, trained_model)
    
    # Wides check
    wides = sample_wpa[sample_wpa["extras_type"] == "wide"]
    if len(wides) > 0:
        np.testing.assert_allclose(wides["batter_wpa"], 0.0, atol=1e-6)
        
    # Pure byes / legbyes check (runs == 0)
    byes_lb = sample_wpa[sample_wpa["extras_type"].isin(["byes", "legbyes"]) & (sample_wpa["runs"] == 0)]
    if len(byes_lb) > 0:
        np.testing.assert_allclose(byes_lb["batter_wpa"], 0.0, atol=1e-6)
        np.testing.assert_allclose(byes_lb["bowler_wpa"], 0.0, atol=1e-6)


def test_non_striker_runout_attribution(valid_deliveries, trained_model):
    """Assert striker is not penalized when non-striker is run out."""
    runouts = valid_deliveries[
        (valid_deliveries["dismissal_type"] == "RunOut")
        & (valid_deliveries["dismissed_player_id"] == valid_deliveries["non_striker_id"])
    ].head(50).copy()
    
    if len(runouts) > 0:
        wpa_df = compute_ball_wpa(runouts, trained_model)
        # Striker must not have negative wicket penalty
        assert (wpa_df["batter_wpa"] >= 0.0).all(), "Striker unfairly penalized on non-striker runout!"
        # Non-striker WPA should capture the negative delta
        assert (wpa_df["non_striker_wpa"] <= 0.0).any()


def test_leverage_index_boundary_conditions(valid_deliveries, trained_model):
    """Assert Leverage Index is non-negative, finite, and accounts for boundary win / all-out loss."""
    sample = valid_deliveries.head(1000).copy()
    sample_wpa = compute_ball_wpa(sample, trained_model)
    sample_li = compute_leverage_index(sample_wpa, trained_model)
    
    assert (sample_li["leverage_index"] >= 0.0).all(), "Negative Leverage Index found!"
    assert not sample_li["leverage_index"].isna().any(), "NaN found in Leverage Index!"
    assert np.isfinite(sample_li["leverage_index"]).all(), "Infinite value found in Leverage Index!"


def test_match_transitions_continuity(valid_deliveries, trained_model):
    """Assert Innings 1 terminal transition and Innings 2 terminal balls are valid probabilities."""
    sample = valid_deliveries.head(2000).copy()
    sample_wpa = compute_ball_wpa(sample, trained_model)
    
    # Check post-probabilities
    assert (sample_wpa["win_prob_post"] >= 0.0).all() and (sample_wpa["win_prob_post"] <= 1.0).all()
    assert (sample_wpa["wpa"] >= -1.0).all() and (sample_wpa["wpa"] <= 1.0).all()
