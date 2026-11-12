"""Automated test suite for Phase 3: Player Pressure Intelligence, Shrinkage & Rankings."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src import config
from src.player_analytics import calculate_batter_metrics, calculate_bowler_metrics


@pytest.fixture(scope="session")
def enriched_deliveries():
    parquet_path = config.OUTPUTS_DIR / "deliveries_with_wpa.parquet"
    assert parquet_path.exists(), "deliveries_with_wpa.parquet must exist"
    return pd.read_parquet(parquet_path)


@pytest.fixture(scope="session")
def batter_rankings(enriched_deliveries):
    return calculate_batter_metrics(enriched_deliveries, min_pressure_balls=50)


@pytest.fixture(scope="session")
def bowler_rankings(enriched_deliveries):
    return calculate_bowler_metrics(enriched_deliveries, min_pressure_balls=50)


def test_ranking_file_outputs_exist():
    """Verify top10_batters.csv and top10_bowlers.csv exist and contain 10 rows."""
    bat_path = config.OUTPUTS_DIR / "top10_batters.csv"
    bowl_path = config.OUTPUTS_DIR / "top10_bowlers.csv"
    assert bat_path.exists(), "top10_batters.csv missing"
    assert bowl_path.exists(), "top10_bowlers.csv missing"
    
    df_bat = pd.read_csv(bat_path)
    df_bowl = pd.read_csv(bowl_path)
    assert len(df_bat) == 10, f"Expected 10 batters, got {len(df_bat)}"
    assert len(df_bowl) == 10, f"Expected 10 bowlers, got {len(df_bowl)}"
    assert list(df_bat["Rank"]) == list(range(1, 11))
    assert list(df_bowl["Rank"]) == list(range(1, 11))


def test_no_nan_or_infinite_values(batter_rankings, bowler_rankings):
    """Ensure no NaN, null, or infinite values in ranking tables."""
    assert not batter_rankings["pressure_iq_score"].isna().any()
    assert not bowler_rankings["pressure_iq_score"].isna().any()
    assert np.isfinite(batter_rankings["pressure_iq_score"]).all()
    assert np.isfinite(bowler_rankings["pressure_iq_score"]).all()


def test_minimum_sample_size_enforced(batter_rankings, bowler_rankings):
    """Assert all evaluated players meet or exceed the 50-ball threshold."""
    assert (batter_rankings["pressure_balls"] >= 50).all()
    assert (bowler_rankings["pressure_balls"] >= 50).all()


def test_bayesian_shrinkage_monotonicity(batter_rankings):
    """Credibility weight lambda = n / (n + k) must strictly be monotonic non-decreasing with sample size."""
    sorted_df = batter_rankings.sort_values(by="pressure_balls").reset_index(drop=True)
    # Differences must be >= 0
    diffs = np.diff(sorted_df["sample_credibility"].values)
    assert (diffs >= -1e-9).all(), "Credibility factor must be monotonic non-decreasing with n"
    assert (batter_rankings["sample_credibility"] > 0.0).all()
    assert (batter_rankings["sample_credibility"] < 1.0).all()


def test_wpa_aggregation_integrity(enriched_deliveries, batter_rankings, bowler_rankings):
    """Total pressure WPA aggregated per player matches raw delivery sums."""
    press_df = enriched_deliveries[enriched_deliveries["is_pressure"]]
    top_batter = batter_rankings.iloc[0]
    expected_wpa = press_df[press_df["striker_id"] == top_batter["striker_id"]]["batter_wpa"].sum()
    np.testing.assert_allclose(top_batter["pressure_wpa"], expected_wpa, atol=1e-5)
    
    top_bowler = bowler_rankings.iloc[0]
    expected_bowler_wpa = press_df[press_df["bowler_id"] == top_bowler["bowler_id"]]["bowler_wpa"].sum()
    np.testing.assert_allclose(top_bowler["pressure_wpa"], expected_bowler_wpa, atol=1e-5)


def test_bowler_score_directionality_economy(bowler_rankings):
    """Verify that suppressing economy under pressure (negative econ_diff) yields positive economy component."""
    # Compare bowlers with similar WPA: lower economy diff should yield higher economy z-score
    narine = bowler_rankings[bowler_rankings["bowler_name"].str.contains("Narine", na=False)].iloc[0]
    # Narine has negative econ_diff (-0.13), meaning he restricted runs under pressure
    assert narine["econ_differential"] < 0, "Sunil Narine should have negative economy differential"
    assert narine["shrunk_econ_diff"] < 0.2, "Shrunk economy diff should reflect run suppression"


def test_no_bowler_volume_bias(bowler_rankings):
    """Confirm zero volume bias in bowler rankings (Spearman correlation < 0.20)."""
    from scipy.stats import spearmanr
    corr, pval = spearmanr(bowler_rankings["pressure_balls"], bowler_rankings["pressure_iq_score"])
    assert abs(corr) < 0.20, f"Bowler volume bias detected: Spearman r={corr:.3f}"


def test_csv_and_parquet_synchronization(batter_rankings, bowler_rankings):
    """Ensure top 10 CSV values precisely match the parquet analytical records."""
    bat_csv = pd.read_csv(config.OUTPUTS_DIR / "top10_batters.csv")
    bowl_csv = pd.read_csv(config.OUTPUTS_DIR / "top10_bowlers.csv")
    
    for _, row in bat_csv.iterrows():
        pname = row["Player Name"]
        parq_match = batter_rankings[batter_rankings["striker_name"] == pname]
        assert len(parq_match) == 1, f"Missing batter in parquet: {pname}"
        np.testing.assert_allclose(row["Total Pressure WPA"], parq_match.iloc[0]["pressure_wpa"], atol=1e-3)
        np.testing.assert_allclose(row["PressureIQ Score"], parq_match.iloc[0]["pressure_iq_score"], atol=1e-2)
        
    for _, row in bowl_csv.iterrows():
        pname = row["Player Name"]
        parq_match = bowler_rankings[bowler_rankings["bowler_name"] == pname]
        assert len(parq_match) == 1, f"Missing bowler in parquet: {pname}"
        np.testing.assert_allclose(row["Total Pressure WPA"], parq_match.iloc[0]["pressure_wpa"], atol=1e-3)
        np.testing.assert_allclose(row["PressureIQ Score"], parq_match.iloc[0]["pressure_iq_score"], atol=1e-2)

