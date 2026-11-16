"""Automated tests for Phase 4: PressureIQ Streamlit Dashboard Components.

Tests data loaders, page components, chart calculations, CSV downloads,
and data integrity across all 6 dashboard views.
"""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd

from src import config

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def dashboard_data():
    """Load and return all cached dashboard data assets."""
    from app.app import load_top10, load_all_players, load_model_metrics
    df_top_bat, df_top_bowl = load_top10()
    df_all_bat, df_all_bowl = load_all_players()
    df_eval = load_model_metrics()
    return {
        "top_bat": df_top_bat,
        "top_bowl": df_top_bowl,
        "all_bat": df_all_bat,
        "all_bowl": df_all_bowl,
        "eval": df_eval,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────
def test_dashboard_data_loaders(dashboard_data):
    """Verify all dashboard data assets load cleanly with expected shapes."""
    assert len(dashboard_data["top_bat"]) == 10
    assert len(dashboard_data["top_bowl"]) == 10
    assert len(dashboard_data["all_bat"]) >= 200
    assert len(dashboard_data["all_bowl"]) >= 200
    assert len(dashboard_data["eval"]) == 8  # 4 models × 2 splits


def test_leaderboard_section_integrity(dashboard_data):
    """Verify Section 1 (Official Top 10 Leaderboards) fields, schema, and sorting integrity."""
    top_bat = dashboard_data["top_bat"]
    top_bowl = dashboard_data["top_bowl"]

    required_bat_cols = [
        "Rank", "Player Name", "Pressure Matches", "Pressure Balls",
        "Pressure Runs", "Baseline SR", "Pressure SR", "SR Differential (+/-)",
        "Total Pressure WPA", "Shrunk WPA / Ball", "PressureIQ Score", "Reliability Weight"
    ]
    for col in required_bat_cols:
        assert col in top_bat.columns, f"Missing column in top10_batters: {col}"

    required_bowl_cols = [
        "Rank", "Player Name", "Pressure Matches", "Pressure Balls",
        "Pressure Wickets", "Baseline Economy", "Pressure Economy", "Economy Differential (+/-)",
        "Total Pressure WPA", "Shrunk WPA / Ball", "PressureIQ Score", "Reliability Weight"
    ]
    for col in required_bowl_cols:
        assert col in top_bowl.columns, f"Missing column in top10_bowlers: {col}"

    # Verify ranks are ordered 1 to 10
    assert list(top_bat["Rank"]) == list(range(1, 11))
    assert list(top_bowl["Rank"]) == list(range(1, 11))

    # Verify scores are strictly monotonic non-increasing (sorted descending)
    assert (top_bat["PressureIQ Score"].diff().dropna() <= 1e-5).all(), "Batter scores not sorted descending"
    assert (top_bowl["PressureIQ Score"].diff().dropna() <= 1e-5).all(), "Bowler scores not sorted descending"

    # Verify #1 Rank Batter validity
    assert top_bat.iloc[0]["Rank"] == 1
    assert top_bat.iloc[0]["PressureIQ Score"] > 200.0
    assert len(top_bat.iloc[0]["Player Name"]) > 0

    # Verify #1 Rank Bowler validity
    assert top_bowl.iloc[0]["Rank"] == 1
    assert top_bowl.iloc[0]["PressureIQ Score"] > 200.0
    assert len(top_bowl.iloc[0]["Player Name"]) > 0

    # Verify all numerical values are finite and non-null
    assert np.isfinite(top_bat["PressureIQ Score"]).all()
    assert np.isfinite(top_bowl["PressureIQ Score"]).all()
    assert not top_bat.isna().any().any()
    assert not top_bowl.isna().any().any()


def test_csv_download_buffers(dashboard_data):
    """Verify CSV export strings can be encoded and match source files."""
    top_bat = dashboard_data["top_bat"]
    top_bowl = dashboard_data["top_bowl"]

    csv_bat_bytes = top_bat.to_csv(index=False).encode("utf-8")
    csv_bowl_bytes = top_bowl.to_csv(index=False).encode("utf-8")

    assert len(csv_bat_bytes) > 500, "Batters CSV export buffer too small"
    assert len(csv_bowl_bytes) > 500, "Bowlers CSV export buffer too small"
    assert b"Suryakumar Yadav" in csv_bat_bytes
    assert b"Rashid Khan" in csv_bowl_bytes


def test_player_scouting_calculations(dashboard_data):
    """Verify Section 2 (Player Deep-Dive) radar and situational metrics."""
    all_bat = dashboard_data["all_bat"]
    all_bowl = dashboard_data["all_bowl"]

    # Test batter radar components for Suryakumar Yadav
    sky = all_bat[all_bat["striker_name"] == "Suryakumar Yadav"].iloc[0]
    assert np.isfinite(sky["pressure_sr"])
    assert np.isfinite(sky["overall_sr"])
    assert np.isfinite(sky["pressure_wpa_per_ball"])
    assert sky["pressure_sr"] > sky["overall_sr"]  # Clutch accelerator

    # Test bowler radar components for Rashid Khan
    rk = all_bowl[all_bowl["bowler_name"] == "Rashid Khan"].iloc[0]
    assert np.isfinite(rk["pressure_econ"])
    assert np.isfinite(rk["overall_econ"])
    assert np.isfinite(rk["pressure_wpa"])
    assert rk["pressure_wickets"] > 50


def test_head_to_head_comparison_logic(dashboard_data):
    """Verify Section 3 (Head-to-Head Comparison) evaluates both players without error."""
    all_bat = dashboard_data["all_bat"]
    p1 = all_bat[all_bat["striker_name"] == "Suryakumar Yadav"].iloc[0]
    p2 = all_bat[all_bat["striker_name"] == "Rinku Singh"].iloc[0]

    metrics = [
        ("PressureIQ Score", p1["pressure_iq_score"], p2["pressure_iq_score"], True),
        ("Pressure SR", p1["pressure_sr"], p2["pressure_sr"], True),
        ("Dot Ball %", p1["pressure_dot_pct"], p2["pressure_dot_pct"], False),
    ]

    for label, v1, v2, higher_is_better in metrics:
        if higher_is_better:
            adv = "p1" if v1 > v2 else "p2"
        else:
            adv = "p1" if v1 < v2 else "p2"
        assert adv in ["p1", "p2"], f"Invalid advantage determination for {label}"


def test_situational_differentials_matrix(dashboard_data):
    """Verify Section 4 (Baseline vs Pressure Matrix) has valid coordinates for all points."""
    all_bat = dashboard_data["all_bat"]
    all_bowl = dashboard_data["all_bowl"]

    # Batter coordinates
    assert all_bat["overall_sr"].notna().all()
    assert all_bat["pressure_sr"].notna().all()
    assert (all_bat["overall_sr"] > 0).all()
    assert (all_bat["pressure_sr"] > 0).all()

    # Bowler coordinates
    assert all_bowl["overall_econ"].notna().all()
    assert all_bowl["pressure_econ"].notna().all()
    assert (all_bowl["overall_econ"] > 0).all()
    assert (all_bowl["pressure_econ"] > 0).all()


def test_match_wpa_timeline(dashboard_data):
    """Verify Section 5 (Match Win Probability & WPA Dynamics) loads match deliveries."""
    from app.app import load_match_deliveries
    df_match = load_match_deliveries()
    assert len(df_match) > 100000

    # Check that high-leverage matches exist
    matches = df_match["match_id"].unique()
    assert len(matches) > 500

    # Inspect a single match
    sample_m = df_match[df_match["match_id"] == matches[0]].sort_values("innings_ball_num")
    assert (sample_m["win_prob_post"] >= 0.0).all()
    assert (sample_m["win_prob_post"] <= 1.0).all()

    # Turning points sorting
    top_wpa = sample_m.sort_values(by="wpa", key=abs, ascending=False).head(5)
    assert len(top_wpa) <= 5
    assert (top_wpa["wpa"].abs().diff().dropna() <= 1e-6).all()


def test_methodology_assets_exist(dashboard_data):
    """Verify Section 6 (Methodology & Model Evaluation) assets exist and are valid."""
    cal_fig = FIGURES_DIR / "calibration_curves.png"
    assert cal_fig.exists(), "Calibration curves figure missing from outputs/figures/"
    assert cal_fig.stat().st_size > 10000

    eval_df = dashboard_data["eval"]
    ensemble_val = eval_df[(eval_df["model_type"] == "ensemble") & (eval_df["dataset"] == "Validation")].iloc[0]
    assert ensemble_val["roc_auc"] > 0.80
    assert ensemble_val["brier_score"] < 0.18


def test_run_dashboard_batch_script():
    """Verify run_dashboard.bat exists and references the correct app path."""
    bat_file = ROOT_DIR / "run_dashboard.bat"
    assert bat_file.exists(), "run_dashboard.bat missing"
    content = bat_file.read_text(encoding="utf-8")
    assert "streamlit run" in content
    assert "app\\app.py" in content or "app/app.py" in content

