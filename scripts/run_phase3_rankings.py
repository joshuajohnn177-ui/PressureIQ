"""Execution script for Phase 3: Player Pressure Intelligence & Top 10 Rankings.

Calculates full player performance metrics, applies empirical Bayes shrinkage,
exports official submission rankings (top10_batters.csv, top10_bowlers.csv),
and generates summary statistics and visualizations.
"""

from pathlib import Path
import sys
import json
import logging
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.player_analytics import calculate_batter_metrics, calculate_bowler_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3Runner")


def run_phase3():
    """Execute complete Phase 3 workflow."""
    logger.info("=== STARTING PHASE 3: PLAYER PRESSURE INTELLIGENCE & RANKINGS ===")
    
    # 1. Load deliveries with WPA
    parquet_path = config.OUTPUTS_DIR / "deliveries_with_wpa.parquet"
    logger.info("Loading enriched deliveries from %s...", parquet_path)
    df = pd.read_parquet(parquet_path)
    
    # 2. Calculate Batting Rankings
    batters_all = calculate_batter_metrics(df, min_pressure_balls=50)
    top10_batters = batters_all.head(10).copy()
    
    # Format Top 10 Batters output
    batter_export_cols = [
        "rank",
        "striker_name",
        "pressure_matches",
        "pressure_balls",
        "pressure_runs",
        "overall_sr",
        "pressure_sr",
        "sr_differential",
        "pressure_wpa",
        "shrunk_wpa_per_ball",
        "pressure_iq_score",
        "sample_credibility"
    ]
    
    top10_batters_clean = top10_batters[batter_export_cols].rename(columns={
        "rank": "Rank",
        "striker_name": "Player Name",
        "pressure_matches": "Pressure Matches",
        "pressure_balls": "Pressure Balls",
        "pressure_runs": "Pressure Runs",
        "overall_sr": "Baseline SR",
        "pressure_sr": "Pressure SR",
        "sr_differential": "SR Differential (+/-)",
        "pressure_wpa": "Total Pressure WPA",
        "shrunk_wpa_per_ball": "Shrunk WPA / Ball",
        "pressure_iq_score": "PressureIQ Score",
        "sample_credibility": "Reliability Weight"
    })
    
    # Round numerical columns for clean presentation
    top10_batters_clean["Baseline SR"] = top10_batters_clean["Baseline SR"].round(2)
    top10_batters_clean["Pressure SR"] = top10_batters_clean["Pressure SR"].round(2)
    top10_batters_clean["SR Differential (+/-)"] = top10_batters_clean["SR Differential (+/-)"].round(2)
    top10_batters_clean["Total Pressure WPA"] = top10_batters_clean["Total Pressure WPA"].round(4)
    top10_batters_clean["Shrunk WPA / Ball"] = top10_batters_clean["Shrunk WPA / Ball"].round(5)
    top10_batters_clean["PressureIQ Score"] = top10_batters_clean["PressureIQ Score"].round(2)
    top10_batters_clean["Reliability Weight"] = top10_batters_clean["Reliability Weight"].round(3)
    
    top10_batters_path = config.OUTPUTS_DIR / "top10_batters.csv"
    top10_batters_clean.to_csv(top10_batters_path, index=False)
    logger.info("Saved Top 10 Batters to %s", top10_batters_path)
    
    # Also save full eligible batters table
    batters_all.to_parquet(config.OUTPUTS_DIR / "all_batters_pressure_metrics.parquet", index=False)
    
    # 3. Calculate Bowling Rankings
    bowlers_all = calculate_bowler_metrics(df, min_pressure_balls=50)
    top10_bowlers = bowlers_all.head(10).copy()
    
    bowler_export_cols = [
        "rank",
        "bowler_name",
        "pressure_matches",
        "pressure_balls",
        "pressure_wickets",
        "overall_econ",
        "pressure_econ",
        "econ_differential",
        "pressure_wpa",
        "shrunk_wpa_per_ball",
        "pressure_iq_score",
        "sample_credibility"
    ]
    
    top10_bowlers_clean = top10_bowlers[bowler_export_cols].rename(columns={
        "rank": "Rank",
        "bowler_name": "Player Name",
        "pressure_matches": "Pressure Matches",
        "pressure_balls": "Pressure Balls",
        "pressure_wickets": "Pressure Wickets",
        "overall_econ": "Baseline Economy",
        "pressure_econ": "Pressure Economy",
        "econ_differential": "Economy Differential (+/-)",
        "pressure_wpa": "Total Pressure WPA",
        "shrunk_wpa_per_ball": "Shrunk WPA / Ball",
        "pressure_iq_score": "PressureIQ Score",
        "sample_credibility": "Reliability Weight"
    })
    
    top10_bowlers_clean["Baseline Economy"] = top10_bowlers_clean["Baseline Economy"].round(2)
    top10_bowlers_clean["Pressure Economy"] = top10_bowlers_clean["Pressure Economy"].round(2)
    top10_bowlers_clean["Economy Differential (+/-)"] = top10_bowlers_clean["Economy Differential (+/-)"].round(2)
    top10_bowlers_clean["Total Pressure WPA"] = top10_bowlers_clean["Total Pressure WPA"].round(4)
    top10_bowlers_clean["Shrunk WPA / Ball"] = top10_bowlers_clean["Shrunk WPA / Ball"].round(5)
    top10_bowlers_clean["PressureIQ Score"] = top10_bowlers_clean["PressureIQ Score"].round(2)
    top10_bowlers_clean["Reliability Weight"] = top10_bowlers_clean["Reliability Weight"].round(3)
    
    top10_bowlers_path = config.OUTPUTS_DIR / "top10_bowlers.csv"
    top10_bowlers_clean.to_csv(top10_bowlers_path, index=False)
    logger.info("Saved Top 10 Bowlers to %s", top10_bowlers_path)
    
    bowlers_all.to_parquet(config.OUTPUTS_DIR / "all_bowlers_pressure_metrics.parquet", index=False)
    
    # 4. Print Verification Summaries
    print("\n" + "="*80)
    print("🏆 OFFICIAL TOP 10 BATTERS UNDER PRESSURE (PressureIQ)")
    print("="*80)
    print(top10_batters_clean.to_string(index=False))
    
    print("\n" + "="*80)
    print("🎯 OFFICIAL TOP 10 BOWLERS UNDER PRESSURE (PressureIQ)")
    print("="*80)
    print(top10_bowlers_clean.to_string(index=False))
    print("="*80 + "\n")
    
    logger.info("=== PHASE 3 COMPLETED SUCCESSFULLY ===")
    return top10_batters_clean, top10_bowlers_clean


if __name__ == "__main__":
    run_phase3()

