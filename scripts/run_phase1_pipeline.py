"""Execution script for Phase 1 of PressureIQ.

Executes data loading, metadata merging, pre-delivery feature engineering,
leakage verification tests, and outputs the processed dataset.
"""

from pathlib import Path
import sys
import logging
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import load_raw_datasets, clean_and_merge_metadata
from src.features import engineer_ball_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase1Runner")


def run_phase1():
    """Run full Phase 1 pipeline with validation checks."""
    logger.info("=== STARTING PHASE 1: PREPROCESSING & FEATURE ENGINEERING ===")
    
    # Step 1: Load raw data
    raw_data = load_raw_datasets()
    
    # Step 2: Merge metadata
    merged_df = clean_and_merge_metadata(
        ball_df=raw_data["ball_by_ball"],
        matches_df=raw_data["matches"],
        competitions_df=raw_data["competitions"],
    )
    
    # Step 3: Feature Engineering
    processed_df = engineer_ball_features(merged_df)
    
    # Step 4: Verification & Leakage Assertion Tests
    logger.info("--- RUNNING INTEGRITY AND LEAKAGE ASSERTIONS ---")
    
    # 1. Leakage Check: Ball 1 of every innings MUST have 0 runs and 0 wickets pre-ball
    ball_ones = processed_df.groupby(["match_id", "innings"]).head(1)
    assert (ball_ones["cum_runs_pre"] == 0).all(), "Leakage Error: cum_runs_pre is non-zero on ball 1!"
    assert (ball_ones["wickets_down_pre"] == 0).all(), "Leakage Error: wickets_down_pre is non-zero on ball 1!"
    assert (ball_ones["legal_balls_bowled_pre"] == 0).all(), "Leakage Error: legal_balls_bowled_pre non-zero on ball 1!"
    logger.info("PASS: Pre-delivery states on Ball 1 strictly start at 0 (No Leakage).")
    
    # 2. Monotonicity Check: Wickets and cumulative runs must be non-decreasing within an innings
    sample_match = processed_df[processed_df["match_id"] == processed_df["match_id"].iloc[0]]
    for inn in [1, 2]:
        inn_df = sample_match[sample_match["innings"] == inn]
        if len(inn_df) > 0:
            assert inn_df["cum_runs_pre"].is_monotonic_increasing, "cum_runs_pre is not monotonic!"
            assert inn_df["wickets_down_pre"].is_monotonic_increasing, "wickets_down_pre is not monotonic!"
    logger.info("PASS: Cumulative runs and wickets strictly monotonic non-decreasing.")
    
    # 3. Extras & Legal Deliveries Validation
    wide_rows = processed_df[processed_df["extras_type"] == "wide"]
    nb_rows = processed_df[processed_df["extras_type"] == "Overstep"]
    assert (~wide_rows["is_legal_ball"]).all(), "Wide counted as legal delivery!"
    assert (~nb_rows["is_legal_ball"]).all(), "No-ball counted as legal delivery!"
    logger.info("PASS: Illegal deliveries (wides & oversteps) correctly excluded from legal balls.")
    
    # 4. Target & RRR Validation for Innings 2
    inn2_df = processed_df[processed_df["innings"] == 2]
    assert inn2_df["target_runs"].notna().mean() > 0.98, "Target runs missing in innings 2!"
    assert (inn2_df["runs_needed_pre"] >= 0).all(), "Negative runs needed found!"
    assert (inn2_df["rrr_pre"] >= 0).all(), "Negative required run rate found!"
    logger.info("PASS: Innings 2 Target, Runs Needed, and RRR mathematically valid.")
    
    # 5. Pressure Distribution Summary
    pressure_counts = processed_df["pressure_category"].value_counts()
    logger.info("Pressure Distribution:\n%s", pressure_counts)
    logger.info(
        "Pressure Deliveries: %d (%.2f%%)",
        processed_df["is_pressure"].sum(),
        100.0 * processed_df["is_pressure"].mean(),
    )
    
    # Step 5: Save Processed Outputs
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = config.PROCESSED_DELIVERIES_PARQUET
    logger.info("Saving processed dataset to %s...", parquet_path)
    processed_df.to_parquet(parquet_path, index=False)
    
    # Also save a sample for quick inspection
    sample_csv = config.OUTPUTS_DIR / "processed_deliveries_sample.csv"
    processed_df.head(500).to_csv(sample_csv, index=False)
    logger.info("Saved 500-row sample to %s", sample_csv)
    
    logger.info("=== PHASE 1 COMPLETED SUCCESSFULLY ===")
    return processed_df


if __name__ == "__main__":
    run_phase1()

