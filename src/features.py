"""Feature engineering module for PressureIQ.

Calculates pre-delivery match state features with strict leakage prevention,
incorporating accurate extras rules, legal balls tracking, current run rate,
target runs, required run rate, match phases, and situational pressure.
"""

from typing import Tuple
import logging
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def compute_delivery_runs(df: pd.DataFrame) -> pd.Series:
    """Compute exact total runs awarded to the team on each delivery.
    
    Updated Dataset Encoding:
    - 'runs': Runs scored directly off the bat by the striker (e.g. 0, 1, 2, 3, 4, 6).
    - 'extras1_runs': Primary extra runs awarded to the batting team:
        - wide: 1 (normal wide), 2 (wide+run), 3, 4, 5 (wide boundary)
        - Overstep / No Ball: 1 (penalty run)
        - byes: 1, 2, 3, 4 (bye boundary), 5
        - legbyes: 1, 2, 3, 4 (legbye boundary), 5
        - other: 1
        - Penalty: 5
    - 'extras_2_runs': Secondary extra runs occurring on the delivery:
        - On Overstep: additional runs scored (e.g. 1 bye/legbye or 4 boundary)
        - On other: additional boundary or overthrow runs (e.g. 1 or 4)

    Cricket Scoring Invariant:
    Total runs on any delivery is the exact sum of runs off the bat plus all extras.
    Because runs and extras_2_runs are mutually exclusive (0 overlaps across 200,999 balls),
    the arithmetic total is:
        total_runs = runs + extras1_runs + extras_2_runs
    """
    runs = df["runs"].fillna(0).astype(int).values
    extras_1 = df["extras1_runs"].fillna(0).astype(int).values
    extras_2 = df["extras_2_runs"].fillna(0).astype(int).values
    
    total_runs = runs + extras_1 + extras_2
    return pd.Series(total_runs, index=df.index, dtype=int)


def compute_match_phases(df: pd.DataFrame) -> pd.Series:
    """Classify delivery into match phase based on overid (1-indexed)."""
    conditions = [
        df["overid"] <= 6,
        (df["overid"] >= 7) & (df["overid"] <= 15),
        df["overid"] >= 16,
    ]
    choices = [config.PHASE_POWERPLAY, config.PHASE_MIDDLE, config.PHASE_DEATH]
    return pd.Series(np.select(conditions, choices, default=config.PHASE_MIDDLE), index=df.index)


def engineer_ball_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer leak-free, pre-delivery match state features across all deliveries.
    
    Guarantees:
    - Every pre-* feature represents the exact game state BEFORE the ball is bowled.
    - Accurately tracks legal deliveries vs extras.
    - Correctly calculates innings 1 total and innings 2 target & required run rate.
    
    Args:
        merged_df: DataFrame with deliveries joined to match metadata.
        
    Returns:
        pd.DataFrame: Augmented DataFrame with pre-delivery state and pressure indicators.
    """
    logger.info("Computing delivery-level runs and legal ball status...")
    df = merged_df.copy()
    
    # 1. Total runs on this ball
    df["ball_total_runs"] = compute_delivery_runs(df)
    
    # 2. Legal ball flag (Wides and No-balls are re-bowled, not legal balls)
    df["is_legal_ball"] = ~df["extras_type"].isin(config.ILLEGAL_DELIVERY_TYPES)
    
    # 3. Match Phase
    df["match_phase"] = compute_match_phases(df)
    
    # 4. Binary Ground Truth Label for Win Probability (Batting team wins?)
    df["is_batting_team_win"] = np.where(
        df["win_team_id"].isna(),
        np.nan,
        (df["batting_team_id"] == df["win_team_id"]).astype(float),
    )
    
    logger.info("Vectorized calculation of cumulative pre-delivery states...")
    inn_group = df.groupby(["match_id", "innings"], sort=False)
    
    # Cumulative runs strictly BEFORE this ball
    df["cum_runs_pre"] = inn_group["ball_total_runs"].cumsum() - df["ball_total_runs"]
    
    # Cumulative wickets strictly BEFORE this ball
    df["wickets_down_pre"] = inn_group["is_wicket"].cumsum() - df["is_wicket"].astype(int)
    
    # Cumulative legal balls bowled strictly BEFORE this ball
    df["legal_balls_bowled_pre"] = inn_group["is_legal_ball"].cumsum() - df["is_legal_ball"].astype(int)
    
    # Calculate max overs per innings (accounting for revised overs in rain-affected matches)
    max_overs = np.where(
        (df["innings"] == 1) & df["first_inn_revised_overs"].notna() & (df["first_inn_revised_overs"] > 0),
        df["first_inn_revised_overs"],
        np.where(
            (df["innings"] == 2) & df["second_inn_revised_overs"].notna() & (df["second_inn_revised_overs"] > 0),
            df["second_inn_revised_overs"],
            config.STANDARD_MAX_OVERS,
        ),
    )
    max_legal_balls = (max_overs * 6).astype(int)
    
    # Legal balls remaining BEFORE this ball
    df["legal_balls_rem_pre"] = np.maximum(0, max_legal_balls - df["legal_balls_bowled_pre"])
    
    # Overs completed before this ball
    df["overs_completed_pre"] = df["legal_balls_bowled_pre"] / 6.0
    
    # Current Run Rate (CRR) before this ball (guarded against division by zero)
    df["crr_pre"] = np.where(
        df["overs_completed_pre"] > 0,
        df["cum_runs_pre"] / np.maximum(df["overs_completed_pre"], 1e-5),
        0.0,
    )
    
    # Compute Innings 1 total runs and Innings 2 Target / RRR
    logger.info("Computing Innings 2 targets and Required Run Rates...")
    inn1_totals = df[df["innings"] == 1].groupby("match_id")["ball_total_runs"].sum()
    inn1_target = df["match_id"].map(inn1_totals) + 1
    
    is_inn2 = df["innings"] == 2
    
    # Target runs for innings 2
    df["target_runs"] = np.where(
        is_inn2,
        np.where(df["revised_target"].notna(), df["revised_target"], inn1_target),
        np.nan,
    )
    
    # Runs needed before delivery
    df["runs_needed_pre"] = np.where(
        is_inn2,
        np.maximum(0, df["target_runs"] - df["cum_runs_pre"]),
        np.nan,
    )
    
    # Required Run Rate before delivery
    df["rrr_pre"] = np.where(
        is_inn2 & (df["legal_balls_rem_pre"] > 0),
        (df["runs_needed_pre"] / np.maximum(df["legal_balls_rem_pre"], 1)) * 6.0,
        np.where(is_inn2 & (df["runs_needed_pre"] > 0), 36.0, 0.0),
    )
    
    # 5. Pressure Situations (Rule-based domain classification)
    logger.info("Classifying situational match pressure...")
    
    # Innings 2 High Pressure conditions:
    # 1. RRR >= 9.5 and balls_remaining <= 60
    # 2. Close finish: runs_needed <= 36 and balls_remaining <= 24
    # 3. Chasing collapse: wickets_down >= 4 in overs <= 10
    # 4. Critical RRR: RRR >= 12.0
    inn2_high_pressure = (
        is_inn2
        & (
            ((df["rrr_pre"] >= config.PRESSURE_INNS2_HIGH_RRR) & (df["legal_balls_rem_pre"] <= config.PRESSURE_INNS2_LATE_BALLS))
            | ((df["runs_needed_pre"] <= config.PRESSURE_INNS2_CLOSE_RUNS) & (df["legal_balls_rem_pre"] <= config.PRESSURE_INNS2_CLOSE_BALLS))
            | ((df["wickets_down_pre"] >= config.PRESSURE_INNS2_COLLAPSE_WKTS) & (df["overid"] <= 10))
            | (df["rrr_pre"] >= config.PRESSURE_INNS2_CRITICAL_RRR)
        )
    )
    
    # Innings 1 High Pressure conditions:
    # 1. Death overs: overs 16 to 20
    # 2. Powerplay top-order collapse: overs 1 to 6 and wickets >= 3
    # 3. Knockout match with >= 2 wickets lost
    is_inn1 = df["innings"] == 1
    inn1_high_pressure = (
        is_inn1
        & (
            (df["overid"] >= 16)
            | ((df["overid"] <= 6) & (df["wickets_down_pre"] >= config.PRESSURE_INNS1_COLLAPSE_WKTS))
            | ((df["knock_out"] == 1) & (df["wickets_down_pre"] >= 2))
        )
    )
    
    df["is_pressure"] = inn2_high_pressure | inn1_high_pressure
    df["pressure_category"] = np.where(df["is_pressure"], "High", "Normal")
    
    logger.info(
        "Feature engineering complete. Pressure deliveries: %d (%.2f%%)",
        df["is_pressure"].sum(),
        100.0 * df["is_pressure"].mean(),
    )
    return df
