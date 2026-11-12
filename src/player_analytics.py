"""Player performance intelligence module for PressureIQ.

Implements empirical Bayes shrinkage, calculates situational pressure metrics
(Strike Rate, Economy, Dot%, Boundary%, WPA/ball), performance differentials,
consistency tracking, and generates robust Top 10 rankings.
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def calculate_batter_metrics(
    df: pd.DataFrame,
    min_pressure_balls: int = config.MIN_PRESSURE_BALLS_FACED_BATTER,
    shrinkage_k: float = 75.0,
) -> pd.DataFrame:
    """Calculate batting pressure performance metrics and empirical Bayes shrunk ratings.
    
    Args:
        df: Enriched deliveries DataFrame with WPA and Leverage Index.
        min_pressure_balls: Minimum threshold of pressure deliveries faced.
        shrinkage_k: Prior weight parameter for empirical Bayes shrinkage.
        
    Returns:
        pd.DataFrame: Comprehensive batter performance and ranking table.
    """
    logger.info("Computing batting performance metrics across baseline and pressure states...")
    
    # Exclude wide deliveries from batter faced deliveries
    legal_bat = df[df["extras_type"] != "wide"].copy()
    
    # 1. Overall Baseline Metrics per Batter
    overall = legal_bat.groupby(["striker_id", "striker_name"]).agg(
        matches=("match_id", "nunique"),
        total_balls=("is_legal_ball", "count"),
        total_runs=("runs", "sum"),
        total_dismissals=("is_wicket", "sum"),
        total_wpa=("batter_wpa", "sum"),
        total_dots=("runs", lambda x: (x == 0).sum()),
        total_boundaries=("runs", lambda x: x.isin([4, 6]).sum()),
    ).reset_index()
    
    overall["overall_sr"] = (overall["total_runs"] / np.maximum(1, overall["total_balls"])) * 100.0
    overall["overall_dot_pct"] = (overall["total_dots"] / np.maximum(1, overall["total_balls"])) * 100.0
    overall["overall_boundary_pct"] = (overall["total_boundaries"] / np.maximum(1, overall["total_balls"])) * 100.0
    
    # 2. Pressure Metrics per Batter
    press_df = legal_bat[legal_bat["is_pressure"]].copy()
    
    # Match-level consistency (standard deviation of match pressure WPA)
    match_press_wpa = press_df.groupby(["striker_id", "striker_name", "match_id"])["batter_wpa"].sum().reset_index()
    consistency = match_press_wpa.groupby(["striker_id", "striker_name"])["batter_wpa"].agg(
        wpa_consistency_std="std",
        matches_with_pos_wpa=lambda x: (x > 0).sum()
    ).reset_index()
    consistency["wpa_consistency_std"] = consistency["wpa_consistency_std"].fillna(0.0)
    
    press = press_df.groupby(["striker_id", "striker_name"]).agg(
        pressure_matches=("match_id", "nunique"),
        pressure_balls=("is_legal_ball", "count"),
        pressure_runs=("runs", "sum"),
        pressure_dismissals=("is_wicket", "sum"),
        pressure_wpa=("batter_wpa", "sum"),
        pressure_dots=("runs", lambda x: (x == 0).sum()),
        pressure_boundaries=("runs", lambda x: x.isin([4, 6]).sum()),
        avg_leverage=("leverage_index", "mean"),
    ).reset_index()
    
    press["pressure_sr"] = (press["pressure_runs"] / np.maximum(1, press["pressure_balls"])) * 100.0
    press["pressure_dot_pct"] = (press["pressure_dots"] / np.maximum(1, press["pressure_balls"])) * 100.0
    press["pressure_boundary_pct"] = (press["pressure_boundaries"] / np.maximum(1, press["pressure_balls"])) * 100.0
    press["pressure_wpa_per_ball"] = press["pressure_wpa"] / np.maximum(1, press["pressure_balls"])
    
    # Merge datasets
    merged = pd.merge(overall, press, on=["striker_id", "striker_name"], how="inner")
    merged = pd.merge(merged, consistency, on=["striker_id", "striker_name"], how="left")
    
    # Filter minimum sample threshold
    filtered = merged[merged["pressure_balls"] >= min_pressure_balls].copy()
    
    # 3. Differentials vs Baseline
    filtered["sr_differential"] = filtered["pressure_sr"] - filtered["overall_sr"]
    filtered["boundary_differential"] = filtered["pressure_boundary_pct"] - filtered["overall_boundary_pct"]
    filtered["dot_differential"] = filtered["pressure_dot_pct"] - filtered["overall_dot_pct"]
    
    # 4. Empirical Bayes Shrinkage Formulation
    # Prior population means
    mu_wpa_pb = filtered["pressure_wpa_per_ball"].mean()
    mu_sr_diff = filtered["sr_differential"].mean()
    
    n_balls = filtered["pressure_balls"]
    shrinkage_factor = n_balls / (n_balls + shrinkage_k)
    
    filtered["shrunk_wpa_per_ball"] = shrinkage_factor * filtered["pressure_wpa_per_ball"] + (1 - shrinkage_factor) * mu_wpa_pb
    filtered["shrunk_sr_diff"] = shrinkage_factor * filtered["sr_differential"] + (1 - shrinkage_factor) * mu_sr_diff
    filtered["sample_credibility"] = shrinkage_factor
    
    # 5. Composite PressureIQ Rating Formulation
    # Combines Shrunk WPA/ball, Shrunk Strike Rate boost, Clutch Win Rate, and Volume Weighting
    filtered["pressure_wpa_total_shrunk"] = filtered["shrunk_wpa_per_ball"] * filtered["pressure_balls"]
    
    # Standardize components for composite score
    wpa_norm = (filtered["pressure_wpa_total_shrunk"] - filtered["pressure_wpa_total_shrunk"].mean()) / (filtered["pressure_wpa_total_shrunk"].std() + 1e-6)
    sr_norm = (filtered["shrunk_sr_diff"] - filtered["shrunk_sr_diff"].mean()) / (filtered["shrunk_sr_diff"].std() + 1e-6)
    cred_norm = filtered["sample_credibility"]
    
    filtered["pressure_iq_score"] = 60.0 * wpa_norm + 25.0 * sr_norm + 15.0 * cred_norm
    
    # Sort and rank
    ranked = filtered.sort_values(by="pressure_iq_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    
    logger.info("Batter rating complete. Ranked %d eligible batters.", len(ranked))
    return ranked


def calculate_bowler_metrics(
    df: pd.DataFrame,
    min_pressure_balls: int = config.MIN_PRESSURE_BALLS_BOWLED_BOWLER,
    shrinkage_k: float = 75.0,
) -> pd.DataFrame:
    """Calculate bowling pressure performance metrics and empirical Bayes shrunk ratings.
    
    Args:
        df: Enriched deliveries DataFrame with WPA and Leverage Index.
        min_pressure_balls: Minimum threshold of pressure legal deliveries bowled.
        shrinkage_k: Prior weight parameter for empirical Bayes shrinkage.
        
    Returns:
        pd.DataFrame: Comprehensive bowler performance and ranking table.
    """
    logger.info("Computing bowling performance metrics across baseline and pressure states...")
    
    # 1. Overall Baseline Metrics per Bowler
    overall = df.groupby(["bowler_id", "bowler_name"]).agg(
        matches=("match_id", "nunique"),
        total_balls=("is_legal_ball", "sum"),
        total_runs_conceded=("ball_total_runs", "sum"),
        total_wickets=("is_wicket", "sum"),
        total_wpa=("bowler_wpa", "sum"),
        total_dots=("runs", lambda x: (x == 0).sum()),
        total_boundaries=("runs", lambda x: x.isin([4, 6]).sum()),
    ).reset_index()
    
    overall["overall_econ"] = (overall["total_runs_conceded"] / np.maximum(1, overall["total_balls"])) * 6.0
    overall["overall_dot_pct"] = (overall["total_dots"] / np.maximum(1, overall["total_balls"])) * 100.0
    overall["overall_boundary_pct"] = (overall["total_boundaries"] / np.maximum(1, overall["total_balls"])) * 100.0
    
    # 2. Pressure Metrics per Bowler
    press_df = df[df["is_pressure"]].copy()
    
    match_press_wpa = press_df.groupby(["bowler_id", "bowler_name", "match_id"])["bowler_wpa"].sum().reset_index()
    consistency = match_press_wpa.groupby(["bowler_id", "bowler_name"])["bowler_wpa"].agg(
        wpa_consistency_std="std",
        matches_with_pos_wpa=lambda x: (x > 0).sum()
    ).reset_index()
    consistency["wpa_consistency_std"] = consistency["wpa_consistency_std"].fillna(0.0)
    
    press = press_df.groupby(["bowler_id", "bowler_name"]).agg(
        pressure_matches=("match_id", "nunique"),
        pressure_balls=("is_legal_ball", "sum"),
        pressure_runs_conceded=("ball_total_runs", "sum"),
        pressure_wickets=("is_wicket", "sum"),
        pressure_wpa=("bowler_wpa", "sum"),
        pressure_dots=("runs", lambda x: (x == 0).sum()),
        pressure_boundaries=("runs", lambda x: x.isin([4, 6]).sum()),
        avg_leverage=("leverage_index", "mean"),
    ).reset_index()
    
    press["pressure_econ"] = (press["pressure_runs_conceded"] / np.maximum(1, press["pressure_balls"])) * 6.0
    press["pressure_dot_pct"] = (press["pressure_dots"] / np.maximum(1, press["pressure_balls"])) * 100.0
    press["pressure_boundary_pct"] = (press["pressure_boundaries"] / np.maximum(1, press["pressure_balls"])) * 100.0
    press["pressure_wpa_per_ball"] = press["pressure_wpa"] / np.maximum(1, press["pressure_balls"])
    
    # Merge datasets
    merged = pd.merge(overall, press, on=["bowler_id", "bowler_name"], how="inner")
    merged = pd.merge(merged, consistency, on=["bowler_id", "bowler_name"], how="left")
    
    # Filter minimum sample threshold
    filtered = merged[merged["pressure_balls"] >= min_pressure_balls].copy()
    
    # 3. Differentials vs Baseline (Negative econ_diff means bowler conceded FEWER runs under pressure!)
    filtered["econ_differential"] = filtered["pressure_econ"] - filtered["overall_econ"]
    filtered["dot_differential"] = filtered["pressure_dot_pct"] - filtered["overall_dot_pct"]
    filtered["boundary_differential"] = filtered["pressure_boundary_pct"] - filtered["overall_boundary_pct"]
    
    # 4. Empirical Bayes Shrinkage Formulation
    mu_wpa_pb = filtered["pressure_wpa_per_ball"].mean()
    mu_econ_diff = filtered["econ_differential"].mean()
    
    n_balls = filtered["pressure_balls"]
    shrinkage_factor = n_balls / (n_balls + shrinkage_k)
    
    filtered["shrunk_wpa_per_ball"] = shrinkage_factor * filtered["pressure_wpa_per_ball"] + (1 - shrinkage_factor) * mu_wpa_pb
    filtered["shrunk_econ_diff"] = shrinkage_factor * filtered["econ_differential"] + (1 - shrinkage_factor) * mu_econ_diff
    filtered["sample_credibility"] = shrinkage_factor
    
    # 5. Composite PressureIQ Rating Formulation
    # Higher WPA, more negative econ_diff (or resisted death inflation), high volume credibility
    filtered["pressure_wpa_total_shrunk"] = filtered["shrunk_wpa_per_ball"] * filtered["pressure_balls"]
    
    wpa_norm = (filtered["pressure_wpa_total_shrunk"] - filtered["pressure_wpa_total_shrunk"].mean()) / (filtered["pressure_wpa_total_shrunk"].std() + 1e-6)
    # Economy suppression: negate econ diff so that lower pressure economy receives a positive boost
    econ_suppression = -filtered["shrunk_econ_diff"]
    econ_norm = (econ_suppression - econ_suppression.mean()) / (econ_suppression.std() + 1e-6)
    cred_norm = filtered["sample_credibility"]
    
    filtered["pressure_iq_score"] = 60.0 * wpa_norm + 25.0 * econ_norm + 15.0 * cred_norm
    
    ranked = filtered.sort_values(by="pressure_iq_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    
    logger.info("Bowler rating complete. Ranked %d eligible bowlers.", len(ranked))
    return ranked

