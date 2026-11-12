"""Data loading and validation module for PressureIQ.

Loads raw CSVs, performs integrity checks, cleans types, and joins match
metadata to delivery-level records.
"""

from typing import Dict, Optional, Tuple
import logging
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_datasets() -> Dict[str, pd.DataFrame]:
    """Load all raw CSV datasets from the Data Set folder.
    
    Returns:
        dict: Dictionary of raw DataFrames.
    """
    logger.info("Loading raw datasets from %s...", config.DATA_DIR)
    
    datasets = {
        "ball_by_ball": pd.read_csv(config.BALL_BY_BALL_FILE),
        "matches": pd.read_csv(config.MATCHES_FILE),
        "competitions": pd.read_csv(config.COMPETITIONS_FILE),
        "teams": pd.read_csv(config.TEAMS_FILE),
        "players": pd.read_csv(config.PLAYERS_FILE),
        "playing_eleven": pd.read_csv(config.PLAYING_ELEVEN_FILE),
    }
    
    for name, df in datasets.items():
        logger.info("Loaded '%s' with shape: %s", name, df.shape)
        
    return datasets


def clean_and_merge_metadata(
    ball_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and merge tournament and match metadata onto delivery-level records.
    
    Args:
        ball_df: Raw ball-by-ball DataFrame.
        matches_df: Raw matches DataFrame.
        competitions_df: Raw competitions DataFrame.
        
    Returns:
        pd.DataFrame: Merged DataFrame sorted by match_date, match_id, innings, innings_ball_num.
    """
    logger.info("Cleaning match and competition metadata...")
    
    matches_clean = matches_df.copy()
    competitions_clean = competitions_df.copy()
    
    # Standardize match_date
    matches_clean["match_date"] = pd.to_datetime(matches_clean["match_date"], errors="coerce")
    
    # Merge competition details into matches
    matches_with_comp = matches_clean.merge(
        competitions_clean[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id",
        how="left",
    )
    
    # Identify IPL vs T20I
    matches_with_comp["is_ipl"] = matches_with_comp["comp_name"].str.contains("IPL", case=False, na=False)
    matches_with_comp["competition_tier"] = matches_with_comp["is_ipl"].map({True: "IPL", False: "T20I"})
    
    # Select relevant match columns
    match_cols_to_merge = [
        "match_id",
        "comp_id",
        "comp_name",
        "season",
        "match_format",
        "competition_tier",
        "match_date",
        "stage",
        "knock_out",
        "venue_id",
        "venue_country",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "toss_win_team_id",
        "toss_opted",
        "win_team_id",
        "result",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "revised_target",
    ]
    
    merged_df = ball_df.merge(
        matches_with_comp[match_cols_to_merge],
        on="match_id",
        how="inner",
    )
    
    # Sort rigorously by match_date, match_id, innings, and innings_ball_num
    merged_df = merged_df.sort_values(
        by=["match_date", "match_id", "innings", "innings_ball_num"]
    ).reset_index(drop=True)
    
    logger.info("Merged dataset created with shape: %s", merged_df.shape)
    return merged_df

