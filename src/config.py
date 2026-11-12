"""Configuration module for PressureIQ.

Centralizes paths, dataset schemas, feature thresholds, and model parameters.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data Set"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Raw Data Files
BALL_BY_BALL_FILE = DATA_DIR / "ball_by_ball.csv"
MATCHES_FILE = DATA_DIR / "matches.csv"
COMPETITIONS_FILE = DATA_DIR / "competitions.csv"
TEAMS_FILE = DATA_DIR / "teams.csv"
PLAYERS_FILE = DATA_DIR / "competition_team_player_mapping.csv"
PLAYING_ELEVEN_FILE = DATA_DIR / "playing_eleven.csv"

# Processed Cache Files
PROCESSED_DELIVERIES_PARQUET = OUTPUTS_DIR / "processed_deliveries.parquet"
PROCESSED_DELIVERIES_CSV = OUTPUTS_DIR / "processed_deliveries.csv.gz"

# Extras Encoding Mapping (Verified against dataset)
# 6: None, 2: Wide, 3: Legbyes, 1: Overstep (no ball), 4: Byes, 5: Other, 7: Penalty
EXTRAS_CODE_MAP = {
    6: "none",
    2: "wide",
    3: "legbyes",
    1: "Overstep",
    4: "byes",
    5: "other",
    7: "Penalty",
}

# Delivery Classification
ILLEGAL_DELIVERY_TYPES = {"wide", "Overstep"}

# Match Phase Definitions (Overs 1-indexed: overid 1 to 20)
PHASE_POWERPLAY = "Powerplay"  # Overs 1 to 6
PHASE_MIDDLE = "Middle"        # Overs 7 to 15
PHASE_DEATH = "Death"          # Overs 16 to 20

# Standard T20 Constants
STANDARD_MAX_OVERS = 20
STANDARD_MAX_BALLS = 120
MAX_WICKETS = 10

# Baseline Rule-based Pressure Criteria (Before ML Leverage Index)
PRESSURE_INNS2_HIGH_RRR = 9.5
PRESSURE_INNS2_CRITICAL_RRR = 12.0
PRESSURE_INNS2_LATE_BALLS = 60
PRESSURE_INNS2_CLOSE_BALLS = 24
PRESSURE_INNS2_CLOSE_RUNS = 36
PRESSURE_INNS2_COLLAPSE_WKTS = 4
PRESSURE_INNS1_COLLAPSE_WKTS = 3

# Minimum Sample Sizes for Rankings (Prevents small-sample noise)
MIN_PRESSURE_BALLS_FACED_BATTER = 50
MIN_PRESSURE_BALLS_BOWLED_BOWLER = 50

# Random Seed for Reproducibility
RANDOM_SEED = 42

