"""Win Probability Added (WPA) & Leverage Index Engine for PressureIQ.

Calculates pre- and post-delivery win probabilities, Win Probability Added (WPA),
state sensitivity, leverage indices, and fine-grained attribution to batters,
bowlers, and non-strikers (handling run-outs, byes, legbyes, and boundary extras).
"""

from typing import Tuple
import logging
import numpy as np
import pandas as pd

from src.win_prob_model import DualInningsWinProbModel, prepare_features, INNINGS_1_FEATURES, INNINGS_2_FEATURES

logger = logging.getLogger(__name__)


def compute_ball_wpa(
    df: pd.DataFrame,
    model: DualInningsWinProbModel,
) -> pd.DataFrame:
    """Compute pre/post win probabilities and WPA across all deliveries with audited attributions.
    
    Audit Corrections:
    - Non-striker run-outs: Attributed specifically to the dismissed player rather than penalizing striker.
    - Pure byes/legbyes: Batter and bowler WPA decoupled from fielding/keeping extras.
    - Wides & No-balls: Correctly attributed as bowling extras.
    - Super overs: Tied terminal state of Innings 2 assigned 0.50 win probability.
    """
    logger.info("Computing pre-delivery win probabilities...")
    df_out = df.copy()
    
    # 1. Compute pre-delivery win probabilities
    df_out["win_prob_pre"] = model.predict_proba(df_out)
    
    # 2. Compute post-delivery win probabilities
    logger.info("Computing post-delivery transitions and terminal state probabilities...")
    df_out["win_prob_post"] = np.nan
    
    grouped = df_out.groupby(["match_id", "innings"], sort=False)
    post_probs = []
    
    # Pre-compute first ball of innings 2 win probability for innings 1 terminal transition
    inn2_ball_ones = df_out[df_out["innings"] == 2].groupby("match_id")["win_prob_pre"].first()
    
    for (match_id, innings), group in grouped:
        n_balls = len(group)
        group_post = np.empty(n_balls, dtype=float)
        
        # Intra-innings: post prob for ball k is pre prob of ball k+1
        if n_balls > 1:
            group_post[:-1] = group["win_prob_pre"].iloc[1:].values
            
        # Terminal ball of this innings
        last_idx = n_balls - 1
        if innings == 1:
            # End of innings 1: batting team win prob is 1 - (innings 2 chasing team win prob at ball 1)
            if match_id in inn2_ball_ones:
                group_post[last_idx] = 1.0 - inn2_ball_ones[match_id]
            else:
                group_post[last_idx] = group["win_prob_pre"].iloc[last_idx]
        elif innings == 2:
            # Check for tied match heading into super over
            res_str = str(group["result"].iloc[last_idx]).lower()
            is_super_over_tie = "super over" in res_str or "eliminator" in res_str or "match tied" in res_str
            
            # If match ended in a tie before super over, the 20-over game finished at parity (0.50)
            if is_super_over_tie and abs(group["runs_needed_pre"].iloc[last_idx] - group["ball_total_runs"].iloc[last_idx]) == 0:
                group_post[last_idx] = 0.50
            else:
                won = group["batting_team_won"].iloc[last_idx]
                group_post[last_idx] = float(won) if pd.notna(won) else group["win_prob_pre"].iloc[last_idx]
        else:
            group_post[last_idx] = group["win_prob_pre"].iloc[last_idx]
            
        post_probs.append(pd.Series(group_post, index=group.index))
        
    df_out["win_prob_post"] = pd.concat(post_probs).sort_index()
    
    # 3. Win Probability Added (Batting team perspective)
    df_out["wpa"] = df_out["win_prob_post"] - df_out["win_prob_pre"]
    
    # 4. Refined Player Attributions (Audited)
    is_wide = df_out["extras_type"] == "wide"
    is_pure_bye_lb = df_out["extras_type"].isin(["byes", "legbyes"]) & (df_out["runs"] == 0)
    is_run_out = df_out["dismissal_type"] == "RunOut"
    non_striker_runout = is_run_out & (df_out["dismissed_player_id"] == df_out["non_striker_id"])
    
    # Bowler attribution:
    # Bowlers receive -wpa, EXCEPT on pure byes and legbyes which are wicketkeeper/fielding extras
    df_out["bowler_wpa"] = np.where(is_pure_bye_lb, 0.0, -df_out["wpa"])
    
    # Striker (Batter) attribution:
    # 1. On wides: 0.0 (striker faced no delivery)
    # 2. On pure byes/legbyes: 0.0 (no runs off bat)
    # 3. On non-striker runouts: Striker receives 0.0 (or bat runs WPA), NOT the wicket loss penalty!
    # 4. Standard deliveries: receives full wpa
    striker_wpa = df_out["wpa"].values.copy()
    striker_wpa = np.where(is_wide.values | is_pure_bye_lb.values, 0.0, striker_wpa)
    # If non-striker was run out, don't penalize striker
    striker_wpa = np.where(non_striker_runout.values, np.maximum(0.0, striker_wpa), striker_wpa)
    df_out["batter_wpa"] = striker_wpa
    
    # Track non-striker runout penalty separately
    df_out["non_striker_wpa"] = np.where(non_striker_runout.values, df_out["wpa"].values, 0.0)
    
    logger.info("WPA calculation complete. Mean WPA: %.5f", df_out["wpa"].mean())
    return df_out


def compute_leverage_index(
    df: pd.DataFrame,
    model: DualInningsWinProbModel,
) -> pd.DataFrame:
    """Compute state sensitivity and normalized Leverage Index (LI) with exact boundary conditions.
    
    Audited Enhancements:
    - Terminal victory condition: if boundary reaches target in Innings 2, p_bound = 1.0.
    - Terminal all-out condition: if 10th wicket falls in Innings 2, p_wkt = 0.0.
    - Innings 1 all-out: legal balls remaining set to 0.
    """
    logger.info("Computing state sensitivity and dynamic Leverage Index...")
    df_out = df.copy()
    
    # 1. Prepare boundary simulated state (+4 runs, +1 legal ball)
    df_bound = df_out.copy()
    df_bound["cum_runs_pre"] = df_bound["cum_runs_pre"] + 4
    df_bound["legal_balls_bowled_pre"] = df_bound["legal_balls_bowled_pre"] + 1
    df_bound["legal_balls_rem_pre"] = np.maximum(0, df_bound["legal_balls_rem_pre"] - 1)
    df_bound["overs_completed_pre"] = df_bound["legal_balls_bowled_pre"] / 6.0
    df_bound["crr_pre"] = np.where(
        df_bound["overs_completed_pre"] > 0,
        df_bound["cum_runs_pre"] / np.maximum(df_bound["overs_completed_pre"], 1e-5),
        0.0,
    )
    df_bound["runs_needed_pre"] = np.maximum(0, df_bound["target_runs"] - df_bound["cum_runs_pre"])
    df_bound["rrr_pre"] = np.where(
        (df_bound["innings"] == 2) & (df_bound["legal_balls_rem_pre"] > 0),
        (df_bound["runs_needed_pre"] / np.maximum(df_bound["legal_balls_rem_pre"], 1)) * 6.0,
        0.0,
    )
    
    # 2. Prepare wicket simulated state (+1 wicket, 0 runs, +1 legal ball)
    df_wkt = df_out.copy()
    df_wkt["wickets_down_pre"] = np.minimum(10, df_wkt["wickets_down_pre"] + 1)
    df_wkt["legal_balls_bowled_pre"] = df_wkt["legal_balls_bowled_pre"] + 1
    df_wkt["legal_balls_rem_pre"] = np.maximum(0, df_wkt["legal_balls_rem_pre"] - 1)
    df_wkt["overs_completed_pre"] = df_wkt["legal_balls_bowled_pre"] / 6.0
    df_wkt["crr_pre"] = np.where(
        df_wkt["overs_completed_pre"] > 0,
        df_wkt["cum_runs_pre"] / np.maximum(df_wkt["overs_completed_pre"], 1e-5),
        0.0,
    )
    df_wkt["rrr_pre"] = np.where(
        (df_wkt["innings"] == 2) & (df_wkt["legal_balls_rem_pre"] > 0),
        (df_wkt["runs_needed_pre"] / np.maximum(df_wkt["legal_balls_rem_pre"], 1)) * 6.0,
        0.0,
    )
    
    # Innings 1 all-out termination
    inn1_all_out = (df_wkt["innings"] == 1) & (df_wkt["wickets_down_pre"] >= 10)
    df_wkt.loc[inn1_all_out, "legal_balls_rem_pre"] = 0
    
    # 3. Predict hypothetical probabilities
    p_curr = df_out["win_prob_pre"].values
    p_bound = model.predict_proba(df_bound)
    p_wkt = model.predict_proba(df_wkt)
    
    # Apply Terminal Boundary Conditions in Innings 2:
    # A. If +4 runs wins the match (runs_needed <= 0): win prob becomes 0.999
    inn2_boundary_win = (df_out["innings"] == 2) & (df_bound["runs_needed_pre"] <= 0)
    p_bound = np.where(inn2_boundary_win.values, 0.999, p_bound)
    
    # B. If +1 wicket loses the match (10th wicket / all out): win prob becomes 0.001
    inn2_wkt_loss = (df_out["innings"] == 2) & (df_wkt["wickets_down_pre"] >= 10)
    p_wkt = np.where(inn2_wkt_loss.values, 0.001, p_wkt)
    
    # State sensitivity: swing between boundary and wicket
    sensitivity = np.abs(p_bound - p_curr) + np.abs(p_curr - p_wkt)
    df_out["state_sensitivity"] = sensitivity
    
    # Normalize by mean sensitivity so mean(LI) == 1.0
    mean_sens = np.mean(sensitivity)
    if mean_sens > 0:
        df_out["leverage_index"] = sensitivity / mean_sens
    else:
        df_out["leverage_index"] = 1.0
        
    logger.info("Leverage Index computed. Mean: %.3f, Median: %.3f, Max: %.3f",
                df_out["leverage_index"].mean(),
                df_out["leverage_index"].median(),
                df_out["leverage_index"].max())
    return df_out
