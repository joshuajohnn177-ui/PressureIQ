"""Phase 2 Training & Evaluation Pipeline for PressureIQ.

Trains, calibrates, compares, and serializes Win Probability models
(Logistic Regression, Random Forest, XGBoost, Ensemble), generates evaluation
plots, and calculates ball-by-ball WPA and Leverage Index.
"""

from pathlib import Path
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.win_prob_model import DualInningsWinProbModel
from src.wpa_engine import compute_ball_wpa, compute_leverage_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2Runner")


def perform_chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Perform strict match-level chronological data splitting."""
    logger.info("Executing chronological match-level split...")
    
    matches_meta = (
        df[["match_id", "match_date"]]
        .drop_duplicates()
        .sort_values(by=["match_date", "match_id"])
        .reset_index(drop=True)
    )
    
    n_matches = len(matches_meta)
    train_end = int(n_matches * train_ratio)
    val_end = int(n_matches * (train_ratio + val_ratio))
    
    train_ids = set(matches_meta.iloc[:train_end]["match_id"])
    val_ids = set(matches_meta.iloc[train_end:val_end]["match_id"])
    test_ids = set(matches_meta.iloc[val_end:]["match_id"])
    
    train_df = df[df["match_id"].isin(train_ids)].copy()
    val_df = df[df["match_id"].isin(val_ids)].copy()
    test_df = df[df["match_id"].isin(test_ids)].copy()
    
    split_info = {
        "total_matches": n_matches,
        "train_matches": len(train_ids),
        "val_matches": len(val_ids),
        "test_matches": len(test_ids),
        "train_date_range": [
            str(matches_meta.iloc[0]["match_date"]),
            str(matches_meta.iloc[train_end - 1]["match_date"]),
        ],
        "val_date_range": [
            str(matches_meta.iloc[train_end]["match_date"]),
            str(matches_meta.iloc[val_end - 1]["match_date"]),
        ],
        "test_date_range": [
            str(matches_meta.iloc[val_end]["match_date"]),
            str(matches_meta.iloc[-1]["match_date"]),
        ],
        "train_deliveries": len(train_df),
        "val_deliveries": len(val_df),
        "test_deliveries": len(test_df),
        "train_win_rate": float(train_df["batting_team_won"].mean()),
        "val_win_rate": float(val_df["batting_team_won"].mean()),
        "test_win_rate": float(test_df["batting_team_won"].mean()),
    }
    
    logger.info(
        "Split Summary: Train=%d matches (%d balls), Val=%d matches (%d balls), Test=%d matches (%d balls)",
        split_info["train_matches"],
        split_info["train_deliveries"],
        split_info["val_matches"],
        split_info["val_deliveries"],
        split_info["test_matches"],
        split_info["test_deliveries"],
    )
    return train_df, val_df, test_df, split_info


def run_phase2():
    """Execute complete Phase 2 workflow with technical audit enhancements."""
    logger.info("=== STARTING PHASE 2: WIN PROBABILITY MODELING & WPA ===")
    
    # 1. Load processed data
    parquet_path = config.PROCESSED_DELIVERIES_PARQUET
    logger.info("Loading preprocessed dataset from %s...", parquet_path)
    df = pd.read_parquet(parquet_path)
    
    # 2. Filter valid resolved matches (exclude abandoned/no-result)
    valid_df = df[df["win_team_id"].notna() & df["innings"].isin([1, 2])].copy()
    valid_df["batting_team_won"] = (valid_df["batting_team_id"] == valid_df["win_team_id"]).astype(int)
    valid_df["is_ipl"] = (valid_df["competition_tier"] == "IPL").astype(int)
    
    logger.info(
        "Filtered to %d resolved deliveries across %d matches.",
        len(valid_df),
        valid_df["match_id"].nunique(),
    )
    
    # 3. Chronological match-level splitting
    train_df, val_df, test_df, split_info = perform_chronological_split(valid_df)
    
    with open(config.OUTPUTS_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4)
        
    # 4. Train & Compare Models
    model_types = ["logistic_regression", "random_forest", "xgboost", "ensemble"]
    fitted_models = {}
    eval_records = []
    
    for m_type in model_types:
        logger.info("Training and evaluating: %s...", m_type)
        model = DualInningsWinProbModel(model_type=m_type)
        model.fit(train_df)
        fitted_models[m_type] = model
        
        # Evaluate on validation
        val_metrics = model.evaluate(val_df, dataset_name="Validation")
        eval_records.append(val_metrics)
        
        # Evaluate on test
        test_metrics = model.evaluate(test_df, dataset_name="Test")
        eval_records.append(test_metrics)
        
        logger.info(
            "[%s] Val ROC-AUC: %.4f, Brier: %.4f | Test ROC-AUC: %.4f, Brier: %.4f",
            m_type,
            val_metrics["roc_auc"],
            val_metrics["brier_score"],
            test_metrics["roc_auc"],
            test_metrics["brier_score"],
        )
        
    eval_df = pd.DataFrame(eval_records)
    eval_df.to_csv(config.OUTPUTS_DIR / "model_evaluation_metrics.csv", index=False)
    with open(config.OUTPUTS_DIR / "model_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=4)
        
    logger.info("\n--- MODEL COMPARISON TABLE ---\n%s", eval_df.to_string(index=False))
    
    # 5. Generate Calibration Curves Plot
    logger.info("Plotting probabilistic calibration curves...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")
    
    colors = {
        "logistic_regression": "blue",
        "random_forest": "forestgreen",
        "xgboost": "crimson",
        "ensemble": "purple",
    }
    labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "ensemble": "Calibrated Ensemble (LR + XGB)",
    }
    
    for m_type, model in fitted_models.items():
        prob_true, prob_pred = model.get_calibration(test_df, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color=colors[m_type], label=f"{labels[m_type]}")
        
    ax.set_title("Reliability Diagram (Calibration Curve on Test Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Predicted Win Probability", fontsize=11)
    ax.set_ylabel("Empirical Win Fraction", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    
    calib_plot_path = config.FIGURES_DIR / "calibration_curves.png"
    plt.tight_layout()
    plt.savefig(calib_plot_path)
    plt.close()
    logger.info("Saved calibration curves plot to %s", calib_plot_path)
    
    # 6. Select Best Production Model
    # Ensemble combines the superior calibration stability of Logistic Regression with the
    # non-linear situational split power of XGBoost
    best_model_type = "ensemble"
    best_model = fitted_models[best_model_type]
    
    model_save_path = config.OUTPUTS_DIR / "win_prob_model.joblib"
    best_model.save(model_save_path)
    
    # 7. Compute Win Probability Added (WPA) & Leverage Index across all deliveries
    logger.info("Applying production model to calculate WPA and Leverage Index across all deliveries...")
    deliveries_with_wpa = compute_ball_wpa(valid_df, best_model)
    deliveries_with_wpa = compute_leverage_index(deliveries_with_wpa, best_model)
    
    # 8. Save Final Phase 2 Enriched Dataset
    enriched_parquet_path = config.OUTPUTS_DIR / "deliveries_with_wpa.parquet"
    logger.info("Saving enriched deliveries dataset to %s...", enriched_parquet_path)
    deliveries_with_wpa.to_parquet(enriched_parquet_path, index=False)
    
    sample_csv_path = config.OUTPUTS_DIR / "deliveries_with_wpa_sample.csv"
    deliveries_with_wpa.head(500).to_csv(sample_csv_path, index=False)
    logger.info("Saved 500-row sample to %s", sample_csv_path)
    
    logger.info("=== PHASE 2 COMPLETED SUCCESSFULLY ===")
    return eval_df, deliveries_with_wpa


if __name__ == "__main__":
    run_phase2()
