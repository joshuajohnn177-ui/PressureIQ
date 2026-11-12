"""Win Probability Modeling Module for PressureIQ.

Implements dual-innings modeling architecture for T20 cricket, supporting
Logistic Regression, Random Forest, XGBoost, and Calibrated Ensembles with
rigorous chronological validation and probabilistic evaluation.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

from src import config

logger = logging.getLogger(__name__)

# Features used for Innings 1 (Target Setting)
INNINGS_1_FEATURES = [
    "cum_runs_pre",
    "wickets_down_pre",
    "legal_balls_rem_pre",
    "crr_pre",
    "projected_score_pre",
    "overid",
    "is_ipl",
    "knock_out",
]

# Features used for Innings 2 (Chasing)
INNINGS_2_FEATURES = [
    "cum_runs_pre",
    "wickets_down_pre",
    "legal_balls_rem_pre",
    "crr_pre",
    "target_runs",
    "runs_needed_pre",
    "rrr_pre",
    "overid",
    "is_ipl",
    "knock_out",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare explicit, leak-free modeling features."""
    df_feat = df.copy()
    
    # Tournament tier flag
    if "is_ipl" not in df_feat.columns:
        if "competition_tier" in df_feat.columns:
            df_feat["is_ipl"] = (df_feat["competition_tier"] == "IPL").astype(int)
        else:
            df_feat["is_ipl"] = 0
            
    # Projected 1st innings total score
    df_feat["projected_score_pre"] = (
        df_feat["cum_runs_pre"] + df_feat["crr_pre"] * (df_feat["legal_balls_rem_pre"] / 6.0)
    )
    
    # Ensure knock_out is clean binary
    df_feat["knock_out"] = df_feat["knock_out"].fillna(0).astype(int)
    
    # Fill safe defaults for chase features in case of any floating-point edge cases
    df_feat["target_runs"] = df_feat["target_runs"].fillna(0.0)
    df_feat["runs_needed_pre"] = df_feat["runs_needed_pre"].fillna(0.0)
    df_feat["rrr_pre"] = df_feat["rrr_pre"].fillna(0.0)
    
    return df_feat


def get_model_instances(model_type: str, random_state: int = config.RANDOM_SEED) -> Tuple[Any, Any]:
    """Factory creating separate Innings 1 and Innings 2 classifier instances."""
    if model_type == "logistic_regression":
        m1 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)),
        ])
        m2 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)),
        ])
    elif model_type == "random_forest":
        m1 = RandomForestClassifier(
            n_estimators=120,
            max_depth=9,
            min_samples_leaf=25,
            random_state=random_state,
            n_jobs=-1,
        )
        m2 = RandomForestClassifier(
            n_estimators=120,
            max_depth=9,
            min_samples_leaf=25,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        m1 = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.06,
            reg_lambda=5.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
        m2 = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.06,
            reg_lambda=5.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    elif model_type == "ensemble":
        # Calibrated blend: Logistic Regression + Regularized XGBoost
        m1 = {
            "lr": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=random_state))]),
            "xgb": XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.06, reg_lambda=5.0, subsample=0.85, colsample_bytree=0.85, random_state=random_state, n_jobs=-1, eval_metric="logloss"),
        }
        m2 = {
            "lr": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=random_state))]),
            "xgb": XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.06, reg_lambda=5.0, subsample=0.85, colsample_bytree=0.85, random_state=random_state, n_jobs=-1, eval_metric="logloss"),
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return m1, m2


class DualInningsWinProbModel:
    """Unified dual-innings win probability model routing deliveries to specialized sub-models."""

    def __init__(self, model_type: str = "xgboost", random_state: int = config.RANDOM_SEED):
        self.model_type = model_type
        self.random_state = random_state
        self.inn1_model, self.inn2_model = get_model_instances(model_type, random_state)
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "DualInningsWinProbModel":
        """Fit sub-models on respective innings data."""
        df_prep = prepare_features(train_df)
        
        # Fit Innings 1 model
        df1 = df_prep[df_prep["innings"] == 1]
        X1 = df1[INNINGS_1_FEATURES]
        y1 = df1["batting_team_won"]
        
        # Fit Innings 2 model
        df2 = df_prep[df_prep["innings"] == 2]
        X2 = df2[INNINGS_2_FEATURES]
        y2 = df2["batting_team_won"]
        
        if self.model_type == "ensemble":
            self.inn1_model["lr"].fit(X1, y1)
            self.inn1_model["xgb"].fit(X1, y1)
            self.inn2_model["lr"].fit(X2, y2)
            self.inn2_model["xgb"].fit(X2, y2)
        else:
            self.inn1_model.fit(X1, y1)
            self.inn2_model.fit(X2, y2)
            
        self.is_fitted = True
        logger.info("Fitted %s DualInningsWinProbModel (Inn1: %d, Inn2: %d)", self.model_type, len(df1), len(df2))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict batting team win probability for each delivery."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
            
        df_prep = prepare_features(df)
        probs = np.full(len(df_prep), 0.5, dtype=float)
        
        # Innings 1 deliveries
        mask1 = df_prep["innings"] == 1
        if mask1.any():
            X1 = df_prep.loc[mask1, INNINGS_1_FEATURES]
            if self.model_type == "ensemble":
                p1_lr = self.inn1_model["lr"].predict_proba(X1)[:, 1]
                p1_xgb = self.inn1_model["xgb"].predict_proba(X1)[:, 1]
                probs[mask1.values] = 0.5 * p1_lr + 0.5 * p1_xgb
            else:
                probs[mask1.values] = self.inn1_model.predict_proba(X1)[:, 1]
            
        # Innings 2 deliveries
        mask2 = df_prep["innings"] == 2
        if mask2.any():
            X2 = df_prep.loc[mask2, INNINGS_2_FEATURES]
            if self.model_type == "ensemble":
                p2_lr = self.inn2_model["lr"].predict_proba(X2)[:, 1]
                p2_xgb = self.inn2_model["xgb"].predict_proba(X2)[:, 1]
                probs[mask2.values] = 0.5 * p2_lr + 0.5 * p2_xgb
            else:
                probs[mask2.values] = self.inn2_model.predict_proba(X2)[:, 1]
            
        # Physical Boundary Conditions:
        # In Innings 2: if target already reached (runs_needed <= 0), WP is 1.0
        is_inn2_target_met = mask2 & (df_prep["runs_needed_pre"] <= 0) & (df_prep["cum_runs_pre"] > 0)
        probs[is_inn2_target_met.values] = 0.999
        
        # Clip to ensure valid probabilities strictly in [0.001, 0.999]
        probs = np.clip(probs, 0.001, 0.999)
        return probs

    def evaluate(self, df: pd.DataFrame, dataset_name: str = "Test") -> Dict[str, float]:
        """Compute evaluation metrics on a dataset partition."""
        y_true = df["batting_team_won"].values
        y_prob = self.predict_proba(df)
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = {
            "dataset": dataset_name,
            "model_type": self.model_type,
            "samples": int(len(df)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "log_loss": float(log_loss(y_true, y_prob)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
        return metrics

    def get_calibration(self, df: pd.DataFrame, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Compute calibration curve prob_true and prob_pred."""
        y_true = df["batting_team_won"].values
        y_prob = self.predict_proba(df)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        return prob_true, prob_pred

    def save(self, filepath: Path) -> None:
        """Serialize trained model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved model to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> "DualInningsWinProbModel":
        """Load trained model from disk."""
        model = joblib.load(filepath)
        logger.info("Loaded model from %s", filepath)
        return model
