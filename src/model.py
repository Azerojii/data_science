from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


@dataclass
class ModelConfig:
    window_days: int = 14
    horizon_days: int = 5
    threshold: float = 0.035  # 3.5% for high-volatility regime (was 1.5% for stable stocks)
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 42
    # Volatility-specific parameters
    atr_stop_loss_multiplier: float = 2.5  # k in StopLoss = Price - (k × ATR_14)


def train_xgboost_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    config: ModelConfig,
) -> Tuple[XGBClassifier, dict]:
    """
    Train an XGBoost classifier with a TimeSeriesSplit.
    Returns:
      model – fitted on the full training set
      metrics – dict with last fold metrics (precision, report text)
    """
    # Basic time-series split (could be extended to rolling expanding windows)
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_precision = -np.inf
    best_model: XGBClassifier | None = None
    last_report: str = ""

    # Exclude metadata and target-related columns from features
    exclude_cols = ["ticker", "date", "future_return"]
    X_values = X.drop(columns=[c for c in exclude_cols if c in X.columns]).values
    y_values = y.values

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_values), start=1):
        print(f"[Model] Fold {fold}/{n_splits} – training...")
        X_train, X_val = X_values[train_idx], X_values[val_idx]
        y_train, y_val = y_values[train_idx], y_values[val_idx]

        # Calculate class weights to handle imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        model = XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            scale_pos_weight=scale_pos_weight,  # Balance classes
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        print(f"[Model] Fold {fold}/{n_splits} – evaluating...")

        y_pred = model.predict(X_val)
        prec = precision_score(y_val, y_pred, zero_division=0)
        report = classification_report(y_val, y_pred, zero_division=0)

        print(f"[Model] Fold {fold}/{n_splits} precision: {prec:.4f}")

        if prec > best_precision:
            best_precision = prec
            best_model = model
            last_report = report

    assert best_model is not None
    metrics = {
        "cv_best_precision": best_precision,
        "cv_classification_report": last_report,
    }
    return best_model, metrics


def evaluate_on_holdout(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Fit model on train set and evaluate on chronological test (2023+).
    """
    # Exclude metadata and target-related columns from features
    exclude_cols = ["ticker", "date", "future_return"]
    X_train_clean = X_train.drop(columns=[c for c in exclude_cols if c in X_train.columns])
    X_test_clean = X_test.drop(columns=[c for c in exclude_cols if c in X_test.columns])
    
    model.fit(X_train_clean.values, y_train.values)

    proba_test = model.predict_proba(X_test_clean.values)[:, 1]
    y_pred_test = (proba_test >= 0.5).astype(int)

    prec = precision_score(y_test, y_pred_test, zero_division=0)
    report = classification_report(y_test, y_pred_test, zero_division=0)

    return {
        "test_precision": prec,
        "test_classification_report": report,
        "test_proba": proba_test,
        "test_pred": y_pred_test,
    }


if __name__ == "__main__":
    print("Model training utilities for XGBoost.")


