from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def split_xy(
    df: pd.DataFrame,
    target_col: str = "target",
    drop_cols: Sequence[str] = ("ticker", "date", "future_return"),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a modeling DataFrame into features (X) and target (y).
    """
    missing = [c for c in (target_col,) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    y = df[target_col].astype(int).copy()
    drop_set = {target_col, *drop_cols}
    x_cols = [c for c in df.columns if c not in drop_set]
    x = df[x_cols].copy()
    return x, y


def evaluate_binary_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    """
    Evaluate binary classification probabilities at a custom decision threshold.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    y_pred = (y_proba_arr >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba_arr))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_proba_arr))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    metrics["coverage"] = float((y_pred == 1).mean())
    return metrics


def threshold_sweep(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """
    Compute precision/recall/F1 trade-offs over a set of probability thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.25, 0.81, 0.05)

    rows = [evaluate_binary_predictions(y_true, y_proba, t) for t in thresholds]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def train_standard_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> dict[str, object]:
    """
    Train three baseline models for quick model-comparison experiments:
      - Logistic Regression
      - Random Forest
      - XGBoost
    """
    models: dict[str, object] = {}

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )
    logistic.fit(x_train, y_train)
    models["LogisticRegression"] = logistic

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    models["RandomForest"] = rf

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(x_train.values, y_train.values)
    models["XGBoost"] = xgb

    return models


def compare_models(
    models: Mapping[str, object],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Evaluate multiple fitted models on a single holdout set.
    """
    rows: list[dict[str, float | str]] = []

    for name, model in models.items():
        if isinstance(model, XGBClassifier):
            y_proba = model.predict_proba(x_test.values)[:, 1]
        else:
            y_proba = model.predict_proba(x_test)[:, 1]

        metrics = evaluate_binary_predictions(y_test, y_proba, threshold=threshold)
        rows.append({"model": name, **metrics})

    return (
        pd.DataFrame(rows)
        .sort_values(["precision", "f1", "recall"], ascending=False)
        .reset_index(drop=True)
    )


def infer_feature_groups(columns: Sequence[str]) -> dict[str, list[str]]:
    """
    Group engineered features into semantic blocks for ablation experiments.
    """
    groups = {
        "volatility_atr": [c for c in columns if "atr_" in c],
        "macro_rates": [c for c in columns if "ten_year_yield" in c],
        "sentiment": [c for c in columns if c.startswith("sent_") or "sentiment_momentum" in c],
        "trend_momentum": [
            c
            for c in columns
            if any(k in c for k in ("rsi_", "macd", "bb_", "ma_", "volume_z"))
        ],
    }
    return groups


def run_xgboost_ablation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_groups: Dict[str, Sequence[str]] | None = None,
    threshold: float = 0.50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Retrain XGBoost while removing each feature group to estimate feature value.
    """
    if feature_groups is None:
        feature_groups = infer_feature_groups(list(x_train.columns))

    experiments: list[dict[str, float | str]] = []

    runs = {"full_model": []}
    for name, cols in feature_groups.items():
        runs[f"drop_{name}"] = list(cols)

    for label, drop_cols in runs.items():
        keep_cols = [c for c in x_train.columns if c not in set(drop_cols)]

        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        model = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(x_train[keep_cols].values, y_train.values)
        y_proba = model.predict_proba(x_test[keep_cols].values)[:, 1]
        metrics = evaluate_binary_predictions(y_test, y_proba, threshold=threshold)
        experiments.append({"experiment": label, "n_features": len(keep_cols), **metrics})

    return pd.DataFrame(experiments).sort_values("precision", ascending=False).reset_index(drop=True)
