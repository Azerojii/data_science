from __future__ import annotations

import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Hybrid Financial Intelligence System API")


# Prefer the notebook's bundled artifacts; fall back to legacy separate files
MODEL_ARTIFACTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "model_artifacts.joblib")
LEGACY_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "xgb_model.joblib")
LEGACY_FEATURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feature_columns.joblib")


class InferenceRequest(BaseModel):
    """
    Simple schema for an inference request.
    In practice, you would send the latest engineered feature row for a given ticker.
    """

    features: Dict[str, float]


def load_model_and_columns():
    # Try notebook artifacts first (model_artifacts.joblib)
    if os.path.exists(MODEL_ARTIFACTS_PATH):
        artifacts = joblib.load(MODEL_ARTIFACTS_PATH)
        model = artifacts["model"]
        feature_columns = artifacts["input_cols"]
        scaler = artifacts.get("scaler", None)
        threshold = artifacts.get("threshold", 0.5)
        return model, feature_columns, scaler, threshold

    # Fall back to legacy pipeline files
    if os.path.exists(LEGACY_MODEL_PATH) and os.path.exists(LEGACY_FEATURES_PATH):
        model = joblib.load(LEGACY_MODEL_PATH)
        feature_columns = joblib.load(LEGACY_FEATURES_PATH)
        return model, feature_columns, None, 0.5

    raise RuntimeError(
        "No model found. Run the notebook or pipeline_train.py first."
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: InferenceRequest) -> dict:
    model, feature_columns, scaler, threshold = load_model_and_columns()

    # Build a single-row array matching training feature order
    row = np.array([[req.features.get(col, 0.0) for col in feature_columns]])

    # Apply scaler if the notebook model was trained with one
    if scaler is not None:
        row = scaler.transform(row)

    proba = model.predict_proba(row)[0, 1]

    signal = "Buy" if proba >= threshold else "Hold"

    return {
        "probability_buy": float(proba),
        "signal": signal,
        "threshold": threshold,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
