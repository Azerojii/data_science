from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Hybrid Financial Intelligence System API")


MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "xgb_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feature_columns.joblib")


class InferenceRequest(BaseModel):
    """
    Simple schema for an inference request.
    In practice, you would send the latest engineered feature row for a given ticker.
    """

    features: Dict[str, float]


def load_model_and_columns():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_COLUMNS_PATH):
        raise RuntimeError(
            "Model or feature column file not found. Train and save the model before running the API."
        )
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: InferenceRequest) -> dict:
    model, feature_columns = load_model_and_columns()

    # Build a single-row DataFrame matching training feature order
    row = np.array([[req.features.get(col, 0.0) for col in feature_columns]])
    proba = model.predict_proba(row)[0, 1]

    # Map to discrete action for the frontend (Buy/Hold)
    signal = "Buy" if proba >= 0.5 else "Hold"

    return {
        "probability_buy": float(proba),
        "signal": signal,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


