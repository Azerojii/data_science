from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


FINBERT_MODEL_NAME = "ProsusAI/finbert"


class FinBertSentiment:
    """
    Thin wrapper around FinBERT to produce per-headline sentiment scores.

    For each text, outputs probabilities for [negative, neutral, positive].
    """

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        probs_list: List[np.ndarray] = []
        texts_list = list(texts)

        for i in range(0, len(texts_list), batch_size):
            batch_texts = texts_list[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)

        return np.vstack(probs_list) if probs_list else np.zeros((0, 3), dtype=float)


def score_headlines_with_finbert(
    df_headlines: pd.DataFrame,
    text_col: str = "headline",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Given a DataFrame of headlines with columns:
        [date, ticker, headline]
    compute FinBERT sentiment probabilities and aggregate to daily features:
        - sent_pos_mean, sent_neg_mean, sent_neu_mean
        - sentiment_momentum = sent_pos_mean - sent_neg_mean
    """
    if df_headlines.empty:
        raise ValueError("Headline DataFrame is empty.")

    model = FinBertSentiment()
    probs = model.predict_proba(df_headlines[text_col].tolist())

    df = df_headlines.copy()
    df["sent_neg"] = probs[:, 0]
    df["sent_neu"] = probs[:, 1]
    df["sent_pos"] = probs[:, 2]

    # Aggregate per date and ticker
    df[date_col] = pd.to_datetime(df[date_col])
    grouped = (
        df.groupby(["ticker", date_col])
        .agg(
            sent_neg_mean=("sent_neg", "mean"),
            sent_neu_mean=("sent_neu", "mean"),
            sent_pos_mean=("sent_pos", "mean"),
        )
        .reset_index()
    )
    grouped["sentiment_momentum"] = grouped["sent_pos_mean"] - grouped["sent_neg_mean"]

    out_path = os.path.join(DATA_DIR, "sentiment_daily.csv")
    grouped.to_csv(out_path, index=False)
    return grouped


def merge_sentiment(
    df_prices_macro: pd.DataFrame, 
    df_sentiment: pd.DataFrame,
    decay_days: float = 0.5  # 12 hours for high-volatility regime (was 2.0 for stable stocks)
) -> pd.DataFrame:
    """
    Merge daily sentiment aggregates with price+macro data, with forward fill to handle
    days without news (weekends/holidays) per ticker.
    
    For high-volatility regimes:
      - decay_days=0.5 (12 hours) for fast-moving stocks like SMCI, PLTR
      - News impact is priced in almost instantly in AI/Biotech sectors
    
    Sentiment decay: sentiment_weight = exp(-age_in_days / decay_days)
    """
    df = df_prices_macro.copy()
    df["date"] = pd.to_datetime(df["date"])
    df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])

    merged_list: List[pd.DataFrame] = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        sent = df_sentiment[df_sentiment["ticker"] == ticker].sort_values("date")
        merged = g.merge(sent, on=["ticker", "date"], how="left")
        
        # Apply exponential decay to sentiment features
        sentiment_cols = [c for c in merged.columns if c.startswith("sent_") or "sentiment_momentum" in c]
        
        # Forward fill with decay weighting
        for col in sentiment_cols:
            last_valid_value = None
            last_valid_date = None
            
            for idx in merged.index:
                current_date = merged.loc[idx, "date"]
                current_value = merged.loc[idx, col]
                
                if pd.notna(current_value):
                    # New sentiment data point
                    last_valid_value = current_value
                    last_valid_date = current_date
                elif last_valid_value is not None:
                    # Apply exponential decay
                    days_elapsed = (current_date - last_valid_date).total_seconds() / 86400
                    decay_weight = np.exp(-days_elapsed / decay_days)
                    merged.loc[idx, col] = last_valid_value * decay_weight
        
        merged_list.append(merged)

    out = pd.concat(merged_list, ignore_index=True)
    
    # Remove duplicate columns (keep first occurrence)
    out = out.loc[:, ~out.columns.duplicated()]
    
    return out


if __name__ == "__main__":
    print("Sentiment utilities for FinBERT.")


