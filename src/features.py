from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import ta


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators per ticker:
      - RSI(14)
      - MACD (12,26,9)
      - Bollinger Bands (20)
      - Moving averages (50, 200)
    """
    df = df.sort_values(["ticker", "date"]).copy()
    
    # Ensure columns are not multi-level and remove duplicates
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
    df = df.loc[:, ~df.columns.duplicated()]
    
    grouped = []

    for ticker, g in df.groupby("ticker", group_keys=False):
        g = g.copy()
        close = g["adj_close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        volume = g["volume"].astype(float)

        g["rsi_14"] = ta.momentum.RSIIndicator(close).rsi()

        macd = ta.trend.MACD(close)
        g["macd"] = macd.macd()
        g["macd_signal"] = macd.macd_signal()
        g["macd_hist"] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(close)
        g["bb_high"] = bb.bollinger_hband()
        g["bb_low"] = bb.bollinger_lband()
        g["bb_width"] = ((bb.bollinger_hband() - bb.bollinger_lband()) / close)

        g["ma_50"] = close.rolling(window=50).mean()
        g["ma_200"] = close.rolling(window=200).mean()
        g["volume_z"] = ((volume - volume.rolling(50).mean()) / (volume.rolling(50).std() + 1e-9))
        
        # ATR (Average True Range) - Critical for high-volatility normalization
        atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        g["atr_14"] = atr_indicator.average_true_range()
        # ATR as % of price (normalized volatility metric)
        g["atr_pct"] = (g["atr_14"] / close) * 100

        grouped.append(g)

    out = pd.concat(grouped, ignore_index=True)
    return out


def add_future_return_and_label(
    df: pd.DataFrame,
    horizon_days: int = 5,
    threshold: float = 0.035,  # Default to 3.5% for high-volatility regime
) -> pd.DataFrame:
    """
    For each ticker and day, compute cumulative future return over `horizon_days`
    and create binary label:
        future_return = close[t+horizon]/close[t] - 1
        target = 1 if future_return > threshold else 0
    
    Note: Threshold adjusted for volatility regime:
      - Stable Large Caps (AAPL, MSFT): 1.5% is significant
      - High-Beta Growth (SMCI, PLTR): 3.5% filters noise from trend
    """
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = []

    for ticker, g in df.groupby("ticker", group_keys=False):
        close = g["adj_close"].astype(float)
        future_price = close.shift(-horizon_days)
        g["future_return"] = future_price / close - 1.0
        g["target"] = (g["future_return"] > threshold).astype(int)
        grouped.append(g)

    out = pd.concat(grouped, ignore_index=True)
    return out


def build_rolling_window_dataset(
    df: pd.DataFrame,
    window_days: int = 14,
    horizon_days: int = 5,
    threshold: float = 0.035,  # High-volatility threshold
) -> pd.DataFrame:
    """
    Construct rolling-window dataset using the last `window_days` for each prediction day.

    Each row contains:
      - ticker, date
      - flattened window statistics (mean/std/last for each feature)
      - target (binary label)
      - future_return (for backtesting)
    """
    df = add_future_return_and_label(df, horizon_days=horizon_days, threshold=threshold)

    feature_cols = [
        "adj_close",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width",
        "ma_50",
        "ma_200",
        "volume_z",
        "atr_14",      # ATR for volatility-aware modeling
        "atr_pct",     # ATR as % of price (normalized)
    ]
    # Include macro and sentiment columns if present
    exclude = {
        "ticker",
        "date",
        "target",
        "future_return",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    macro_sent_cols = [c for c in df.columns if c not in feature_cols and c not in exclude]
    feature_cols = feature_cols + macro_sent_cols

    rows = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        for idx in range(window_days, len(g) - horizon_days):
            window = g.iloc[idx - window_days : idx]
            current = g.iloc[idx].copy()

            stats = {}
            for col in feature_cols:
                if col not in window.columns:
                    continue
                series = window[col].astype(float)
                # Use skipna=True to ignore NaN values in calculations
                stats[f"{col}_mean"] = series.mean(skipna=True)
                stats[f"{col}_std"] = series.std(skipna=True)
                # For last value, use the last non-NaN value if available
                if series.notna().any():
                    stats[f"{col}_last"] = series.dropna().iloc[-1] if len(series.dropna()) > 0 else np.nan
                else:
                    stats[f"{col}_last"] = np.nan

            row = {
                "ticker": current["ticker"],
                "date": current["date"],
                "target": g.loc[idx, "target"],
                "future_return": g.loc[idx, "future_return"],
                **stats,
            }
            
            # Skip rows where target or future_return is NaN
            if pd.isna(row["target"]) or pd.isna(row["future_return"]):
                continue
                
            rows.append(row)

    dataset = pd.DataFrame(rows)
    
    # Only drop rows where target is NaN (future_return NaN already filtered above)
    # Fill remaining NaN values with 0 (for features where warm-up caused NaN)
    dataset = dataset[dataset["target"].notna()].copy()
    
    # Fill NaN in feature columns with 0 (conservative approach for missing indicators)
    feature_cols_to_fill = [c for c in dataset.columns if c not in ["ticker", "date", "target", "future_return"]]
    dataset[feature_cols_to_fill] = dataset[feature_cols_to_fill].fillna(0)
    
    dataset = dataset.reset_index(drop=True)
    return dataset


def build_rolling_window_features(
    df: pd.DataFrame,
    window_days: int = 14,
    horizon_days: int = 5,
    threshold: float = 0.035,  # High-volatility threshold
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Backwards-compatible helper that returns only (X, y) from the full rolling-window dataset.
    """
    dataset = build_rolling_window_dataset(
        df,
        window_days=window_days,
        horizon_days=horizon_days,
        threshold=threshold,
    )
    X = dataset.drop(columns=["target"])
    y = dataset["target"].astype(int)
    return X, y


if __name__ == "__main__":
    print("This module defines feature engineering utilities.")


