"""
Quick script to recalculate equity curve with the fixed backtest function.
No need to retrain the model - just reload and recalculate.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd
from backtest import compute_equity_curve, annualized_return, max_drawdown, sharpe_ratio

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Load existing test signals
print("Loading test signals...")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_signals.csv"), parse_dates=["date"])
print(f"Loaded {len(test_df)} predictions for {test_df['ticker'].nunique()} tickers")

# Recalculate equity curve with fixed function
print("\nRecalculating equity curve with proper multi-ticker aggregation...")
backtest_threshold = 0.25
test_df["signal"] = (test_df["proba"] >= backtest_threshold).astype(int)

equity = compute_equity_curve(
    test_df,
    prob_col="proba",
    signal_col="signal",
    return_col="future_return",
    threshold=backtest_threshold,
)
test_df["equity_curve"] = equity.values

# Calculate metrics
strat_returns = test_df["signal"] * test_df["future_return"]
daily_returns = test_df.groupby('date').apply(
    lambda x: (x["signal"] * x["future_return"]).mean()
)

ann_ret = annualized_return(equity.drop_duplicates())
mdd = max_drawdown(equity.drop_duplicates())
sharpe = sharpe_ratio(daily_returns)

print("\n=== CORRECTED Backtest Results ===")
print(f"Starting capital: $1.00")
print(f"Final equity: ${equity.iloc[-1]:.2f}")
print(f"Total return: {(equity.iloc[-1] - 1) * 100:.2f}%")
print(f"Annualized Return: {ann_ret * 100:.2f}%")
print(f"Max Drawdown: {mdd * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")

# Save corrected signals
test_df.to_csv(os.path.join(DATA_DIR, "test_signals.csv"), index=False)
print(f"\n✅ Saved corrected test_signals.csv")
