from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from backtest import annualized_return, compute_equity_curve, max_drawdown, sharpe_ratio
from data_ingestion import download_price_data, load_macro_data, merge_price_and_macro
from features import add_technical_indicators, build_rolling_window_dataset
from model import ModelConfig, evaluate_on_holdout, train_xgboost_time_series
from sentiment import merge_sentiment, score_headlines_with_finbert


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def maybe_add_sentiment(df_prices_macro: pd.DataFrame) -> pd.DataFrame:
    """
    If a headlines file exists (data/news_headlines.csv with columns: date,ticker,headline),
    run FinBERT and merge daily sentiment features; otherwise, return the input unchanged.
    """
    headlines_path = os.path.join(DATA_DIR, "news_headlines.csv")
    if not os.path.exists(headlines_path):
        return df_prices_macro

    df_headlines = pd.read_csv(headlines_path)
    sent_daily = score_headlines_with_finbert(df_headlines, text_col="headline", date_col="date")
    df_with_sent = merge_sentiment(df_prices_macro, sent_daily)
    return df_with_sent


def main() -> None:
    print("[1/9] Downloading price data...")
    prices = download_price_data()
    print(f"[1/9] Prices downloaded: {prices['ticker'].nunique()} tickers, {len(prices)} rows.")

    try:
        print("[2/9] Loading macro data from CSV...")
        macro = load_macro_data()
    except FileNotFoundError:
        # If no macro data file exists yet, create a dummy constant series so the pipeline can run.
        macro = (
            prices[["date"]]
            .drop_duplicates()
            .assign(ten_year_yield=2.0)
            .reset_index(drop=True)
        )
        macro.to_csv(os.path.join(DATA_DIR, "macro_10y_yield.csv"), index=False)

    print("[3/9] Merging prices with macro data...")
    df = merge_price_and_macro(prices, macro)
    print(f"[3/9] Merged shape: {df.shape}")

    # 2. Optionally add sentiment if headlines file is available
    df = maybe_add_sentiment(df)

    # 3. Add technical indicators
    print("[4/9] Adding technical indicators...")
    df = add_technical_indicators(df)
    print(f"[4/9] After indicators, columns: {len(df.columns)}")

    # 4. Build full rolling-window dataset (includes future_return)
    config = ModelConfig()
    print("[5/9] Building rolling-window dataset...")
    dataset = build_rolling_window_dataset(
        df,
        window_days=config.window_days,
        horizon_days=config.horizon_days,
        threshold=config.threshold,
    )
    print(f"[5/9] Rolling-window dataset shape: {dataset.shape}")

    # 5. Split into train/test by date (2006–2022 train, 2023+ test)
    dataset = dataset.sort_values("date").reset_index(drop=True)

    cutoff_date = pd.Timestamp("2023-01-01")
    train_mask = dataset["date"] < cutoff_date
    test_mask = ~train_mask

    train_df = dataset[train_mask].reset_index(drop=True)
    test_df = dataset[test_mask].reset_index(drop=True)
    print(f"[6/9] Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # 6. Train model with time-series CV on train set
    print("[7/9] Training XGBoost with time-series CV...")
    model, cv_metrics = train_xgboost_time_series(X_train, y_train, config)

    # 7. Evaluate on chronological holdout (2023+)
    test_metrics = evaluate_on_holdout(model, X_train, y_train, X_test, y_test)

    print("=== Cross-Validation (Train) ===")
    print("Best CV precision:", cv_metrics["cv_best_precision"])
    print(cv_metrics["cv_classification_report"])

    print("=== Test (2023+) ===")
    print("Test precision:", test_metrics["test_precision"])
    print(test_metrics["test_classification_report"])

    # 8. Backtest using test-period signals (NON-OVERLAPPING periods)
    print("[8/9] Running backtest on test-period signals...")
    test_df = test_df.copy()
    test_df["proba"] = test_metrics["test_proba"]
    
    # STRATEGY UPDATE: Long-Only for Safety
    # The classification model predicts "Will it go up > 3.5%?".
    # Low probability means "It won't go up significantly", NOT "It will crash".
    # Therefore, shorting on low probability is dangerous in a bull market.
    # We use:
    #   - Signal 1 (BUY): High confidence it will go up.
    #   - Signal 0 (CASH): Low confidence, stay in cash (protect capital).
    
    buy_threshold = 0.50  # Slightly stricter threshold
    
    # 0 = CASH (formatted as 'SELL/HOLD' in UI), 1 = BUY
    test_df["signal"] = np.where(test_df["proba"] >= buy_threshold, 1, 0)
    
    print(f"Generated signals with threshold {buy_threshold}:")
    print(f"  Buy (1): {sum(test_df['signal'] == 1)}")
    print(f"  Cash (0): {sum(test_df['signal'] == 0)}")
    
    # Since we have 5-day forward returns (horizon_days=5), we should only
    # use non-overlapping periods to avoid inflating returns
    all_dates = sorted(test_df["date"].unique())
    # Ensure config object is available - use default from model.py if not imported
    horizon = config.horizon_days if 'config' in locals() else 5
    non_overlapping_dates = [all_dates[i] for i in range(0, len(all_dates), horizon)]
    test_df_no_overlap = test_df[test_df["date"].isin(non_overlapping_dates)].copy()
    
    # Calculate strategy returns for non-overlapping periods
    # signal: 1=long, 0=cash
    test_df_no_overlap["strategy_return"] = test_df_no_overlap["signal"] * test_df_no_overlap["future_return"]
    
    # Equal-weight portfolio: average returns by date
    daily_returns = test_df_no_overlap.groupby("date")["strategy_return"].mean()
    equity_curve_no_overlap = (1 + daily_returns).cumprod()
    
    # Map equity back to non-overlapping dataframe
    equity_dict = equity_curve_no_overlap.to_dict()
    test_df_no_overlap["equity_curve"] = test_df_no_overlap["date"].map(equity_dict)
    
    # For the full dataframe, forward fill equity values
    test_df["equity_curve"] = 1.0
    for date in non_overlapping_dates:
        if date in equity_dict:
            test_df.loc[test_df["date"] == date, "equity_curve"] = equity_dict[date]
    test_df["equity_curve"] = test_df.groupby("ticker")["equity_curve"].ffill()
    
    # Calculate metrics (using non-overlapping returns)
    ann_ret = annualized_return(equity_curve_no_overlap, periods_per_year=252/horizon)
    mdd = max_drawdown(equity_curve_no_overlap)
    sharpe = sharpe_ratio(daily_returns, periods_per_year=252/horizon)

    print("=== Backtest (2023+ signals) ===")
    print("Annualized Return:", ann_ret)
    print("Max Drawdown:", mdd)
    print("Sharpe Ratio:", sharpe)

    # 9. Save artifacts for dashboard and API
    feature_columns = [c for c in X_train.columns if c not in ("ticker", "date", "future_return")]
    joblib.dump(model, os.path.join(DATA_DIR, "xgb_model.joblib"))
    joblib.dump(feature_columns, os.path.join(DATA_DIR, "feature_columns.joblib"))

    dataset.to_csv(os.path.join(DATA_DIR, "rolling_window_dataset.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_signals.csv"), index=False)

    print("[9/9] Model, feature columns, and datasets saved in 'data/' directory.")
    print("      Sharpe Ratio on test-period backtest:", sharpe)


if __name__ == "__main__":
    main()

