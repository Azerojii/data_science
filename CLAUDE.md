# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hybrid Financial Intelligence System — an ML-based trading strategy using XGBoost to predict 5-day stock price movements. Supports two regimes: high-volatility stocks (SMCI, CRSP, PLTR) and stable large-caps (AAPL, NVDA, TSLA, MSFT, AMZN).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (download data → engineer features → train → evaluate → backtest)
cd src && python pipeline_train.py

# Launch Streamlit dashboard
streamlit run dashboard.py

# Start FastAPI backend
cd src && uvicorn api:app --reload

# ATR stop-loss calculator
python src/atr_stop_loss.py SMCI 825.0 --multiplier 2.5 --account-size 100000 --risk 0.02
```

## Architecture

**Pipeline flow** (`src/pipeline_train.py` orchestrates everything):

1. **Data ingestion** (`data_ingestion.py`) — yfinance for OHLCV, FRED API for 10Y yield, Finnhub for news
2. **Feature engineering** (`features.py`) — technical indicators (RSI, MACD, Bollinger Bands, ATR, MAs, Volume Z-score) + 14-day rolling window aggregates (~54 features total)
3. **Sentiment** (`sentiment.py`) — FinBERT scoring with exponential decay (optional, requires `FINNHUB_API_KEY`)
4. **Model training** (`model.py`) — XGBoost binary classifier with TimeSeriesSplit CV (5 folds), class-balanced via `scale_pos_weight`
5. **Backtesting** (`backtest.py`) — non-overlapping 5-day periods, equity curves, Sharpe ratio, max drawdown
6. **Deployment** — Streamlit dashboard (`dashboard.py`) or FastAPI (`api.py`)

**Target variable:** `target = 1 if (close[t+5]/close[t] - 1) > threshold else 0`

**Signal mapping:** BUY (prob ≥ 0.50), CASH (prob < 0.50). Long-only strategy.

## Regime Configuration

Edit `src/config_volatility_regime.py` line ~70 to switch regimes, then re-run `pipeline_train.py`.

| Parameter | HIGH_VOLATILITY | STABLE_LARGE_CAP |
|-----------|----------------|-------------------|
| Success threshold | 3.5% | 1.5% |
| ATR stop multiplier | 2.5x | 2.0x |
| Sentiment decay | 0.5 days | 2.0 days |
| Data start | 2020-01-01 | 2006-01-01 |

## Key Data Files

- `data/xgb_model.joblib` — trained model
- `data/feature_columns.joblib` — feature column list for inference
- `data/rolling_window_dataset.csv` — full feature-engineered dataset
- `data/test_signals.csv` — predictions on 2023+ test set

## Environment Variables

Set in `.env`: `FRED_API_KEY` (macro data) and `FINNHUB_API_KEY` (news sentiment). Macro and sentiment are optional — the pipeline runs without them.
