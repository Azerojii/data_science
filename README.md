## Hybrid Financial Intelligence System – Capstone Project

This repository implements a **Hybrid Financial Intelligence System** with support for multiple volatility regimes:

- **Objective**: Supervised binary classification to predict whether a stock will exceed a volatility-adjusted return threshold over the next 5 trading days.
- **Regimes**: 
  - **High-Volatility** (default): High-beta growth stocks (SMCI, CRSP, PLTR) with 5-15% weekly swings
  - **Stable Large-Cap**: Big Tech (AAPL, NVDA, TSLA, MSFT, AMZN) with 1-3% movements
- **Modalities**: Technical indicators (including ATR), macro factors, and sentiment (FinBERT with exponential decay).
- **Models**: Rolling-window feature construction + XGBoost classifier, with volatility-normalized features.
- **Risk Management**: ATR-based stop losses and position sizing for high-volatility regimes.
- **Deployment Goal**: Fast, modern dashboard with real-time inference and sell signals.

### Key Features for High-Volatility Trading

1. **Dynamic Target Thresholding**: 3.5% threshold for high-beta stocks (vs 1.5% for stable)
2. **ATR-Based Features**: Average True Range normalization for volatility-aware predictions
3. **Sentiment Decay**: 12-hour half-life for fast-moving stocks (vs 48h for stable)
4. **ATR Stop Losses**: Formula: `StopLoss = Entry - (2.5 × ATR_14)`
5. **Three-Way Signals**: BUY (long), SELL (short), or HOLD (cash)

This is a starting implementation skeleton; you can run and extend it in your local environment.

### Project Structure

- `data/` – raw and intermediate CSV files (prices, macro, sentiment, rolling-window dataset, test signals).
- `notebooks/` – exploratory notebooks (EDA, experiments) – you can add your own here.
- `src/` – Python source code:
  - `data_ingestion.py` – download/load price and macro data with forward-filled macro.
  - `features.py` – technical indicators (including ATR), rolling windows, target generation.
  - `sentiment.py` – FinBERT-based sentiment scoring with exponential decay.
  - `model.py` – XGBoost training with volatility-adjusted thresholds.
  - `backtest.py` – backtesting utilities (equity curve, annualized return, Sharpe, max drawdown).
  - `config_volatility_regime.py` – **NEW**: Regime configuration (high-vol vs stable).
  - `atr_stop_loss.py` – **NEW**: ATR-based stop loss calculator and position sizing.
  - `api.py` – FastAPI backend for real-time inference.
  - `pipeline_train.py` – full training + evaluation + backtest pipeline.
- `dashboard.py` – Streamlit front-end with BUY/SELL/HOLD signals and premium dark theme.
- `requirements.txt` – main Python dependencies.

You can adapt this structure to match your course requirements or personal preferences.

### How to Run the Full Pipeline (A → Z)

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **(Optional) Add macro and news sentiment data**:
- **Automatic via APIs (recommended)**:
  - Get a **FRED API key** from the St. Louis Fed website and set it as an environment variable:
    - `FRED_API_KEY=your_key_here`
  - Get a **Finnhub API key** from their website and set:
    - `FINNHUB_API_KEY=your_key_here`
  - Then run:

```bash
cd src
python live_ingest.py
```

  This will:
  - Update `data/macro_10y_yield.csv` with the latest 10Y Treasury Yield (DGS10) from FRED.
  - Update/append `data/news_headlines.csv` with the latest company news headlines for Big Tech tickers.

- **Manual CSV option** (if you prefer):
  - Macro: place `macro_10y_yield.csv` in `data/` with columns: `date,ten_year_yield`.
  - News: place `news_headlines.csv` in `data/` with columns: `date,ticker,headline`.

3. **Train, evaluate, and backtest**:

```bash
cd src
python pipeline_train.py
```

This will:
- Download OHLCV prices (Big Tech).
- Merge macro + (optionally) FinBERT sentiment.
- Build rolling-window features and labels.
- Train XGBoost with time-series CV.
- Evaluate on a 2023+ holdout.
- Run a backtest and print **Sharpe ratio**, **annualized return**, and **max drawdown**.
- Save:
  - `data/xgb_model.joblib`
  - `data/feature_columns.joblib`
  - `data/rolling_window_dataset.csv`
  - `data/test_signals.csv`

4. **Run the FastAPI backend** (optional, for external full-stack use):

```bash
cd src
uvicorn api:app --reload
```

5. **Launch the modern dashboard (Streamlit)**:

From the project root:

```bash
streamlit run dashboard.py
```

The dashboard includes:
- **Live Signals** – BUY (green), SELL (red), or HOLD (gray) for each ticker with confidence scores
- **Equity Curve** – Portfolio performance following algorithm signals (non-overlapping periods)
- **Performance Metrics** – Annualized return, Sharpe ratio, max drawdown for each ticker
- **Recent News** – Latest headlines with FinBERT sentiment scores per ticker

### Switching Volatility Regimes

The system supports two pre-configured regimes. To switch between them:

**Option 1: Edit config file** (recommended for permanent changes)
```python
# In src/config_volatility_regime.py, change line:
REGIME = HIGH_VOLATILITY  # or STABLE_LARGE_CAP
```

**Option 2: Compare regimes**
```bash
cd src
python config_volatility_regime.py
```

This prints a comparison table:

| Parameter | High-Volatility | Stable Large-Cap |
|-----------|----------------|------------------|
| Tickers | SMCI, CRSP, PLTR | AAPL, NVDA, TSLA, MSFT, AMZN |
| Success Threshold | 3.5% | 1.5% |
| Sentiment Half-Life | 12h | 48h |
| ATR Stop Multiplier | 2.5x | 2.0x |

After changing regimes, re-run `python src/pipeline_train.py` to retrain on the new ticker universe.

### Using ATR-Based Stop Losses

Calculate risk-adjusted stop losses for position management:

**Command-line tool:**
```bash
# Calculate stop loss for SMCI at $825 entry
python src/atr_stop_loss.py SMCI 825.0

# With custom parameters
python src/atr_stop_loss.py SMCI 825.0 --multiplier 2.5 --account-size 100000 --risk 0.02
```

**Output example:**
```
==============================================================
ATR-BASED STOP LOSS: SMCI
==============================================================

Entry Price:        $825.00
ATR (14-day):       $45.00
Multiplier:         2.5x

Stop Loss:          $712.50
Risk per Share:     $112.50 (13.6%)

--------------------------------------------------------------
POSITION SIZING
--------------------------------------------------------------
Account Size:       $100,000
Risk per Trade:     2.0% ($2,000)

Recommended Shares: 17
Position Value:     $14,025
Total Risk:         $1,913
Risk % of Account:  1.91%
==============================================================
```

**Python usage:**
```python
from atr_stop_loss import calculate_stop_loss, get_current_atr, calculate_position_size

# Get current ATR
atr = get_current_atr("SMCI")

# Calculate stop for long position
stop = calculate_stop_loss(entry_price=825.0, atr=atr, multiplier=2.5)

# Size position for 2% account risk
shares = calculate_position_size(
    account_size=100000,
    entry_price=825.0,
    stop_price=stop,
    risk_per_trade=0.02
)
```

### Theory: Why Different Regimes Need Different Parameters

**1. Target Threshold (1.5% vs 3.5%)**
- High-volatility stocks like SMCI can move ±5% daily on noise. A 1.5% move means nothing.
- For stable stocks, 1.5% often represents genuine momentum from catalysts.
- Solution: Higher threshold (3.5%) filters noise from signal in high-beta names.

**2. ATR-Based Features**
- Model learns to normalize price movements: $10 move in $500 stock with $50 ATR ≠ $10 move in $100 stock with $3 ATR.
- ATR% (ATR/Price) provides scale-invariant volatility measurement.

**3. Sentiment Decay (12h vs 48h)**
- AI/Biotech/Meme stocks: News priced in within hours.
- Blue chips: Earnings/product launches have multi-day impact.
- Solution: Exponential decay with shorter half-life for high-volatility.

**4. ATR Stop-Loss (2.5x vs 2.0x)**
- High-vol needs wider stops to avoid being "shaken out" by normal intraday swings.
- Formula: `StopLoss = Entry - (k × ATR_14)` where k=2.5 for high-vol, 2.0 for stable.

### Signal Types

The system generates **three-way signals**:

- **BUY (signal=1)**: Probability ≥ 0.48 → Go long, expecting >3.5% gain
- **SELL (signal=-1)**: Probability ≤ 0.30 → Go short, expecting price decline
- **HOLD (signal=0)**: 0.30 < prob < 0.48 → Stay in cash, uncertain direction

Backtest handles long/short positioning automatically:
- BUY: Earn +future_return when position moves up
- SELL: Earn -future_return (profit when position moves down)
- HOLD: No position, 0% return


