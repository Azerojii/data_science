# Hybrid Financial Intelligence System — Full Project & ML Guide (for Engineers)

This document is your **reference only**: a complete, engineer-friendly explanation of the project and all ML aspects. No finance background assumed.

---

## 1. What This Project Does (One Paragraph)

We build an **end-to-end ML pipeline** that predicts a **binary outcome**: *“Will this stock go up by more than 3.5% over the next 5 trading days?”* (Yes = 1, No = 0.) We use **historical daily data** for three tickers (SMCI, CRSP, PLTR): prices, volume, and one macro series (10-year US Treasury yield). From that we derive about 40 **features** (rolling statistics of technical indicators), split data **by time** (train before 2023, test from 2023), train **three classifiers** (XGBoost, Random Forest, Logistic Regression), pick one and tune the **decision threshold**, then evaluate with standard **ML metrics** and a **backtest** (simulated trading). So: **supervised binary classification** with a strict **temporal train/test split** and optional strategy evaluation via backtest.

---

## 2. ML in Plain Words

### 2.1 What we do

- **Task:** Supervised **binary classification**. Each sample = one (date, ticker). **Input:** ~40 numbers (features). **Output:** 0 or 1 (label).
- **Label definition:** 1 if the price **5 days later** is more than 3.5% higher than today’s close; else 0.
- **Training:** Fit a model on **past** data (before 2023) so it learns which feature values tend to go with 1 vs 0.
- **Testing:** Evaluate on **future** data (from 2023) to measure how well it **generalizes**. We never use future information when training.

### 2.2 Why split by time (no shuffling)

- If we **shuffled** dates, the model could indirectly use “future” information (e.g. patterns that only exist when you mix past and future). That would **overstate** performance.
- **Time-based split:** train on past, test on future → performance is **realistic** and comparable to “would this have worked in 2023+?”

### 2.3 Train vs test in this project

- **Train:** all data **before 2023-01-01** (e.g. ~2,038 samples after preprocessing).
- **Test (holdout):** all data **from 2023-01-01** onward (e.g. ~2,319 samples).
- We do **not** use test data for training or for choosing the model; we only use it once at the end to report metrics and backtest.

---

## 3. Data & Features (Full Description)

### 3.1 Raw data

- **Source:** Yahoo Finance (daily OHLCV per stock), FRED (10-year US Treasury yield, one time series).
- **OHLCV:** Open, High, Low, Close (four prices per day), Volume (number of shares traded). “Close” is the price we use for returns.
- **Macro:** One number per day (10Y yield). Merged to each (date, ticker) and **forward-filled** for weekends/holidays so every row has a value.
- **Size:** 4,414 rows (one per date × ticker), 3 tickers, 2020–2026. No missing values after preprocessing.

### 3.2 The 12 “base” features (derived time series)

From OHLCV + macro we compute **12 numeric series** per ticker:

| Feature        | What it is (engineer-friendly) |
|----------------|-------------------------------|
| RSI(14)        | Momentum index 0–100 from recent gains/losses (overbought/oversold). |
| MACD (12,26,9) | Trend: difference between fast and slow moving averages; we use line, signal, histogram. |
| Bollinger width| Volatility: (upper band − lower band) / price; bands = 20-day ± 2σ. |
| MA 50 / MA 200 | 50-day and 200-day rolling mean of close price. |
| ATR(14), ATR%  | Average daily high–low range; ATR% = ATR/price × 100 (normalized volatility). |
| Volume Z-score | (volume − 50-day mean) / 50-day std (volume anomaly). |
| 10Y yield      | US government bond rate (macro), same for all tickers on a given date. |

(Plus adjusted close, etc., as in the code; the exact list is in `src/features.py`.)

### 3.3 Rolling window (how we get ~40 features)

- Raw indicators **change scale** over time. To give the model “recent context” and more stable inputs, we use a **14-day rolling window**.
- For **each** of the 12 (or so) base series we compute over the **last 14 trading days**:
  - **mean**
  - **std**
  - **last** (value at the current day)
- So: 12 series × 3 stats ≈ **36 features**, plus identifiers → final matrix **4,357 rows × 40 columns** (after dropping NaN warm-up rows).

---

## 4. Target (Label) Definition

- At each date \( t \), for each ticker, we look at **close price today** and **close price 5 trading days later**.
- **Return:** \( r_{t,t+5} = \frac{\text{close}_{t+5}}{\text{close}_t} - 1 \).
- **Label:** \( y_t = 1 \) if \( r_{t,t+5} > 0.035 \) (3.5%), else \( y_t = 0 \).
- So we are **not** predicting the exact return; we only predict **above 3.5% or not** (binary). In the dataset, about 34% are class 1, 66% class 0; we handle imbalance with e.g. `scale_pos_weight` in XGBoost.

---

## 5. Models We Use

- **XGBoost:** Gradient boosting (trees); we use it as the main model, with 5-fold **time-series cross-validation** on the training set (folds respect time order).
- **Random Forest:** Ensemble of trees; we compare it on the same train/test split.
- **Logistic Regression:** Linear model + sigmoid; baseline. We use **MinMaxScaler** on features before training.
- We compare them on **precision**, **recall**, **F1**, **ROC-AUC** on the **test set** (holdout from 2023). All three are only slightly above random (AUC > 0.50); stock prediction is noisy, so this is expected.

---

## 6. ML Metrics (What We Report)

- **Accuracy:** Fraction of correct predictions (right 0 or 1). Can be misleading when classes are imbalanced (we have ~34% positives).
- **Precision:** Of all rows we **predicted as 1**, how many were **actually 1**? Important when “false positives” are costly (e.g. bad trades).
- **Recall:** Of all **actual 1s**, how many did we **predict as 1**? Trade-off with precision.
- **F1:** Harmonic mean of precision and recall; balances the two.
- **ROC-AUC:** Area under ROC curve (model’s ability to rank positives vs negatives). 0.5 = random; > 0.5 = better than random.
- We also use **confusion matrix** (TP, TN, FP, FN) to see where the model is wrong.

---

## 7. Threshold Tuning (Why Not 0.5?)

- The model outputs a **probability** \( P(\text{class } 1) \). We turn that into a decision: “predict 1” if probability **≥ threshold**, else “predict 0”.
- **Default threshold = 0.5** is not necessarily best. **Higher threshold** → we predict 1 less often → fewer “buy” signals, often **higher precision** but **lower recall**.
- We **sweep** thresholds (e.g. 0.30, 0.35, 0.40, …) and, for each, run a **backtest** (see below). We pick the threshold that gives the best **Sharpe ratio** (risk-adjusted return) on the test period. In the notebook that is **0.40**.

---

## 8. Backtest (What It Is, for an Engineer)

- **Backtest:** Simulate a **trading strategy** on **historical** (here: test) data, **without** using future information. We only use the model’s predictions at each date as they would have been available then.
- **Our rule:** On each day we get a probability from the model. If probability ≥ **threshold** (e.g. 0.40), we “BUY” and hold for 5 days; otherwise we stay in “cash”. We use **non-overlapping 5-day windows** so each day’s decision is applied to a disjoint 5-day block (no double-counting).
- **Metrics we report:**
  - **Annualized return:** If we had run this strategy for a year, what return would we get (annualized)?
  - **Sharpe ratio:** (Return − risk-free rate) / volatility; higher = better **risk-adjusted** return.
  - **Max drawdown:** Largest peak-to-trough drop in the equity curve; lower (e.g. −28%) = more risk.

So: backtest = **replay the strategy on test data** to see if the ML signal would have been profitable and how risky.

---

## 9. Pipeline Summary (Order of Operations)

1. **Data:** Download OHLCV + 10Y yield; merge, forward-fill.
2. **Features:** Add 12 base indicators; build 14-day rolling mean/std/last → 36 features; add target and IDs.
3. **Split:** Train = before 2023; Test = from 2023 (chronological).
4. **Train:** Fit XGBoost (and RF, LR) on train; optional 5-fold time-series CV for hyperparameters.
5. **Evaluate:** Predict on test; compute accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
6. **Threshold:** Sweep threshold; for each, run backtest on test; choose threshold by Sharpe (e.g. 0.40).
7. **Backtest:** Run strategy with chosen threshold on test; report annualized return, Sharpe, max drawdown.
8. **Deploy (optional):** Streamlit dashboard, FastAPI, export signals (e.g. `test_signals.csv`).

---

## 10. Recap for You (Engineer Takeaways)

- **Problem:** Binary classification: “up > 3.5% in 5 days?” (0/1).
- **Data:** Daily OHLCV + 10Y yield; 12 base series → 14-day rolling mean/std/last → ~40 features; 4,357 samples after preprocessing.
- **ML:** Train on past (before 2023), test on future (from 2023). Three models (XGB, RF, LR); XGB used with threshold 0.40.
- **Metrics:** Precision, recall, F1, ROC-AUC, confusion matrix; then backtest (annualized return, Sharpe, max drawdown).
- **Why time split:** Avoid leakage; measure realistic, deployable performance.
- **Why threshold tuning:** 0.5 is arbitrary; we choose threshold by backtest (e.g. Sharpe) to match our “strategy” objective.

All of this is implemented in the notebook and in `src/` (e.g. `features.py`, `model.py`, `backtest.py`). This file is your **standalone reference** — keep it next to the slides when you need the full story.
