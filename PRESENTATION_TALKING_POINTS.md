# Talking Points — Slide by Slide

Use this as your cheat sheet while presenting. One section per slide; bullets = what to say or remember.

---

## Slide 1 — Title
- Say: title, subtitle, your names, institute, supervisor, date.
- Optional: “We’ll walk through an ML pipeline that predicts short-term stock moves.”

---

## Slide 2 — Outline
- Briefly name the sections: Introduction, Data, EDA, Feature Engineering, Model Training & Evaluation, Deployment, Conclusion.
- “We’ll follow the pipeline from data to deployment.”

---

## Slide 3 — What This Project Does (No Finance Needed)
- **One sentence:** We use historical daily data to train a model that answers: “Will this asset go up more than 3.5% in the next 5 days?” Yes or no.
- **Why 3.5% and 5 days:** Clear binary target; 3.5% is meaningful for volatile stocks; 5 days gives many examples.
- **Stocks:** SMCI, CRSP, PLTR — volatile so 3.5% moves happen often enough to learn from.

---

## Slide 4 — Project Overview
- **Objective:** End-to-end ML pipeline: binary classification — predict if price rises >3.5% in next 5 trading days.
- **Target stocks:** SMCI, CRSP, PLTR. **Tech:** Python (pandas, scikit-learn, XGBoost), Streamlit, FastAPI.
- **Pipeline in short:** (1) Get data → build features, (2) Split by time (train before 2023, test from 2023), (3) Train 3 models, pick one, tune threshold, (4) Evaluate with metrics + backtest.

---

## Slide 5 — ML in Plain Words (for Engineers)
- **What we do:** Supervised binary classification. Input = ~40 numbers per (day, stock). Output = 0 or 1.
- **Training:** Fit on past data (before 2023). **Testing:** Evaluate on future data (from 2023) to see generalization.
- **Why split by time:** Shuffling would let the model “cheat” with future info. Time split = realistic performance.

---

## Slide 6 — Data Sources & Dataset
- **Sources:** Yahoo Finance → daily OHLCV per stock. FRED → 10-year US Treasury yield (one series).
- **Size:** 4,414 rows (one per date × ticker), 2020–2026, 3 tickers. No missing values after preprocessing.

---

## Slide 7 — Preprocessing Steps
- **Steps in order:** (1) Download OHLCV, merge 10Y yield, forward-fill. (2) Compute 12 derived series (RSI, MACD, etc.). (3) Rolling mean/std/last over 14 days → 36 features. (4) Define label: 1 if next 5-day return > 3.5%, else 0. (5) Drop NaN warm-up rows.
- **Result:** 4,357 rows × 40 columns, ready for ML.

---

## Slide 8 — Dataset & Features — Raw Columns
- **Raw columns:** OHLCV (Open, High, Low, Close, Volume); Macro (10Y yield, forward-filled).
- **Next:** We derive 12 time series from these, then apply a 14-day rolling window (mean, std, last) so the model sees recent context.

---

## Slide 9 — Dataset & Features — The 12 Base Features
- **12 base series:** RSI, MACD, Bollinger width, MA 50/200, ATR & ATR%, Volume Z-score, 10Y yield (and a few more in code).
- **Rolling:** For each series, mean, std, last over 14 days → 36 features. Final matrix 4,357 × 40.

---

## Slide 10 — Target & Macro Context
- **What we predict:** “Will this stock go up more than 3.5% over the next 5 trading days?” Binary yes/no. Formula: label = 1 if 5-day return > 0.035.
- **10Y yield:** One daily series from FRED; we use it only as an input feature (macro regime). Chart shows how it evolves; model uses it as a number.

---

## Slide 11 — EDA — Stock Prices
- **Chart:** Normalized prices for the three tickers over time.
- **Observations:** SMCI has big swings; PLTR uptrend since late 2022; CRSP its own pattern. Daily volatility around 3.8–5.2%.
- **Target:** ~34% positive (up >3.5%), ~66% negative; we handle imbalance with e.g. scale_pos_weight in XGBoost.

---

## Slide 12 — EDA — Distributions & Correlations
- **Left:** 5-day forward return distribution. **Right:** Target class balance (~34% vs 66%).
- One line: “This is the distribution we’re trying to predict; the model learns to separate the two classes.”

---

## Slide 13 — Feature Engineering
- **12 base indicators:** RSI, MACD, Bollinger, MAs, ATR, volume Z-score, 10Y yield.
- **Rolling window (14 days):** For each indicator we take mean, std, last → 12×3 = 36 features. Captures recent trend and variability.
- **Why rolling:** Raw indicators change scale over time; rolling stats help generalization.

---

## Slide 14 — Pipeline Architecture
- **Flow:** Data ingestion → Feature engineering → Chrono split (2023) → Train (XGB, RF, LR) → 5-fold TimeSeries CV → Holdout evaluation.
- **Train:** Before 2023-01-01 (~2,038 samples). **Test:** From 2023-01-01 (~2,319 samples). **Scaling:** MinMaxScaler. **CV metric:** Precision (5-fold).

---

## Slide 15 — Model Comparison — Results
- **Table:** XGBoost, Random Forest, Logistic Regression — Accuracy, Precision, F1, ROC-AUC. All slightly above random (AUC > 0.50).
- **Confusion matrix:** Where the model is right/wrong (TP, TN, FP, FN).
- **Takeaways:** Stock prediction is noisy; these numbers are realistic; Logistic Regression is competitive with trees.

---

## Slide 16 — Threshold Tuning (from Notebook)
- **Problem:** Default threshold 0.5 may not be best. We sweep thresholds and evaluate with backtest (Sharpe ratio).
- **Chart:** Precision, recall, coverage vs threshold. Higher threshold → fewer trades, often better precision.
- **Choice:** Best threshold by Sharpe = 0.40. Trade-off: precision vs coverage.

---

## Slide 17 — Feature Importance (XGBoost)
- **Chart:** Top 15 features (rolling mean/std/last of RSI, ATR, MACD, 10Y yield, etc.).
- One line: “These are the inputs that drive predictions most; we also did ablation in the notebook to see which groups help.”

---

## Slide 18 — ROC & Model Comparison
- **Left:** ROC curve for XGBoost on test set. **Right:** Model comparison on test set.
- One line: “ROC shows ranking ability; comparison summarizes how the three models perform.”

---

## Slide 19 — Backtest & Equity Curve
- **Chart:** Equity curve at threshold 0.40.
- **Setup:** Non-overlapping 5-day windows; long-only / cash; BUY when P(up) ≥ 0.40.
- **Metrics:** Annualized return, Sharpe ratio, max drawdown on holdout; threshold chosen for risk-adjusted performance.

---

## Slide 20 — Live Dashboard
- We have a Streamlit dashboard (and FastAPI) for serving predictions. (If you have a screenshot, show it and say what the user can do: e.g. see signals, thresholds, tickers.)

---

## Slide 21 — Conclusion
- **What we built:** (1) End-to-end ML pipeline, (2) Chronological train/test split (no leakage), (3) Three models compared, (4) Streamlit + FastAPI.
- **Lessons:** Stock prediction is hard — 53% accuracy is realistic; more features didn’t help much; temporal validation is critical; simple pipelines are valuable.

---

## Slide 22 — Future Work
- **Model:** Sentiment (e.g. FinBERT), deep learning (LSTM, Transformers), ensembles, calibration.
- **Engineering:** Docker, automated retraining, monitoring and drift, CI/CD.

---

## Slide 23 — Thank You
- Thank the audience. “Questions?” Names and institute again.
