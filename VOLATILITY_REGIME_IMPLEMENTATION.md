# High-Volatility Regime Implementation Summary

## 🎯 Objective
Convert the model from stable large-cap stocks to high-beta growth stocks to capture larger alpha through volatility arbitrage.

## 📊 What Changed

### 1. Ticker Universe
**Before:** Stable Large Caps
- AAPL, NVDA, TSLA, MSFT, AMZN
- Typical 5-day move: 1-3%
- Data since: 2006

**After:** High-Beta Growth  
- **SMCI** (Super Micro Computer - Semiconductors)
- **CRSP** (CRISPR Therapeutics - Biotech)
- **PLTR** (Palantir - AI/Data Analytics)
- Typical 5-day move: 5-15%
- Data since: 2020 (recent behavior more relevant)

### 2. Target Labeling Threshold
**Before:** `threshold = 0.015` (1.5%)
- Appropriate for stable stocks where 1.5% is significant

**After:** `threshold = 0.035` (3.5%)
- Higher bar filters noise from trend
- In SMCI, a 1.5% move is intraday chop; 3.5%+ represents real momentum

**Implementation:**
```python
# src/model.py - ModelConfig
threshold: float = 0.035  # Was 0.015

# src/features.py - add_future_return_and_label()
threshold: float = 0.035  # Default parameter updated
```

### 3. ATR Feature Engineering
**New Features Added:**
- `atr_14`: 14-day Average True Range (absolute dollars)
- `atr_pct`: ATR as percentage of price (normalized)

**Why ATR Matters:**
- Normalizes price movements across different volatility regimes
- Model learns: "$10 move with $50 ATR" ≠ "$10 move with $3 ATR"
- Critical for comparing SMCI ($800, ATR=$45) vs CRSP ($70, ATR=$5)

**Implementation:**
```python
# src/features.py - add_technical_indicators()
atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
g["atr_14"] = atr_indicator.average_true_range()
g["atr_pct"] = (g["atr_14"] / close) * 100
```

**Rolling Window Features:**
- Added `atr_14_mean`, `atr_14_std`, `atr_14_last` to feature set
- Model now has access to recent volatility context

### 4. Sentiment Decay Scaling
**Before:** Simple forward-fill (implicit 48h+ half-life)
- Appropriate for blue chips where news digests slowly

**After:** Exponential decay with 12-hour half-life
- Formula: `weight = exp(-days_elapsed / 0.5)`
- At 12 hours (0.5 days): sentiment retains 36.8% of original strength
- At 24 hours: 13.5% remaining
- At 48 hours: 1.8% remaining (effectively zero)

**Why Faster Decay?**
- AI/Biotech/Meme stocks react within hours to news
- Clinical trial results, product launches priced in immediately
- 2-day-old news is stale and misleading for high-beta names

**Implementation:**
```python
# src/sentiment.py - merge_sentiment()
def merge_sentiment(
    df_prices_macro: pd.DataFrame, 
    df_sentiment: pd.DataFrame,
    decay_days: float = 0.5  # 12 hours for high-volatility
) -> pd.DataFrame:
    # ... exponential decay logic ...
    decay_weight = np.exp(-days_elapsed / decay_days)
    merged.loc[idx, col] = last_valid_value * decay_weight
```

### 5. ATR-Based Stop Loss Model
**New Tool:** `atr_stop_loss.py`

**Formula:**
```
StopLoss = Entry_Price - (k × ATR_14)

Where:
  k = 2.5 for high-volatility (wider stops)
  k = 2.0 for stable stocks (tighter stops)
```

**Example:**
- Entry: $825.00 (SMCI)
- ATR_14: $45.00
- k: 2.5
- **Stop: $825 - (2.5 × $45) = $712.50**
- Risk per share: $112.50 (13.6%)

**Position Sizing:**
```python
risk_per_trade = 2%  # of account
account_size = $100,000
risk_amount = $2,000

shares = risk_amount / risk_per_share
       = $2,000 / $112.50
       = 17 shares

position_value = 17 × $825 = $14,025
```

## 📁 Files Modified

| File | Changes | Purpose |
|------|---------|---------|
| `data_ingestion.py` | Changed `BIG_TECH_TICKERS` → `HIGH_BETA_TICKERS`<br>Start date 2006 → 2020 | New ticker universe |
| `model.py` | `threshold: 0.015` → `0.035`<br>Added `atr_stop_loss_multiplier` | Volatility-adjusted labeling |
| `features.py` | Added ATR indicators<br>Updated default thresholds<br>Added ATR to feature list | Volatility normalization |
| `sentiment.py` | Added exponential decay<br>`decay_days=0.5` parameter | Fast sentiment decay |
| **NEW** `config_volatility_regime.py` | Regime definitions<br>Comparison tool | Easy regime switching |
| **NEW** `atr_stop_loss.py` | Stop loss calculator<br>Position sizing<br>CLI tool | Risk management |
| `README.md` | Added regime documentation<br>Usage examples | User guide |

## 🚀 How to Use

### Option 1: Quick Test (Single Command)
```bash
cd src
python pipeline_train.py
```
This will automatically use the high-volatility configuration (SMCI, CRSP, PLTR with 3.5% threshold).

### Option 2: Switch Regimes
```python
# Edit src/config_volatility_regime.py line 62:
REGIME = HIGH_VOLATILITY  # or STABLE_LARGE_CAP

# Then re-run:
python src/pipeline_train.py
```

### Option 3: View Regime Comparison
```bash
python src/config_volatility_regime.py
```
Output:
```
================================================================================
VOLATILITY REGIME COMPARISON
================================================================================

Parameter                      High-Volatility           Stable Large-Cap         
--------------------------------------------------------------------------------
Tickers                        SMCI, CRSP, PLTR          AAPL, NVDA, TSLA, MSFT...
Typical 5-day Move             5-15%                     1-3%
Data Start                     2020-01-01                2006-01-01
Success Threshold              3.5%                      1.5%
Sentiment Half-Life            12h                       48h
ATR Stop Multiplier            2.5x                      2.0x
```

### Calculate Stop Losses
```bash
# For SMCI at $825 entry
python src/atr_stop_loss.py SMCI 825.0

# Custom parameters
python src/atr_stop_loss.py SMCI 825.0 --multiplier 2.5 --account-size 100000 --risk 0.02
```

## 📈 Expected Performance Differences

### High-Volatility Regime
**Pros:**
- Larger absolute returns (5-15% moves vs 1-3%)
- More alpha generation potential
- Better signal-to-noise with 3.5% threshold

**Cons:**
- Higher volatility → larger drawdowns
- More false signals during range-bound periods
- Requires active monitoring

**Best For:**
- Aggressive traders
- Shorter time horizons
- High risk tolerance

### Stable Large-Cap Regime
**Pros:**
- Lower volatility → smoother equity curve
- More consistent signals
- Lower psychological stress

**Cons:**
- Smaller absolute returns per trade
- Less alpha potential
- 1.5% threshold may trigger on noise

**Best For:**
- Conservative traders
- Longer time horizons
- Lower risk tolerance

## 🔧 Technical Details

### Feature Count Changes
**Before:** ~48 features
- Technical indicators (RSI, MACD, BB, MA)
- Macro (10Y yield)
- Sentiment (if available)

**After:** ~54 features
- All previous features +
- `atr_14_mean`, `atr_14_std`, `atr_14_last` (×3)
- `atr_pct_mean`, `atr_pct_std`, `atr_pct_last` (×3)

### Model Training Adjustments
No changes to XGBoost hyperparameters needed. The model automatically:
1. Learns ATR importance through feature selection
2. Adjusts to higher threshold (3.5% vs 1.5%)
3. Uses decayed sentiment appropriately

### Backtest Methodology
Still uses non-overlapping periods:
- Signals every 5 days (matching horizon)
- Long/short positioning based on BUY/SELL signals
- Equal-weight portfolio across tickers

## 📚 Mathematical Foundations

### ATR Stop Loss Derivation
```
Goal: Risk 2% of account per trade

Given:
  A = account size ($100,000)
  r = risk per trade (2% = 0.02)
  E = entry price ($825)
  k = stop multiplier (2.5)
  ATR = average true range ($45)

Stop Loss Calculation:
  S = E - (k × ATR)
  S = 825 - (2.5 × 45)
  S = $712.50

Risk Per Share:
  R = E - S
  R = 825 - 712.50
  R = $112.50

Position Size:
  N = (A × r) / R
  N = (100,000 × 0.02) / 112.50
  N = 2,000 / 112.50
  N ≈ 17 shares
```

### Sentiment Decay Function
```
w(t) = exp(-t / τ)

Where:
  w(t) = weight at time t
  t = time since news (in days)
  τ = decay constant (0.5 days = 12 hours)

Examples:
  w(0.5 days)  = exp(-0.5/0.5) = e^(-1) = 0.368 → 36.8% weight
  w(1.0 days)  = exp(-1.0/0.5) = e^(-2) = 0.135 → 13.5% weight
  w(2.0 days)  = exp(-2.0/0.5) = e^(-4) = 0.018 → 1.8% weight
```

## 🎓 Next Steps

1. **Run Initial Training:**
   ```bash
   cd src
   python pipeline_train.py
   ```

2. **Monitor Performance:**
   - Check test precision (should be >50%)
   - Verify Sharpe ratio (positive = good)
   - Review equity curve for smoothness

3. **Iterate on Thresholds:**
   - If too many false positives: increase `buy_threshold` (0.48 → 0.52)
   - If too conservative: decrease `buy_threshold` (0.48 → 0.45)
   - Adjust `sell_threshold` similarly (currently 0.30)

4. **Test Live:**
   - Use `src/atr_stop_loss.py` for position management
   - Monitor dashboard for real-time signals
   - Paper trade first before real capital

## ⚠️ Risk Warnings

1. **High-beta stocks are volatile** - expect large swings
2. **Backtests ≠ future performance** - always paper trade first
3. **News dependency** - sentiment features require headline data
4. **Stop losses can gap** - ATR stops are guidelines, not guarantees
5. **Correlation risk** - SMCI/PLTR may move together (sector exposure)

## 📞 Support

If issues arise:
1. Check `data/` folder for required CSVs
2. Verify `requirements.txt` packages installed
3. Review error messages in terminal output
4. Check README.md for detailed instructions

---
**Implementation Date:** February 2026  
**Author:** AI Stock Intelligence System  
**Version:** 2.0 (High-Volatility Regime)
