"""
High-Volatility Regime Configuration

This module defines configuration parameters for trading high-beta growth stocks
vs stable large-cap stocks. Based on the thesis that different volatility regimes
require different modeling approaches.

Usage:
    from config_volatility_regime import REGIME, get_regime_config
    config = get_regime_config()
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RegimeConfig:
    """Configuration for a specific volatility regime"""
    
    # Ticker universe
    tickers: List[str]
    
    # Data parameters
    start_date: str
    
    # Target labeling
    threshold: float  # Minimum return to label as "success"
    
    # Sentiment decay
    sentiment_decay_days: float  # Half-life for news impact
    
    # Risk management
    atr_stop_loss_multiplier: float
    
    # Model parameters
    window_days: int = 14
    horizon_days: int = 5
    
    # Description
    name: str = ""
    description: str = ""


# ============================================================================
# REGIME DEFINITIONS
# ============================================================================

HIGH_VOLATILITY = RegimeConfig(
    name="High-Volatility Regime",
    description="High-beta growth stocks with 5-15% weekly swings",
    tickers=["SMCI", "CRSP", "PLTR"],
    start_date="2020-01-01",  # Recent data more relevant for these stocks
    threshold=0.035,  # 3.5% - filters noise from trend
    sentiment_decay_days=0.5,  # 12 hours - news priced in instantly
    atr_stop_loss_multiplier=2.5,  # Wider stops for volatility
)

STABLE_LARGE_CAP = RegimeConfig(
    name="Stable Large-Cap Regime",
    description="FAANG/Big Tech with 1-3% weekly movements",
    tickers=["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"],
    start_date="2006-01-01",  # Can use longer history
    threshold=0.015,  # 1.5% - meaningful for stable stocks
    sentiment_decay_days=2.0,  # 48 hours - slower news absorption
    atr_stop_loss_multiplier=2.0,  # Tighter stops for lower volatility
)

# Current active regime
REGIME = HIGH_VOLATILITY  # Change to STABLE_LARGE_CAP to switch


def get_regime_config() -> RegimeConfig:
    """Get the currently active regime configuration"""
    return REGIME


def print_regime_comparison():
    """Print comparison table of both regimes"""
    print("\n" + "="*80)
    print("VOLATILITY REGIME COMPARISON")
    print("="*80)
    print(f"\n{'Parameter':<30} {'High-Volatility':<25} {'Stable Large-Cap':<25}")
    print("-"*80)
    
    params = [
        ("Tickers", ", ".join(HIGH_VOLATILITY.tickers), ", ".join(STABLE_LARGE_CAP.tickers)),
        ("Typical 5-day Move", "5-15%", "1-3%"),
        ("Data Start", HIGH_VOLATILITY.start_date, STABLE_LARGE_CAP.start_date),
        ("Success Threshold", f"{HIGH_VOLATILITY.threshold:.1%}", f"{STABLE_LARGE_CAP.threshold:.1%}"),
        ("Sentiment Half-Life", f"{HIGH_VOLATILITY.sentiment_decay_days*24:.0f}h", f"{STABLE_LARGE_CAP.sentiment_decay_days*24:.0f}h"),
        ("ATR Stop Multiplier", f"{HIGH_VOLATILITY.atr_stop_loss_multiplier:.1f}x", f"{STABLE_LARGE_CAP.atr_stop_loss_multiplier:.1f}x"),
    ]
    
    for name, high_vol, stable in params:
        print(f"{name:<30} {high_vol:<25} {stable:<25}")
    
    print("\n" + "="*80)
    print(f"CURRENT REGIME: {REGIME.name}")
    print("="*80 + "\n")


# ============================================================================
# RATIONALE & THEORY
# ============================================================================

RATIONALE = """
WHY DIFFERENT REGIMES REQUIRE DIFFERENT PARAMETERS:

1. TARGET THRESHOLD (1.5% vs 3.5%)
   - High-Volatility: A 1.5% move in SMCI or PLTR is "noise" within daily chop.
     Only movements >3.5% represent genuine directional momentum worth trading.
   - Stable: A 1.5% move in AAPL or MSFT is significant, often driven by 
     fundamental catalysts rather than intraday volatility.

2. ATR-BASED FEATURE ENGINEERING
   - The model learns to "normalize" price movements using ATR.
   - Example: A $10 move means nothing for a $500 stock with ATR=$50,
     but is massive for a $100 stock with ATR=$3.
   - ATR% (ATR/Price) provides scale-invariant volatility measurement.

3. SENTIMENT DECAY (12h vs 48h)
   - High-Volatility: AI/Biotech/Meme stocks react within hours to news.
     48-hour-old sentiment is stale and misleading.
   - Stable: Blue-chip stocks digest news over days. Earnings calls,
     product launches have multi-day impact windows.

4. ATR STOP-LOSS MULTIPLIER (2.5x vs 2.0x)
   - High-Volatility: Need wider stops (2.5 × ATR) to avoid being shaken out
     by normal intraday swings that don't invalidate the thesis.
   - Stable: Tighter stops (2.0 × ATR) work fine since volatility is lower,
     and false stops are less common.

IMPLEMENTATION FORMULA:
   StopLoss = Entry_Price - (k × ATR_14)
   
   Where k = 2.5 for high-vol, 2.0 for stable.
   
   This is dynamically adjusted per-ticker per-day based on current volatility.

HIGH-BETA SELECTION CRITERIA:
   - Historical Volatility (StdDev) > 40% annualized
   - Beta > 1.5 relative to SPY
   - Average Daily Range > 5% of price
   
   Examples: SMCI (semiconductors), CRSP (biotech), PLTR (AI/data analytics)
"""


if __name__ == "__main__":
    print_regime_comparison()
    print(RATIONALE)
