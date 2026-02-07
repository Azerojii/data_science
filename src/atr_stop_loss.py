"""
ATR-Based Stop Loss Calculator

Implements volatility-scaled stop losses for risk management in high-volatility regimes.

Formula:
    StopLoss = Entry_Price - (k × ATR_14)
    
Where:
    - k = stop loss multiplier (2.0-2.5 typically)
    - ATR_14 = 14-day Average True Range

Theory:
    High-volatility stocks need wider stops to avoid being "stopped out" by normal
    price action that doesn't invalidate the trade thesis. Using a fixed percentage
    stop (e.g., -5%) works poorly because volatility varies across tickers and time.
    
    ATR automatically adjusts:
    - Wider stops during volatile periods (prevents premature exits)
    - Tighter stops during calm periods (reduces risk per trade)

Usage:
    from atr_stop_loss import calculate_stop_loss, get_current_atr
    
    # Get current ATR for a ticker
    atr = get_current_atr("SMCI")
    
    # Calculate stop loss for a new position
    entry_price = 825.0
    stop_price = calculate_stop_loss(entry_price, atr, multiplier=2.5)
    
    # Calculate position size based on risk
    risk_per_trade = 0.02  # Risk 2% of portfolio
    position_size = calculate_position_size(
        account_size=100000,
        entry_price=825.0,
        stop_price=stop_price,
        risk_per_trade=0.02
    )
"""

import os
from datetime import datetime, timedelta

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    multiplier: float = 2.5
) -> float:
    """
    Calculate stop loss price using ATR-based volatility scaling.
    
    Args:
        entry_price: Entry price for the position
        atr: Current 14-day Average True Range
        multiplier: ATR multiplier (2.0 for stable, 2.5 for high-volatility)
    
    Returns:
        Stop loss price (below entry for long, above for short)
    
    Example:
        >>> calculate_stop_loss(entry_price=825.0, atr=45.0, multiplier=2.5)
        712.5  # Stop is 2.5 × $45 = $112.50 below entry
    """
    stop_distance = multiplier * atr
    stop_loss_price = entry_price - stop_distance
    return round(stop_loss_price, 2)


def calculate_position_size(
    account_size: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float = 0.02
) -> int:
    """
    Calculate position size based on fixed risk per trade.
    
    Args:
        account_size: Total account equity
        entry_price: Entry price per share
        stop_price: Stop loss price per share
        risk_per_trade: Fraction of account to risk (default 2%)
    
    Returns:
        Number of shares to buy/sell
    
    Example:
        >>> calculate_position_size(
        ...     account_size=100000,
        ...     entry_price=825.0,
        ...     stop_price=712.5,
        ...     risk_per_trade=0.02
        ... )
        17  # Risk $2000 / $112.50 per share ≈ 17 shares
    """
    risk_amount = account_size * risk_per_trade
    risk_per_share = abs(entry_price - stop_price)
    
    if risk_per_share == 0:
        return 0
    
    shares = int(risk_amount / risk_per_share)
    return shares


def get_current_atr(ticker: str, dataset_path: str = None) -> float:
    """
    Get the most recent ATR value for a ticker from the dataset.
    
    Args:
        ticker: Stock ticker symbol
        dataset_path: Path to rolling_window_dataset.csv (optional)
    
    Returns:
        Current ATR_14 value
    
    Raises:
        ValueError: If ticker not found or no ATR data available
    """
    if dataset_path is None:
        dataset_path = os.path.join(DATA_DIR, "rolling_window_dataset.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run 'python src/pipeline_train.py' first to generate features."
        )
    
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    ticker_data = df[df["ticker"] == ticker]
    
    if ticker_data.empty:
        raise ValueError(f"Ticker '{ticker}' not found in dataset")
    
    # Get most recent row
    latest = ticker_data.sort_values("date", ascending=False).iloc[0]
    
    # Look for ATR in rolling window features (e.g., "atr_14_last")
    atr_cols = [c for c in latest.index if "atr_14" in c.lower()]
    if not atr_cols:
        raise ValueError(f"No ATR features found for {ticker}")
    
    # Prefer "_last" value (most recent in window)
    atr_col = next((c for c in atr_cols if "last" in c), atr_cols[0])
    atr_value = latest[atr_col]
    
    if pd.isna(atr_value) or atr_value == 0:
        raise ValueError(f"Invalid ATR value for {ticker}: {atr_value}")
    
    return float(atr_value)


def generate_stop_loss_table(
    tickers: list,
    entry_prices: dict,
    multiplier: float = 2.5,
    risk_per_trade: float = 0.02,
    account_size: float = 100000
) -> pd.DataFrame:
    """
    Generate a stop loss table for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        entry_prices: Dict mapping ticker -> entry price
        multiplier: ATR stop loss multiplier
        risk_per_trade: Risk fraction per trade
        account_size: Total account size
    
    Returns:
        DataFrame with stop losses and position sizes
    
    Example:
        >>> tickers = ["SMCI", "CRSP", "PLTR"]
        >>> entries = {"SMCI": 825.0, "CRSP": 68.5, "PLTR": 45.2}
        >>> table = generate_stop_loss_table(tickers, entries)
        >>> print(table)
    """
    results = []
    
    for ticker in tickers:
        try:
            atr = get_current_atr(ticker)
            entry = entry_prices.get(ticker)
            
            if entry is None:
                print(f"Warning: No entry price provided for {ticker}, skipping...")
                continue
            
            stop = calculate_stop_loss(entry, atr, multiplier)
            risk_per_share = entry - stop
            shares = calculate_position_size(account_size, entry, stop, risk_per_trade)
            position_value = shares * entry
            
            results.append({
                "Ticker": ticker,
                "Entry Price": f"${entry:.2f}",
                "ATR_14": f"${atr:.2f}",
                "Stop Loss": f"${stop:.2f}",
                "Risk/Share": f"${risk_per_share:.2f}",
                "Shares": shares,
                "Position Value": f"${position_value:,.0f}",
                "Risk Amount": f"${shares * risk_per_share:.0f}",
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    return pd.DataFrame(results)


# ============================================================================
# CLI TOOL
# ============================================================================

def main():
    """Command-line interface for stop loss calculator"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate ATR-based stop losses for high-volatility stocks"
    )
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("entry_price", type=float, help="Entry price")
    parser.add_argument(
        "--multiplier", "-k", 
        type=float, 
        default=2.5,
        help="ATR multiplier (default: 2.5)"
    )
    parser.add_argument(
        "--account-size", "-a",
        type=float,
        default=100000,
        help="Account size in dollars (default: 100000)"
    )
    parser.add_argument(
        "--risk", "-r",
        type=float,
        default=0.02,
        help="Risk per trade as decimal (default: 0.02 for 2%%)"
    )
    
    args = parser.parse_args()
    
    try:
        # Get ATR
        atr = get_current_atr(args.ticker)
        
        # Calculate stop loss
        stop = calculate_stop_loss(args.entry_price, atr, args.multiplier)
        risk_per_share = args.entry_price - stop
        risk_pct = (risk_per_share / args.entry_price) * 100
        
        # Calculate position size
        shares = calculate_position_size(
            args.account_size,
            args.entry_price,
            stop,
            args.risk
        )
        position_value = shares * args.entry_price
        total_risk = shares * risk_per_share
        
        # Display results
        print("\n" + "="*60)
        print(f"ATR-BASED STOP LOSS: {args.ticker}")
        print("="*60)
        print(f"\nEntry Price:        ${args.entry_price:.2f}")
        print(f"ATR (14-day):       ${atr:.2f}")
        print(f"Multiplier:         {args.multiplier}x")
        print(f"\nStop Loss:          ${stop:.2f}")
        print(f"Risk per Share:     ${risk_per_share:.2f} ({risk_pct:.1f}%)")
        print("\n" + "-"*60)
        print("POSITION SIZING")
        print("-"*60)
        print(f"Account Size:       ${args.account_size:,.0f}")
        print(f"Risk per Trade:     {args.risk:.1%} (${args.account_size * args.risk:,.0f})")
        print(f"\nRecommended Shares: {shares}")
        print(f"Position Value:     ${position_value:,.0f}")
        print(f"Total Risk:         ${total_risk:,.0f}")
        print(f"Risk % of Account:  {(total_risk/args.account_size)*100:.2f}%")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}\n")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
