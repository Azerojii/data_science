from __future__ import annotations

import numpy as np
import pandas as pd


def compute_equity_curve(
    df_signals: pd.DataFrame,
    prob_col: str = "proba",
    signal_col: str = "signal",
    return_col: str = "future_return",
    threshold: float = 0.5,
) -> pd.Series:
    """
    Long/short strategy for multi-ticker portfolio:
      - signal=1 (BUY): go long for the next period
      - signal=-1 (SELL): go short for the next period
      - signal=0 (HOLD): stay in cash
    
    For multiple tickers, we aggregate returns by date (equal-weight portfolio).
    Returns a pandas Series with cumulative equity curve starting at 1.0.
    """
    df = df_signals.copy()
    
    # Calculate per-ticker strategy returns (signal already contains -1, 0, or 1)
    df['strategy_return'] = df[signal_col] * df[return_col]
    
    # If we have multiple tickers, aggregate by date (equal-weight portfolio)
    if 'ticker' in df.columns and df['ticker'].nunique() > 1:
        # Group by date and take mean of all positions (equal-weight)
        daily_returns = df.groupby('date')['strategy_return'].mean()
        equity = (1 + daily_returns).cumprod()
        
        # Map back to original dataframe
        equity_dict = equity.to_dict()
        result = df['date'].map(equity_dict)
        result.name = "equity_curve"
        return result
    else:
        # Single ticker case
        equity = (1 + df['strategy_return']).cumprod()
        equity.name = "equity_curve"
        return equity


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized return given a daily equity curve.
    """
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    years = len(equity_curve) / periods_per_year
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown of an equity curve.
    """
    cum_max = equity_curve.cummax()
    drawdowns = equity_curve / cum_max - 1.0
    return drawdowns.min()


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio of a return series.
    """
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    mean_excess = excess.mean()
    std_excess = excess.std()
    daily_sharpe = mean_excess / std_excess
    return daily_sharpe * np.sqrt(periods_per_year)


if __name__ == "__main__":
    print("Backtesting utilities (equity curve, Sharpe, drawdown).")


