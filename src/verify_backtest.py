"""
Verify equity curve calculation manually
"""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_signals.csv"), parse_dates=["date"])

# Get first few unique dates
dates = sorted(test_df['date'].unique())[:5]

print("Manual Equity Curve Verification")
print("=" * 60)
print("\nStarting equity: $1.00\n")

equity = 1.0
for i, date in enumerate(dates):
    day_data = test_df[test_df['date'] == date]
    
    print(f"\n{'='*60}")
    print(f"Day {i+1}: {date.date()}")
    print(f"{'='*60}")
    
    # Show each ticker's contribution
    total_return = 0
    n_signals = 0
    
    for _, row in day_data.iterrows():
        ticker = row['ticker']
        signal = int(row['signal'])
        future_ret = float(row['future_return'])
        contrib = signal * future_ret
        total_return += contrib
        n_signals += 1
        
        print(f"  {ticker}: signal={signal}, future_return={future_ret:+.4f}, " + 
              f"contribution={contrib:+.6f}")
    
    # Average return across tickers
    avg_return = total_return / n_signals
    print(f"\n  Average daily return: {avg_return:.6f} ({avg_return*100:.4f}%)")
    print(f"  Number of positions: {n_signals}")
    
    # Update equity
    equity *= (1 + avg_return)
    print(f"  New equity: ${equity:.4f}")
    
    # Compare with actual
    actual_equity = day_data.iloc[0]['equity_curve']
    print(f"  Actual in file: ${actual_equity:.4f}")
    print(f"  Match: {'✓' if abs(equity - actual_equity) < 0.01 else '✗ MISMATCH!'}")

print(f"\n\n{'='*60}")
print(f"After 5 days: ${equity:.2f}")
print(f"Return: {(equity - 1) * 100:.2f}%")
print(f"{'='*60}\n")
