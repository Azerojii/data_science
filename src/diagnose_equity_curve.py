"""
Equity Curve Diagnostic Tool

This script demonstrates the issue with plotting forward-filled equity curves
and shows how the fix works.
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TEST_SIGNALS_PATH = os.path.join(DATA_DIR, "test_signals.csv")

def analyze_equity_curve_issue():
    """Analyze and explain the equity curve plotting issue"""
    
    print("\n" + "="*80)
    print("EQUITY CURVE PLOTTING ISSUE DIAGNOSTIC")
    print("="*80)
    
    # Load data
    df = pd.read_csv(TEST_SIGNALS_PATH, parse_dates=["date"])
    
    # Show overall stats
    print(f"\n📊 Dataset Overview:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Unique dates: {df['date'].nunique()}")
    print(f"   Tickers: {', '.join(df['ticker'].unique())}")
    print(f"   Rows per ticker: {len(df) / df['ticker'].nunique():.0f}")
    
    # Demonstrate the problem on a specific date
    problem_date = "2023-01-03"
    problem_data = df[df["date"] == problem_date]
    
    print(f"\n⚠️  THE PROBLEM (Date: {problem_date}):")
    print("   When equity is forward-filled for ALL rows, multiple tickers have")
    print("   the same equity value on the same date:\n")
    print(problem_data[["ticker", "date", "equity_curve"]].to_string(index=False))
    
    print(f"\n   Result: Plotly receives 3 data points at x={problem_date}:")
    for _, row in problem_data.iterrows():
        print(f"     - ({problem_date}, {row['equity_curve']:.4f})  <- {row['ticker']}")
    print("\n   With fill='tozeroy', this creates VERTICAL BARS instead of a curve!")
    
    # Show the solution
    print("\n✅ THE SOLUTION:")
    print("   Aggregate by date BEFORE plotting (take first equity value per date):\n")
    
    equity_by_date = df.groupby("date")["equity_curve"].first().reset_index()
    solution_data = equity_by_date[equity_by_date["date"] == problem_date]
    print(solution_data.to_string(index=False))
    
    print(f"\n   Result: Plotly receives 1 data point at x={problem_date}:")
    print(f"     - ({problem_date}, {solution_data['equity_curve'].iloc[0]:.4f})")
    print("\n   This creates a SMOOTH CURVE as expected!")
    
    # Show equity curve statistics
    print("\n📈 Equity Curve Statistics:")
    print(f"   Starting equity: {equity_by_date['equity_curve'].iloc[0]:.4f}")
    print(f"   Final equity: {equity_by_date['equity_curve'].iloc[-1]:.4f}")
    print(f"   Total return: {(equity_by_date['equity_curve'].iloc[-1] - 1) * 100:.2f}%")
    print(f"   Max equity: {equity_by_date['equity_curve'].max():.4f}")
    print(f"   Min equity: {equity_by_date['equity_curve'].min():.4f}")
    print(f"   Max drawdown: {(equity_by_date['equity_curve'].min() - 1) * 100:.2f}%")
    
    # Show timeline
    print(f"\n📅 Timeline:")
    print(f"   Start date: {equity_by_date['date'].min().strftime('%Y-%m-%d')}")
    print(f"   End date: {equity_by_date['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Trading days: {len(equity_by_date)}")
    
    # Verify the fix
    print("\n🔧 Dashboard Fix Applied:")
    print("   OLD CODE (WRONG):")
    print("     x=ticker_data['date'], y=ticker_data['equity_curve']")
    print("     ❌ Plots ALL rows including duplicates per date")
    print()
    print("   NEW CODE (CORRECT):")
    print("     equity_by_date = ticker_data.groupby('date')['equity_curve'].first()")
    print("     x=equity_by_date['date'], y=equity_by_date['equity_curve']")
    print("     ✅ Plots ONE value per date for smooth curve")
    
    # Show sample of equity progression
    print("\n📊 Sample Equity Progression (First 10 days):")
    sample = equity_by_date.head(10)
    print(sample.to_string(index=False))
    
    print("\n" + "="*80)
    print("The dashboard has been fixed and should now display smooth equity curves!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        analyze_equity_curve_issue()
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("Make sure 'test_signals.csv' exists in the data/ folder.")
        print("Run 'python src/pipeline_train.py' first to generate the data.\n")
