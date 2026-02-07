import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from data_ingestion import download_price_data, BIG_TECH_TICKERS
import pandas as pd

print("Downloading price data...")
prices = download_price_data(tickers=BIG_TECH_TICKERS)

print(f"\nDataFrame shape: {prices.shape}")
print(f"\nColumns: {list(prices.columns)}")
print(f"\nUnique tickers: {sorted(prices['ticker'].unique())}")
print(f"\nRows per ticker:")
print(prices.groupby('ticker').size())
print(f"\nFirst few rows:")
print(prices.head(10))
print(f"\nLast few rows:")
print(prices.tail(10))

# Check the saved file
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
csv_path = os.path.join(data_dir, "prices_daily.csv")
print(f"\n\nChecking saved CSV file: {csv_path}")
with open(csv_path, 'r') as f:
    lines = f.readlines()
    print(f"First 3 lines of CSV:")
    for i, line in enumerate(lines[:3]):
        print(f"Line {i}: {line[:150]}...")
