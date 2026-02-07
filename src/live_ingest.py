"""
Utility script for continuously updating macro and news data from APIs.

You can schedule this script (e.g., Windows Task Scheduler, cron) to run daily.
It uses:
  - FRED API for 10Y Treasury Yield (DGS10)
  - Finnhub API for company news headlines

Environment variables required:
  - FRED_API_KEY
  - FINNHUB_API_KEY
"""

from __future__ import annotations
import os
import sys
from dotenv import load_dotenv

load_dotenv()

import os
from datetime import datetime

from data_ingestion import BIG_TECH_TICKERS, fetch_news_headlines_finnhub, update_macro_from_fred


def run_daily_updates() -> None:
    if not os.getenv("FRED_API_KEY"):
        print("FRED_API_KEY not set – skipping macro update.")
    else:
        macro = update_macro_from_fred()
        print(f"Updated macro data to {macro['date'].max().date()}")

    if not os.getenv("FINNHUB_API_KEY"):
        print("FINNHUB_API_KEY not set – skipping news update.")
    else:
        today = datetime.today().strftime("%Y-%m-%d")
        # For daily updates you might set start=today; here we use a short lookback to be safe.
        start = today
        fetch_news_headlines_finnhub(
            api_key=os.environ["FINNHUB_API_KEY"],
            tickers=BIG_TECH_TICKERS,
            start=start,
            end=today,
            save=True,
        )
        print("Fetched latest news headlines.")


if __name__ == "__main__":
    run_daily_updates()


