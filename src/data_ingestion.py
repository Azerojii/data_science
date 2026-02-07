import os
from datetime import datetime, timedelta
from typing import List

from dotenv import load_dotenv
import pandas as pd
import requests
import yfinance as yf

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# High-Beta Growth Stocks (High-Volatility Regime)
# These tickers exhibit 5-15% 5-day swings vs 1.5% for stable large caps
HIGH_BETA_TICKERS = ["SMCI", "CRSP", "PLTR"]
BIG_TECH_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]  # Legacy stable tickers


def download_price_data(
    tickers: List[str] = HIGH_BETA_TICKERS,  # Default to high-volatility regime
    start: str = "2020-01-01",  # Focus on recent data for high-beta behavior
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data for given tickers using yfinance and save to CSV.

    Returns a tidy DataFrame with columns:
        ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    df_list: List[pd.DataFrame] = []
    for ticker in tickers:
        hist = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if hist.empty:
            continue
        hist = hist.reset_index()
       
        # Flatten MultiIndex columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [col[0] if isinstance(col, tuple) else col for col in hist.columns]
        
        # Standardize column names
        hist.columns = [str(col).lower().replace(" ", "_") for col in hist.columns]
        
        # Rename to our standard names
        col_map = {}
        for col in hist.columns:
            if "date" in col:
                col_map[col] = "date"
            elif "open" in col:
                col_map[col] = "open"
            elif "high" in col:
                col_map[col] = "high"
            elif "low" in col:
                col_map[col] = "low"
            elif "adj" in col and "close" in col:
                col_map[col] = "adj_close"
            elif col == "close":
                col_map[col] = "close"
            elif "volume" in col:
                col_map[col] = "volume"
        
        hist = hist.rename(columns=col_map)
        
        # Select only columns we need
        required_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        hist = hist[[col for col in required_cols if col in hist.columns]].copy()
        hist["ticker"] = ticker
        df_list.append(hist)

    if not df_list:
        raise RuntimeError("No price data downloaded. Check tickers or internet connection.")

    prices = pd.concat(df_list, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, "prices_daily.csv")
    prices.to_csv(out_path, index=False)
    return prices


def load_price_data(path: str | None = None) -> pd.DataFrame:
    """Load previously downloaded price data."""
    if path is None:
        path = os.path.join(DATA_DIR, "prices_daily.csv")
    return pd.read_csv(path, parse_dates=["date"])


def load_macro_data() -> pd.DataFrame:
    """
    Placeholder for macro factors (e.g., 10Y Treasury Yield).

    For now, this expects a CSV `macro_10y_yield.csv` with:
        date, ten_year_yield
    saved in the data directory. You can generate this from FRED or another source.
    """
    path = os.path.join(DATA_DIR, "macro_10y_yield.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected macro data at {path}. Please create it with columns: date,ten_year_yield."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_10y_yield_from_fred(
    api_key: str,
    start: str = "2006-01-01",
    end: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch 10-Year Treasury Yield (DGS10) from the FRED API.

    You can get a free API key from the St. Louis Fed:
      https://fred.stlouisfed.org/

    The result is saved to `data/macro_10y_yield.csv` with:
        date, ten_year_yield
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DGS10",
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    obs = data.get("observations", [])
    records = []
    for o in obs:
        value = o.get("value")
        if value in (".", None, ""):
            continue
        records.append(
            {
                "date": pd.to_datetime(o["date"]),
                "ten_year_yield": float(value),
            }
        )

    if records:
        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["date", "ten_year_yield"])

    if save:
        out_path = os.path.join(DATA_DIR, "macro_10y_yield.csv")
        df.to_csv(out_path, index=False)

    return df


def update_macro_from_fred() -> pd.DataFrame:
    """
    Convenience wrapper: reads FRED_API_KEY from env and appends latest DGS10 data
    to `macro_10y_yield.csv`. Use this in a scheduled job for continuous updates.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY environment variable is not set.")

    path = os.path.join(DATA_DIR, "macro_10y_yield.csv")
    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        last_date = existing["date"].max().date()
        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        existing = None
        start = "2006-01-01"

    end = datetime.today().strftime("%Y-%m-%d")
    if start > end:
        return load_macro_data()

    latest = fetch_10y_yield_from_fred(api_key, start=start, end=end, save=False)
    if existing is not None and not latest.empty:
        combined = (
            pd.concat([existing, latest], ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
    elif existing is not None:
        combined = existing
    else:
        combined = latest

    combined.to_csv(path, index=False)
    return combined


def merge_price_and_macro(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily prices with macro factors using forward fill for non-trading days.

    This implements the 'Forward Filling' logic from the blueprint.
    """
    prices = prices.copy()
    macro = macro.copy()

    # Flatten any multi-level indexes to regular columns to avoid merge level errors
    if not isinstance(prices.index, pd.RangeIndex):
        prices = prices.reset_index(drop=True)
    if not isinstance(macro.index, pd.RangeIndex):
        macro = macro.reset_index(drop=True)

    # Ensure we have a `date` column on both sides (handle variations like 'Date')
    if "date" not in prices.columns:
        if "Date" in prices.columns:
            prices = prices.rename(columns={"Date": "date"})
        else:
            raise KeyError("Prices DataFrame has no 'date' column or index.")

    if "date" not in macro.columns:
        if "Date" in macro.columns:
            macro = macro.rename(columns={"Date": "date"})
        else:
            raise KeyError("Macro DataFrame has no 'date' column or index.")

    # Ensure both have date as datetime
    prices["date"] = pd.to_datetime(prices["date"])
    macro["date"] = pd.to_datetime(macro["date"])

    # Expand macro to full calendar and forward-fill
    full_range = pd.date_range(macro["date"].min(), prices["date"].max(), freq="D")
    macro_full = (
        macro.set_index("date")
        .reindex(full_range)
        .ffill()
        .rename_axis("date")
        .reset_index()
    )

    # Debug: Check for multi-level columns
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.to_flat_index()
        prices.columns = [col[0] if isinstance(col, tuple) else col for col in prices.columns]
    if isinstance(macro_full.columns, pd.MultiIndex):
        macro_full.columns = macro_full.columns.to_flat_index()
        macro_full.columns = [col[0] if isinstance(col, tuple) else col for col in macro_full.columns]

    merged = prices.merge(macro_full, on="date", how="left")
    return merged


def fetch_news_headlines_finnhub(
    api_key: str,
    tickers: List[str] = BIG_TECH_TICKERS,
    start: str = "2006-01-01",
    end: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch company news headlines for each ticker using the Finnhub API.

    You can get a free API key at:
      https://finnhub.io/

    Requires env var FINNHUB_API_KEY or pass api_key explicitly.

    Saves `data/news_headlines.csv` with:
        date, ticker, headline

    Note: For production, you may want to handle pagination, rate limits, and richer fields.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    headers = {}

    all_records: list[dict] = []

    for ticker in tickers:
        params = {"symbol": ticker, "from": start, "to": end, "token": api_key}
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        news_items = resp.json()
        for item in news_items:
            # Finnhub returns time as UNIX timestamp seconds
            dt = datetime.utcfromtimestamp(item.get("datetime", 0))
            headline = item.get("headline") or ""
            if not headline:
                continue
            all_records.append(
                {
                    "date": dt.date(),
                    "ticker": ticker,
                    "headline": headline,
                }
            )

    df = pd.DataFrame(all_records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        if save:
            out_path = os.path.join(DATA_DIR, "news_headlines.csv")
            if os.path.exists(out_path):
                existing = pd.read_csv(out_path, parse_dates=["date"])
                df = (
                    pd.concat([existing, df], ignore_index=True)
                    .drop_duplicates(subset=["ticker", "date", "headline"])
                    .sort_values(["ticker", "date"])
                )
            df.to_csv(out_path, index=False)

    return df


if __name__ == "__main__":
    # Simple manual tests (optional)
    df_prices = download_price_data()
    print("Downloaded prices:", df_prices.head())

    # Example: update macro from FRED if FRED_API_KEY is set
    if os.getenv("FRED_API_KEY"):
        df_macro = update_macro_from_fred()
        print("Macro rows:", len(df_macro))

    # Example: fetch headlines from Finnhub if FINNHUB_API_KEY is set
    if os.getenv("FINNHUB_API_KEY"):
        df_news = fetch_news_headlines_finnhub(os.getenv("FINNHUB_API_KEY"))
        print("News rows:", len(df_news))


