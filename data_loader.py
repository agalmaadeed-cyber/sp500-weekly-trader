"""
data_loader.py
Download S&P 500 daily data via yfinance
"""

import yfinance as yf
import pandas as pd
import requests
import io
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data_cache"
START_FULL   = "2010-01-01"   # for backtest
START_RECENT = "2010-01-01"   # for scanner (faster download)


def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 ticker list from Wikipedia"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=10
        )
        tables = pd.read_html(io.BytesIO(r.content))
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        return tickers
    except Exception:
        # Fallback list
        return [
            "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","JNJ","V",
            "PG","UNH","HD","MA","DIS","ADBE","NFLX","XOM","CVX","PFE",
            "KO","PEP","ABBV","MRK","TMO","ABT","CRM","ACN","AVGO","COST",
            "WMT","BAC","MCD","NEE","LIN","ORCL","TXN","QCOM","HON","PM",
            "UPS","LOW","AMGN","IBM","GS","CAT","BA","MMM","RTX","SPGI",
        ]


def download_ticker(ticker: str, start: str = START_FULL) -> pd.DataFrame | None:
    """Download a single ticker"""
    try:
        raw = yf.download(ticker, start=start, interval="1wk", auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 30:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]
        df.index.name = "date"
        return df
    except Exception:
        return None


def download_batch(tickers: list[str], start: str = START_FULL,
                   use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Download a batch of tickers with optional cache"""
    CACHE_DIR.mkdir(exist_ok=True)
    data = {}

    for ticker in tickers:
        cache_file = CACHE_DIR / f"{ticker}.parquet"

        if use_cache and cache_file.exists():
            try:
                data[ticker] = pd.read_parquet(cache_file)
                continue
            except Exception:
                pass

        df = download_ticker(ticker, start)
        if df is not None:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
            data[ticker] = df

    return data


def get_latest_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Fetch recent data for the scanner.
    Downloads START_RECENT to today -- fast and sufficient for signal detection.
    No cache dependency so it always works on Streamlit Cloud.
    """
    data = {}

    for ticker in tickers:
        df = download_ticker(ticker, start=START_RECENT)
        if df is not None:
            data[ticker] = df

    return data
