"""Download and prepare daily price data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLU", "XLV"]
START = "2010-01-01"


def load_prices(
    tickers: list[str] = TICKERS,
    start: str = START,
    end: str | None = None,
    cache: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return adjusted daily closes, one column per ticker.

    Prices are adjusted for splits and dividends, so returns reflect total
    return. If a cache path is given, the download is saved there and reused
    on later runs unless refresh=True.
    """
    if cache is not None and cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        return prices[tickers]

    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    prices = raw["Close"][tickers].dropna(how="any")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache, float_format="%.6f")
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns in percent (the scale the arch package expects)."""
    return (100 * np.log(prices / prices.shift(1))).dropna(how="any")
