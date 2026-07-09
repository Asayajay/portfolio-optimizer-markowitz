"""Historical price data and the return/covariance estimates the optimizer needs."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_price_data(
    tickers: Sequence[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily adjusted close prices for the given tickers.

    Returns a DataFrame indexed by date, one column per ticker. Tickers with
    no data in the window (typos, delistings) are dropped rather than
    silently zero-filled, since a NaN column would poison the covariance
    matrix.
    """
    raw = yf.download(
        list(tickers), start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="any")

    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"No price data returned for: {sorted(missing)}")

    return prices[list(tickers)]


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (not log) daily returns, one row shorter than the price history."""
    return prices.pct_change().dropna(how="any")


def annualize_mean_returns(daily_returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    return daily_returns.mean() * trading_days


def annualize_covariance(daily_returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    return daily_returns.cov() * trading_days


def estimate_statistics(
    prices: pd.DataFrame, trading_days: int = 252
) -> tuple[pd.Series, pd.DataFrame]:
    """Convenience wrapper: prices in, (annualized mean returns, annualized covariance) out."""
    daily_returns = compute_daily_returns(prices)
    mean_returns = annualize_mean_returns(daily_returns, trading_days)
    cov_matrix = annualize_covariance(daily_returns, trading_days)
    return mean_returns, cov_matrix


def train_test_split_prices(
    prices: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a price history in time: an estimation window and a holdout window.

    Used to check whether weights fit on the training window still look good
    on data the optimizer never saw, rather than only in-sample.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_idx = int(len(prices) * train_fraction)
    return prices.iloc[:split_idx], prices.iloc[split_idx:]


def to_numpy(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The optimizer works in plain numpy; this is the one place pandas exits."""
    return mean_returns.to_numpy(), cov_matrix.to_numpy()
