"""The sanity check: is the optimized portfolio actually better than equal-weight?

Two comparisons, and they can disagree:

- `summarize_portfolios` compares expected return/volatility/Sharpe as *estimated
  from the same data the optimizer was fit on*. The optimized portfolio will
  always win here almost by construction -- it was built to maximize exactly
  this number on exactly this data.
- `historical_growth` replays both portfolios' actual day-by-day returns over a
  price history and tracks cumulative growth of $1. Run this on a holdout
  window the optimizer never saw, and it's the honest test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimizer import portfolio_return, portfolio_std, sharpe_ratio


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def summarize_portfolios(
    portfolios: dict[str, np.ndarray],
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> pd.DataFrame:
    """One row per named portfolio: expected return, volatility, Sharpe ratio."""
    rows = {
        name: {
            "expected_return": portfolio_return(weights, mean_returns),
            "volatility": portfolio_std(weights, cov_matrix),
            "sharpe_ratio": sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate),
        }
        for name, weights in portfolios.items()
    }
    return pd.DataFrame(rows).T


def historical_growth(prices: pd.DataFrame, portfolios: dict[str, np.ndarray]) -> pd.DataFrame:
    """Cumulative growth of $1 for each named portfolio, replayed on real daily returns.

    `prices` should be a window the weights were *not* fit on to make this a
    genuine out-of-sample check rather than circular reasoning.
    """
    daily_returns = prices.pct_change().dropna(how="any")
    growth = {
        name: (1.0 + daily_returns.to_numpy() @ weights).cumprod()
        for name, weights in portfolios.items()
    }
    return pd.DataFrame(growth, index=daily_returns.index)


def total_return(growth_series: pd.Series) -> float:
    """Total return over the window, from the growth-of-$1 series."""
    return float(growth_series.iloc[-1] / growth_series.iloc[0] - 1.0)


def annualized_volatility_from_growth(growth_series: pd.Series, trading_days: int = 252) -> float:
    daily_returns = growth_series.pct_change().dropna()
    return float(daily_returns.std() * np.sqrt(trading_days))
