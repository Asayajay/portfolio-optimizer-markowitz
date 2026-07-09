"""CLI entry point: fetch data, optimize, compare against equal-weight, plot.

Run with `python -m src.main`. All tunable parameters live in config.yaml;
see --help for the handful of things you can override per-run.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.backtest import (
    equal_weight_portfolio,
    historical_growth,
    summarize_portfolios,
    total_return,
)
from src.config import Config, load_config
from src.data import (
    estimate_statistics,
    fetch_price_data,
    train_test_split_prices,
)
from src.optimizer import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    optimize_for_target,
    portfolio_return,
    portfolio_std,
)
from src.visualize import plot_efficient_frontier

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def _print_allocation(title: str, tickers: list[str], weights: np.ndarray) -> None:
    print(f"\n{title}")
    allocation = (
        pd.Series(weights, index=tickers, name="weight")
        .sort_values(ascending=False)
        .map(lambda w: f"{w:.2%}")
    )
    print(allocation.to_string())


def _estimation_window_sensitivity(prices: pd.DataFrame, config: Config) -> None:
    """Refit the max-Sharpe portfolio on two different lookback windows and show
    how much the allocation and its expected Sharpe move -- the point being that
    "the optimal portfolio" is only optimal for the data it was estimated on.
    """
    half = len(prices) // 2
    windows = {"full history": prices, "most recent half": prices.iloc[half:]}

    weights_by_window = {}
    for label, window_prices in windows.items():
        mean_returns, cov_matrix = estimate_statistics(window_prices, config.trading_days_per_year)
        weights = max_sharpe_portfolio(
            mean_returns.to_numpy(),
            cov_matrix.to_numpy(),
            config.risk_free_rate,
            allow_short=config.allow_short,
        )
        weights_by_window[label] = weights

    diff = np.abs(weights_by_window["full history"] - weights_by_window["most recent half"])
    print("\nEstimation-window sensitivity check (max-Sharpe portfolio)")
    print(
        "  Same optimizer, same tickers, two lookback windows. Average absolute "
        f"weight shift per ticker: {diff.mean():.2%}. Largest single shift: "
        f"{diff.max():.2%} ({config.tickers[diff.argmax()]})."
    )
    print(
        "  This is the estimation risk Markowitz optimization doesn't show you on "
        "the frontier plot: the 'optimal' weights move with the lookback window, "
        "not just with the market."
    )


def run(config: Config, output_path: str = "output/efficient_frontier.html") -> None:
    print(f"Fetching price history for {len(config.tickers)} tickers "
          f"from {config.start_date} to {config.end_date or 'today'}...")
    prices = fetch_price_data(config.tickers, config.start_date, config.end_date)
    train_prices, test_prices = train_test_split_prices(prices, config.train_test_split)
    print(
        f"Estimation window: {train_prices.index[0].date()} to {train_prices.index[-1].date()} "
        f"({len(train_prices)} trading days)"
    )
    print(
        f"Holdout window:    {test_prices.index[0].date()} to {test_prices.index[-1].date()} "
        f"({len(test_prices)} trading days)"
    )

    mean_returns, cov_matrix = estimate_statistics(train_prices, config.trading_days_per_year)
    mean_returns_arr, cov_matrix_arr = mean_returns.to_numpy(), cov_matrix.to_numpy()

    chosen_weights = optimize_for_target(
        mean_returns_arr,
        cov_matrix_arr,
        target_return=config.target_return,
        target_risk=config.target_risk,
        risk_free_rate=config.risk_free_rate,
        allow_short=config.allow_short,
    )
    equal_weights = equal_weight_portfolio(len(config.tickers))
    min_vol_weights = min_variance_portfolio(
        mean_returns_arr, cov_matrix_arr, allow_short=config.allow_short
    )
    max_sharpe_weights = max_sharpe_portfolio(
        mean_returns_arr, cov_matrix_arr, config.risk_free_rate, allow_short=config.allow_short
    )

    query = (
        f"target return {config.target_return:.2%}" if config.target_return is not None
        else f"target risk {config.target_risk:.2%}" if config.target_risk is not None
        else "max Sharpe ratio (default)"
    )
    _print_allocation(f"Optimal allocation for: {query}", config.tickers, chosen_weights)

    print("\nIn-sample stats (estimated on the training window -- expect the "
          "optimizer to look good here almost by construction):")
    in_sample = summarize_portfolios(
        {"Optimized": chosen_weights, "Equal-weight": equal_weights},
        mean_returns_arr,
        cov_matrix_arr,
        config.risk_free_rate,
    )
    print(in_sample.to_string(formatters={
        "expected_return": "{:.2%}".format,
        "volatility": "{:.2%}".format,
        "sharpe_ratio": "{:.3f}".format,
    }))

    print("\nOut-of-sample check (actual returns on the holdout window the "
          "optimizer never saw):")
    growth = historical_growth(
        test_prices, {"Optimized": chosen_weights, "Equal-weight": equal_weights}
    )
    for name in growth.columns:
        print(f"  {name:<12} total return: {total_return(growth[name]):+.2%}")

    optimized_oos = total_return(growth["Optimized"])
    equal_oos = total_return(growth["Equal-weight"])
    if optimized_oos > equal_oos:
        edge = optimized_oos - equal_oos
        print(f"  -> Optimization beat equal-weight out-of-sample by {edge:.2%} over this window.")
    else:
        gap = equal_oos - optimized_oos
        print(
            f"  -> Equal-weight actually beat the optimized portfolio out-of-sample by {gap:.2%} "
            "over this window. The optimizer fit the training window's mean/covariance, not "
            "the future -- this is the expected failure mode of mean-variance optimization on "
            "a short history, not a bug."
        )

    _estimation_window_sensitivity(prices, config)

    asset_returns = mean_returns_arr
    asset_risks = np.sqrt(np.diag(cov_matrix_arr))
    frontier_risks, frontier_returns, _ = efficient_frontier(
        mean_returns_arr, cov_matrix_arr, config.frontier_points, config.allow_short
    )
    plot_efficient_frontier(
        frontier_risks,
        frontier_returns,
        min_vol_point=(
            portfolio_std(min_vol_weights, cov_matrix_arr),
            portfolio_return(min_vol_weights, mean_returns_arr),
        ),
        max_sharpe_point=(
            portfolio_std(max_sharpe_weights, cov_matrix_arr),
            portfolio_return(max_sharpe_weights, mean_returns_arr),
        ),
        equal_weight_point=(
            portfolio_std(equal_weights, cov_matrix_arr),
            portfolio_return(equal_weights, mean_returns_arr),
        ),
        asset_risks=asset_risks,
        asset_returns=asset_returns,
        asset_labels=config.tickers,
        output_path=output_path,
    )
    print(f"\nEfficient frontier chart saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Markowitz mean-variance portfolio optimizer")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--target-return", type=float, default=None, help="Override target annualized return")
    parser.add_argument("--target-risk", type=float, default=None, help="Override target annualized volatility")
    parser.add_argument("--output", default="output/efficient_frontier.html", help="Path to save the frontier chart")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.target_return is not None:
        config = Config(**{**config.__dict__, "target_return": args.target_return, "target_risk": None})
    elif args.target_risk is not None:
        config = Config(**{**config.__dict__, "target_risk": args.target_risk, "target_return": None})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    run(config, args.output)


if __name__ == "__main__":
    main()
