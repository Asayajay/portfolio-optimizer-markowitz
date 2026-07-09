"""Unit tests for the equal-weight comparison in src/backtest.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    equal_weight_portfolio,
    historical_growth,
    summarize_portfolios,
    total_return,
)


class TestEqualWeightPortfolio:
    def test_weights_are_equal_and_sum_to_one(self):
        weights = equal_weight_portfolio(4)
        assert np.allclose(weights, 0.25)
        assert weights.sum() == pytest.approx(1.0)


class TestSummarizePortfolios:
    def test_returns_one_row_per_named_portfolio(self):
        mean_returns = np.array([0.10, 0.20])
        cov_matrix = np.array([[0.04, 0.0], [0.0, 0.09]])
        portfolios = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}

        summary = summarize_portfolios(portfolios, mean_returns, cov_matrix, risk_free_rate=0.03)

        assert set(summary.index) == {"A", "B"}
        assert summary.loc["A", "expected_return"] == pytest.approx(0.10)
        assert summary.loc["B", "volatility"] == pytest.approx(0.3)


class TestHistoricalGrowth:
    def test_growth_of_one_asset_matches_its_own_price_path(self):
        # A portfolio 100% in "A" should track A's price path exactly (normalized to 1.0 at day one).
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        prices = pd.DataFrame({"A": [100, 110, 121, 110], "B": [50, 50, 50, 50]}, index=dates)

        growth = historical_growth(prices, {"AllA": np.array([1.0, 0.0])})

        expected = prices["A"] / prices["A"].iloc[0]
        assert np.allclose(growth["AllA"].to_numpy(), expected.iloc[1:].to_numpy())

    def test_total_return_matches_start_to_end_change(self):
        growth = pd.Series([1.0, 1.1, 1.2, 0.9])
        assert total_return(growth) == pytest.approx(-0.1)
