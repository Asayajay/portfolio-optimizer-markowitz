"""Unit tests for return/covariance calculations in src/data.py.

Uses a small hand-built price series so the expected numbers can be checked
by hand, rather than trusting the function under test to grade itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    annualize_covariance,
    annualize_mean_returns,
    compute_daily_returns,
    estimate_statistics,
    train_test_split_prices,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "A": [100, 102, 101, 103, 105],
            "B": [50, 49, 51, 52, 52],
        },
        index=dates,
    )


class TestComputeDailyReturns:
    def test_returns_one_fewer_row_than_prices(self, prices):
        returns = compute_daily_returns(prices)
        assert len(returns) == len(prices) - 1

    def test_matches_hand_computed_pct_change(self, prices):
        returns = compute_daily_returns(prices)
        assert returns["A"].iloc[0] == pytest.approx((102 - 100) / 100)
        assert returns["B"].iloc[0] == pytest.approx((49 - 50) / 50)

    def test_no_nans_remain(self, prices):
        returns = compute_daily_returns(prices)
        assert not returns.isna().any().any()


class TestAnnualization:
    def test_mean_return_scales_by_trading_days(self, prices):
        daily_returns = compute_daily_returns(prices)
        annualized = annualize_mean_returns(daily_returns, trading_days=252)
        assert annualized["A"] == pytest.approx(daily_returns["A"].mean() * 252)

    def test_covariance_scales_by_trading_days(self, prices):
        daily_returns = compute_daily_returns(prices)
        annualized_cov = annualize_covariance(daily_returns, trading_days=252)
        daily_cov = daily_returns.cov()
        assert annualized_cov.loc["A", "B"] == pytest.approx(daily_cov.loc["A", "B"] * 252)

    def test_covariance_matrix_is_symmetric(self, prices):
        daily_returns = compute_daily_returns(prices)
        cov = annualize_covariance(daily_returns)
        assert np.allclose(cov.to_numpy(), cov.to_numpy().T)

    def test_estimate_statistics_matches_manual_pipeline(self, prices):
        mean_returns, cov_matrix = estimate_statistics(prices, trading_days=252)
        daily_returns = compute_daily_returns(prices)
        assert np.allclose(mean_returns.to_numpy(), (daily_returns.mean() * 252).to_numpy())
        assert np.allclose(cov_matrix.to_numpy(), (daily_returns.cov() * 252).to_numpy())


class TestTrainTestSplit:
    def test_split_preserves_all_rows_with_no_overlap(self, prices):
        train, test = train_test_split_prices(prices, train_fraction=0.6)
        assert len(train) + len(test) == len(prices)
        assert train.index[-1] < test.index[0]

    def test_train_fraction_out_of_range_raises(self, prices):
        with pytest.raises(ValueError):
            train_test_split_prices(prices, train_fraction=1.5)
        with pytest.raises(ValueError):
            train_test_split_prices(prices, train_fraction=0.0)
