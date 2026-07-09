"""Unit tests for the Markowitz optimization logic in src/optimizer.py.

All of these work on small synthetic mean/covariance inputs -- no network
calls, no dependence on what the market happened to do today.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.optimizer import (
    efficient_frontier,
    max_return_for_risk,
    max_sharpe_portfolio,
    min_variance_portfolio,
    optimize_for_target,
    portfolio_return,
    portfolio_std,
    portfolio_variance,
    sharpe_ratio,
)

# Three assets with distinct risk/return so there's a meaningfully different
# min-variance vs. max-Sharpe vs. equal-weight portfolio to check against.
MEAN_RETURNS = np.array([0.08, 0.15, 0.12])
COV_MATRIX = np.array(
    [
        [0.020, 0.005, 0.002],
        [0.005, 0.045, 0.010],
        [0.002, 0.010, 0.030],
    ]
)
RISK_FREE_RATE = 0.03


def _assert_valid_long_only_weights(weights: np.ndarray, n_assets: int) -> None:
    assert weights.shape == (n_assets,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert (weights >= -1e-8).all(), f"found a negative weight: {weights}"


class TestPortfolioMath:
    def test_portfolio_return_is_weighted_average(self):
        weights = np.array([0.5, 0.25, 0.25])
        expected = 0.5 * 0.08 + 0.25 * 0.15 + 0.25 * 0.12
        assert portfolio_return(weights, MEAN_RETURNS) == pytest.approx(expected)

    def test_single_asset_variance_matches_its_own_variance(self):
        weights = np.array([1.0, 0.0, 0.0])
        assert portfolio_variance(weights, COV_MATRIX) == pytest.approx(COV_MATRIX[0, 0])

    def test_portfolio_std_is_sqrt_of_variance(self):
        weights = np.array([0.3, 0.4, 0.3])
        variance = portfolio_variance(weights, COV_MATRIX)
        assert portfolio_std(weights, COV_MATRIX) == pytest.approx(np.sqrt(variance))

    def test_sharpe_ratio_zero_vol_is_negative_infinity(self):
        # Degenerate case: a covariance matrix of all zeros has no risk to divide by.
        weights = np.array([1.0, 0.0, 0.0])
        zero_cov = np.zeros((3, 3))
        assert sharpe_ratio(weights, MEAN_RETURNS, zero_cov, RISK_FREE_RATE) == -np.inf


class TestMinVariancePortfolio:
    def test_weights_sum_to_one_and_are_non_negative(self):
        weights = min_variance_portfolio(MEAN_RETURNS, COV_MATRIX)
        _assert_valid_long_only_weights(weights, 3)

    def test_beats_or_matches_equal_weight_variance(self):
        # By definition, the minimum-variance portfolio can't have higher
        # variance than any other fully-invested long-only portfolio.
        min_var_weights = min_variance_portfolio(MEAN_RETURNS, COV_MATRIX)
        equal_weights = np.full(3, 1 / 3)
        assert portfolio_variance(min_var_weights, COV_MATRIX) <= portfolio_variance(
            equal_weights, COV_MATRIX
        ) + 1e-9

    def test_pinning_a_target_return_hits_it(self):
        target = 0.10
        weights = min_variance_portfolio(MEAN_RETURNS, COV_MATRIX, target_return=target)
        _assert_valid_long_only_weights(weights, 3)
        assert portfolio_return(weights, MEAN_RETURNS) == pytest.approx(target, abs=1e-4)

    def test_infeasible_target_return_raises(self):
        # Long-only, fully-invested: no combination of these assets can
        # return more than the single best asset (15%).
        with pytest.raises(RuntimeError):
            min_variance_portfolio(MEAN_RETURNS, COV_MATRIX, target_return=0.50)


class TestMaxSharpePortfolio:
    def test_weights_are_valid(self):
        weights = max_sharpe_portfolio(MEAN_RETURNS, COV_MATRIX, RISK_FREE_RATE)
        _assert_valid_long_only_weights(weights, 3)

    def test_has_highest_sharpe_among_the_named_candidates(self):
        max_sharpe_weights = max_sharpe_portfolio(MEAN_RETURNS, COV_MATRIX, RISK_FREE_RATE)
        best_sharpe = sharpe_ratio(max_sharpe_weights, MEAN_RETURNS, COV_MATRIX, RISK_FREE_RATE)

        for candidate in (
            np.full(3, 1 / 3),
            min_variance_portfolio(MEAN_RETURNS, COV_MATRIX),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ):
            assert best_sharpe >= sharpe_ratio(
                candidate, MEAN_RETURNS, COV_MATRIX, RISK_FREE_RATE
            ) - 1e-6


class TestMaxReturnForRisk:
    def test_respects_the_risk_cap(self):
        target_risk = 0.15
        weights = max_return_for_risk(MEAN_RETURNS, COV_MATRIX, target_risk)
        _assert_valid_long_only_weights(weights, 3)
        assert portfolio_std(weights, COV_MATRIX) <= target_risk + 1e-4


class TestOptimizeForTarget:
    def test_defaults_to_max_sharpe(self):
        default = optimize_for_target(MEAN_RETURNS, COV_MATRIX, risk_free_rate=RISK_FREE_RATE)
        explicit = max_sharpe_portfolio(MEAN_RETURNS, COV_MATRIX, RISK_FREE_RATE)
        assert np.allclose(default, explicit)

    def test_rejects_both_target_return_and_target_risk(self):
        with pytest.raises(ValueError):
            optimize_for_target(MEAN_RETURNS, COV_MATRIX, target_return=0.10, target_risk=0.15)


class TestEfficientFrontier:
    def test_frontier_is_non_decreasing_in_risk_and_return(self):
        risks, returns, weights_list = efficient_frontier(MEAN_RETURNS, COV_MATRIX, n_points=20)
        assert len(risks) == len(returns) == len(weights_list)
        assert len(risks) > 5  # most of the 20 requested points should be feasible

        # Efficient frontier points are sorted by construction (increasing
        # target return), and for a genuine frontier risk should rise with it.
        assert np.all(np.diff(returns) >= -1e-9)
        assert np.all(np.diff(risks) >= -1e-6)

    def test_every_frontier_point_has_valid_weights(self):
        _, _, weights_list = efficient_frontier(MEAN_RETURNS, COV_MATRIX, n_points=10)
        for weights in weights_list:
            _assert_valid_long_only_weights(weights, 3)


class TestShortSellingToggle:
    """allow_short=True should be able to reach lower variance than long-only,
    since it's a strict relaxation of the same constraint set (bounds widen,
    everything else -- objective, equality constraint -- stays identical)."""

    def test_short_selling_relaxation_does_not_increase_min_variance(self):
        long_only = min_variance_portfolio(MEAN_RETURNS, COV_MATRIX, allow_short=False)
        with_short = min_variance_portfolio(MEAN_RETURNS, COV_MATRIX, allow_short=True)
        assert portfolio_variance(with_short, COV_MATRIX) <= portfolio_variance(
            long_only, COV_MATRIX
        ) + 1e-9

    def test_short_selling_allows_negative_weights_bound(self):
        weights = min_variance_portfolio(MEAN_RETURNS, COV_MATRIX, allow_short=True)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        # Not asserting a negative weight appears (depends on the inputs) --
        # just that the bound permits it and the fully-invested constraint holds.
