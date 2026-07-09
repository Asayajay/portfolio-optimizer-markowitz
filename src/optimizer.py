"""Markowitz mean-variance optimization: efficient frontier, min-variance, and max-Sharpe portfolios.

Everything here works in plain numpy arrays (mean_returns: shape (n,), cov_matrix:
shape (n, n)) so it has no opinion about tickers or pandas -- that lives in
src/data.py and src/main.py.

Long/short is a single switch. `_bounds()` is the only place that decides what a
weight is allowed to be; every optimizer below takes `allow_short` and defers to
it rather than hardcoding a [0, 1] bound. Flipping the default to short-selling
later means changing one call site, not re-deriving the constraint logic.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_SOLVER_OPTIONS = {"maxiter": 1000, "ftol": 1e-12}


def _bounds(n_assets: int, allow_short: bool) -> tuple[tuple[float, float], ...]:
    """Per-asset weight bounds. Long-only clamps to [0, 1]; short allows [-1, 1]."""
    if allow_short:
        return tuple((-1.0, 1.0) for _ in range(n_assets))
    return tuple((0.0, 1.0) for _ in range(n_assets))


def _fully_invested_constraint() -> dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _equal_weights(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    return float(weights @ mean_returns)


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(weights @ cov_matrix @ weights)


def portfolio_std(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(np.sqrt(max(portfolio_variance(weights, cov_matrix), 0.0)))


def sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    volatility = portfolio_std(weights, cov_matrix)
    if volatility == 0:
        return -np.inf
    return (portfolio_return(weights, mean_returns) - risk_free_rate) / volatility


def _solve(objective, n_assets, constraints, allow_short, x0=None):
    result = minimize(
        objective,
        x0 if x0 is not None else _equal_weights(n_assets),
        method="SLSQP",
        bounds=_bounds(n_assets, allow_short),
        constraints=constraints,
        options=_SOLVER_OPTIONS,
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed to converge: {result.message}")
    return result.x


def min_variance_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: float | None = None,
    allow_short: bool = False,
) -> np.ndarray:
    """Weights that minimize variance, optionally pinned to an exact target return.

    With no target_return this is the global minimum-variance portfolio. With
    one, it's a single point on the efficient frontier -- sweeping target_return
    across its feasible range and calling this repeatedly is how
    `efficient_frontier()` traces the whole curve.
    """
    n = len(mean_returns)
    constraints = [_fully_invested_constraint()]
    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w: w @ mean_returns - target_return}
        )
    return _solve(lambda w: portfolio_variance(w, cov_matrix), n, constraints, allow_short)


def max_sharpe_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    allow_short: bool = False,
) -> np.ndarray:
    """Weights that maximize (return - risk_free_rate) / volatility."""
    n = len(mean_returns)
    constraints = [_fully_invested_constraint()]

    def neg_sharpe(w):
        return -sharpe_ratio(w, mean_returns, cov_matrix, risk_free_rate)

    return _solve(neg_sharpe, n, constraints, allow_short)


def max_return_for_risk(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_risk: float,
    allow_short: bool = False,
) -> np.ndarray:
    """Weights that maximize return subject to volatility <= target_risk.

    This is the "I have a risk budget, tell me the best portfolio at or under
    it" query -- the mirror image of pinning a target return in
    `min_variance_portfolio`.
    """
    n = len(mean_returns)
    constraints = [
        _fully_invested_constraint(),
        {"type": "ineq", "fun": lambda w: target_risk**2 - portfolio_variance(w, cov_matrix)},
    ]
    return _solve(lambda w: -portfolio_return(w, mean_returns), n, constraints, allow_short)


def efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 60,
    allow_short: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Trace the efficient frontier by sweeping target return and minimizing variance at each.

    Returns (risks, returns, weights_list), each of length <= n_points --
    points where the target return turned out infeasible (can happen right at
    the top of the range under long-only bounds) are skipped rather than
    raising.

    The range runs from the minimum-variance portfolio's return up to the
    single highest-returning asset's return, which is the max achievable
    return under long-only, fully-invested constraints (put everything in the
    best performer).
    """
    min_var_weights = min_variance_portfolio(mean_returns, cov_matrix, allow_short=allow_short)
    low_return = portfolio_return(min_var_weights, mean_returns)
    high_return = mean_returns.max()

    target_returns = np.linspace(low_return, high_return, n_points)
    risks, returns, weights_list = [], [], []
    for target in target_returns:
        try:
            weights = min_variance_portfolio(
                mean_returns, cov_matrix, target_return=target, allow_short=allow_short
            )
        except RuntimeError:
            continue
        risks.append(portfolio_std(weights, cov_matrix))
        returns.append(portfolio_return(weights, mean_returns))
        weights_list.append(weights)

    return np.array(risks), np.array(returns), weights_list


def optimize_for_target(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: float | None = None,
    target_risk: float | None = None,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
) -> np.ndarray:
    """The single "give me an allocation" entry point used by main.py.

    Pass a target_return, a target_risk, or neither (defaults to the
    max-Sharpe portfolio). Passing both is ambiguous and rejected.
    """
    if target_return is not None and target_risk is not None:
        raise ValueError("Specify at most one of target_return / target_risk, not both")
    if target_return is not None:
        return min_variance_portfolio(
            mean_returns, cov_matrix, target_return=target_return, allow_short=allow_short
        )
    if target_risk is not None:
        return max_return_for_risk(
            mean_returns, cov_matrix, target_risk=target_risk, allow_short=allow_short
        )
    return max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate, allow_short=allow_short)
