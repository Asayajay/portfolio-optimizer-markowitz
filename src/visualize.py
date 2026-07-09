"""The efficient frontier chart: risk vs. return, with the two named portfolios marked."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

# Validated categorical palette (dataviz skill, light surface #fcfcfb) --
# chosen so frontier line, the two highlighted portfolios, and the equal-weight
# baseline stay distinct for colorblind viewers (worst adjacent CVD dE 13.3).
_COLOR_FRONTIER = "#2a78d6"  # blue
_COLOR_MIN_VOL = "#008300"  # green
_COLOR_MAX_SHARPE = "#e34948"  # red
_COLOR_EQUAL_WEIGHT = "#eb6834"  # orange
_COLOR_ASSETS = "#898781"  # muted ink, for context points rather than a series
_COLOR_GRIDLINE = "#e1e0d9"


def plot_efficient_frontier(
    frontier_risks: np.ndarray,
    frontier_returns: np.ndarray,
    min_vol_point: tuple[float, float],
    max_sharpe_point: tuple[float, float],
    equal_weight_point: tuple[float, float],
    asset_risks: Sequence[float],
    asset_returns: Sequence[float],
    asset_labels: Sequence[str],
    output_path: str | None = None,
) -> go.Figure:
    """Build the frontier chart and optionally save it as a static HTML file.

    Risk is annualized volatility (x-axis), return is annualized expected
    return (y-axis) -- both as decimals, displayed as percentages.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frontier_risks,
            y=frontier_returns,
            mode="lines",
            name="Efficient frontier",
            line=dict(color=_COLOR_FRONTIER, width=2),
            hovertemplate="Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(asset_risks),
            y=list(asset_returns),
            mode="markers+text",
            name="Individual assets",
            text=list(asset_labels),
            textposition="top center",
            textfont=dict(color=_COLOR_ASSETS, size=11),
            marker=dict(color=_COLOR_ASSETS, size=8, line=dict(color="#fcfcfb", width=2)),
            hovertemplate="%{text}<br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[equal_weight_point[0]],
            y=[equal_weight_point[1]],
            mode="markers",
            name="Equal-weight portfolio",
            marker=dict(
                color=_COLOR_EQUAL_WEIGHT,
                size=14,
                symbol="x",
                line=dict(color="#fcfcfb", width=2),
            ),
            hovertemplate="Equal-weight<br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[min_vol_point[0]],
            y=[min_vol_point[1]],
            mode="markers",
            name="Min-variance portfolio",
            marker=dict(
                color=_COLOR_MIN_VOL,
                size=14,
                symbol="diamond",
                line=dict(color="#fcfcfb", width=2),
            ),
            hovertemplate="Min-variance<br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_point[0]],
            y=[max_sharpe_point[1]],
            mode="markers",
            name="Max-Sharpe portfolio",
            marker=dict(
                color=_COLOR_MAX_SHARPE,
                size=16,
                symbol="star",
                line=dict(color="#fcfcfb", width=2),
            ),
            hovertemplate="Max-Sharpe<br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis=dict(
            title="Annualized volatility (risk)",
            tickformat=".0%",
            gridcolor=_COLOR_GRIDLINE,
            zeroline=False,
        ),
        yaxis=dict(
            title="Annualized expected return",
            tickformat=".0%",
            gridcolor=_COLOR_GRIDLINE,
            zeroline=False,
        ),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(252,252,251,0.8)"),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color="#0b0b0b"),
        width=900,
        height=600,
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig
