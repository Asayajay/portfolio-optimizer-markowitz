"""Loads config.yaml into a typed, validated object so the rest of the code
never touches raw dict keys."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    tickers: list[str]
    start_date: str
    end_date: str | None
    risk_free_rate: float
    allow_short: bool
    target_return: float | None
    target_risk: float | None
    trading_days_per_year: int
    frontier_points: int
    train_test_split: float


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        tickers=list(raw["tickers"]),
        start_date=raw["start_date"],
        end_date=raw.get("end_date"),
        risk_free_rate=float(raw.get("risk_free_rate", 0.0)),
        allow_short=bool(raw.get("allow_short", False)),
        target_return=raw.get("target_return"),
        target_risk=raw.get("target_risk"),
        trading_days_per_year=int(raw.get("trading_days_per_year", 252)),
        frontier_points=int(raw.get("frontier_points", 60)),
        train_test_split=float(raw.get("train_test_split", 0.7)),
    )
