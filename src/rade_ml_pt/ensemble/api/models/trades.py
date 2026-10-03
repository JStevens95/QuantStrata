"""Pydantic response schemas for the ``/prism/v1/trades`` endpoint."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class TradeMetric(BaseModel):
    """Per-trade aggregate metrics over all scenarios in one split.

    Mirrors ``trade_metrics_{split}.parquet`` (B3 schema v1).
    """

    cluster_id: str
    trade_id: str
    split: str
    mae: float
    mse: float
    rmse: float
    max_ae: float
    p95_ae: float
    p99_ae: float
    mean_residual: float = Field(..., description="mean(predicted − actual)")
    std_residual: float = Field(..., description="std(predicted − actual)")
    n_scenarios: int


class TradesResponse(BaseModel):
    """Collection response for ``GET /prism/v1/trades``."""

    split: str
    n_trades: int
    trades: List[TradeMetric]
