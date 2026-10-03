"""Pydantic response schemas for portfolio-timeseries endpoints."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PortfolioTimeseriesResponse(BaseModel):
    """Long-format portfolio predictions vs targets series for one split.

    Columnar (parallel arrays) rather than row-oriented: plotting
    libraries (Plotly, D3, Recharts) all consume arrays directly, and
    the JSON payload is ~3× smaller than an equivalent list-of-objects
    at 5k scenarios.

    All arrays share the same length ``n_scenarios`` and are aligned
    positionally.  ``scenario_idx[i]`` corresponds to ``predictions[i]``
    and ``targets[i]`` for the same scenario.

    Phase 3.2 — ``space`` identifies the PnL units of every measure
    column.  Measure-column types are nullable (``List[Optional[float]]``)
    so original-space requests on runs with partial / no scaler
    coverage can serialise NaN positions as JSON ``null`` (a plain
    ``List[float]`` would reject NaN).
    """

    split: str
    n_scenarios: int
    space: Literal["scaled", "original"] = Field(
        "scaled",
        description=(
            "PnL units of the measure columns.  ``scaled`` is the "
            "model's z-space output; ``original`` is inverse-scaled, "
            "notional-sign-restored PnL in source currency units."
        ),
    )

    scenario_idx: List[int] = Field(..., description="0-based index within the split.")
    scenario_label: List[str] = Field(..., description="Human-readable label, typically a date.")
    predictions: List[Optional[float]] = Field(..., description="Portfolio predicted PnL per scenario.")
    targets: List[Optional[float]] = Field(..., description="Portfolio actual (ground-truth) PnL per scenario.")
    error: List[Optional[float]] = Field(..., description="predictions − targets")
    abs_error: List[Optional[float]]
    squared_error: List[Optional[float]]
