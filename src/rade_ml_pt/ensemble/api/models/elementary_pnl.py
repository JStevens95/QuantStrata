"""Pydantic response schemas for the ``/prism/v1/elementary-pnl`` endpoint.

Mirrors ``members/{cluster_id}/elementary_pnl.parquet``, which the eval
pipeline stages from the training registry alongside
``training_curves.parquet`` and ``trade_universe.json``.

The parquet's natural shape is row=scenario, col=trade — i.e. one
column per elementary trade id, values are the raw PnL of that hedge
instrument under each scenario.  For the JSON wire format we transpose
to columnar (``{trade_id: [values]}``) for the same reasons as
:class:`~src.rade_ml_pt.ensemble.api.models.training_curves.TrainingCurvesResponse`:
the UI can plot any subset of trades without re-zipping rows
client-side, and the payload stays compact when only a few trades are
selected.

Elementary trades are *model inputs* (the atomic legs / hedge
instruments that compose target structures), so this endpoint
deliberately does not return predictions or targets — the chart on
the UI side plots the raw PnL directly.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ElementaryPnlResponse(BaseModel):
    """Per-scenario PnL for one or more elementary trades in a cluster.

    Columnar layout (parallel arrays).  ``scenario_idx`` is a 0-indexed
    integer axis shared by every series in ``pnl``; values are the raw
    PnL of each elementary trade under each scenario.
    """

    cluster_id: str
    n_scenarios: int = Field(
        ..., description="Length of every series in ``pnl``.",
    )
    n_trades: int = Field(
        ..., description="Number of trades returned in ``pnl``.",
    )

    scenario_idx: List[int] = Field(
        ...,
        description=(
            "0-indexed scenario axis shared by every series in ``pnl``."
        ),
    )
    pnl: Dict[str, List[float]] = Field(
        ...,
        description=(
            "``{trade_id: [pnl_per_scenario]}`` — one entry per "
            "elementary trade requested via ``trade_ids``.  Trades "
            "that don't exist in the cluster's parquet are silently "
            "omitted; consumers can compare ``len(pnl)`` against "
            "``len(trade_ids)`` to detect drops."
        ),
    )


__all__ = ["ElementaryPnlResponse"]
