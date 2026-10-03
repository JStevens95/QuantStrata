"""Pydantic response schemas for the training-curves endpoint.

Mirrors ``members/{cluster_id}/training_curves.parquet``, which the eval
pipeline stages from the training registry (trainer-side contract
§11.15.1).  Shape:

* ``epoch``       int    (required — monotonically increasing)
* ``train_loss``  float  (required — always emitted by the trainer)
* *other cols*    float  (any additional per-epoch series — e.g.
                          ``val_loss``, ``mae``, ``val_mae``)

The response is columnar (parallel arrays) so the UI can plot any metric
without having to re-zip rows client-side, and so the payload stays
compact even for long training runs.  ``metrics`` lists the optional
series names (excluding ``epoch`` + ``train_loss``) so consumers can
drive a metric picker without inspecting every column of ``series``.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class TrainingCurvesResponse(BaseModel):
    """Per-epoch training curves for one cluster member.

    ``series`` is a ``{metric_name: [values]}`` dict where every value
    array has length ``n_epochs``.  ``train_loss`` is always present;
    other keys are optional and depend on what the trainer emitted.
    ``epoch`` lives alongside the series dict for convenience — it is
    redundant with ``range(n_epochs)`` but the explicit representation
    keeps the x-axis correct even if the trainer ever switches to a
    non-zero start epoch.
    """

    cluster_id: str
    n_epochs: int
    epoch: List[int]
    series: Dict[str, List[float]] = Field(
        ...,
        description=(
            "Per-metric per-epoch values.  Always contains "
            "'train_loss'; other keys are optional per-epoch series "
            "the trainer emitted."
        ),
    )
    metrics: List[str] = Field(
        ...,
        description=(
            "Names of the optional metric series (i.e. keys of "
            "`series` excluding 'train_loss').  Surfaces the set of "
            "columns available for a UI metric picker."
        ),
    )


__all__ = ["TrainingCurvesResponse"]
