"""Pydantic response schemas for the ``/prism/v1/group-correlations`` endpoint.

Backs B6 in the PRISM eval contract — long-format pairwise residual
correlations between groups defined by each cluster attribute (currency,
desk, product, etc.) for a given split.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class GroupCorrelationPair(BaseModel):
    """One upper-triangular pair within one attribute."""

    attribute: str = Field(
        ...,
        description=(
            "Cluster-attribute key defining the grouping, "
            "e.g. ``currency_code`` or ``desk``."
        ),
    )
    group_a: str = Field(
        ..., description="First group value within *attribute* (e.g. ``GBP``)."
    )
    group_b: str = Field(
        ..., description="Second group value within *attribute* (e.g. ``USD``)."
    )
    split: str = Field(..., description="train / val / test")
    rho: float = Field(
        ...,
        description="Pearson correlation of the two groups' residual series.",
    )
    n_scenarios: int = Field(
        ...,
        description="Number of scenarios behind the correlation (same as split length).",
    )


class GroupCorrelationsResponse(BaseModel):
    split: str
    attributes_available: List[str] = Field(
        ...,
        description=(
            "Every ``attribute`` value present in the parquet for this split. "
            "Unaffected by the ``attribute`` query filter — the client can "
            "populate an attribute dropdown from a single call."
        ),
    )
    pairs: List[GroupCorrelationPair]
