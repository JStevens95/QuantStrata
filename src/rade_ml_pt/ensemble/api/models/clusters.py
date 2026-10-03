"""Pydantic response schemas for the ``/prism/v1/clusters`` endpoint.

``cluster_attributes.parquet`` has a **dynamic** column set — every run
has ``cluster_id`` and ``n_trades``, but the additional attribute
columns (asset class, currency, desk, product, ...) depend on the
ensemble config.  We expose that variability through an ``attributes``
bag on each row rather than hard-coding fields.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ClusterInfo(BaseModel):
    """Metadata for a single cluster."""

    cluster_id: str
    n_trades: int
    attributes: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description=(
            "Free-form attribute bag — keys vary per ensemble.  Typical "
            "keys: asset_class, currency_code, desk, product_code."
        ),
    )


class ClustersResponse(BaseModel):
    """Collection response for ``GET /prism/v1/clusters``."""

    attribute_names: List[str] = Field(
        ...,
        description=(
            "Union of attribute keys present across all clusters, sorted.  "
            "Useful for rendering tables with consistent column order."
        ),
    )
    clusters: List[ClusterInfo]
