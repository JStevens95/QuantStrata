"""``/prism/v1/group-correlations`` — pairwise residual correlations.

Reads ``group_correlations_{split}.parquet`` (B6 in the PRISM eval
contract).  One row per upper-triangular group-pair × attribute;
optionally filtered to a single attribute for the Cross-Cluster tab's
dropdown-driven view.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.group_correlations import (
    GroupCorrelationPair,
    GroupCorrelationsResponse,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["group-correlations"])


@router.get("/group-correlations", response_model=GroupCorrelationsResponse)
def get_group_correlations(
    split: str = Query(..., description="train / val / test"),
    attribute: Optional[str] = Query(
        None,
        description=(
            "Filter to pairs within one cluster attribute (e.g. "
            "``currency_code``). Omit to return every attribute."
        ),
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> GroupCorrelationsResponse:
    # Probe for file-existence once up-front so missing-split 404s are
    # attributed to the artifact rather than the attribute filter.
    try:
        attrs = reader.group_correlation_attributes(split)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"group_correlations_{split}.parquet not found. "
                f"Splits available: {reader.available_splits()}."
            ),
        )

    if attribute is not None and attribute not in attrs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"attribute '{attribute}' not present in split '{split}'. "
                f"Attributes available: {attrs}."
            ),
        )

    df = reader.group_correlations(split, attribute=attribute)
    pairs = [
        GroupCorrelationPair(**row) for row in df.to_dict(orient="records")
    ]
    return GroupCorrelationsResponse(
        split=split,
        attributes_available=attrs,
        pairs=pairs,
    )
