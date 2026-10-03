"""``/prism/v1/portfolio`` — portfolio-level timeseries per split.

Reads ``portfolio_timeseries_{split}.parquet`` (B1 in the PRISM eval
contract, schema v3) and returns its eight measure columns as parallel
arrays.  ``?space=scaled|original`` selects which PnL space's columns
to surface as the canonical measure columns — see
:func:`src.rade_ml_pt.ensemble.api.services.reader._apply_space`.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import pandas as pd

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.portfolio import (
    PortfolioTimeseriesResponse,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["portfolio"])


def _nullable_floats(values) -> List[Optional[float]]:
    """Coerce a parquet column to a JSON-safe ``List[Optional[float]]``.

    ``float('nan')`` is not valid JSON and Pydantic's ``List[float]``
    would also reject it — so original-space requests on runs without
    scaler coverage need NaN positions to round-trip as ``null``.  This
    helper centralises that coercion for the portfolio + cluster
    timeseries routers.
    """
    return [None if pd.isna(x) else float(x) for x in values]


@router.get("/portfolio", response_model=PortfolioTimeseriesResponse)
def get_portfolio_timeseries(
    split: str = Query(..., description="train / val / test"),
    space: Literal["scaled", "original"] = Query(
        "scaled",
        description=(
            "PnL space for the measure columns.  ``scaled`` (default) "
            "returns the model's z-space output; ``original`` returns "
            "inverse-scaled, notional-sign-restored PnL.  Original-space "
            "values are ``null`` for runs / scenarios where the writer "
            "had no scaler coverage."
        ),
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> PortfolioTimeseriesResponse:
    """Return the portfolio predicted/actual series for one split."""
    try:
        df = reader.portfolio_timeseries(split, space=space)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"portfolio_timeseries_{split}.parquet not found. "
                f"Splits available: {reader.available_splits()}."
            ),
        )

    cols = df.to_dict(orient="list")

    return PortfolioTimeseriesResponse(
        split=split,
        space=space,
        n_scenarios=len(df),
        scenario_idx=[int(x) for x in cols["scenario_idx"]],
        scenario_label=[str(x) for x in cols["scenario_label"]],
        predictions=_nullable_floats(cols["predictions"]),
        targets=_nullable_floats(cols["targets"]),
        error=_nullable_floats(cols["error"]),
        abs_error=_nullable_floats(cols["abs_error"]),
        squared_error=_nullable_floats(cols["squared_error"]),
    )
