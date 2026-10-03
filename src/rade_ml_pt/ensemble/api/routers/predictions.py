"""``/prism/v1/predictions`` — raw per-member prediction shards.

Streams the ``members/{cluster_id}/predictions/{split}.npz`` (scaled)
or ``{split}_original.npz`` (original) file verbatim.  Each NPZ carries
four arrays:

- ``predictions``     — ``(n_scenarios, n_trades_in_cluster)``, float32.
- ``targets``         — same shape, float32.
- ``trade_ids``       — ``(n_trades_in_cluster,)``, unicode.
- ``scenario_labels`` — ``(n_scenarios,)``, unicode.

``?space=scaled|original`` (Phase 3.4) selects which shard to stream.
The client loads the file with ``np.load(io.BytesIO(content))`` and
computes residuals locally.  Keeping the API binary-agnostic lets the
payload stay on the disk/memory boundary (Python never builds a list of
floats), so the Prediction Explorer stays fast even at 25 trades × 5 000
scenarios (≈ 500 KB per shard).

Not mtime-cached: shards are big, random-accessed, and change rarely in
practice.  ``FileResponse`` uses ``sendfile`` under the hood and serves
Range requests automatically.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["predictions"])


@router.get(
    "/predictions",
    response_class=FileResponse,
    # ``response_model=None`` disables FastAPI's attempt to derive a
    # Pydantic response schema from the ``-> FileResponse`` return
    # annotation.  Without it FastAPI raises
    # ``FastAPIError: Invalid args for response field`` because
    # ``FileResponse`` is a Starlette response (not a Pydantic-friendly
    # type).  ``response_class`` already carries the OpenAPI content type.
    response_model=None,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": (
                "NPZ file containing ``predictions`` and ``targets`` arrays, "
                "each shaped ``(n_scenarios, n_trades_in_cluster)``, plus "
                "self-describing ``trade_ids`` and ``scenario_labels``."
            ),
        },
        404: {
            "description": (
                "Shard not found for the given cluster / split / space.  "
                "Common case: ``space=original`` request on a run with no "
                "scaler coverage (no ``_original.npz`` files were written)."
            ),
        },
    },
)
def get_predictions(
    cluster_id: str = Query(..., description="Cluster identifier."),
    split:      str = Query(..., description="train / val / test"),
    space:      Literal["scaled", "original"] = Query(
        "scaled",
        description=(
            "PnL space — ``scaled`` (default) streams the legacy "
            "``{split}.npz``; ``original`` streams the Phase 3.1 "
            "``{split}_original.npz`` written when scaler coverage "
            "was available at eval time."
        ),
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> FileResponse:
    path = reader.member_predictions_path(cluster_id, split, space=space)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"predictions NPZ not found for cluster '{cluster_id}', "
                f"split '{split}', space '{space}' at {path}. Check that "
                f"the eval pipeline ran for this cluster/split combination "
                f"and (for space='original') that scaler artefacts were "
                f"available."
            ),
        )

    # Filename surfaces the space suffix so downloaded shards from the
    # two PnL spaces don't collide in the user's downloads folder.
    suffix = "" if space == "scaled" else "_original"
    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename=f"{cluster_id}_{split}{suffix}.npz",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
