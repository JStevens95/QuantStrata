"""Registry endpoints — config data, cluster mapping, member configs."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from src.ui.apps.ensemble_analytics_db.api.dependencies import get_cache
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

router = APIRouter(prefix="/registry")


@router.get("/cluster-ids")
def get_cluster_ids(
    cache: ArtifactCache = Depends(get_cache),
) -> List[str]:
    return cache.cluster_ids


@router.get("/cluster-mapping")
def get_cluster_mapping(
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, List[str]]:
    return cache.cluster_mapping


@router.get("/member-versions")
def get_member_versions(
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, str]:
    return cache.member_versions


@router.get("/member-configs")
def get_member_configs(
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, Dict[str, Any]]:
    return cache.member_configs


@router.get("/ensemble-version")
def get_ensemble_version(
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, str]:
    return {"version": cache.ensemble_version}


@router.get("/ensemble-metrics/{split}")
def get_ensemble_metrics(
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, float]:
    return cache.ensemble_metrics.get(split, {})


@router.get("/member-metrics/{split}")
def get_member_metrics(
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, Dict[str, Any]]:
    return cache.member_metrics.get(split, {})


@router.get("/graph-data/{cluster_id}")
def get_graph_data(
    cluster_id: str,
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, Any]:
    data = cache.get_graph_data(cluster_id)
    if not data:
        return {}
    result: Dict[str, Any] = {}
    gr = data.get("graph_results", {})
    if gr:
        import numpy as np
        serialised = {}
        for k, v in gr.items():
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    t = v.detach().cpu()
                    if t.is_sparse:
                        t = t.to_dense()
                    serialised[k] = t.numpy().tolist()
                    continue
            except ImportError:
                pass
            if isinstance(v, np.ndarray):
                serialised[k] = v.tolist()
            elif hasattr(v, "tolist"):
                serialised[k] = v.tolist()
            else:
                serialised[k] = v
        result["graph_results"] = serialised
    result["trade_universe"] = data.get("trade_universe", {})
    return result


@router.get("/market-data/{cluster_id}")
def get_market_data(
    cluster_id: str,
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, Any]:
    data = cache.get_market_data(cluster_id)
    if not data:
        return {}
    import numpy as np
    result: Dict[str, Any] = {}
    for asset, rfs in data.items():
        result[asset] = {
            rf: arr.tolist() if isinstance(arr, np.ndarray) else arr
            for rf, arr in rfs.items()
        }
    return result


@router.get("/cluster-display/{cluster_id}")
def get_cluster_display(
    cluster_id: str,
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, Any]:
    return cache.get_cluster_display(cluster_id)


@router.get("/convergence-png/{cluster_id}")
def get_convergence_png(
    cluster_id: str,
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, Any]:
    """Return base64-encoded training convergence PNG for a cluster."""
    import base64

    png_bytes = cache.get_convergence_png(cluster_id)
    if png_bytes is None:
        return {"png_base64": None}
    return {"png_base64": base64.b64encode(png_bytes).decode("ascii")}


@router.get("/elementary-pnl/{cluster_id}")
def get_elementary_pnl(
    cluster_id: str,
    cache: ArtifactCache = Depends(get_cache),
) -> Dict[str, Any]:
    """Return elementary PnL data as column-oriented JSON."""
    import pandas as pd

    path = cache.get_elementary_pnl_path(cluster_id)
    if path is None:
        return {"columns": [], "data": []}
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        return {"columns": [], "data": []}
    df = pd.read_parquet(p)
    return {
        "columns": [str(c) for c in df.columns.tolist()],
        "data": df.values.tolist(),
    }
