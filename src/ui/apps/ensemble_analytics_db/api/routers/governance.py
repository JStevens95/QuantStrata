"""Governance endpoints — version listing, comparison, config inspection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.ui.apps.ensemble_analytics_db.api.config import get_settings
from src.ui.apps.ensemble_analytics_db.api.dependencies import get_cache
from src.ui.apps.ensemble_analytics_db.api.models.graph import (
    MetricDelta,
    VersionCompareResponse,
)
from src.ui.apps.ensemble_analytics_db.api.models.metadata import (
    VersionListEntry,
    VersionListResponse,
    VersionMetaResponse,
)
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

router = APIRouter(prefix="/governance")


@router.get("/version", response_model=VersionMetaResponse)
def get_active_version(cache: ArtifactCache = Depends(get_cache)):
    """Return metadata for the currently loaded ensemble version."""
    settings = get_settings()
    m = cache.manifest
    return VersionMetaResponse(
        version=settings.resolved_version,
        n_clusters=len(m.get("cluster_ids", [])),
        n_trades=len(cache.trade_cluster_map),
        splits_available=m.get("splits_available", []),
        cluster_ids=m.get("cluster_ids", []),
    )


@router.get("/versions", response_model=VersionListResponse)
def list_versions(cache: ArtifactCache = Depends(get_cache)):
    """List all available ensemble versions found on disk."""
    settings = get_settings()
    registry_ens_dir = settings.registry_path / "ensemble"
    versions = []
    if registry_ens_dir.exists():
        for d in sorted(registry_ens_dir.iterdir()):
            if d.is_dir() and (d / "ensemble_config.json").exists():
                try:
                    cfg = json.loads((d / "ensemble_config.json").read_text())
                    versions.append(VersionListEntry(
                        version=d.name,
                        n_clusters=cfg.get("n_members", 0),
                        n_trades=len(cfg.get("all_trade_ids", [])),
                    ))
                except Exception:
                    versions.append(VersionListEntry(version=d.name))

    return VersionListResponse(
        versions=versions,
        active_version=settings.resolved_version,
    )


@router.get("/compare", response_model=VersionCompareResponse)
def compare_versions(
    compare_version: str = Query(..., description="Version to compare against"),
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
):
    """Compare ensemble metrics between the active version and another."""
    settings = get_settings()
    current_metrics = cache.ensemble_metrics.get(split, {})
    if not current_metrics:
        raise HTTPException(404, f"No metrics for active version, split '{split}'")

    suffix = "" if split == "test" else f"_{split}"
    compare_eval = (
        settings.artifacts_path / "ensemble" / compare_version / "evaluation"
    )
    compare_path = compare_eval / f"ensemble_metrics{suffix}.json"
    if not compare_path.exists():
        raise HTTPException(
            404, f"No metrics found for version '{compare_version}', split '{split}'"
        )

    compare_metrics = json.loads(compare_path.read_text())

    all_keys = sorted(set(current_metrics) | set(compare_metrics))
    deltas = []
    for k in all_keys:
        cur = current_metrics.get(k, 0.0)
        cmp = compare_metrics.get(k, 0.0)
        delta = cur - cmp
        pct = (delta / cmp * 100) if cmp != 0 else None
        deltas.append(MetricDelta(
            metric=k, current=cur, compare=cmp, delta=delta, pct_change=pct,
        ))

    return VersionCompareResponse(
        current_version=settings.resolved_version,
        compare_version=compare_version,
        split=split,
        deltas=deltas,
    )
