"""``/prism/v1/governance/*`` — cross-version registry endpoints.

The single endpoint shipped here returns one row per registered
ensemble version (the *Model Registry* table that anchors the
governance page).  Sign-offs / approvals / audit-log endpoints will
join this router in Stage 2 once the producer side exists; the route
prefix is already namespaced under ``/governance/`` so adding them is
a single ``router.get(...)`` away.

Why settings (not the reader) provide the paths
-----------------------------------------------
Every other PRISM router reads the active ensemble's bundle through
:func:`get_reader` — but governance is *cross-version* by definition.
The :class:`~src.rade_ml_pt.ensemble.api.services.reader.ArtifactReader`
holds a single :class:`~..services.paths.ArtifactPaths` keyed on the
active version, so it can't enumerate the full registry.  We pull
``registry_path`` / ``artifacts_path`` straight from settings instead
and delegate the walk to
:mod:`src.rade_ml_pt.ensemble.api.services.governance`.
"""
from __future__ import annotations

from fastapi import APIRouter

from src.rade_ml_pt.ensemble.api.config import get_settings
from src.rade_ml_pt.ensemble.api.models.governance import (
    GovernanceRegistryResponse,
)
from src.rade_ml_pt.ensemble.api.services.governance import (
    build_governance_registry,
)

router = APIRouter(prefix="/prism/v1/governance", tags=["governance"])


@router.get("/registry", response_model=GovernanceRegistryResponse)
def get_governance_registry() -> GovernanceRegistryResponse:
    """Return the full ensemble registry (one row per version)."""
    settings = get_settings()
    return build_governance_registry(
        registry_dir=settings.registry_path,
        artifacts_dir=settings.artifacts_path,
        active_version=settings.resolved_version,
    )
