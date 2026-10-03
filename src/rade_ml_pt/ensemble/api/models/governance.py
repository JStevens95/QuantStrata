"""Pydantic schemas for the ``/prism/v1/governance/*`` endpoints.

Governance is a *cross-version* concern — every other PRISM endpoint
serves the active ensemble's evaluation bundle, but the governance
page needs to enumerate every version registered under the
``EnsembleRegistry`` so reviewers can see the full release history.

The shape of one row is deliberately wide because the UI Registry
table renders nine columns (status, version, created_by, created_at,
promoted_at, commit_sha, mae_test, n_clusters, n_trades) and we'd
rather pay the JSON inflation than do nine separate lookups in the
front-end.

Audit-log + sign-offs are intentionally *not* on the wire yet — the
V1 of the page seeds those panels from static placeholders so we can
ship the registry view immediately.  Re-introducing them is a Stage-2
add: extend this file with two more response types and add matching
endpoints; the UI is already set up to swallow the new fields.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GovernanceRegistryRow(BaseModel):
    """One row of the model registry — one ensemble version."""

    version: str = Field(
        ..., description="Concrete version directory name (e.g. ``ens_20260417_142233_abc123``).",
    )
    tags: List[str] = Field(
        default_factory=list,
        description=(
            "All tags pointing at this version in ``index.json`` (e.g. "
            "``['production']``, ``['latest']``, or both)."
        ),
    )
    status: str = Field(
        ...,
        description=(
            "Lifecycle bucket derived from the tag set: "
            "``'production'`` (carries the ``production`` tag), "
            "``'staging'`` (``staging`` tag), "
            "``'candidate'`` (the active / ``latest`` version when no "
            "production / staging tag has been set yet), or "
            "``'archived'`` (no tags pointing at it)."
        ),
    )
    created_at: str = Field(
        ...,
        description=(
            "ISO-formatted UTC timestamp.  Parsed from the version directory "
            "name when possible (versions follow the ``ens_YYYYMMDD_HHMMSS_*`` "
            "convention) and falls back to the directory's ``mtime``."
        ),
    )
    promoted_at: Optional[str] = Field(
        None,
        description=(
            "When the version was last tagged ``production`` — taken from the "
            "mtime of ``index.json`` for now (V1 placeholder).  Re-promotion "
            "events will get a proper audit trail in Stage 2."
        ),
    )
    created_by: str = Field(
        ...,
        description=(
            "Author / pipeline identity.  Currently a static seed (``pipeline-bot``) "
            "until the registry stamps a real author into version metadata."
        ),
    )
    commit_sha: str = Field(
        ...,
        description=(
            "Short git SHA proxy — the trailing hash component of the version "
            "directory name (the registry already embeds an md5-derived 6-char "
            "tag).  Replaced with the real source-control SHA in Stage 2."
        ),
    )
    n_members: int = Field(
        ...,
        description="Number of cluster members in this version's ``ensemble_config.json``.",
    )
    n_trades: int = Field(
        ...,
        description=(
            "Total trade count across all clusters' ``cluster_mapping`` entries — "
            "the population the ensemble was trained on."
        ),
    )
    aggregation: str = Field(
        ...,
        description="Aggregation strategy from ``ensemble_config.json`` (e.g. ``concat``, ``stacking``).",
    )
    has_evaluation: bool = Field(
        ...,
        description=(
            "Whether ``{artifacts_dir}/ensemble/{version}/evaluation/`` exists — "
            "tells the UI whether the row's *Open* button should land on the "
            "Evaluation tab or display 'Eval pending'."
        ),
    )
    mae_test: Optional[float] = Field(
        None,
        description=(
            "Test-split MAE if the evaluation bundle was published.  ``None`` "
            "for versions that were registered but never evaluated, or "
            "evaluations that didn't include a ``test`` split."
        ),
    )
    rmse_test: Optional[float] = Field(
        None,
        description="Test-split RMSE; same caveats as :attr:`mae_test`.",
    )
    coverage_test: Optional[float] = Field(
        None,
        description=(
            "Fraction of test trades the ensemble produced predictions for "
            "(0–1, expected to be ~1.0 for a healthy run).  ``None`` when "
            "evaluation hasn't run."
        ),
    )


class GovernanceRegistryResponse(BaseModel):
    """Collection response for ``GET /prism/v1/governance/registry``."""

    active_version: str = Field(
        ...,
        description=(
            "The currently-served version — the row whose ``version`` matches "
            "this gets a *Selected* outline in the UI table."
        ),
    )
    rows: List[GovernanceRegistryRow] = Field(
        ...,
        description=(
            "Every registered version, sorted newest-first by "
            ":attr:`GovernanceRegistryRow.created_at`."
        ),
    )
