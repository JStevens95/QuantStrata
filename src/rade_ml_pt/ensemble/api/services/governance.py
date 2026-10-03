"""Governance service — walks the cross-version ensemble registry.

Sits *next to* :mod:`reader` rather than on it because the
:class:`~src.rade_ml_pt.ensemble.api.services.reader.ArtifactReader` is
scoped to a single ensemble version (the active one), whereas
governance needs every version that's ever been registered.

What we read
------------
* ``{registry_dir}/ensemble/index.json`` — the tag → version map.  We
  invert it so each version carries the list of tags pointing at it.
* ``{registry_dir}/ensemble/{version}/ensemble_config.json`` — for
  ``n_members``, ``n_trades`` and ``aggregation``.
* The version directory's ``mtime`` — fallback for ``created_at`` when
  the version name doesn't follow the ``ens_YYYYMMDD_HHMMSS_*`` convention.
* ``{artifacts_dir}/ensemble/{version}/evaluation/ensemble_metrics.parquet`` —
  for ``mae_test`` / ``rmse_test`` / ``coverage_test``.  Optional —
  versions registered but not yet evaluated leave these as ``None``.

What we *don't* read
--------------------
* No git lookups, no author resolution — V1 seeds those columns with
  static placeholders and we'll add a real source-of-truth in Stage 2
  (probably stamped at registration time by the eval pipeline).
* No audit-log persistence — the UI seeds that panel with mocked
  events for the same Stage-2 reason.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.rade_ml_pt.ensemble.api.models.governance import (
    GovernanceRegistryResponse,
    GovernanceRegistryRow,
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────

_INDEX_FILENAME = "index.json"
_CONFIG_FILENAME = "ensemble_config.json"
_EVAL_METRICS_FILENAME = "ensemble_metrics.parquet"

# Until the registry stamps real authors, every row gets the same
# placeholder — keeps the table column populated without lying about
# provenance.  Replaced with a real lookup in Stage 2.
_DEFAULT_AUTHOR = "pipeline-bot"

# Versions follow ``ens_YYYYMMDD_HHMMSS_<6 char md5>``; we extract
# the timestamp + the trailing hash separately.  Anything that
# doesn't match falls back to filesystem mtime + a synthetic SHA.
_VERSION_PATTERN = re.compile(
    r"^ens_(?P<ts>\d{8}_\d{6})_(?P<sha>[a-fA-F0-9]+)$",
)


# ── Public entry point ────────────────────────────────────────────────


def build_governance_registry(
    registry_dir: Path,
    artifacts_dir: Path,
    active_version: str,
) -> GovernanceRegistryResponse:
    """Walk every registered ensemble version and return the registry payload.

    Parameters
    ----------
    registry_dir
        ``settings.registry_path`` — the *parent* of the ``ensemble/``
        subtree.  Mirrors the directory the eval pipeline registers
        versions into.
    artifacts_dir
        ``settings.artifacts_path`` — the *parent* of the per-version
        ``evaluation/`` subtree.  Used to discover whether a version has
        evaluation artefacts and to read its ``ensemble_metrics`` parquet.
    active_version
        The version the rest of the API is currently serving — bubbled
        up to the response so the UI can decorate that row.
    """
    registry_root = registry_dir / "ensemble"

    if not registry_root.exists():
        logger.warning("Governance: registry root missing at %s", registry_root)
        return GovernanceRegistryResponse(
            active_version=active_version, rows=[],
        )

    tag_index = _load_index(registry_root)
    version_to_tags = _invert_tag_index(tag_index)
    index_path = registry_root / _INDEX_FILENAME
    index_mtime = (
        _to_iso_utc(_safe_mtime(index_path)) if index_path.exists() else None
    )

    rows: List[GovernanceRegistryRow] = []
    for version_dir in sorted(p for p in registry_root.iterdir() if p.is_dir()):
        config_path = version_dir / _CONFIG_FILENAME
        if not config_path.exists():
            # Half-written / aborted registration — skip rather than
            # blow up the whole governance page for it.
            continue

        try:
            row = _build_row(
                version_dir=version_dir,
                config_path=config_path,
                tags=version_to_tags.get(version_dir.name, []),
                artifacts_dir=artifacts_dir,
                promoted_at=index_mtime,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception(
                "Governance: failed to build registry row for %s — skipping",
                version_dir.name,
            )
            del exc
            continue

        rows.append(row)

    rows.sort(key=lambda r: r.created_at, reverse=True)

    return GovernanceRegistryResponse(
        active_version=active_version, rows=rows,
    )


# ── Row builder ───────────────────────────────────────────────────────


def _build_row(
    *,
    version_dir: Path,
    config_path: Path,
    tags: List[str],
    artifacts_dir: Path,
    promoted_at: Optional[str],
) -> GovernanceRegistryRow:
    """Build one :class:`GovernanceRegistryRow` from one version directory."""
    version = version_dir.name

    config = json.loads(config_path.read_text())
    cluster_mapping: Dict[str, List[str]] = config.get("cluster_mapping", {}) or {}
    n_members = len(cluster_mapping)
    n_trades = sum(len(v) for v in cluster_mapping.values())
    aggregation = str(config.get("aggregation", "concat"))

    created_at = _resolve_created_at(version, version_dir)
    commit_sha = _resolve_commit_sha(version)
    status = _classify_status(tags)

    # Evaluation artefacts — best-effort.  If the parquet is missing or
    # malformed we keep the row but leave the metrics as ``None``; the
    # UI renders an em-dash.
    eval_dir = artifacts_dir / "ensemble" / version / "evaluation"
    has_evaluation = eval_dir.exists()

    mae_test, rmse_test, coverage_test = _read_test_metrics(
        eval_dir / _EVAL_METRICS_FILENAME,
    )

    return GovernanceRegistryRow(
        version=version,
        tags=tags,
        status=status,
        created_at=created_at,
        promoted_at=promoted_at if status == "production" else None,
        created_by=_DEFAULT_AUTHOR,
        commit_sha=commit_sha,
        n_members=n_members,
        n_trades=n_trades,
        aggregation=aggregation,
        has_evaluation=has_evaluation,
        mae_test=mae_test,
        rmse_test=rmse_test,
        coverage_test=coverage_test,
    )


# ── Registry index helpers ────────────────────────────────────────────


def _load_index(registry_root: Path) -> Dict[str, str]:
    """Return ``{tag: version}`` from ``index.json``, or empty dict."""
    index_path = registry_root / _INDEX_FILENAME
    if not index_path.exists():
        return {}
    try:
        return json.loads(index_path.read_text()) or {}
    except json.JSONDecodeError:
        logger.exception("Governance: malformed index.json at %s", index_path)
        return {}


def _invert_tag_index(index: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert the ``{tag: version}`` map into ``{version: [tags...]}``.

    Tags are sorted to give the response a stable order across calls.
    """
    out: Dict[str, List[str]] = {}
    for tag, version in index.items():
        out.setdefault(version, []).append(tag)
    for v in out:
        out[v].sort()
    return out


# ── Status / lifecycle classification ─────────────────────────────────


def _classify_status(tags: List[str]) -> str:
    """Bucket a tag list into one of four lifecycle states.

    Resolution order:

    1. ``production`` — the most authoritative tag wins.
    2. ``staging`` — second-rank tag.
    3. ``candidate`` — has any tag (typically ``latest``) but neither of
       the two promoted tags above.
    4. ``archived`` — no tags pointing at it (rolled past or
       superseded).
    """
    tag_set = {t.lower() for t in tags}
    if "production" in tag_set:
        return "production"
    if "staging" in tag_set:
        return "staging"
    if tag_set:
        return "candidate"
    return "archived"


# ── created_at / commit_sha resolution ────────────────────────────────


def _resolve_created_at(version: str, version_dir: Path) -> str:
    """Parse the timestamp from the version name, falling back to mtime."""
    match = _VERSION_PATTERN.match(version)
    if match:
        try:
            dt = datetime.strptime(match.group("ts"), "%Y%m%d_%H%M%S")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    return _to_iso_utc(_safe_mtime(version_dir))


def _resolve_commit_sha(version: str) -> str:
    """Return the trailing hash from a registry version, or a synthetic stub."""
    match = _VERSION_PATTERN.match(version)
    if match:
        return match.group("sha")
    # Stable but synthetic — use the directory name as the source so
    # the UI gets *something* unique instead of a placeholder string.
    return version[:8]


def _safe_mtime(path: Path) -> float:
    """Return ``path.stat().st_mtime`` with a 0-fallback so callers never raise."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _to_iso_utc(epoch_seconds: float) -> str:
    """Format a POSIX timestamp as UTC ISO-8601 string."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


# ── Test metrics extraction ───────────────────────────────────────────


def _read_test_metrics(
    metrics_path: Path,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Pull ``mae`` / ``rmse`` / ``coverage`` for the ``test`` split, if present.

    Returns ``(None, None, None)`` for any version whose evaluation
    bundle hasn't been produced (or any malformed parquet) so the
    governance row still renders.
    """
    if not metrics_path.exists():
        return None, None, None

    try:
        df = pq.read_table(metrics_path).to_pandas()
    except Exception:  # pragma: no cover — defensive
        logger.exception(
            "Governance: failed to read ensemble_metrics at %s", metrics_path,
        )
        return None, None, None

    if df.empty or "split" not in df.columns:
        return None, None, None

    test_rows = df[df["split"].astype(str).str.lower() == "test"]
    if test_rows.empty:
        return None, None, None

    row = test_rows.iloc[0]
    return (
        _maybe_float(row.get("mae")),
        _maybe_float(row.get("rmse")),
        _maybe_float(row.get("coverage")),
    )


def _maybe_float(value: Any) -> Optional[float]:
    """Coerce parquet cell to ``float`` or ``None`` (handles NaN)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
