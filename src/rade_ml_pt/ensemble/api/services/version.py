"""Resolve ensemble version tags (e.g. ``"latest"``) to concrete version directories.

Mirrors the lookup logic in
:class:`src.rade_ml_pt.ensemble.registry.EnsembleRegistry` so the API and
the pipeline agree on how a tag like ``"latest"`` resolves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def resolve_version(
    version_or_tag: str,
    registry_dir: Path,
    artifacts_dir: Path,
) -> str:
    """Resolve a version string or tag to an actual version directory name.

    Resolution order:

    1. Look up *version_or_tag* as a key in ``{registry_dir}/ensemble/index.json``
       — this handles tags like ``"latest"`` and any custom aliases.
    2. Check for a version directory ``{registry_dir}/ensemble/{version_or_tag}/``.
    3. Check for a version directory ``{artifacts_dir}/ensemble/{version_or_tag}/``.
    4. Raise :class:`FileNotFoundError` listing available tags.
    """
    ens_reg = registry_dir / "ensemble"
    index_path = ens_reg / "index.json"

    if index_path.exists():
        index = json.loads(index_path.read_text())
        if version_or_tag in index:
            return index[version_or_tag]

    if (ens_reg / version_or_tag).is_dir():
        return version_or_tag

    ens_art = artifacts_dir / "ensemble"
    if (ens_art / version_or_tag).is_dir():
        return version_or_tag

    available: List[str] = []
    if index_path.exists():
        available = list(json.loads(index_path.read_text()).keys())

    raise FileNotFoundError(
        f"Cannot resolve ensemble version '{version_or_tag}'. "
        f"Not found in index.json or as a directory. "
        f"Available tags: {available}"
    )


def list_versions(registry_dir: Path) -> List[str]:
    """Return every tag recorded in ``{registry_dir}/ensemble/index.json``."""
    index_path = registry_dir / "ensemble" / "index.json"
    if not index_path.exists():
        return []
    return sorted(json.loads(index_path.read_text()).keys())
