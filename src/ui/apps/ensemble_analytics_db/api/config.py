"""
Application settings loaded from environment variables or passed directly.

Supports two usage patterns:

1. **Env vars / .env** — for ``uvicorn`` CLI launch::

       export PRISM_ARTIFACTS_DIR=/path/to/artifacts
       export PRISM_REGISTRY_DIR=/path/to/registry
       export PRISM_ENSEMBLE_VERSION=latest
       uvicorn ...api.main:app --port 8000

2. **Programmatic** — for script-based launch::

       from api.config import Settings, set_settings
       settings = Settings(
           artifacts_dir="/path/to/artifacts",
           registry_dir="/path/to/registry",
           ensemble_version="latest",
       )
       set_settings(settings)

The ``eval_dir`` is **resolved automatically** from ``artifacts_dir``
and ``ensemble_version`` using the same path convention as the eval
pipeline: ``{artifacts_dir}/ensemble/{version}/evaluation/``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """PRISM API configuration.

    Environment variables are prefixed with ``PRISM_`` and are
    case-insensitive.  For example, ``PRISM_ARTIFACTS_DIR`` maps to
    ``artifacts_dir``.
    """

    # ── Required ──────────────────────────────────────────────────
    artifacts_dir: str
    registry_dir: str

    # ── Version (resolves "latest" automatically) ─────────────────
    ensemble_version: str = "latest"

    # ── Server ────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]
    log_level: str = "info"

    # ── Resolved paths ────────────────────────────────────────────

    @property
    def registry_path(self) -> Path:
        return Path(self.registry_dir)

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def resolved_version(self) -> str:
        """Resolve tags like ``"latest"`` via the ensemble registry index."""
        return _resolve_version(
            self.ensemble_version, self.registry_path, self.artifacts_path,
        )

    @property
    def eval_path(self) -> Path:
        """``{artifacts_dir}/ensemble/{resolved_version}/evaluation/``"""
        return (
            self.artifacts_path / "ensemble" / self.resolved_version / "evaluation"
        )

    model_config = {"env_prefix": "PRISM_", "env_file": ".env"}


# ── Singleton management ──────────────────────────────────────────

_settings: Optional[Settings] = None


def set_settings(settings: Settings) -> None:
    """Inject settings programmatically (before app startup)."""
    global _settings
    _settings = settings


def get_settings() -> Settings:
    """Return the active settings (env-based or programmatic)."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings


# ── Version resolution helper ─────────────────────────────────────

def _resolve_version(
    version_or_tag: str,
    registry_dir: Path,
    artifacts_dir: Path,
) -> str:
    """Resolve a version string or tag to an actual version directory name.

    Uses the same ``index.json`` lookup as
    :class:`~src.rade_ml_pt.ensemble.registry.EnsembleRegistry`:

    1. Read ``{registry_dir}/ensemble/index.json``.
    2. If *version_or_tag* is a key in the index, return its value.
    3. Otherwise check if a directory with that name exists.
    4. Raise if nothing matches.
    """
    ens_reg = registry_dir / "ensemble"
    index_path = ens_reg / "index.json"

    # Step 1: check the registry index (handles "latest" and custom tags)
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if version_or_tag in index:
            return index[version_or_tag]

    # Step 2: direct version directory in registry
    if (ens_reg / version_or_tag).is_dir():
        return version_or_tag

    # Step 3: direct version directory in artifacts
    ens_art = artifacts_dir / "ensemble"
    if (ens_art / version_or_tag).is_dir():
        return version_or_tag

    available = []
    if index_path.exists():
        available = list(json.loads(index_path.read_text()).keys())

    raise FileNotFoundError(
        f"Cannot resolve version '{version_or_tag}'. "
        f"Not found in index.json or as a directory. "
        f"Available tags: {available}"
    )
