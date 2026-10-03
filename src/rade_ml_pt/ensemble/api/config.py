"""PRISM API settings.

Two launch modes are supported:

1. **Env vars / .env** — for ``uvicorn`` CLI launch::

       export PRISM_ARTIFACTS_DIR=/path/to/artifacts
       export PRISM_REGISTRY_DIR=/path/to/registry
       export PRISM_ENSEMBLE_VERSION=latest
       uvicorn src.rade_ml_pt.ensemble.api.app:get_app --factory --port 8000

2. **Programmatic** — for script-based launch::

       from src.rade_ml_pt.ensemble.api.config import Settings, set_settings
       set_settings(Settings(
           artifacts_dir="/path/to/artifacts",
           registry_dir="/path/to/registry",
           ensemble_version="latest",
       ))
       from src.rade_ml_pt.ensemble.api.app import create_app
       app = create_app()

The server is scoped to a single ensemble version resolved at startup;
to serve a different version, restart the server (or point at a
different artifacts tree).  ``"latest"`` is re-resolved on every start.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings

from src.rade_ml_pt.ensemble.api.services.version import resolve_version


class Settings(BaseSettings):
    """PRISM API configuration.

    Environment variables use the ``PRISM_`` prefix and are
    case-insensitive (``PRISM_ARTIFACTS_DIR`` → ``artifacts_dir``).
    """

    # ── Required ──────────────────────────────────────────────────
    artifacts_dir: str
    registry_dir: str

    # ── Version (resolves "latest" on startup) ────────────────────
    ensemble_version: str = "latest"

    # ── Server ────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]
    log_level: str = "info"

    # ── Resolved paths ────────────────────────────────────────────
    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def registry_path(self) -> Path:
        return Path(self.registry_dir)

    @property
    def resolved_version(self) -> str:
        """Concrete version directory name (e.g. ``"20260417_143022"``)."""
        return resolve_version(
            self.ensemble_version, self.registry_path, self.artifacts_path,
        )

    model_config = {"env_prefix": "PRISM_", "env_file": ".env"}


# ── Singleton management ──────────────────────────────────────────

_settings: Optional[Settings] = None


def set_settings(settings: Settings) -> None:
    """Inject settings programmatically before app startup."""
    global _settings
    _settings = settings


def get_settings() -> Settings:
    """Return the active settings — loads from env vars on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
