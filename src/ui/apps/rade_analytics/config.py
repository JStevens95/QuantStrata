"""Application settings for rade_analytics.

Environment-driven via pydantic-settings so the same code runs unchanged
in dev (localhost API), CI (mocked fixture server), and prod.  Prefix
all env vars with ``RADE_UI_`` to disambiguate from the FastAPI server's
``PRISM_*`` vars.

Example::

    RADE_UI_API_URL=http://localhost:8000 \\
    RADE_UI_CACHE_DEFAULT_TIMEOUT=120 \\
    python -m src.ui.apps.rade_analytics.app
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings


class RadeUiSettings(BaseSettings):
    """Runtime settings for the Rade Analytics Dash app."""

    # ── API wiring ────────────────────────────────────────────────
    api_url: str = "http://localhost:8000"
    api_timeout_s: float = 30.0
    api_http2: bool = False

    # ── Dash server ───────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False
    log_level: str = "info"

    # ── Cache policy ──────────────────────────────────────────────
    # ``SimpleCache`` for dev, ``RedisCache`` in prod.  See
    # flask_caching docs for the full list.  Everything prefixed
    # ``cache_`` is passed straight into ``Cache(config=...)``.
    cache_type: Literal["SimpleCache", "NullCache", "RedisCache"] = "SimpleCache"
    cache_default_timeout: int = 300
    cache_redis_url: Optional[str] = None

    # ── Branding / UX toggles ─────────────────────────────────────
    theme: Literal["dark", "light"] = "dark"
    show_version_picker: bool = True

    model_config = {
        "env_prefix": "RADE_UI_",
        "env_file": ".env",
        "extra": "ignore",
    }

    def cache_config(self) -> dict:
        """Translate settings into a ``flask_caching.Cache`` config dict."""
        cfg: dict = {
            "CACHE_TYPE": self.cache_type,
            "CACHE_DEFAULT_TIMEOUT": self.cache_default_timeout,
        }
        if self.cache_type == "RedisCache" and self.cache_redis_url:
            cfg["CACHE_REDIS_URL"] = self.cache_redis_url
        return cfg


_settings: Optional[RadeUiSettings] = None


def set_settings(settings: RadeUiSettings) -> None:
    """Override the process-wide settings (used by the app factory)."""
    global _settings
    _settings = settings


def get_settings() -> RadeUiSettings:
    """Lazy accessor — env vars are read on first access."""
    global _settings
    if _settings is None:
        _settings = RadeUiSettings()
    return _settings
