"""Dash app factory — the single entry point for the Rade UI.

Composes every layer B.1–B.5 has built so far:

* Settings                (:mod:`.config`)
* Typed API client        (:mod:`src.rade_ml_pt.ensemble.api.client`)
* Backend cache + adapter (:mod:`.data.backend`)
* App shell + head        (:mod:`.layouts`)
* Callbacks + router      (:mod:`.callbacks` / :mod:`.router`)

Usage
-----
.. code-block:: python

    from src.ui.apps.rade_analytics.app import create_app

    app = create_app()
    app.run(debug=True)

    # WSGI
    server = create_app().server

Run directly for local development::

    python -m src.ui.apps.rade_analytics.app

Environment variables (prefixed ``RADE_UI_``) override config defaults
— see :class:`RadeUiSettings`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# ── React 18 pin ────────────────────────────────────────────────────
# ``dash-mantine-components`` ≥ 0.14 requires the React 18 ``useId``
# hook.  Dash 2.x ships React 16.14 by default, which surfaces as the
# browser-side "(0, r.UseId) is not a function" runtime error when any
# DMC component renders.  This call *must* run before ``dash.Dash`` is
# constructed, so we do it at import time.
import dash._dash_renderer  # noqa: E402
dash._dash_renderer._set_react_version("18.2.0")

from dash import Dash  # noqa: E402
from flask_caching import Cache  # noqa: E402

from src.rade_ml_pt.ensemble.api.client import RadeApiClient  # noqa: E402

from .callbacks import register_all  # noqa: E402
from .config import RadeUiSettings, set_settings  # noqa: E402
from .data.backend import NoOpCache, RadeBackend  # noqa: E402
from .layouts import INDEX_STRING, META_TAGS, build_root  # noqa: E402


log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# Files in ``assets/`` Dash should NOT auto-serve.  ``tailwind.*`` are
# reference sources kept alongside the compiled ``rade.css`` so a
# future edit doesn't lose context; ``README.md`` is reviewer
# documentation.  Everything else (``rade.css``, ``logo.svg``,
# ``favicon.svg``) is served as-is.
_ASSETS_IGNORE = r"tailwind\.(config\.js|input\.css)|README\.md"

_APP_TITLE = "Rade Analytics"


# ─────────────────────────────────────────────────────────────────────
# Internal builders
# ─────────────────────────────────────────────────────────────────────


def _build_cache(settings: RadeUiSettings, server: object):
    """Return the cache object :class:`RadeBackend` will use.

    Chooses between :class:`NoOpCache` (``cache_type="NullCache"``) and
    a real :class:`flask_caching.Cache` bound to ``server`` (any other
    cache type).  Both satisfy the ``CacheLike`` protocol.
    """
    if settings.cache_type == "NullCache":
        log.info("Rade UI cache: disabled (NoOpCache)")
        return NoOpCache()

    log.info(
        "Rade UI cache: %s (timeout=%ss)",
        settings.cache_type,
        settings.cache_default_timeout,
    )
    cache = Cache(config=settings.cache_config())
    cache.init_app(server)
    return cache


def _build_backend(settings: RadeUiSettings, server: object) -> RadeBackend:
    """Instantiate the typed API client + :class:`RadeBackend`."""
    client = RadeApiClient(
        base_url=settings.api_url,
        timeout=settings.api_timeout_s,
        http2=settings.api_http2,
    )
    cache = _build_cache(settings, server)
    return RadeBackend(
        client=client,
        cache=cache,
        default_timeout_s=settings.cache_default_timeout,
    )


# ─────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────


def create_app(settings: Optional[RadeUiSettings] = None) -> Dash:
    """Build a fully wired Dash app.

    Parameters
    ----------
    settings
        Optional pre-built :class:`RadeUiSettings`.  If ``None``, one
        is constructed from environment variables (``RADE_UI_*``) and
        defaults.

    Returns
    -------
    Dash
        A Dash app with ``layout`` set, callbacks registered and the
        :class:`RadeBackend` available under
        ``app.server.config["rade_backend"]`` for ad-hoc introspection
        (callbacks hold a reference via closure — use the config slot
        only for tests / REPL poking).
    """
    cfg = settings or RadeUiSettings()
    set_settings(cfg)

    assets_dir = Path(__file__).resolve().parent / "assets"

    app = Dash(
        __name__,
        title=_APP_TITLE,
        update_title=None,
        index_string=INDEX_STRING,
        meta_tags=META_TAGS,
        assets_folder=str(assets_dir),
        assets_ignore=_ASSETS_IGNORE,
        suppress_callback_exceptions=True,
    )

    backend = _build_backend(cfg, app.server)
    app.server.config["rade_backend"] = backend
    app.server.config["rade_settings"] = cfg

    app.layout = build_root()

    register_all(app, backend)

    log.info(
        "Rade UI initialised — api=%s cache=%s debug=%s",
        cfg.api_url,
        cfg.cache_type,
        cfg.debug,
    )
    return app


# ─────────────────────────────────────────────────────────────────────
# ``python -m src.ui.apps.rade_analytics.app`` entry point
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Local dev entry — honours ``RADE_UI_HOST`` / ``RADE_UI_PORT``."""
    cfg = RadeUiSettings()
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app(cfg)
    app.run(
        host=cfg.host,
        port=cfg.port,
        debug=cfg.debug,
    )


if __name__ == "__main__":
    main()


__all__ = ["create_app", "main"]
