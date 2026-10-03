"""End-to-end smoke test for the Overview page with *no* real backend.

Phase D.3 ships the live callback that wires the overview to
:class:`RadeBackend`.  This script runs the **full** Rade app (router,
shell, callbacks, session store, split toggle) against a
:class:`MockRadeBackend` so you can click around the overview, toggle
splits, and verify every slot re-renders without the FastAPI server
running.

The mock now lives in a shared helper (``_mock_backend.py``) so the
Evaluation-Portfolio preview (06_…) and every future preview script
drive against the same synthetic dataset.

Run from the project root::

    python examples/rade_analytics/05_overview_preview_live.py

Then open http://localhost:8052.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── React 18 pin ────────────────────────────────────────────────────
# Must happen *before* ``dash.Dash`` is imported.
import dash._dash_renderer  # noqa: E402
dash._dash_renderer._set_react_version("18.2.0")

# Make ``_mock_backend`` importable via sibling-file lookup.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dash import Dash  # noqa: E402

from _mock_backend import MockRadeBackend  # noqa: E402  (sibling import)
from src.ui.apps.rade_analytics.callbacks import register_all  # noqa: E402
from src.ui.apps.rade_analytics.config import RadeUiSettings, set_settings  # noqa: E402
from src.ui.apps.rade_analytics.layouts import (  # noqa: E402
    INDEX_STRING,
    META_TAGS,
    build_shell,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rade.preview.overview_live")


def build_preview_app() -> Dash:
    """Compose a fully-wired Rade app against :class:`MockRadeBackend`."""
    settings = RadeUiSettings(
        api_url="http://mock",
        cache_type="NullCache",  # synthetic data is already deterministic
        debug=True,
    )
    set_settings(settings)

    assets_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "ui" / "apps" / "rade_analytics" / "assets"
    )

    app = Dash(
        __name__,
        title="Rade — Overview live preview (mock)",
        update_title=None,
        index_string=INDEX_STRING,
        meta_tags=META_TAGS,
        assets_folder=str(assets_dir),
        assets_ignore=r"tailwind\.(config\.js|input\.css)|README\.md",
        suppress_callback_exceptions=True,
    )

    backend = MockRadeBackend()
    app.server.config["rade_backend"] = backend
    app.server.config["rade_settings"] = settings

    app.layout = build_shell()
    register_all(app, backend)

    log.info("Preview ready — landing on / with MockRadeBackend")
    return app


if __name__ == "__main__":
    build_preview_app().run(
        debug=True,
        host="0.0.0.0",
        port=8052,
    )
