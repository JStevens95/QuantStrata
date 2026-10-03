"""End-to-end smoke test for the Evaluation → Portfolio sub-tab.

Phase E.1 ships the full Portfolio callback suite (group-by Select,
violin / scatter render, leaderboard, click-to-focus on the scatter).
This script runs the **full** Rade app against :class:`MockRadeBackend`
so you can:

* Navigate to ``/evaluation/portfolio``.
* Toggle the split from the topbar and watch rows 1-3 redraw.
* Pick a break-down dimension (Desk, Product, Currency, Asset class,
  Cluster) and watch rows 4-5 populate.
* Click a point on the grouped scatter to **focus** on that category;
  double-click the plot (or click "× Show all") to clear focus.
* Apply filter-bar filters and confirm every row honours them.

Run from the project root::

    python examples/rade_analytics/06_portfolio_preview_live.py

Then open http://localhost:8053.  The 8050-series ports run the real
app; 8052+ are all mock-backed preview scripts (05, 06, …) so they can
coexist.
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
log = logging.getLogger("rade.preview.portfolio_live")


def build_preview_app() -> Dash:
    """Compose a fully-wired Rade app against :class:`MockRadeBackend`."""
    settings = RadeUiSettings(
        api_url="http://mock",
        cache_type="NullCache",
        debug=True,
    )
    set_settings(settings)

    assets_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "ui" / "apps" / "rade_analytics" / "assets"
    )

    app = Dash(
        __name__,
        title="Rade — Portfolio live preview (mock)",
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

    log.info(
        "Preview ready — open http://localhost:8053/evaluation/portfolio "
        "(starts at / so use the sidebar or navigate directly)"
    )
    return app


if __name__ == "__main__":
    build_preview_app().run(
        debug=True,
        host="0.0.0.0",
        port=8053,
    )
