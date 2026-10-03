"""End-to-end smoke test for the Evaluation → Trade-Graph sub-tab.

Phase E.3 ships the Cytoscape network, the side-panel cards (Selected
Trade, Legend, Cluster stats, Ensemble summary) and the two secondary
charts (density histogram + edges vs nodes scatter).  This script runs
the **full** Rade app against :class:`MockRadeBackend` so you can:

* Navigate to ``/evaluation/trade-graph``.
* Switch clusters from the header band and see the graph re-render.
* Tap a node and watch the Selected-Trade card fill (trade id +
  cluster badge + attribute list + deep-dive button enables).
* Drag the "Min weight" slider and see low-weight edges disappear
  without a round-trip (threshold filter is client-side on the
  stored nodes/edges payload).
* Toggle the layout (``cose``, ``concentric``, …) and confirm the
  Cytoscape component re-runs the layout.
* Eyeball the density histogram and edges-vs-nodes scatter — the
  selected cluster should light up in both.

Run from the project root::

    python examples/rade_analytics/08_trade_graph_preview_live.py

Then open http://localhost:8054.  The 8050-series ports run the real
app; 8052+ are all mock-backed preview scripts (05, 06, 08, …) so they
can coexist on the same machine.
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
log = logging.getLogger("rade.preview.trade_graph_live")


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
        title="Rade — Trade-Graph live preview (mock)",
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
        "Preview ready — open http://localhost:8054/evaluation/trade-graph"
    )
    return app


if __name__ == "__main__":
    build_preview_app().run(
        debug=True,
        host="0.0.0.0",
        port=8054,
    )
