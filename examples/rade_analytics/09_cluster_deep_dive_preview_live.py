"""End-to-end smoke test for the Evaluation → Cluster Deep-Dive sub-tab.

Phase E.4 ships the full Cluster Deep-Dive page (Row 2 revision):

* Sticky header with cluster picker + attribute chips (no KPI strip —
  KPIs now live in Row 2).
* Row 2 — 2×2 KPI grid (MAE, RMSE, R², Coverage) on the left, training
  curves chart on the right with a chip-row metric picker.  Coverage
  uses a ``|error| ≤ MAE`` tolerance; R² is computed client-side from
  the cluster timeseries.
* Row 3 — residual-over-time (rolling mean + ±1σ band) and predicted-
  vs-target-PnL with a rose shaded error band.
* Row 4 — per-trade residual violin (target / elementary) and per-trade
  scatter (mean_residual × MAE, coloured by trade type).  Clicking a
  scatter point highlights the corresponding trade id; the focus chip
  in the header clears it.
* Row 5 — trades AgGrid.  Row click also updates the selected trade,
  which the scatter re-renders to emerald-ring the chosen point.
* Header "Trade-Graph" button navigates to the Trade-Graph sub-tab
  with the current cluster pinned; the reciprocal "Open in Cluster
  Deep Dive" button on the Trade-Graph tab navigates back here,
  priming the cluster + selected trade.

Run from the project root::

    python examples/rade_analytics/09_cluster_deep_dive_preview_live.py

Then open http://localhost:8055/evaluation/cluster.

You can also deep-link into a specific cluster via ``?cid=``::

    http://localhost:8055/evaluation/cluster?cid=cluster_04
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── React 18 pin — must happen *before* ``dash.Dash`` is imported.
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
log = logging.getLogger("rade.preview.cluster_deep_dive_live")


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
        title="Rade — Cluster Deep-Dive live preview (mock)",
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
        "Preview ready — open http://localhost:8055/evaluation/cluster"
    )
    return app


if __name__ == "__main__":
    build_preview_app().run(
        debug=True,
        host="0.0.0.0",
        port=8055,
    )
