"""Visual-only preview of the Rade Overview / landing page.

Run this without an API, without the router, without the rest of the
app wiring — purely to inspect the Overview layout (topbar breadcrumb
+ split toggle, KPI strip, portfolio PnL chart, cluster health
heatmap, insights row, quick-actions footer) while iterating visuals.

What this script DOES
---------------------
* Mounts the real sidebar (``build_sidebar_content``) and the real
  refactored topbar (``build_topbar``) inside ``dmc.AppShell`` so you
  see exactly what the shell will look like in the full app.
* Slots :func:`build_overview` straight into ``AppShellMain`` — no
  router, no content callback, no ``dcc.Location``.
* Populates the topbar version ``Select`` with mock version strings on
  page load so the dropdown isn't stuck on its placeholder.

What this script does NOT do
----------------------------
* No navigation — sidebar items are real links but there's nowhere to
  navigate to, so clicks reload the preview page.
* No real data — every KPI / chart / list row uses the hardcoded
  placeholders baked into ``layouts/overview.py``.  The Phase D.3
  callback will overwrite all of it once real data is wired in.
* No session store — the split toggle defaults to ``"test"`` and has
  no persistence.

For the end-to-end flow (splash → enter → overview → real data) use
the real app factory::

    python -m src.ui.apps.rade_analytics.app

Usage
-----
From the project root::

    python examples/rade_analytics/03_overview_preview.py

Then open http://localhost:8051 (different port to the splash preview
so both can run side-by-side).
"""
from __future__ import annotations

from pathlib import Path


# ── React 18 pin ────────────────────────────────────────────────────
# ``dash-mantine-components`` ≥ 0.14 relies on the React 18 ``useId``
# hook.  Dash 2.x ships React 16.14 by default, causing the
# "(0, r.UseId) is not a function" runtime error.  This must run
# *before* ``dash.Dash`` is imported / constructed.
import dash._dash_renderer  # noqa: E402
dash._dash_renderer._set_react_version("18.2.0")

import dash_mantine_components as dmc  # noqa: E402
from dash import Dash, Input, Output, dcc, html  # noqa: E402
from dash.exceptions import PreventUpdate  # noqa: E402

from src.ui.apps.rade_analytics.components.sidebar import build_sidebar_content  # noqa: E402
from src.ui.apps.rade_analytics.components.topbar import TOPBAR_IDS, build_topbar  # noqa: E402
from src.ui.apps.rade_analytics.layouts.head import INDEX_STRING, META_TAGS  # noqa: E402
from src.ui.apps.rade_analytics.layouts.overview import build_overview  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Mock data — tweak these to preview different scenarios
# ─────────────────────────────────────────────────────────────────────

_MOCK_VERSIONS = [
    "v2026.04.17-a1b2c",
    "v2026.04.10-f3e4d7",
    "v2026.04.03-9876a2",
    "v2026.03.27-112233",
]
_MOCK_ACTIVE_VERSION = _MOCK_VERSIONS[0]


# ─────────────────────────────────────────────────────────────────────
# Dash app
# ─────────────────────────────────────────────────────────────────────

ASSETS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src" / "ui" / "apps" / "rade_analytics" / "assets"
)

app = Dash(
    __name__,
    title="Rade — Overview preview",
    update_title=None,
    meta_tags=META_TAGS,
    index_string=INDEX_STRING,
    assets_folder=str(ASSETS_DIR),
    # Don't pre-register DMC CSS — we rely on rade.css only.
    suppress_callback_exceptions=True,
)


def _preview_shell() -> dmc.MantineProvider:
    """Outer tree — MantineProvider + AppShell with overview slotted in.

    Deliberately omits ``dcc.Location`` and the session / toast stores
    so we don't pull in router / callback machinery that isn't relevant
    at the skeleton-review stage.
    """
    return dmc.MantineProvider(
        forceColorScheme="dark",
        children=[
            # Single Interval to fire the one-shot "populate mocks"
            # callback below — exactly mirrors the pattern in
            # 02_splash_preview.py.
            dcc.Interval(
                id="preview-bootstrap-interval",
                interval=1,
                max_intervals=1,
            ),
            dmc.AppShell(
                header={"height": 60},
                navbar={"width": 220, "breakpoint": "sm"},
                padding=0,
                children=[
                    dmc.AppShellHeader(
                        children=build_topbar(),
                        className=(
                            "bg-slate-950/80 border-b border-slate-800 "
                            "backdrop-blur-sm"
                        ),
                    ),
                    dmc.AppShellNavbar(
                        children=build_sidebar_content(),
                        className="bg-slate-900 border-r border-slate-800",
                    ),
                    dmc.AppShellMain(
                        children=build_overview(),
                        className="bg-slate-950",
                    ),
                ],
            ),
        ],
    )


app.layout = _preview_shell()


# ─────────────────────────────────────────────────────────────────────
# Mock bootstrap — populate dynamic topbar slots on first tick
# ─────────────────────────────────────────────────────────────────────


@app.callback(
    Output(TOPBAR_IDS["version_select"], "data"),
    Output(TOPBAR_IDS["version_select"], "value"),
    Input("preview-bootstrap-interval", "n_intervals"),
)
def _bootstrap_topbar(n_intervals: int | None):
    """Fill the version ``Select`` so the dropdown shows real-looking data.

    Fires exactly once (``max_intervals=1`` on the ``dcc.Interval``) so
    the preview isn't stuck at the default "version" placeholder.
    """
    if not n_intervals:
        raise PreventUpdate

    data = [{"label": v, "value": v} for v in _MOCK_VERSIONS]
    return data, _MOCK_ACTIVE_VERSION


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Port 8051 so this can run alongside the real app on 8050 and the
    # splash preview on 8050 without clashing.
    app.run(debug=True, host="0.0.0.0", port=8051)
