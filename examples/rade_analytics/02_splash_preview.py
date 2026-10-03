"""Visual-only preview of the Rade splash page.

Run this without an API, without the router, without the shell — purely
to inspect the splash layout (gradient backdrop, brand block, status
strip, version picker, CTA) while iterating on visuals.

A tiny ``dcc.Interval`` fires once on page load and populates the
dynamic slots with plausible mock data so the page renders in its
"live" state rather than the default "Connecting..." placeholders.

Clicking the **Enter Rade** button is intentionally a no-op here — no
navigation, no session write — because the rest of the app isn't
mounted.  For end-to-end testing of the splash flow, use the real app
factory instead::

    python -m src.ui.apps.rade_analytics.app

Usage
-----
From the project root::

    python examples/rade_analytics/02_splash_preview.py

Then open http://localhost:8050.
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

from src.ui.apps.rade_analytics.layouts.head import INDEX_STRING, META_TAGS  # noqa: E402
from src.ui.apps.rade_analytics.layouts.splash import SPLASH_IDS, build_splash  # noqa: E402


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
_MOCK_API_URL = "/tmp/rade_artifacts (preview)"


# ─────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────

# Point Dash at the real assets folder so rade.css, logo.svg and
# favicon.svg all load exactly as they do in production.
_ASSETS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "ui" / "apps" / "rade_analytics" / "assets"
)

app = Dash(
    __name__,
    title="Rade Splash Preview",
    update_title=None,
    index_string=INDEX_STRING,
    meta_tags=META_TAGS,
    assets_folder=str(_ASSETS_DIR),
    assets_ignore=r"tailwind\.(config\.js|input\.css)|README\.md",
)


app.layout = dmc.MantineProvider(
    forceColorScheme="dark",
    children=[
        # Fires once 100 ms after page load so we can flip the splash
        # out of its "Connecting..." state into the mock "Live" state.
        dcc.Interval(
            id="_preview_tick",
            interval=100,
            max_intervals=1,
        ),
        html.Div(
            className="min-h-screen bg-slate-950 text-slate-100",
            children=build_splash(),
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Mock bootstrap callback — matches the real splash bootstrap's output
# shape so the layout behaves identically to a live backend response.
# ─────────────────────────────────────────────────────────────────────

@app.callback(
    Output(SPLASH_IDS["status_dot"],     "className"),
    Output(SPLASH_IDS["status_label"],   "children"),
    Output(SPLASH_IDS["api_url"],        "children"),
    Output(SPLASH_IDS["active_version"], "children"),
    Output(SPLASH_IDS["version_select"], "data"),
    Output(SPLASH_IDS["version_select"], "value"),
    Output(SPLASH_IDS["version_select"], "disabled"),
    Output(SPLASH_IDS["enter_btn"],      "disabled"),
    Input("_preview_tick", "n_intervals"),
)
def _mock_bootstrap(n_intervals):
    if not n_intervals:
        raise PreventUpdate

    return (
        "rade-status-dot rade-status-dot--ok",         # status dot
        "Live",                                        # status label
        _MOCK_API_URL,                                 # api url
        _MOCK_ACTIVE_VERSION,                          # headline version
        [{"label": v, "value": v} for v in _MOCK_VERSIONS],
        _MOCK_ACTIVE_VERSION,                          # select value
        False,                                         # select enabled
        False,                                         # enter btn enabled
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
