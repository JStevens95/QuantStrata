"""Visual-only preview of the Rade Evaluation page (Phase E.0 skeleton).

Run this without an API, without the full router, without the rest of
the app wiring — purely to inspect the Evaluation shell (topbar + real
sidebar + global filter bar + 4 sub-tabs) while iterating visuals.

What this script DOES
---------------------
* Mounts the real sidebar (``build_sidebar_content``) and the real
  refactored topbar (``build_topbar``) inside ``dmc.AppShell`` so the
  chrome matches the full app.
* Slots the real :func:`build_evaluation` tree into ``AppShellMain``
  (filter bar + sub-tab pills + stub content).
* Populates the filter bar dropdowns with plausible mock options so
  interactive collapse / chip / clear-all behaviour can be exercised.
* Wires a **minimal subset** of the Evaluation callbacks locally — just
  enough for the collapse toggle and filter chip rendering to work
  without needing ``dcc.Location``, ``session-store`` or the top-level
  router.

What this script does NOT do
----------------------------
* No URL-driven sub-tab switching — clicking a tab does nothing here.
  (In the full app, ``evaluation_cb._sync_from_url`` handles this; it
  requires ``dcc.Location`` which we omit.)
* No real backend — filter dropdown options are hardcoded.
* No session persistence — filters reset every refresh.

For the end-to-end flow use the real app factory::

    python -m src.ui.apps.rade_analytics.app

Usage
-----
From the project root::

    python examples/rade_analytics/04_evaluation_preview.py

Then open http://localhost:8052 (different port to splash / overview
previews so all three can run side-by-side).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── React 18 pin ────────────────────────────────────────────────────
# ``dash-mantine-components`` ≥ 0.14 relies on the React 18 ``useId``
# hook.  Dash 2.x ships React 16.14 by default, causing the
# "(0, r.UseId) is not a function" runtime error.  This must run
# *before* ``dash.Dash`` is imported / constructed.
import dash._dash_renderer  # noqa: E402
dash._dash_renderer._set_react_version("18.2.0")

import dash_mantine_components as dmc  # noqa: E402
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update  # noqa: E402
from dash.exceptions import PreventUpdate  # noqa: E402

from src.ui.apps.rade_analytics.components.evaluation_filter_bar import (  # noqa: E402
    EVAL_FILTER_IDS,
    render_filter_chips,
)
from src.ui.apps.rade_analytics.components.sidebar import build_sidebar_content  # noqa: E402
from src.ui.apps.rade_analytics.components.topbar import (  # noqa: E402
    TOPBAR_IDS,
    build_topbar,
)
from src.ui.apps.rade_analytics.data.session import EvaluationFilters  # noqa: E402
from src.ui.apps.rade_analytics.layouts.evaluation import build_evaluation  # noqa: E402
from src.ui.apps.rade_analytics.layouts.head import INDEX_STRING, META_TAGS  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Mock data — tweak these to preview different scenarios
# ─────────────────────────────────────────────────────────────────────

_MOCK_VERSIONS = [
    "v2026.04.17-a1b2c",
    "v2026.04.10-f3e4d7",
    "v2026.04.03-9876a2",
]
_MOCK_ACTIVE_VERSION = _MOCK_VERSIONS[0]


def _options(values: List[str]) -> List[Dict[str, str]]:
    return [{"label": v, "value": v} for v in values]


_MOCK_ASSET_CLASSES = _options(["Rates", "FX", "Credit", "Commodities", "Equities"])
_MOCK_CURRENCIES    = _options(["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "ZAR"])
_MOCK_DESKS         = _options(["EUR Rates", "USD Rates", "G10 FX", "EM Macro", "IG Credit"])
_MOCK_PRODUCTS      = _options([
    "IR Swap", "XCCY Swap", "FX Forward", "FX Swap", "IR Swaption",
    "Bond Future", "Credit Default Swap",
])


# ─────────────────────────────────────────────────────────────────────
# Dash app
# ─────────────────────────────────────────────────────────────────────

ASSETS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src" / "ui" / "apps" / "rade_analytics" / "assets"
)

app = Dash(
    __name__,
    title="Rade — Evaluation preview",
    update_title=None,
    meta_tags=META_TAGS,
    index_string=INDEX_STRING,
    assets_folder=str(ASSETS_DIR),
    suppress_callback_exceptions=True,
)


def _preview_shell() -> dmc.MantineProvider:
    """Outer tree — MantineProvider + AppShell with evaluation slotted in.

    Deliberately omits ``dcc.Location`` / session / toast stores so we
    don't pull in the router.  A single ``dcc.Interval`` drives the
    one-shot topbar bootstrap, exactly mirroring the overview preview.
    """
    return dmc.MantineProvider(
        forceColorScheme="dark",
        children=[
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
                        children=build_evaluation("/evaluation/portfolio"),
                        className="bg-slate-950",
                    ),
                ],
            ),
        ],
    )


app.layout = _preview_shell()


# ─────────────────────────────────────────────────────────────────────
# Mock bootstrap — populate topbar + filter dropdowns on first tick
# ─────────────────────────────────────────────────────────────────────


@app.callback(
    Output(TOPBAR_IDS["version_select"],   "data"),
    Output(TOPBAR_IDS["version_select"],   "value"),
    Output(EVAL_FILTER_IDS["asset_class"], "data"),
    Output(EVAL_FILTER_IDS["currency"],    "data"),
    Output(EVAL_FILTER_IDS["desk"],        "data"),
    Output(EVAL_FILTER_IDS["product"],     "data"),
    Input("preview-bootstrap-interval", "n_intervals"),
)
def _bootstrap(n_intervals: Optional[int]):
    """Fill every dropdown so the preview isn't stuck on placeholders."""
    if not n_intervals:
        raise PreventUpdate
    return (
        [{"label": v, "value": v} for v in _MOCK_VERSIONS],
        _MOCK_ACTIVE_VERSION,
        _MOCK_ASSET_CLASSES,
        _MOCK_CURRENCIES,
        _MOCK_DESKS,
        _MOCK_PRODUCTS,
    )


# ─────────────────────────────────────────────────────────────────────
# Local filter-bar callbacks
#
# These mimic ``callbacks/evaluation_cb.py`` but without session-store
# (which this preview intentionally omits).  The state lives entirely
# in the DOM; refresh = reset.
# ─────────────────────────────────────────────────────────────────────


@app.callback(
    Output(EVAL_FILTER_IDS["collapse"],   "opened"),
    Input(EVAL_FILTER_IDS["toggle_btn"],  "n_clicks"),
    State(EVAL_FILTER_IDS["collapse"],    "opened"),
    prevent_initial_call=True,
)
def _toggle_drawer(n_clicks: Optional[int], currently_open: Optional[bool]) -> bool:
    if not n_clicks:
        raise PreventUpdate
    return not bool(currently_open)


@app.callback(
    Output(EVAL_FILTER_IDS["chips"],        "children"),
    Output(EVAL_FILTER_IDS["toggle_label"], "children"),
    Output(EVAL_FILTER_IDS["clear_all"],    "style"),
    Input(EVAL_FILTER_IDS["asset_class"],   "value"),
    Input(EVAL_FILTER_IDS["currency"],      "value"),
    Input(EVAL_FILTER_IDS["desk"],          "value"),
    Input(EVAL_FILTER_IDS["product"],       "value"),
    Input(EVAL_FILTER_IDS["date_range"],    "value"),
)
def _render_chips(
    asset_class: Optional[List[str]],
    currency:    Optional[List[str]],
    desk:        Optional[List[str]],
    product:     Optional[List[str]],
    date_range:  Optional[List[Optional[str]]],
) -> Tuple[List[Any], str, Dict[str, Any]]:
    df = (date_range or [None, None])[0] if date_range else None
    dt = (date_range or [None, None])[1] if date_range and len(date_range) > 1 else None

    filters = EvaluationFilters(
        asset_class=list(asset_class or []),
        currency=list(currency or []),
        desk=list(desk or []),
        product=list(product or []),
        date_from=df,
        date_to=dt,
    )
    chips = render_filter_chips(filters.to_dict())
    count = filters.active_chip_count()
    label = f"{count} active" if count else ""
    style = {} if count else {"display": "none"}
    return chips, label, style


@app.callback(
    Output(EVAL_FILTER_IDS["asset_class"], "value", allow_duplicate=True),
    Output(EVAL_FILTER_IDS["currency"],    "value", allow_duplicate=True),
    Output(EVAL_FILTER_IDS["desk"],        "value", allow_duplicate=True),
    Output(EVAL_FILTER_IDS["product"],     "value", allow_duplicate=True),
    Output(EVAL_FILTER_IDS["date_range"],  "value", allow_duplicate=True),
    Input({"type": "eval-filter-chip-close", "dimension": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _clear_single_filter(n_clicks_list: List[Optional[int]]):
    if not any(n_clicks_list or []):
        raise PreventUpdate
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict):
        raise PreventUpdate

    dim = triggered.get("dimension")
    mapping = {
        "asset_class": (0, []),
        "currency":    (1, []),
        "desk":        (2, []),
        "product":     (3, []),
        "date":        (4, [None, None]),
    }
    if dim not in mapping:
        raise PreventUpdate

    idx, empty_value = mapping[dim]
    updates: List[Any] = [no_update] * 5
    updates[idx] = empty_value
    return tuple(updates)


@app.callback(
    Output(EVAL_FILTER_IDS["asset_class"], "value", allow_duplicate=True),
    Output(EVAL_FILTER_IDS["currency"],    "value", allow_duplicate=True),
    Output(EVAL_FILTER_IDS["desk"],        "value", allow_duplicate=True),
    Output(EVAL_FILTER_IDS["product"],     "value", allow_duplicate=True),
    Output(EVAL_FILTER_IDS["date_range"],  "value", allow_duplicate=True),
    Input(EVAL_FILTER_IDS["clear_all"],    "n_clicks"),
    Input(EVAL_FILTER_IDS["reset_btn"],    "n_clicks"),
    prevent_initial_call=True,
)
def _clear_all(clear_clicks: Optional[int], reset_clicks: Optional[int]):
    if not (clear_clicks or reset_clicks):
        raise PreventUpdate
    return [], [], [], [], [None, None]


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Port 8052 so this runs alongside splash (8050) and overview
    # (8051) previews without clashing.
    app.run(debug=True, host="0.0.0.0", port=8052)
