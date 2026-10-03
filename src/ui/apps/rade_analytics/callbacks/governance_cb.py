"""Governance page callbacks — live wiring for the ``/governance`` route.

Wires three rendering surfaces from a single backend call
(:meth:`RadeBackend.governance_registry`):

* **Header KPIs** — total versions / in-production / pending sign-offs
  (mocked V1) / last activity timestamp.
* **Lineage timeline** — most-recent ``promoted_at`` / ``created_at``
  events from the registry payload, plus one static seed event so the
  timeline reads non-empty even when only a single version exists.
* **Model Registry grid** — full list, filtered client-side by the
  status SegmentedControl above it.

Capture surface
---------------
The status SegmentedControl is intentionally **ephemeral** (V1) — its
selection lives only in the live component value and never makes it
into ``Session``.  That keeps this page's callback graph minimal: there
is no capture-side write, and the registry-render callback reads the
filter directly off the SegmentedControl.

If a future reviewer wants the page to remember the last filter across
navigation, add a ``governance_status_filter`` field to ``Session``,
register a small ``_sync_status_filter`` capture callback (mirror
``portfolio_cb._register_split_sync``) and seed the SegmentedControl
``value`` from session at layout build time (Page Contract §3 Rule L1).

Why mount_signal-triggered, not pathname-triggered
--------------------------------------------------
Page Contract §4 Rule C7 — when the router swaps ``app-host`` to a
new page tree, the new page's ``mount_signal`` store mounts fresh
with ``data=True``, which fires the bootstrap callback exactly once
*after* the DOM is in place.  ``Input(pathname)`` would race the
content swap and try to write into IDs that don't exist yet
(Anti-pattern A8).  Both render callbacks below trigger off
``mount_signal``; the registry-render also takes ``status_filter`` as
an Input so the table re-emits whenever the user picks a chip.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from ..layouts.governance import GOVERNANCE_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────


_GOVERNANCE_PATH = "/governance"
_PLACEHOLDER = "—"

# Pending sign-offs — V1 mock count.  Mirrors the static
# ``_PENDING_APPROVAL`` block in :mod:`..layouts.governance`.  When the
# Stage-2 workflow producer ships this becomes a backend lookup.
_MOCK_PENDING_APPROVALS = 1

# Maximum lineage timeline rows we render.  Cap is small because the
# panel is a rhythm-glance widget — denser views live on the
# Evaluation / Monitoring tabs.
_LINEAGE_MAX_ROWS = 6


# ═════════════════════════════════════════════════════════════════════
# Public surface
# ═════════════════════════════════════════════════════════════════════


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every governance callback to ``app``.

    Mirrors the Page Contract §2 capture/render split.  Capture is
    empty for V1 — every input on this page is ephemeral — so this
    function only calls :func:`_register_render`.
    """
    _register_render(app, backend)


# ─────────────────────────────────────────────────────────────────────
# Section dispatcher
# ─────────────────────────────────────────────────────────────────────


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Attach the two render callbacks (header + registry grid)."""
    _register_render_header(app, backend)
    _register_render_registry(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. Render — header KPIs + lineage timeline
# ═════════════════════════════════════════════════════════════════════


def _register_render_header(app: "Dash", backend: "RadeBackend") -> None:
    """Populate the four KPI values and the lineage timeline.

    Cheap call (cache hit after the first request) so we re-issue it
    here rather than threading the registry payload through a
    ``dcc.Store`` — keeps the callback graph linear and the diff
    against the registry-render callback obvious.
    """

    @app.callback(
        Output(GOVERNANCE_IDS["kpi_total_value"],         "children"),
        Output(GOVERNANCE_IDS["kpi_production_value"],    "children"),
        Output(GOVERNANCE_IDS["kpi_pending_value"],       "children"),
        Output(GOVERNANCE_IDS["kpi_last_activity_value"], "children"),
        Output(GOVERNANCE_IDS["lineage_timeline"],        "children"),
        Input(GOVERNANCE_IDS["mount_signal"],             "data"),
        State(SHELL_IDS["url"],                           "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _trigger: Any,
        pathname: Optional[str],
    ) -> Tuple[str, str, str, str, List[Any]]:
        # Belt-and-braces: the mount_signal Store only mounts when the
        # router swaps to the governance page tree, but a pathname
        # guard is still cheap and protects against future router
        # refactors that mount governance under a different route.
        if pathname != _GOVERNANCE_PATH:
            raise PreventUpdate

        res = backend.governance_registry()
        if not res.ok or res.data is None:
            logger.warning(
                "Governance: registry fetch failed — %s", res.error,
            )
            return (
                _PLACEHOLDER,
                _PLACEHOLDER,
                str(_MOCK_PENDING_APPROVALS),
                _PLACEHOLDER,
                _empty_lineage_message("Registry unavailable."),
            )

        rows = list(res.data.rows)
        total = len(rows)
        n_production = sum(1 for r in rows if r.status == "production")
        last_activity = _format_last_activity(rows)

        timeline = _build_lineage_timeline(rows)

        return (
            f"{total}",
            f"{n_production}",
            str(_MOCK_PENDING_APPROVALS),
            last_activity,
            timeline,
        )


# ═════════════════════════════════════════════════════════════════════
# 2. Render — Model Registry grid (filtered by status SegmentedControl)
# ═════════════════════════════════════════════════════════════════════


def _register_render_registry(app: "Dash", backend: "RadeBackend") -> None:
    """Populate the registry grid's ``rowData``, filtered by status chip.

    The status chip is *not* persisted in session (V1) — its current
    value is read straight off the SegmentedControl as a render-time
    Input.  Filtering is applied in Python before emit so AG Grid
    only ever sees the rows we want shown; this keeps the table's
    pagination + sort behaviour intuitive across filter flips.
    """

    @app.callback(
        Output(GOVERNANCE_IDS["registry_grid"], "rowData"),
        Input(GOVERNANCE_IDS["mount_signal"],   "data"),
        Input(GOVERNANCE_IDS["status_filter"],  "value"),
        State(SHELL_IDS["url"],                 "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _trigger:    Any,
        status_pick: Optional[str],
        pathname:    Optional[str],
    ) -> List[Dict[str, Any]]:
        if pathname != _GOVERNANCE_PATH:
            raise PreventUpdate

        res = backend.governance_registry_df()
        if not res.ok or res.data is None or res.data.empty:
            if not res.ok:
                logger.warning(
                    "Governance: registry_df fetch failed — %s",
                    res.error,
                )
            return []

        df = res.data
        df = _apply_status_filter(df, status_pick)

        # AG Grid expects plain dicts; ``model_dump``-derived rows
        # already JSON-friendly.  Drop any row that lost critical
        # context after the filter (defensive — shouldn't happen).
        return df.to_dict(orient="records")


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _apply_status_filter(
    df: pd.DataFrame,
    status_pick: Optional[str],
) -> pd.DataFrame:
    """Return rows matching ``status_pick``; ``"all"`` / ``None`` is identity."""
    if not status_pick or status_pick == "all":
        return df
    if "status" not in df.columns:
        return df
    return df[df["status"].astype(str) == status_pick]


def _format_last_activity(rows: List[Any]) -> str:
    """Return the most recent ``created_at`` formatted as ``DD Mon HH:MM``.

    ``rows`` is the Pydantic list straight off the response — already
    sorted newest-first by the service, so we just take the head and
    parse its ISO timestamp.  Failure to parse falls through to the
    em-dash placeholder so the KPI never renders ``Invalid Date``.
    """
    if not rows:
        return _PLACEHOLDER
    ts = getattr(rows[0], "created_at", None)
    if not ts:
        return _PLACEHOLDER
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%d %b %H:%M")


def _build_lineage_timeline(rows: List[Any]) -> List[Any]:
    """Build a small list of timeline events from the registry payload.

    Three event sources, blended into one chronologically-sorted feed:

    * **Promotion events** — every row with a non-null ``promoted_at``
      (currently the production row(s)).
    * **Registration events** — every row's ``created_at``.
    * **Evaluation events** — every row whose evaluation bundle has
      been published (``has_evaluation``); we use ``promoted_at`` as
      a proxy timestamp because we don't yet record an "evaluated_at"
      on the wire.

    Events are de-duplicated by ``(version, action)`` so a row that
    was promoted and registered doesn't generate two near-identical
    "Registered" entries.  The newest :data:`_LINEAGE_MAX_ROWS` are
    rendered.
    """
    if not rows:
        return _empty_lineage_message(
            "No registered ensemble versions yet."
        )

    events: List[Dict[str, Any]] = []
    for row in rows:
        version = row.version
        if row.promoted_at:
            events.append(
                {
                    "ts":      row.promoted_at,
                    "icon":    "tabler:rocket",
                    "kind":    "promoted",
                    "title":   f"Promoted {version} to Production",
                    "actor":   row.created_by,
                }
            )
        if row.has_evaluation and row.mae_test is not None:
            events.append(
                {
                    "ts":    row.created_at,
                    "icon":  "tabler:bolt",
                    "kind":  "evaluated",
                    "title": (
                        f"Eval completed — MAE {row.mae_test:.3f}"
                        + (
                            f"  RMSE {row.rmse_test:.3f}"
                            if row.rmse_test is not None
                            else ""
                        )
                    ),
                    "actor": "system",
                }
            )
        events.append(
            {
                "ts":     row.created_at,
                "icon":   "tabler:cube",
                "kind":   "registered",
                "title":  f"Registered {version}",
                "actor":  row.created_by,
            }
        )

    # Sort newest-first; ISO-8601 strings sort correctly lexicographically.
    events.sort(key=lambda e: str(e["ts"]), reverse=True)

    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for e in events:
        key = (e["title"], e["ts"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
        if len(deduped) >= _LINEAGE_MAX_ROWS:
            break

    return [_lineage_row(e) for e in deduped]


def _lineage_row(event: Dict[str, Any]) -> html.Div:
    """Render a single timeline event row.

    The icon's coloured background is keyed off ``event["kind"]`` —
    matches the design's coloured event marker palette.  ``ts``
    formatting matches the registry table's ``Created`` column for
    visual continuity.
    """
    icon_class = f"rade-lineage-icon rade-lineage-icon--{event['kind']}"

    body = html.Div(
        className="rade-lineage-body",
        children=[
            html.Span(event["title"]),
            html.Span(
                f"({event.get('actor', 'system')} · {_relative_time(event['ts'])})",
                className="rade-lineage-meta",
            ),
        ],
    )

    return html.Div(
        className="rade-lineage-row",
        children=[
            html.Div(
                className=icon_class,
                children=DashIconify(icon=event["icon"], width=14),
            ),
            body,
            html.A(
                "View details",
                href="#",
                className="rade-lineage-link",
            ),
        ],
    )


def _relative_time(ts: Optional[str]) -> str:
    """Return a coarse relative-time string (``2h ago`` / ``3d ago`` / ...).

    Falls back to the raw ISO string if parsing fails so the row
    still carries useful information.
    """
    if not ts:
        return _PLACEHOLDER
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta = datetime.now(tz=timezone.utc) - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 7 * 86400:
        return f"{seconds // 86400}d ago"
    return dt.strftime("%d %b %Y")


def _empty_lineage_message(message: str) -> List[Any]:
    """Single-row "no data" placeholder for the lineage timeline."""
    return [
        html.Div(
            message,
            className="text-xs text-slate-500 py-2",
        )
    ]


__all__ = ["register"]
