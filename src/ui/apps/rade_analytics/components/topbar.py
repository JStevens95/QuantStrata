"""Topbar — stateless content for ``dmc.AppShellHeader``.

Matches the ``rade_landing_dashboard.png`` design contract:

    ┌───────────────────────────────────────────────────────────────┐
    │ Dashboard ›  Overview               [ver ▾] [split] 🔔  🔍     │
    └───────────────────────────────────────────────────────────────┘

The left slot is a two-segment breadcrumb: the first segment is the
static "Dashboard" app-root label, the second segment is the active
page title — the router callback (Phase D.3) updates the second
segment's ``children`` whenever ``url.pathname`` changes.

The right slot is the global scope-control cluster.  In order:

* ``version_select`` — active ensemble version.  Populated by the
  session-bootstrap callback from ``GET /prism/v1/versions``; emits
  to ``session.active_version``.
* ``split_toggle``   — active dataset split (train / val / test).
  Writes to ``session.split``; every split-scoped page (overview,
  evaluation, monitoring, data-quality) reads from the session store.
* ``alerts_btn``     — bell icon; shows unread count as a ``dmc.Badge``
  when the alerts queue is non-empty.  Opens the alerts drawer in F.
* ``search_btn``     — spotlight-style search.  Opens the command
  palette in F; hidden behind a placeholder callback for now.

Design spec anchors
-------------------
* Layout (§5): 60 px header, flex justify-between, sticky to top.
* Components (§6): DMC Select, SegmentedControl, ActionIcon, Badge.
* Accessibility (§10): every ActionIcon carries an ``aria-label``.
"""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


# Every dynamic id in the topbar lives here so callbacks never
# hardcode strings.  Keep in sync with ``callbacks/router_cb.py`` +
# ``callbacks/shell_cb.py`` (Phase D.3 / D.4).
TOPBAR_IDS = {
    # Two-segment breadcrumb.  The first segment is immutable
    # ("Dashboard"); the second is updated by the router.
    "breadcrumb_root":   "topbar-breadcrumb-root",
    "breadcrumb_page":   "topbar-breadcrumb-page",

    # Right-side controls.
    "version_select":    "topbar-version-select",
    "split_toggle":      "topbar-split-toggle",
    # Phase 3.3 — global Scaled / Original PnL-units toggle.  Drives
    # the ``space=`` query param threaded through every space-aware
    # backend call.  Persisted in ``session.pnl_space``.
    "space_toggle":      "topbar-space-toggle",
    "alerts_btn":        "topbar-alerts-btn",
    "alerts_badge":      "topbar-alerts-badge",
    "search_btn":        "topbar-search-btn",
}


# Default segments for the split toggle — must match
# ``Session.split`` literal union in ``data/session.py``.
_SPLIT_OPTIONS = [
    {"label": "Train", "value": "train"},
    {"label": "Val",   "value": "val"},
    {"label": "Test",  "value": "test"},
]


# Segments for the PnL-space toggle.  Must match ``PNL_SPACES`` in
# ``data/session.py`` and the ``Literal`` accepted by the API's
# ``?space=`` query param (Phase 3.2).
_SPACE_OPTIONS = [
    {"label": "Scaled",   "value": "scaled"},
    {"label": "Original", "value": "original"},
]


def _breadcrumb(breadcrumb_title: str) -> html.Div:
    """Left slot — ``Dashboard › {breadcrumb_title}``.

    Rendered as plain HTML (not DMC) so the page segment can be set at
    build time without wrestling with DMC's internal state.  The router
    bakes the title in when it builds the chrome on each top-level
    navigation, so no separate breadcrumb-update callback is needed.
    """
    return html.Div(
        className="rade-breadcrumb",
        children=[
            html.Span(
                "Dashboard",
                id=TOPBAR_IDS["breadcrumb_root"],
                className="rade-breadcrumb-item",
            ),
            DashIconify(
                icon="tabler:chevron-right",
                width=14,
                className="rade-breadcrumb-separator",
            ),
            html.Span(
                breadcrumb_title or "Overview",
                id=TOPBAR_IDS["breadcrumb_page"],
                className="rade-breadcrumb-item rade-breadcrumb-item--active",
            ),
        ],
    )


def _controls(current_split: str, current_pnl_space: str) -> dmc.Group:
    """Right slot — version / split / PnL-space / alerts / search."""
    return dmc.Group(
        gap="sm",
        align="center",
        wrap="nowrap",
        children=[
            # Version picker.  ``data=[]`` is filled by the session
            # bootstrap callback on first tick.
            dmc.Select(
                id=TOPBAR_IDS["version_select"],
                placeholder="version",
                data=[],
                w=160,
                size="xs",
                clearable=False,
                leftSection=DashIconify(
                    icon="tabler:git-branch", width=14,
                ),
            ),

            # Split segmented control.  Initial value is sourced from
            # ``session.split`` by the chrome builder so the user's
            # choice survives chrome rebuilds (top-level navigation).
            dmc.SegmentedControl(
                id=TOPBAR_IDS["split_toggle"],
                data=_SPLIT_OPTIONS,
                value=current_split or "test",
                size="xs",
                radius="sm",
                color="violet",
            ),

            # PnL-space segmented control (Phase 3.3).  Mirrors the
            # split toggle's wiring shape: initial value is sourced
            # from ``session.pnl_space`` by the chrome builder; the
            # capture callback in ``overview_cb`` persists changes
            # back into the session store.  Cyan keeps it visually
            # distinct from the violet split toggle so users can
            # parse the two side-by-side at a glance.
            dmc.SegmentedControl(
                id=TOPBAR_IDS["space_toggle"],
                data=_SPACE_OPTIONS,
                value=current_pnl_space or "scaled",
                size="xs",
                radius="sm",
                color="cyan",
            ),

            # Alerts bell — DMC's "Indicator" wraps the ActionIcon so
            # the unread-count badge pins top-right without layout shift.
            dmc.Indicator(
                id=TOPBAR_IDS["alerts_badge"],
                disabled=True,
                color="rose",
                size=8,
                offset=4,
                children=dmc.ActionIcon(
                    id=TOPBAR_IDS["alerts_btn"],
                    variant="subtle",
                    color="gray",
                    size="lg",
                    children=DashIconify(icon="tabler:bell", width=16),
                    **{"aria-label": "Open alerts"},
                ),
            ),

            # Search — opens the command palette (implemented in F).
            dmc.ActionIcon(
                id=TOPBAR_IDS["search_btn"],
                variant="subtle",
                color="gray",
                size="lg",
                children=DashIconify(icon="tabler:search", width=16),
                **{"aria-label": "Search"},
            ),
        ],
    )


def build_topbar(
    *,
    breadcrumb_title:  str = "",
    current_split:     str = "test",
    current_pnl_space: str = "scaled",
) -> dmc.Group:
    """Build the topbar's inner layout: breadcrumb left, controls right.

    Parameters
    ----------
    breadcrumb_title
        Second segment of the breadcrumb; the chrome passes the active
        page's title here on each rebuild.
    current_split
        Initial value for the split segmented control; the chrome
        sources this from ``session.split`` so it reflects the user's
        last choice across chrome rebuilds.
    current_pnl_space
        Initial value for the PnL-space segmented control (Phase 3.3);
        chrome sources this from ``session.pnl_space`` for the same
        survive-rebuilds reason as ``current_split``.

    Returned as a ``dmc.Group`` so the caller can drop it straight
    into ``dmc.AppShellHeader(children=...)``.
    """
    return dmc.Group(
        justify="space-between",
        align="center",
        wrap="nowrap",
        h="100%",
        px="lg",
        children=[
            _breadcrumb(breadcrumb_title),
            _controls(current_split, current_pnl_space),
        ],
    )


__all__ = [
    "TOPBAR_IDS",
    "build_topbar",
]
