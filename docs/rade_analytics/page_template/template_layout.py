"""TEMPLATE — Layout module skeleton for a new Rade Analytics page.

DO NOT IMPORT THIS FILE.  Copy it to
``src/ui/apps/rade_analytics/layouts/<your_page>.py`` and replace every
``# TODO:`` marker.  See ``docs/rade_analytics/page_template/README.md``
for the 15-minute add-a-page workflow.

What this template demonstrates
-------------------------------
* **Pure layout** (Page Contract §3 Rule L1) — ``build_template(*, session)``
  takes a :class:`Session` and bakes initial input values into the layout
  at build time.  No hydration callback after mount.
* **Mount tripwire** — a ``dcc.Store(id=mount_signal, data=True)`` mounted
  at the layout root.  The bootstrap callback in ``template_cb.py``
  triggers off this Store's ``data`` Input so it fires exactly once per
  fresh mount of the page (Page Contract §3 Rule L4).
* **Stable id contract** — every component id lives in :data:`TEMPLATE_IDS`
  so callbacks never hardcode strings (Page Contract §3 Rule L3).
* **Tailwind / dmc layout primitives** — header band → KPI card → chart
  body, matching the rest of the app.

Anatomy
-------
::

    Row 0 · mount tripwire (invisible)
    Row 1 · Header band (context picker · open-related-page link)
    Row 2 · Single KPI card  ·  Single time-series chart
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ...components.chart_container import ChartContainer
from ...components.kpi_card import KpiCard
from ...data.session import Session


# TODO: Rename TEMPLATE_IDS → <YOUR_PAGE>_IDS, and prefix every id value
# with "your-page-" so the Dash dev-tools error messages tell you which
# page they came from.
TEMPLATE_IDS: Dict[str, str] = {
    "root":                "template-root",

    # Row 1 — Header band
    "context_select":      "template-context-select",
    "open_related_btn":    "template-open-related-btn",

    # Row 2 — KPI card
    "kpi_value":           "template-kpi-value",
    "kpi_card":            "template-kpi-card",

    # Row 2 — Chart
    "main_chart":          "template-main-chart",

    # Mount tripwire — Page Contract §3 Rule L4.  A memory-store seeded
    # with ``data=True`` at build time; the bootstrap callback uses it
    # as its trigger Input so it fires *exactly once* per fresh mount,
    # *after* the parent has finished writing the new content tree.
    # Pathname-as-Input would race with that write; top-level-store
    # would only fire on cross-top-level navigation.
    "mount_signal":        "template-mount-signal",

    # Optional ephemeral stores (uncomment if your page needs them)
    # "store_<thing>":     "template-<thing>-store",
}


# ─────────────────────────────────────────────────────────────────────
# Row 1 — Header band
# ─────────────────────────────────────────────────────────────────────


def _header_band(
    *,
    initial_context_id: Optional[str] = None,
) -> html.Div:
    """Sticky header — context picker + cross-link.

    Parameters
    ----------
    initial_context_id
        Seed for the :class:`dmc.Select`'s ``value`` prop.  Sourced
        from ``session`` at build time (Page Contract §3 Rule L1) —
        the bootstrap callback will populate the ``data`` (option
        list) prop after a backend lookup; the seeded ``value`` then
        reads against that fresh list.  When ``None``, the picker
        shows its placeholder until the user / bootstrap picks one.
    """
    context_picker = html.Div(
        className="flex flex-col gap-1 min-w-[220px]",
        children=[
            html.Span(
                # TODO: replace with your page's context noun (e.g. "Cluster", "Desk", "Run").
                "Context",
                className="text-[11px] uppercase tracking-wider text-slate-400",
            ),
            dmc.Select(
                id=TEMPLATE_IDS["context_select"],
                # ``data`` is intentionally empty here — the bootstrap
                # callback populates it after a backend lookup.
                # Seeding ``value`` from session means the picker
                # remembers the user's choice across page navigation.
                data=[],
                value=initial_context_id,
                # TODO: replace with a context-appropriate placeholder.
                placeholder="Select…",
                searchable=True,
                clearable=False,
                size="sm",
            ),
        ],
    )

    open_related_btn = dmc.Button(
        # TODO: rename + retarget the cross-link, or delete this if your
        # page doesn't have a partner page.
        "Related page",
        id=TEMPLATE_IDS["open_related_btn"],
        variant="light",
        color="violet",
        size="sm",
        leftSection=DashIconify(icon="tabler:share-2", width=16),
    )

    return html.Div(
        className="rade-card flex flex-col gap-3 sticky top-0 z-10",
        children=[
            html.Div(
                className="flex items-end gap-4 flex-wrap",
                children=[
                    context_picker,
                    html.Div(className="flex-1"),  # spacer
                    open_related_btn,
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — KPI + Chart
# ─────────────────────────────────────────────────────────────────────


def _row_kpi_chart() -> html.Div:
    """Two-column row — KPI card on the left, time-series chart on the right.

    The KPI card and chart both render placeholder content (an em-dash
    value, an empty figure) so the page never reflows on initial paint;
    render callbacks replace the placeholders with real values once the
    backend fetch returns.
    """
    return html.Div(
        # 2/5 : 3/5 split on wide screens; collapses to a single stack
        # on narrow viewports.
        className="grid grid-cols-1 lg:grid-cols-5 gap-3 items-stretch",
        children=[
            html.Div(
                className="lg:col-span-2 flex flex-col gap-3",
                children=[
                    KpiCard(
                        # TODO: pick a metric label that matches what
                        # the render callback computes (e.g. "MAE",
                        # "Trade count", "Coverage").
                        label="Headline metric",
                        value="—",
                        card_id=TEMPLATE_IDS["kpi_card"],
                        value_id=TEMPLATE_IDS["kpi_value"],
                        icon="tabler:chart-dots",
                    ),
                ],
            ),
            html.Div(
                className="lg:col-span-3 flex flex-col gap-3",
                children=[
                    ChartContainer(
                        # TODO: pick a title that matches the chart.
                        title="Time-series",
                        chart_id=TEMPLATE_IDS["main_chart"],
                    ),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Public entrypoint — build_<your_page>
# ─────────────────────────────────────────────────────────────────────


def build_template(*, session: Session) -> html.Div:
    """Build the page layout, seeded from :class:`Session`.

    Parameters
    ----------
    session
        The active session.  Drives initial values for every input on
        the page — e.g. ``session.evaluation.<your_page>_context_id``
        for the context picker.  Page Contract §3 Rule L1 mandates
        that every input that *can* read from session, *does* —
        eliminating the need for a hydration callback after mount.

    Notes
    -----
    The layout function is **pure** — same session in, same DOM out.
    All side effects (backend fetches, option-list population) happen
    in the bootstrap callback in ``<your_page>_cb._register_render``.
    """
    # TODO: replace with the actual session field your page reads.
    # If the field doesn't exist yet, add it to
    # ``data/session.py::EvaluationFilters`` (or the session root
    # for non-Evaluation pages) following the existing pattern.
    initial_context_id: Optional[str] = None  # session.evaluation.<your_field>

    return html.Div(
        id=TEMPLATE_IDS["root"],
        className="flex flex-col gap-4 p-4",
        children=[
            # Mount tripwire — Page Contract §3 Rule L4.
            dcc.Store(
                id=TEMPLATE_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            _header_band(initial_context_id=initial_context_id),
            _row_kpi_chart(),
        ],
    )


__all__ = [
    "TEMPLATE_IDS",
    "build_template",
]
