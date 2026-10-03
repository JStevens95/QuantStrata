"""Risk Management page layout (Phase 4.0 — eval-data pivot).

Portfolio-level tail-risk decomposition for the **trained portfolio**
(train / val / test eval splits — not inference runs).  The page
answers "what does the tail of the trained portfolio look like?"  in
two complementary views: a PnL distribution histogram showing where
VaR / CVaR sit, and a cluster-level decomposition (waterfall +
tornado + donut + tail table) showing which clusters drive the tail.

Page anatomy
------------

* **Page header** — title + subtitle (split + ensemble + source).
* **Source toggle** — Predicted (default) / Actual.  Page-local
  segmented control: the rest of the app's pnl_space toggle does not
  apply here because risk talk only makes sense in original notional
  units.
* **KPI strip** — six tiles (VaR 95 / VaR 99 / CVaR 95 / CVaR 99 /
  Skewness / Excess-kurtosis) computed off the portfolio PnL series.
* **PnL distribution** — histogram with VaR / CVaR vertical markers
  and a shaded left-tail.  Canonical risk-management visual.
* **Cluster waterfall** — scenario-pinned P&L decomposition,
  defaults to the worst portfolio loss.
* **Tornado + donut grid** — clusters ranked by |contribution| on
  the left, contribution share donut on the right.
* **Tail-conditional table** — top-N worst scenarios with cluster
  enrichment columns.  ``top_rf`` / ``top_trade_type`` columns are
  placeholdered (``—``) until Phase 4.1 / 4.2 land.
* **Footer** — source manifest path.

Data flow
---------

Single hydrate callback fires on
``(pathname, session-store, source_toggle)`` and pulls

* ``backend.portfolio_df(session.split, space="original")``
* ``backend.cluster_timeseries_df(session.split, space="original")``

so the page respects the topbar split toggle but always reads
original notional units.  See ``callbacks/risk_management_cb.py``.

Design-spec anchors
-------------------
* §1 — typography & palette (Inter UI text, JetBrains-Mono numerics,
  violet → cyan brand).
* §5 — page anatomy (KPI strip on top, hero chart, supporting grid,
  tabular content, footer caption).
* §6 — ``rade-card`` / ``rade-stress-mini-kpi`` shared utilities.
* §11 — top-level nav taxonomy (Risk Management is a peer of
  Overview, Evaluation, Inference, etc.).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import dcc, html

from ..components.chart_container import ChartContainer
from ..figures.risk_management_charts import (
    empty_cluster_donut,
    empty_cluster_tornado,
    empty_cluster_waterfall,
    empty_pnl_distribution,
)

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────
# DOM identifiers — every dynamic id used by the page lives here so
# callbacks never hard-code strings.
# ─────────────────────────────────────────────────────────────────────

RISK_MANAGEMENT_IDS: Dict[str, str] = {
    "root":                 "risk-management-root",
    "subtitle":             "risk-management-subtitle",

    # Mount tripwire — Page Contract §3 Rule L4.  ``data=True`` at
    # build time fires the render callback exactly once per fresh
    # mount of the page.  Lives *inside* the layout chunk the router
    # swaps in, so it can't race the router's content swap on
    # cross-page navigation (see Anti-pattern A8).
    "mount_signal":         "risk-management-mount-signal",

    # Page-local Predicted/Actual segmented control.  Not synced to
    # ``Session`` — the choice is deliberately page-local so the user
    # can flip between model-view and historical-actual without
    # affecting other pages.
    "source_toggle":        "risk-management-source-toggle",

    # KPI tiles (six-card row at top of page).
    "kpi_var_95_value":     "risk-management-kpi-var-95-value",
    "kpi_var_99_value":     "risk-management-kpi-var-99-value",
    "kpi_cvar_95_value":    "risk-management-kpi-cvar-95-value",
    "kpi_cvar_99_value":    "risk-management-kpi-cvar-99-value",
    "kpi_skew_value":       "risk-management-kpi-skew-value",
    "kpi_kurt_value":       "risk-management-kpi-kurt-value",

    # Distribution hero — histogram with VaR / CVaR markers.
    "distribution_chart":   "risk-management-distribution-chart",

    # Shared scenario filter — drives the waterfall + tornado + donut
    # decomposition trio.  Default value is ``"__all__"`` (see
    # ``ALL_SCENARIOS_VALUE`` below) which means "aggregate across
    # every scenario" — preserves Phase 4.0 macro-view behaviour.
    "scenario_filter":      "risk-management-scenario-filter",

    # Cluster decomposition — scenario-pinned waterfall.
    "waterfall_chart":      "risk-management-waterfall-chart",
    "waterfall_caption":    "risk-management-waterfall-caption",

    # Supporting grid — magnitude-ranked tornado + signed donut.
    "tornado_chart":        "risk-management-tornado-chart",
    "donut_chart":          "risk-management-donut-chart",

    # Tail-conditional table — populated by the callback.
    "tail_table_container": "risk-management-tail-table-container",

    # Footer — source manifest line.
    "footer_caption":       "risk-management-footer-caption",
}


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

_PLACEHOLDER: str = "—"

# Defaults captured here so the layout, the callback's empty state
# and the segmented-control initial value stay in lock-step.
DEFAULT_SOURCE: str = "predicted"
_SOURCE_OPTIONS = [
    {"label": "Predicted", "value": "predicted"},
    {"label": "Actual",    "value": "actual"},
]

# Sentinel value used by the scenario filter to mean "aggregate
# across all scenarios" — i.e. the canonical Phase 4.0 macro view.
# Lives in module scope so the callback can import + compare without
# duplicating the literal.
ALL_SCENARIOS_VALUE: str = "__all__"
ALL_SCENARIOS_LABEL: str = "All scenarios (aggregate)"


_PAGE_SUBTITLE_DEFAULT: str = (
    "Tail-risk decomposition for the trained portfolio — "
    "waiting for evaluation data."
)


# ─────────────────────────────────────────────────────────────────────
# Page header + source toggle row
# ─────────────────────────────────────────────────────────────────────

def _page_header() -> html.Div:
    """Title row mirrored on every Rade top-level page."""
    return html.Div(
        className="flex flex-col gap-1 mb-2",
        children=[
            html.Div(
                "Risk Management",
                className="text-2xl font-semibold text-slate-100",
            ),
            html.Div(
                _PAGE_SUBTITLE_DEFAULT,
                id=RISK_MANAGEMENT_IDS["subtitle"],
                className="text-sm text-slate-400",
            ),
        ],
    )


def _source_toggle_row() -> html.Div:
    """Predicted / Actual segmented control + its inline label.

    Cyan colour-keyed to distinguish from the violet split toggle in
    the topbar; small label on the left so the user understands what
    the toggle drives without needing tooltip discovery.
    """
    return html.Div(
        className="flex items-center gap-3 mb-2",
        children=[
            html.Span(
                "Source",
                className=(
                    "text-xs uppercase tracking-wider "
                    "text-slate-500 font-semibold"
                ),
            ),
            dmc.SegmentedControl(
                id=RISK_MANAGEMENT_IDS["source_toggle"],
                data=_SOURCE_OPTIONS,
                value=DEFAULT_SOURCE,
                size="xs",
                radius="sm",
                color="violet",
            ),
            html.Span(
                "Predicted = model-view of tail risk · Actual = "
                "historical PnL on the same scenarios.",
                className="text-xs text-slate-500 italic",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# KPI strip — six label-over-value tiles (NOT the spark KpiCard).
# ─────────────────────────────────────────────────────────────────────

def _kpi_tile(label: str, value_id: str) -> html.Div:
    """One KPI tile (label + big mono value).

    Uses ``rade-stress-mini-kpi`` styling for visual continuity with
    the Inference page's stress tiles; the parent grid lays six of
    them in a single row on large screens.
    """
    return html.Div(
        className="rade-stress-mini-kpi",
        children=[
            html.Div(label.upper(), className="rade-stress-mini-label"),
            html.Div(
                _PLACEHOLDER,
                id=value_id,
                className="rade-stress-mini-value font-mono",
            ),
        ],
    )


def _kpi_strip() -> html.Div:
    """Six-tile KPI row.

    Phase 4.0 expansion vs the original four-tile strip — we now
    surface both 95% and 99% confidence levels for VaR / CVaR so the
    user can see how much of the tail the standard cut-off is hiding.
    """
    return html.Div(
        className=(
            "grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3"
        ),
        children=[
            _kpi_tile("VaR 95%",  RISK_MANAGEMENT_IDS["kpi_var_95_value"]),
            _kpi_tile("VaR 99%",  RISK_MANAGEMENT_IDS["kpi_var_99_value"]),
            _kpi_tile("CVaR 95%", RISK_MANAGEMENT_IDS["kpi_cvar_95_value"]),
            _kpi_tile("CVaR 99%", RISK_MANAGEMENT_IDS["kpi_cvar_99_value"]),
            _kpi_tile("Skewness",        RISK_MANAGEMENT_IDS["kpi_skew_value"]),
            _kpi_tile("Excess kurtosis", RISK_MANAGEMENT_IDS["kpi_kurt_value"]),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Charts — distribution (NEW), waterfall (hero), tornado + donut grid
# ─────────────────────────────────────────────────────────────────────

def _distribution_card() -> html.Div:
    """PnL distribution histogram card (Phase 4.0 addition).

    Anchored above the waterfall because the histogram answers the
    "where does VaR sit on the curve?" question that the KPI strip's
    numbers raise.  Reading flow: KPIs → distribution → which
    clusters drive that distribution.
    """
    return ChartContainer(
        title="Portfolio PnL distribution",
        subtitle=(
            "Histogram of per-scenario portfolio PnL across the active "
            "split.  Dashed markers show VaR (amber) and CVaR (rose) "
            "at 95% confidence; the shaded region is the left tail."
        ),
        graph_id=RISK_MANAGEMENT_IDS["distribution_chart"],
        figure=empty_pnl_distribution(),
        height=320,
    )


def _scenario_filter_row() -> html.Div:
    """Page-local scenario filter — shared across the cluster trio.

    Slots between the distribution hero and the waterfall card so the
    user understands its scope visually: it scopes the three cluster
    charts below (waterfall, tornado, donut) without touching the KPI
    strip, distribution histogram or tail table — all of which keep
    their split-wide framing.

    Options are populated by the render callback off the active
    ``portfolio_df``.  At build time we ship only the "All scenarios"
    sentinel so the dropdown never renders empty during cold-mount.
    """
    return html.Div(
        className=(
            "rade-card flex flex-col sm:flex-row sm:items-center "
            "gap-3 px-4 py-3"
        ),
        children=[
            html.Div(
                className="flex flex-col min-w-0",
                children=[
                    html.Span(
                        "Cluster decomposition · Scenario filter",
                        className=(
                            "text-xs uppercase tracking-wider "
                            "text-slate-500 font-semibold"
                        ),
                    ),
                    html.Span(
                        "Filters the waterfall, cluster sensitivity and "
                        "contribution share charts.  Default aggregates "
                        "across every scenario in the split.",
                        className="text-[11px] text-slate-500 italic",
                    ),
                ],
            ),
            html.Div(
                className="sm:ml-auto min-w-[16rem]",
                children=[
                    dmc.Select(
                        id=RISK_MANAGEMENT_IDS["scenario_filter"],
                        data=[
                            {
                                "value": ALL_SCENARIOS_VALUE,
                                "label": ALL_SCENARIOS_LABEL,
                            },
                        ],
                        value=ALL_SCENARIOS_VALUE,
                        searchable=True,
                        clearable=False,
                        nothingFoundMessage="No scenarios available",
                        size="xs",
                        radius="sm",
                    ),
                ],
            ),
        ],
    )


def _waterfall_card() -> html.Div:
    return html.Div(
        className="flex flex-col gap-2",
        children=[
            ChartContainer(
                title="P&L decomposition — by cluster",
                subtitle=(
                    "How each cluster contributes to the portfolio "
                    "total.  Defaults to the worst-loss scenario in "
                    "the split; the scenario filter above pins it to "
                    "a specific event.  Phase 4.2 swaps this for a "
                    "risk-factor-level waterfall."
                ),
                graph_id=RISK_MANAGEMENT_IDS["waterfall_chart"],
                figure=empty_cluster_waterfall(),
                height=360,
                actions=[
                    html.Div(
                        id=RISK_MANAGEMENT_IDS["waterfall_caption"],
                        className="text-[11px] text-slate-500 italic",
                        children=(
                            "Awaiting evaluation data — scenario pin "
                            "appears once data loads."
                        ),
                    ),
                ],
            ),
        ],
    )


def _tornado_card() -> html.Div:
    return ChartContainer(
        title="Cluster sensitivity",
        subtitle=(
            "Clusters ranked by absolute contribution across all "
            "scenarios in the split.  Sign-coloured: emerald = "
            "positive, rose = negative."
        ),
        graph_id=RISK_MANAGEMENT_IDS["tornado_chart"],
        figure=empty_cluster_tornado(),
        height=360,
    )


def _donut_card() -> html.Div:
    return ChartContainer(
        title="Contribution share",
        subtitle=(
            "|Contribution| share per cluster; signed colour. "
            "Phase 4.2 nests this into a three-ring "
            "RF-group → trade-type → cluster sunburst."
        ),
        graph_id=RISK_MANAGEMENT_IDS["donut_chart"],
        figure=empty_cluster_donut(),
        height=360,
    )


def _tornado_donut_row() -> html.Div:
    """Tornado (2/3 width) + donut (1/3 width).

    The tornado is a horizontal-bar chart whose readability scales
    directly with horizontal real-estate (longer bars + room for
    inline value labels), whereas the donut is bounded by its
    circular geometry and doesn't benefit from extra width.  Hence
    the asymmetric grid — tornado gets ``lg:col-span-2`` so it
    occupies twice the donut's column width on large screens.
    """
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-3 gap-3",
        children=[
            html.Div(
                className="min-w-0 lg:col-span-2",
                children=[_tornado_card()],
            ),
            html.Div(
                className="min-w-0 lg:col-span-1",
                children=[_donut_card()],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Tail-conditional table — placeholder before hydration
# ─────────────────────────────────────────────────────────────────────

def _tail_table_placeholder() -> List[Any]:
    return [
        html.Div(
            "Tail-conditional table",
            className="text-sm font-semibold text-slate-200",
        ),
        html.Div(
            "Scenarios at or below the 5th-percentile portfolio loss. "
            "``Top RF`` and ``Top trade type`` columns light up in "
            "Phase 4.1 / 4.2 once trade attributes are exposed.",
            className="text-xs text-slate-500",
        ),
        html.Div(
            "Awaiting evaluation data — the table will appear here.",
            className="text-xs text-slate-500 italic px-2 py-3",
        ),
    ]


def _tail_table_card() -> html.Div:
    return html.Div(
        id=RISK_MANAGEMENT_IDS["tail_table_container"],
        className="rade-card flex flex-col gap-2 min-w-0",
        children=_tail_table_placeholder(),
    )


# ─────────────────────────────────────────────────────────────────────
# Page footer
# ─────────────────────────────────────────────────────────────────────

def _page_footer() -> html.Div:
    """Subtle source-of-truth line at page bottom.

    Hydrated by the callback to show the evaluation manifest path;
    placeholder text until then.
    """
    return html.Div(
        className="flex justify-end mt-6",
        children=[
            html.Span(
                "Source: awaiting evaluation data …",
                id=RISK_MANAGEMENT_IDS["footer_caption"],
                className="text-xs text-slate-500 font-mono",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Page composition
# ─────────────────────────────────────────────────────────────────────

def build_risk_management(
    *,
    session: Optional["Session"] = None,
) -> html.Div:
    """Compose the Risk Management page.

    Follows Page Contract §2.1 (uniform ``build_*(session=...)``
    signature); ``session`` is accepted for parity with every other
    page builder.  It's currently unused at build time because the
    callback hydrates from ``session-store`` directly — but kept so
    future enhancements (e.g. honouring ``session.cluster_id`` on
    deep-dive) can wire in without changing the build signature.
    """
    del session  # unused at build time; callback reads session-store

    return html.Div(
        id=RISK_MANAGEMENT_IDS["root"],
        className="rade-page flex flex-col gap-4 p-4",
        children=[
            # Mount tripwire — Page Contract §3 Rule L4.  Lives at the
            # root of the swapped layout chunk so it fires *after* the
            # router has finished mounting every dependent Output.
            dcc.Store(
                id=RISK_MANAGEMENT_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            _page_header(),
            _source_toggle_row(),
            _kpi_strip(),
            _distribution_card(),
            _scenario_filter_row(),
            _waterfall_card(),
            _tornado_donut_row(),
            _tail_table_card(),
            _page_footer(),
        ],
    )


__all__ = [
    "ALL_SCENARIOS_LABEL",
    "ALL_SCENARIOS_VALUE",
    "DEFAULT_SOURCE",
    "RISK_MANAGEMENT_IDS",
    "build_risk_management",
]
