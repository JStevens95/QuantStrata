"""Risk Management page callbacks (Phase 4.0 — eval-data pivot).

Two callbacks drive the whole page: one capture, one render — see
Page Contract §2.2.  Both follow §4 Rule C7 (mount-signal trigger) +
Rule C4 (pathname as ``State``, never ``Input``) + Rule C5
(``prevent_initial_call="initial_duplicate"`` on mount-driven fires).

Capture (``_register_scenario_filter_sync``)
--------------------------------------------
Owns the page-local scenario-filter dropdown — writes its ``data``
(option list) and resets its ``value`` to ``ALL_SCENARIOS_VALUE``
whenever the underlying eval data shape changes (mount, session
split, source toggle).  The dropdown's ``value`` is therefore only
ever an ``Input`` to render and never an ``Output`` of render —
critical because Input-and-Output on the same property in the same
callback can manifest as silent figure-output drops when the
trigger edge is the loopback itself.

Render (``_register_render``)
-----------------------------
Paints every chart / KPI / table off the resolved
``(split, source, scenario)`` tuple.  Trigger inputs:

1. ``Input(<PAGE>_IDS["mount_signal"], "data")`` — fires once per
   fresh mount of the page's layout chunk.  Lives *inside* the
   swapped chunk so it can't race the router's content swap
   (Anti-pattern A8).
2. ``Input(SHELL_IDS["session_store"], "data")`` — re-fires when the
   topbar split toggle (or any other shared session field) changes.
3. ``Input(<PAGE>_IDS["source_toggle"], "value")`` — page-internal
   trigger when the user flips Predicted ↔ Actual.
4. ``Input(<PAGE>_IDS["scenario_filter"], "value")`` — page-internal
   trigger when the user picks a single scenario for the cluster
   decomposition trio (waterfall + tornado + donut).

Gate state (Rule C4)
--------------------
``State(SHELL_IDS["url"], "pathname")`` — used only to early-exit when
the user is no longer on ``/risk-management``.  Never an ``Input``.

Behaviour
---------
The callback always reads the eval parquets in *original* PnL space
(risk talk only makes sense in currency units, so the topbar's
pnl_space toggle is intentionally ignored here).  It threads a
single ``pnl_column`` parameter — ``"predictions"`` for Predicted,
``"targets"`` for Actual — through every chart / KPI builder so the
same code paths serve both views.

Outputs cover:

* page subtitle (split + ensemble version + source)
* 6 KPI tile values (VaR 95 / VaR 99 / CVaR 95 / CVaR 99 / skew / kurt)
* 4 figures (PnL distribution, cluster waterfall, tornado, donut)
* waterfall caption (scenario pin + portfolio total for that pin)
* tail-conditional table body
* footer caption (eval manifest path)

Empty states are surface-specific: when the active split has no eval
data, when the requested PnL column is all-NaN (Phase 3.1 / 3.4 "no
scaler coverage" case), or when the cluster summary is missing, the
callback emits a targeted message into the right component instead
of crashing or flat-zeroing the chart.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from ..data.session import Session
from ..figures.risk_management_charts import (
    build_cluster_donut,
    build_cluster_tornado,
    build_cluster_waterfall,
    build_pnl_distribution,
    compute_tail_stats,
    compute_tail_table,
    empty_cluster_donut,
    empty_cluster_tornado,
    empty_cluster_waterfall,
    empty_pnl_distribution,
)
from ..layouts.risk_management import (
    ALL_SCENARIOS_LABEL,
    ALL_SCENARIOS_VALUE,
    DEFAULT_SOURCE,
    RISK_MANAGEMENT_IDS,
)
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

_PLACEHOLDER:     str = "—"
_RISK_MGMT_ROUTE: str = "/risk-management"

# Map between the source segmented-control value and the canonical
# column name produced by ``ArtifactReader._apply_space("original")``.
# Centralised so the callback never carries the string mapping inline.
_PNL_COLUMN_BY_SOURCE: Dict[str, str] = {
    "predicted": "predictions",
    "actual":    "targets",
}


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Risk Management page callback to ``app``.

    Page Contract §2.2 — the single ``register()`` entry point
    delegates to capture / render section helpers.

    * **Capture** — ``_register_scenario_filter_sync`` writes the
      scenario-filter dropdown's ``data`` (option list) + ``value``
      (reset to "All scenarios") whenever the underlying eval data
      changes (split / source flip).  Lives separately from the
      render callback so the dropdown's ``value`` is ONLY ever an
      ``Input`` to render, never an ``Output`` of it — avoiding the
      Input+Output-on-same-property loopback that can cause Dash to
      silently drop figure updates triggered by the same property.
    * **Render** — ``_register_render`` paints every chart / KPI /
      table off the resolved (split, source, scenario) tuple.
    """
    _register_scenario_filter_sync(app, backend)
    _register_render(app, backend)


# ═════════════════════════════════════════════════════════════════════
# Capture — sync scenario-filter dropdown options from the active data
# ═════════════════════════════════════════════════════════════════════

def _register_scenario_filter_sync(app: "Dash", backend: "RadeBackend") -> None:
    """Rebuild the scenario-filter dropdown when the eval data changes.

    Triggers only on data-shape-changing inputs (mount, session,
    source).  Resets ``value`` to ``ALL_SCENARIOS_VALUE`` on every
    fire so the user lands on the canonical "aggregate" view after a
    split or source flip — the previously-selected scenario may not
    even exist in the new split.
    """

    @app.callback(
        Output(RISK_MANAGEMENT_IDS["scenario_filter"], "data"),
        Output(RISK_MANAGEMENT_IDS["scenario_filter"], "value"),
        Input(RISK_MANAGEMENT_IDS["mount_signal"],     "data"),
        Input(SHELL_IDS["session_store"],              "data"),
        Input(RISK_MANAGEMENT_IDS["source_toggle"],    "value"),
        State(SHELL_IDS["url"],                        "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _sync_scenario_filter(
        _mount:       Any,
        session_data: Optional[Dict[str, Any]],
        source:       Optional[str],
        pathname:     Optional[str],
    ) -> Tuple[List[Dict[str, str]], str]:
        if pathname != _RISK_MGMT_ROUTE:
            raise PreventUpdate

        session = Session.from_store(session_data)
        split   = session.split
        source  = source if source in _PNL_COLUMN_BY_SOURCE else DEFAULT_SOURCE
        pnl_col = _PNL_COLUMN_BY_SOURCE[source]

        portfolio_res = backend.portfolio_df(split, space="original")
        portfolio_df  = (
            portfolio_res.data
            if portfolio_res.ok and portfolio_res.data is not None
            else pd.DataFrame()
        )

        options = _scenario_options(portfolio_df, pnl_col=pnl_col)
        logger.info(
            "[risk-mgmt] sync: split=%s source=%s n_options=%d (incl. sentinel)",
            split, source, len(options),
        )
        return options, ALL_SCENARIOS_VALUE


# ═════════════════════════════════════════════════════════════════════
# Render — mount_signal + session + source_toggle → KPIs + figures + table
# ═════════════════════════════════════════════════════════════════════

def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Fetch eval parquets and paint every panel on the page.

    Page Contract §4 Rule C7 — fires off ``mount_signal`` (not
    ``pathname``) so the initial render lands *after* the router has
    finished mounting every Output's component.  ``session-store`` is
    the second trigger Input so topbar split changes re-paint the
    page; the page-local ``source_toggle`` is the third so flipping
    Predicted ↔ Actual re-paints without touching session;
    ``scenario_filter.value`` is the fourth so the user's per-page
    scenario drilldown re-paints the cluster trio (waterfall +
    tornado + donut).

    Crucially, ``scenario_filter.value`` is ONLY an Input here — its
    options + reset are wired through the separate sync callback
    above, so this render never both reads and writes the same
    property in a single fire (a pattern that can manifest as
    silent figure-output drops when the trigger is the loopback
    edge itself).
    """

    @app.callback(
        # Subtitle (one line under the page title)
        Output(RISK_MANAGEMENT_IDS["subtitle"],          "children"),
        # KPI strip (6 values)
        Output(RISK_MANAGEMENT_IDS["kpi_var_95_value"],  "children"),
        Output(RISK_MANAGEMENT_IDS["kpi_var_99_value"],  "children"),
        Output(RISK_MANAGEMENT_IDS["kpi_cvar_95_value"], "children"),
        Output(RISK_MANAGEMENT_IDS["kpi_cvar_99_value"], "children"),
        Output(RISK_MANAGEMENT_IDS["kpi_skew_value"],    "children"),
        Output(RISK_MANAGEMENT_IDS["kpi_kurt_value"],    "children"),
        # Four figures
        Output(RISK_MANAGEMENT_IDS["distribution_chart"], "figure"),
        Output(RISK_MANAGEMENT_IDS["waterfall_chart"],    "figure"),
        Output(RISK_MANAGEMENT_IDS["tornado_chart"],      "figure"),
        Output(RISK_MANAGEMENT_IDS["donut_chart"],        "figure"),
        # Waterfall caption (pinned-scenario meta line)
        Output(RISK_MANAGEMENT_IDS["waterfall_caption"],  "children"),
        # Tail table body
        Output(RISK_MANAGEMENT_IDS["tail_table_container"], "children"),
        # Footer source line
        Output(RISK_MANAGEMENT_IDS["footer_caption"],    "children"),
        # Trigger Inputs — Rule C7.  ``mount_signal`` lives inside
        # the swapped layout chunk and can't race the router's
        # content swap (vs ``Input(pathname)`` = Anti-pattern A8).
        # ``session_store`` re-fires when the topbar split toggle
        # writes.  ``source_toggle.value`` + ``scenario_filter.value``
        # are page-internal (not in ``Session``) so we read them
        # directly as render triggers.  The scenario filter dropdown
        # is wired as Input ONLY here — its options (data) + reset
        # (value) live in the separate sync callback above, so this
        # render is never both an Input and Output on the same
        # property (which Dash can handle but which has historically
        # caused silent figure-output drops on us).
        Input(RISK_MANAGEMENT_IDS["mount_signal"],       "data"),
        Input(SHELL_IDS["session_store"],                "data"),
        Input(RISK_MANAGEMENT_IDS["source_toggle"],      "value"),
        Input(RISK_MANAGEMENT_IDS["scenario_filter"],    "value"),
        # Gate State — Rule C4.  Pathname is never an ``Input`` on
        # page-scoped renders; it gates, never triggers.
        State(SHELL_IDS["url"],                          "pathname"),
        # Rule C5 — mount-signal-driven renders use
        # ``"initial_duplicate"`` so the mount edge fires the first
        # paint.  ``False`` is forbidden on route-bound subtrees.
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:        Any,
        session_data:  Optional[Dict[str, Any]],
        source:        Optional[str],
        scenario_sel:  Optional[str],
        pathname:      Optional[str],
    ) -> Tuple[Any, ...]:
        # Page Contract §4 Rule C4 — early-exit when the user is not
        # on this page.  Other Inputs (session-store) fire globally;
        # gating here keeps cross-page nav free of useless eval-data
        # fetches.
        if pathname != _RISK_MGMT_ROUTE:
            raise PreventUpdate

        session = Session.from_store(session_data)
        split   = session.split
        source  = source if source in _PNL_COLUMN_BY_SOURCE else DEFAULT_SOURCE
        pnl_col = _PNL_COLUMN_BY_SOURCE[source]

        # Fetch the two eval parquets we need.  Both come back
        # already-renamed to canonical ``predictions`` / ``targets``
        # via ``ArtifactReader._apply_space("original")``.
        portfolio_res = backend.portfolio_df(split, space="original")
        clusters_res  = backend.cluster_timeseries_df(split, space="original")

        portfolio_df = (
            portfolio_res.data
            if portfolio_res.ok and portfolio_res.data is not None
            else pd.DataFrame()
        )
        clusters_df = (
            clusters_res.data
            if clusters_res.ok and clusters_res.data is not None
            else pd.DataFrame()
        )

        # Ensemble version + manifest path for the meta lines.  These
        # come off the eval Overview endpoint (cheap, cached, always
        # available when there's any eval data on disk).
        ensemble_version, manifest_path = _eval_meta(backend)

        # ── Empty / failure states ──────────────────────────────────
        if not portfolio_res.ok:
            reason = portfolio_res.error or "portfolio fetch failed"
            return _empty_state(
                reason=f"Could not load portfolio data: {reason}",
                source=source,
                split=split,
                ensemble_version=ensemble_version,
                manifest_path=manifest_path,
            )
        if portfolio_df.empty:
            return _empty_state(
                reason=(
                    f"No portfolio data for split '{split}'.  Pick "
                    f"another split or re-run evaluation."
                ),
                source=source,
                split=split,
                ensemble_version=ensemble_version,
                manifest_path=manifest_path,
            )
        if pnl_col not in portfolio_df.columns or portfolio_df[pnl_col].isna().all():
            # Phase 3.1 / 3.4 "no scaler coverage" case — the original
            # column is wholly NaN.  Tell the user that flipping
            # source might recover the chart (the other side may have
            # data depending on the eval run).
            other_side = "Actual" if source == "predicted" else "Predicted"
            return _empty_state(
                reason=(
                    f"Original-space {source} PnL is unavailable for "
                    f"this evaluation run (no scaler coverage).  Try "
                    f"flipping the source toggle to {other_side}."
                ),
                source=source,
                split=split,
                ensemble_version=ensemble_version,
                manifest_path=manifest_path,
            )

        # ── Scenario filter resolution ──────────────────────────────
        # The capture callback (``_register_scenario_filter_sync``)
        # owns the dropdown's options + reset behaviour; this render
        # callback only READS the current selection.  Sentinel value
        # → aggregate-across-scenarios, anything else → single-scenario
        # decomposition.  An unknown / unexpected value (race during
        # data refresh) also falls back to aggregate so the page
        # never paints a single-scenario view off an invalid label.
        scenario_arg: Optional[str] = (
            None if (scenario_sel is None or scenario_sel == ALL_SCENARIOS_VALUE)
            else str(scenario_sel)
        )

        # Diagnostic log — kept around because the loopback bug we
        # had earlier was hard to spot without seeing both ends of the
        # filter wire on every fire.  Cheap (one log line per render).
        logger.info(
            "[risk-mgmt] render: split=%s source=%s scenario_sel=%r "
            "scenario_arg=%r clusters_df.shape=%s portfolio_df.shape=%s",
            split, source, scenario_sel, scenario_arg,
            getattr(clusters_df,  "shape", None),
            getattr(portfolio_df, "shape", None),
        )

        # ── KPIs ────────────────────────────────────────────────────
        stats = compute_tail_stats(portfolio_df, pnl_column=pnl_col)

        # ── Figures ─────────────────────────────────────────────────
        # Distribution + tail table stay split-wide regardless of the
        # scenario filter — they describe the whole tail, and would
        # collapse to a single bar / single row in single-scenario
        # mode (which is what the waterfall already shows).
        distribution_fig = build_pnl_distribution(
            portfolio_df, pnl_column=pnl_col, stats=stats,
        )

        # Waterfall: when "All" → pin to the worst scenario in the
        # split (legacy default); when a specific scenario → pin to
        # that one so the trio is consistent.
        pinned_scenario = (
            scenario_arg if scenario_arg is not None
            else _resolve_worst_scenario(portfolio_df, pnl_col)
        )
        waterfall_fig = build_cluster_waterfall(
            clusters_df,
            scenario_label=pinned_scenario,
            pnl_column=pnl_col,
        )
        waterfall_caption = _waterfall_caption_text(
            portfolio_df,
            pinned_scenario,
            pnl_column=pnl_col,
            user_selected=scenario_arg is not None,
        )

        # Tornado + donut honour the scenario filter directly: ``None``
        # → aggregate-across-scenarios (old default), specific label →
        # single-scenario decomposition.
        tornado_fig = build_cluster_tornado(
            clusters_df, pnl_column=pnl_col, scenario_label=scenario_arg,
        )
        donut_fig   = build_cluster_donut(
            clusters_df, pnl_column=pnl_col, scenario_label=scenario_arg,
        )

        # ── Tail table ──────────────────────────────────────────────
        tail_df   = compute_tail_table(
            portfolio_df, clusters_df, pnl_column=pnl_col, n=10,
        )
        tail_body = _tail_table_card_body(tail_df)

        # ── Subtitle + footer ───────────────────────────────────────
        subtitle = _build_subtitle(
            split=split,
            ensemble_version=ensemble_version,
            source=source,
            n_scenarios=len(portfolio_df),
        )
        footer = _format_footer(manifest_path)

        return (
            subtitle,
            _format_currency(stats["var_95"]),
            _format_currency(stats["var_99"]),
            _format_currency(stats["cvar_95"]),
            _format_currency(stats["cvar_99"]),
            _format_signed(stats["skew"], digits=2),
            _format_signed(stats["kurt"], digits=2),
            distribution_fig,
            waterfall_fig,
            tornado_fig,
            donut_fig,
            waterfall_caption,
            tail_body,
            footer,
        )


# ─────────────────────────────────────────────────────────────────────
# Empty-state assembly — keeps the callback return arity stable
# ─────────────────────────────────────────────────────────────────────

def _empty_state(
    *,
    reason:           str,
    source:           str,
    split:            str,
    ensemble_version: Optional[str],
    manifest_path:    Optional[str],
) -> Tuple[Any, ...]:
    """Build the full empty-state output tuple.

    Painted whenever we cannot render the real charts — keeps the
    six KPIs as em-dashes, every figure on its dedicated empty
    placeholder, and prints ``reason`` into the tail-table card so
    the user knows exactly why nothing rendered.  The scenario
    filter is left untouched here (its own sync callback owns its
    state) so the dropdown doesn't visibly flicker on transient
    empty-state passes.
    """
    subtitle = _build_subtitle(
        split=split,
        ensemble_version=ensemble_version,
        source=source,
        n_scenarios=None,
    )
    return (
        subtitle,
        _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER,
        _PLACEHOLDER, _PLACEHOLDER,
        empty_pnl_distribution(),
        empty_cluster_waterfall(),
        empty_cluster_tornado(),
        empty_cluster_donut(),
        "Awaiting evaluation data — scenario pin appears once data loads.",
        _empty_tail_table(reason),
        _format_footer(manifest_path),
    )


def _empty_tail_table(reason: str) -> List[Any]:
    """Tail-table placeholder with a real explanation string."""
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
            reason,
            className="text-xs text-slate-500 italic px-2 py-3",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# Meta helpers (subtitle / footer / formatting)
# ─────────────────────────────────────────────────────────────────────

def _eval_meta(backend: "RadeBackend") -> Tuple[Optional[str], Optional[str]]:
    """Resolve ensemble version + manifest path from the Overview API.

    Used for the page subtitle / footer copy.  Failures are silent —
    the page still renders the rest of its panels, just with em-dashes
    in those two cosmetic slots.
    """
    overview_res = backend.overview()
    if not overview_res.ok or overview_res.data is None:
        return None, None
    ensemble_version = getattr(overview_res.data, "version", None)
    manifest_path    = (
        f"evaluation/{ensemble_version}/manifest.json"
        if ensemble_version else None
    )
    return ensemble_version, manifest_path


def _build_subtitle(
    *,
    split:            str,
    ensemble_version: Optional[str],
    source:           str,
    n_scenarios:      Optional[int],
) -> Any:
    """Subtitle line under the page title.

    Compact one-liner with violet ``run`` ⇒ slate ``ensemble`` ⇒
    chip-like source pill so the user can read the four facts the
    page is conditioned on (split, version, source, scenario count)
    at a glance.
    """
    source_label = "Predicted" if source == "predicted" else "Actual"
    parts: List[Any] = [
        html.Span("Split ", className="text-slate-500"),
        html.Span(
            split.capitalize(),
            className="font-mono text-violet-300",
        ),
    ]
    if ensemble_version:
        parts.extend([
            html.Span(" · ensemble ", className="text-slate-500"),
            html.Span(
                ensemble_version,
                className="font-mono text-slate-300",
            ),
        ])
    parts.extend([
        html.Span(" · source ", className="text-slate-500"),
        html.Span(
            source_label,
            className="font-mono text-slate-300",
        ),
    ])
    if n_scenarios:
        parts.extend([
            html.Span(" · ", className="text-slate-500"),
            html.Span(
                f"{int(n_scenarios):,} scenarios",
                className="text-slate-300",
            ),
        ])
    return html.Span(parts)


def _format_currency(value: Optional[float], *, digits: int = 0) -> str:
    """Pretty-print a PnL-style number; ``—`` for missing/NaN."""
    if value is None:
        return _PLACEHOLDER
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(v):
        return _PLACEHOLDER
    return f"{v:,.{digits}f}"


def _format_signed(value: Optional[float], *, digits: int = 2) -> str:
    """Pretty-print a unitless statistic (e.g. skew, kurt); ``—`` if NaN."""
    if value is None:
        return _PLACEHOLDER
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(v):
        return _PLACEHOLDER
    return f"{v:+.{digits}f}"


def _scenario_options(
    portfolio_df: pd.DataFrame,
    *,
    pnl_col:      str,
    cap:          int = 250,
) -> List[Dict[str, str]]:
    """Build the scenario-filter dropdown options.

    First entry is the "All scenarios" sentinel (default).  The rest
    are the individual scenarios sorted **by worst PnL first** so the
    dropdown is most useful for the events the user is most likely to
    investigate (deep tail, then everything else).

    Scenario labels are pretty-printed as ``<label> · PnL <signed>``
    so the dropdown doubles as a quick "which scenario lost what"
    reference without forcing a click-through to the waterfall.

    ``cap`` keeps the dropdown sane on very large splits — beyond
    250 scenarios the long-tail entries are dropped (Phase 4.x can
    swap in a search/typeahead-only model if we need more).
    """
    head: List[Dict[str, str]] = [
        {"value": ALL_SCENARIOS_VALUE, "label": ALL_SCENARIOS_LABEL},
    ]
    if (
        portfolio_df is None
        or portfolio_df.empty
        or pnl_col not in portfolio_df.columns
        or "scenario_label" not in portfolio_df.columns
    ):
        return head

    sub = portfolio_df[["scenario_label", pnl_col]].copy()
    sub = sub.dropna(subset=[pnl_col])
    if sub.empty:
        return head
    sub = sub.sort_values(pnl_col, ascending=True).head(cap)

    options = head + [
        {
            "value": str(row["scenario_label"]),
            "label": f"{row['scenario_label']} · PnL {row[pnl_col]:+,.0f}",
        }
        for _, row in sub.iterrows()
    ]
    return options


def _resolve_worst_scenario(
    portfolio_df: pd.DataFrame,
    pnl_col:      str,
) -> Optional[str]:
    """Find the worst-portfolio-loss scenario label.

    Returns ``None`` for an empty / malformed frame so the figure
    builder can fall back to its empty placeholder.
    """
    if (
        portfolio_df is None
        or portfolio_df.empty
        or pnl_col not in portfolio_df.columns
        or "scenario_label" not in portfolio_df.columns
    ):
        return None
    series = portfolio_df[pnl_col].dropna()
    if series.empty:
        return None
    worst_idx = series.idxmin()
    return str(portfolio_df.loc[worst_idx, "scenario_label"])


def _waterfall_caption_text(
    portfolio_df:   pd.DataFrame,
    pinned_scenario: Optional[str],
    *,
    pnl_column:    str,
    user_selected: bool = False,
) -> Any:
    """One-line note under the waterfall stating the pinned scenario.

    Tells the user *why* the pin is on this scenario — either
    ``"Worst-loss scenario:"`` (default, automatic) or
    ``"User-selected scenario:"`` (when the scenario filter dropdown
    is on a specific event).  Keeps the cluster trio's intent
    legible at a glance.
    """
    if pinned_scenario is None:
        return "Awaiting evaluation data — scenario pin appears once data loads."
    if (
        portfolio_df is None
        or portfolio_df.empty
        or "scenario_label" not in portfolio_df.columns
    ):
        return f"Pinned: {pinned_scenario}"
    row = portfolio_df.loc[
        portfolio_df["scenario_label"].astype(str) == str(pinned_scenario)
    ]
    if row.empty or pnl_column not in row.columns:
        return f"Pinned: {pinned_scenario}"
    total = float(row[pnl_column].iloc[0])
    lead  = (
        "User-selected scenario: " if user_selected
        else "Worst-loss scenario: "
    )
    return html.Span([
        html.Span(lead, className="text-slate-500"),
        html.Span(
            str(pinned_scenario),
            className="font-mono text-violet-300",
        ),
        html.Span(" · portfolio PnL ", className="text-slate-500"),
        html.Span(
            f"{total:,.0f}",
            className="font-mono " + (
                "text-rose-300" if total < 0 else "text-emerald-300"
            ),
        ),
        html.Span(
            " · Phase 4.2 enables RF-level decomposition.",
            className="text-slate-600",
        ),
    ])


def _format_footer(manifest_path: Optional[str]) -> str:
    if manifest_path is None:
        return "Source: evaluation manifest not yet resolved …"
    return f"Source: {manifest_path}"


# ─────────────────────────────────────────────────────────────────────
# Tail table — HTML table builder
# ─────────────────────────────────────────────────────────────────────

def _tail_table_card_body(tail_df: pd.DataFrame) -> List[Any]:
    """Populated tail table.

    Five real columns (``Scenario``, ``Date``, ``PnL``, ``Top
    cluster``, ``Share of tail loss``) plus two placeholder columns
    (``Top RF``, ``Top trade type``) rendered as em-dash so the
    layout is stable when Phase 4.1 / 4.2 fill them.

    ``Share of tail loss`` is bounded 0-100% and sums to 100% across
    the displayed rows — it answers "which of the worst-N scenarios
    drives this tail?".  Replaces the old ``VaR contrib %`` column,
    which could exceed 100% (a scenario worse than the CVaR-95
    average breaches the implied 0-100 scale).
    """
    header_block: List[Any] = [
        html.Div(
            "Tail-conditional table",
            className="text-sm font-semibold text-slate-200",
        ),
        html.Div(
            f"Scenarios at or below the 5th-percentile portfolio loss · "
            f"showing top {len(tail_df)} worst.  ``Share of tail loss`` "
            f"sums to 100% across the rows.",
            className="text-xs text-slate-500",
        ),
    ]

    if tail_df is None or tail_df.empty:
        header_block.append(html.Div(
            "No tail scenarios — portfolio PnL series did not yield any "
            "below-quantile observations.",
            className="text-xs text-slate-500 italic px-2 py-3",
        ))
        return header_block

    header = html.Thead(html.Tr(className="rade-diag-header", children=[
        html.Th("Scenario",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Date",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Portfolio PnL",
            className="text-right px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Top cluster",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Cluster PnL",
            className="text-right px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Top RF",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-500"),
        html.Th("Top trade type",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-500"),
        html.Th("Share of tail loss",
            className="text-right px-2 py-1.5 text-xs font-semibold text-slate-300"),
    ]))

    body_rows: List[Any] = []
    for _, row in tail_df.iterrows():
        scenario   = str(row.get("scenario_label", ""))
        date_str   = _date_from_scenario_label(scenario)
        pnl_value  = float(row.get("portfolio_pnl") or 0.0)
        top_cluster = str(row.get("top_cluster") or _PLACEHOLDER)
        cluster_pnl = (
            float(row.get("top_cluster_pnl"))
            if pd.notna(row.get("top_cluster_pnl"))
            else None
        )
        share_pct = float(row.get("share_of_tail_loss") or 0.0)

        body_rows.append(html.Tr(
            className="rade-diag-row border-t border-slate-800",
            children=[
                html.Td(
                    scenario,
                    className="rade-grid-mono px-2 py-1.5 text-xs text-violet-300",
                ),
                html.Td(
                    date_str,
                    className="px-2 py-1.5 text-xs text-slate-400",
                ),
                html.Td(
                    f"{pnl_value:,.0f}",
                    className=(
                        "rade-grid-mono px-2 py-1.5 text-xs text-right " +
                        ("text-rose-300" if pnl_value < 0 else "text-emerald-300")
                    ),
                ),
                html.Td(
                    top_cluster,
                    className="rade-grid-mono px-2 py-1.5 text-xs text-slate-300",
                ),
                html.Td(
                    f"{cluster_pnl:,.0f}" if cluster_pnl is not None else _PLACEHOLDER,
                    className=(
                        "rade-grid-mono px-2 py-1.5 text-xs text-right " +
                        ("text-rose-300" if (cluster_pnl or 0) < 0 else "text-slate-400")
                    ),
                ),
                html.Td(
                    _PLACEHOLDER,
                    className="px-2 py-1.5 text-xs text-slate-600 italic",
                ),
                html.Td(
                    _PLACEHOLDER,
                    className="px-2 py-1.5 text-xs text-slate-600 italic",
                ),
                html.Td(
                    _share_of_tail_loss_cell(share_pct),
                    className="px-2 py-1.5",
                ),
            ],
        ))

    table = html.Table(
        className="w-full text-xs",
        children=[header, html.Tbody(body_rows)],
    )

    return header_block + [
        html.Div(
            className="overflow-y-auto min-w-0",
            style={"maxHeight": "360px"},
            children=table,
        ),
    ]


def _share_of_tail_loss_cell(pct: float) -> Any:
    """Mini horizontal bar showing share of the displayed tail loss.

    ``pct`` is bounded [0, 100] by the upstream formula, so the
    inline bar width is a direct read with no clipping logic needed.
    Replaces the legacy ``_var_contribution_cell`` whose input
    column could exceed 100% (scenarios worse than CVaR-95 average).
    """
    width = max(0.0, min(pct, 100.0))
    return html.Div(
        className="flex items-center gap-2 justify-end",
        children=[
            html.Span(
                f"{pct:.1f}%",
                className="rade-grid-mono text-xs text-slate-300",
            ),
            html.Div(
                className=(
                    "h-1.5 rounded-full bg-slate-800 overflow-hidden"
                ),
                style={"width": "60px"},
                children=html.Div(
                    style={
                        "width":      f"{width}%",
                        "background": "#f43f5e",
                        "height":     "100%",
                    },
                ),
            ),
        ],
    )


def _date_from_scenario_label(label: str) -> str:
    """Best-effort date extraction from a scenario label.

    Many of our scenario labels follow a ``yyyy-mm-dd``-shaped
    convention (eval splits + new-scenario folder names from the
    data pipeline).  If the input doesn't parse, we silently fall
    back to ``—`` so the column reads cleanly across mixed schemes.
    """
    if not label:
        return _PLACEHOLDER
    try:
        ts = pd.to_datetime(label, errors="coerce")
    except Exception:  # pragma: no cover — defensive
        return _PLACEHOLDER
    if pd.isna(ts):
        return _PLACEHOLDER
    return ts.strftime("%Y-%m-%d")


__all__ = ["register"]
