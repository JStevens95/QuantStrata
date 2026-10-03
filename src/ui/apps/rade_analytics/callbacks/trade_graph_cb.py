"""Evaluation → Trade-Graph sub-tab callbacks (Phase E.3 rebuild).

Page-Contract structure (§2 capture / render split)
---------------------------------------------------
:func:`register` delegates to two section helpers.

Capture callbacks (input gestures → session writes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* :func:`_register_sync_header_to_session`     — cluster / layout /
  color-by / threshold widgets → session.
* :func:`_register_sync_node_tap_to_session`   — Cytoscape ``tapNodeData``
  → ``session.trade_graph_selected_trade_id``.
* :func:`_register_sync_neighbours_k_to_session` — popover NumberInput
  → ``session.trade_graph_neighbour_k``.
* :func:`_register_sync_neighbour_click_to_session` — pattern-matching
  click on a neighbour row → ``session.trade_graph_selected_trade_id``.
* :func:`_register_sync_deep_dive_button`      — "Open Deep Dive" button
  → writes deep_dive session fields + URL navigation.

Render callbacks (state → DOM, no session writes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* :func:`_register_bootstrap`                  — mount tripwire fires
  once per fresh mount; populates the cluster Select's ``data`` and
  handles the two narrow capture-edges (URL deep-link, fresh-user
  default).
* :func:`_register_render_graph`               — cluster / threshold /
  color-by / layout → cytoscape ``elements``, mini-map ``elements``,
  ``stylesheet``, pane status, store_graph, cluster-stats grid.
  Only render callback that writes the ``store_graph`` payload.
* :func:`_register_render_selected_card`       — ``store_graph`` +
  selected_trade_id → trade_id text, chip strip, metrics row,
  neighbours-button enabled state, deep-dive button enabled state.
* :func:`_register_render_neighbours_list`     — store_graph + selected
  + k → popover scroll-list children.
* :func:`_register_render_legend`              — color-by + store_graph
  → legend body children.
* :func:`_register_render_threshold_label`     — slider value → label
  (kept here so the slider doesn't leak threshold-state to the rest
  of the app via the session-store on every drag).
* :func:`_register_render_density_chart`       — ensemble graph_stats →
  density distribution figure.
* :func:`_register_render_edges_vs_nodes_chart` — same → edges-vs-nodes
  scatter figure.

Clientside callbacks
~~~~~~~~~~~~~~~~~~~~
* "Fit view" button — clientside ``cy.fit()`` via
  :data:`_FIT_VIEW_CLIENTSIDE`.
* "Export PNG" button — clientside download via
  :data:`_EXPORT_PNG_CLIENTSIDE`.

Why pathname-gating
-------------------
Every render callback returns ``no_update`` (or :class:`PreventUpdate`)
when ``pathname`` isn't ``/evaluation/trade-graph``.  Page Contract §4
Rule C2 — cheap, idempotent, prevents render storms when the user
navigates between sub-tabs.

Why the ``store_graph`` ephemeral store
---------------------------------------
``RadeBackend.trade_graph(cluster_id)`` is the heaviest single fetch
on the page (full node + edge list).  We pay it once per cluster
selection and cache the deserialised payload in a memory store.
Threshold filter, color-by toggle, neighbours popover, selected-card
re-renders all read from the store.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from dash import (
    ALL,
    ClientsideFunction,
    Input,
    Output,
    State,
    ctx,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

from ..components.kpi_card import KpiCard
from ..data.session import (
    DEFAULT_TRADE_GRAPH_LAYOUT,
    EVALUATION_TRADE_GRAPH_COLOR_BY,
    EVALUATION_TRADE_GRAPH_LAYOUTS,
    Session,
)
from ..figures import (
    build_legend_body,
    build_stylesheet,
    density_distribution,
    edges_vs_nodes_scatter,
    empty_figure,
)
from ..layouts.evaluation.trade_graph import (
    COLOR_BY_LABELS,
    TRADE_GRAPH_IDS,
)
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


_TRADE_GRAPH_PATH = "/evaluation/trade-graph"
_PLACEHOLDER      = "—"


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Trade-Graph sub-tab callback to ``app``.

    Mirrors the template_cb.py structure — one capture section, one
    render section, plus the clientside callbacks at the end.
    """
    _register_capture(app)
    _register_render(app, backend)
    _register_clientside(app)


# ─────────────────────────────────────────────────────────────────────
# Section dispatchers
# ─────────────────────────────────────────────────────────────────────


def _register_capture(app: "Dash") -> None:
    _register_sync_header_to_session(app)
    _register_sync_node_tap_to_session(app)
    _register_sync_neighbours_k_to_session(app)
    _register_sync_neighbour_click_to_session(app)
    # Deep-dive button → URL navigation lives in
    # ``cluster_deep_dive_cb._register_navigate_from_trade_graph`` so
    # the navigation logic sits next to the page that consumes the
    # deep-link.  We only own the button's enabled/disabled state
    # (rendered out of ``_register_render_selected_card`` below).


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    _register_bootstrap(app, backend)
    _register_render_graph(app, backend)
    _register_render_selected_card(app)
    _register_render_neighbours_list(app)
    _register_render_legend(app)
    _register_render_threshold_label(app)
    _register_render_density_chart(app, backend)
    _register_render_edges_vs_nodes_chart(app, backend)


# ═════════════════════════════════════════════════════════════════════
# CAPTURE — input gestures → session writes
# ═════════════════════════════════════════════════════════════════════


def _register_sync_header_to_session(app: "Dash") -> None:
    """Header widgets → session.

    The four header inputs (cluster, layout, color-by, threshold) all
    converge on a single capture callback.  ``ctx.triggered_id`` tells
    us which one fired so we mutate the matching session field and
    leave the others alone.

    Threshold is included here (not in a separate callback) so dragging
    the slider doesn't write to session on every frame — we already use
    ``updatemode="mouseup"`` on the slider to throttle to release.
    """

    @app.callback(
        Output(SHELL_IDS["session_store"],            "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["cluster_select"],      "value"),
        Input(TRADE_GRAPH_IDS["layout_radio"],        "value"),
        Input(TRADE_GRAPH_IDS["color_by_select"],     "value"),
        Input(TRADE_GRAPH_IDS["threshold_slider"],    "value"),
        State(SHELL_IDS["session_store"],             "data"),
        prevent_initial_call=True,
    )
    def _sync(
        cluster:     Optional[str],
        layout_name: Optional[str],
        color_by:    Optional[str],
        threshold:   Optional[float],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        changed = False

        if trigger == TRADE_GRAPH_IDS["cluster_select"]:
            new_cluster = cluster if cluster else None
            if ev.trade_graph_cluster_id != new_cluster:
                ev.trade_graph_cluster_id = new_cluster
                # Changing cluster invalidates the selected trade — the
                # node id would no longer match anything in the new
                # graph.
                ev.trade_graph_selected_trade_id = None
                # Reset threshold to 0 so the slider starts at "show
                # every edge" with the new cluster's weight range.
                # The render callback rebuilds slider min/max/step/
                # marks from the new payload's edge weights.
                ev.trade_graph_weight_threshold = 0.0
                changed = True

        elif trigger == TRADE_GRAPH_IDS["layout_radio"]:
            if (
                layout_name in EVALUATION_TRADE_GRAPH_LAYOUTS
                and ev.trade_graph_layout != layout_name
            ):
                ev.trade_graph_layout = layout_name
                changed = True

        elif trigger == TRADE_GRAPH_IDS["color_by_select"]:
            if (
                color_by in EVALUATION_TRADE_GRAPH_COLOR_BY
                and ev.trade_graph_color_by != color_by
            ):
                ev.trade_graph_color_by = color_by
                changed = True

        elif trigger == TRADE_GRAPH_IDS["threshold_slider"]:
            # Slider max is per-cluster now, so we can't clamp at
            # 1.0 here — accept any non-negative value and let the
            # render callback's ``_slider_props_for_payload`` clamp
            # if it ends up out of range for the next cluster.
            try:
                new_threshold = max(0.0, float(threshold))
            except (TypeError, ValueError):
                raise PreventUpdate
            if abs(ev.trade_graph_weight_threshold - new_threshold) > 1e-9:
                ev.trade_graph_weight_threshold = new_threshold
                changed = True

        if not changed:
            raise PreventUpdate
        return session.to_store()


def _register_sync_node_tap_to_session(app: "Dash") -> None:
    """Cytoscape node tap → ``session.trade_graph_selected_trade_id``."""

    @app.callback(
        Output(SHELL_IDS["session_store"],     "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["cytoscape"],    "tapNodeData"),
        State(SHELL_IDS["session_store"],      "data"),
        prevent_initial_call=True,
    )
    def _on_tap(
        node_data:    Optional[Dict[str, Any]],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not node_data:
            raise PreventUpdate
        trade_id = node_data.get("id")
        if not trade_id:
            raise PreventUpdate

        session = Session.from_store(session_data)
        if session.evaluation.trade_graph_selected_trade_id == trade_id:
            # Re-tap on the same node is a no-op — saves a session round
            # trip + the cascade of render callbacks that would follow.
            raise PreventUpdate

        session.evaluation.trade_graph_selected_trade_id = trade_id
        return session.to_store()


def _register_sync_neighbours_k_to_session(app: "Dash") -> None:
    """Popover NumberInput → ``session.trade_graph_neighbour_k``."""

    @app.callback(
        Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["selected_neighbours_k_input"], "value"),
        State(SHELL_IDS["session_store"], "data"),
        prevent_initial_call=True,
    )
    def _sync_k(
        new_k:        Any,
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            new_k_int = int(new_k)
        except (TypeError, ValueError):
            raise PreventUpdate
        new_k_int = max(1, min(20, new_k_int))

        session = Session.from_store(session_data)
        if session.evaluation.trade_graph_neighbour_k == new_k_int:
            raise PreventUpdate
        session.evaluation.trade_graph_neighbour_k = new_k_int
        return session.to_store()


def _register_sync_neighbour_click_to_session(app: "Dash") -> None:
    """Click on a neighbour row inside the popover → drill to that trade.

    Each row's id is a pattern-matching dict
    (``{"type": "tg-neighbour-row", "trade_id": <id>}``); a single
    callback handles every row's click via the ``ALL`` wildcard.
    """

    @app.callback(
        Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
        Input(
            {"type": "tg-neighbour-row", "trade_id": ALL},
            "n_clicks",
        ),
        State(SHELL_IDS["session_store"], "data"),
        prevent_initial_call=True,
    )
    def _on_click(
        n_clicks_list: Sequence[Optional[int]],
        session_data:  Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # The ``ALL`` pattern fires this callback on initial render
        # too (every n_clicks=None).  Filter to genuine clicks.
        if not n_clicks_list or not any(n_clicks_list):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or "trade_id" not in triggered:
            raise PreventUpdate

        target_trade = triggered["trade_id"]
        session = Session.from_store(session_data)
        if session.evaluation.trade_graph_selected_trade_id == target_trade:
            raise PreventUpdate

        session.evaluation.trade_graph_selected_trade_id = target_trade
        return session.to_store()


# ═════════════════════════════════════════════════════════════════════
# RENDER — bootstrap (mount tripwire)
# ═════════════════════════════════════════════════════════════════════


def _register_bootstrap(app: "Dash", backend: "RadeBackend") -> None:
    """Populate the cluster Select on fresh mount of the page.

    The ``mount_signal`` Store fires this callback once per fresh
    mount (Page Contract §3 Rule L4).  We fetch the cluster list from
    the backend and write the option set to the Select.

    Override edges
    ~~~~~~~~~~~~~~
    * **Fresh-user default** — if ``session.trade_graph_cluster_id``
      is unset and we have clusters, we pick the first one and write
      both ``Select.value`` and ``session`` in the same return tuple.
    * **Stale session id** — if the session-stored cluster id no
      longer exists in the backend (e.g. version flipped), we fall
      back to the first option and overwrite session.

    URL deep-link (``?cluster=<id>``) is intentionally *not* wired
    yet — falls under the broader "deep-link the eval sub-tabs"
    Stage 4.x effort.  The callback structure leaves a clean seam for
    that follow-up.
    """

    @app.callback(
        Output(TRADE_GRAPH_IDS["cluster_select"], "data"),
        Output(TRADE_GRAPH_IDS["cluster_select"], "value", allow_duplicate=True),
        Output(SHELL_IDS["session_store"],        "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["mount_signal"],    "data"),
        State(SHELL_IDS["session_store"],         "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _bootstrap(
        _trigger:     Any,
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], Any, Any]:
        res = backend.clusters_df()
        if not res.ok or res.data is None or res.data.empty:
            return [], no_update, no_update

        df = res.data
        options = [
            {"value": cid, "label": cid}
            for cid in sorted(df["cluster_id"].unique())
        ]

        session = Session.from_store(session_data)
        ev = session.evaluation
        valid_ids = {o["value"] for o in options}

        # Fresh-user default — neither the trade-graph nor the global
        # cluster_id is set.
        if not ev.trade_graph_cluster_id and not session.cluster_id:
            default_value = options[0]["value"]
            ev.trade_graph_cluster_id = default_value
            return options, default_value, session.to_store()

        # Stale session id — fall back to first option.
        active = ev.trade_graph_cluster_id or session.cluster_id
        if active not in valid_ids:
            default_value = options[0]["value"]
            ev.trade_graph_cluster_id = default_value
            return options, default_value, session.to_store()

        # Steady state — option list only; the layout-time seeded
        # value already matches session, no value-side write needed.
        return options, no_update, no_update


# ═════════════════════════════════════════════════════════════════════
# RENDER — graph elements + stylesheet + cluster-stats card
# ═════════════════════════════════════════════════════════════════════


def _register_render_graph(app: "Dash", backend: "RadeBackend") -> None:
    """Cluster / threshold / color-by → graph elements + stylesheet.

    Single fetch per cluster — node + edge lists are stashed in
    ``store_graph`` for downstream callbacks (selected-card,
    neighbours popover, legend).  Color-by changes don't re-fetch;
    they just rebuild the stylesheet from the cached payload.

    Trade attributes (residual gradient, asset_class / currency /
    product categorical modes) are **enriched** here from
    :meth:`RadeBackend.trades_df` so the stylesheet helper can
    paint nodes by the chosen mode.
    """

    @app.callback(
        Output(TRADE_GRAPH_IDS["cytoscape"],          "elements"),
        Output(TRADE_GRAPH_IDS["cytoscape"],          "stylesheet"),
        Output(TRADE_GRAPH_IDS["cytoscape"],          "layout"),
        Output(TRADE_GRAPH_IDS["cytoscape_minimap"],  "elements"),
        Output(TRADE_GRAPH_IDS["cytoscape_minimap"],  "layout"),
        Output(TRADE_GRAPH_IDS["pane_status"],        "children"),
        Output(TRADE_GRAPH_IDS["cluster_stats_grid"], "children"),
        Output(TRADE_GRAPH_IDS["store_graph"],        "data"),
        Output(TRADE_GRAPH_IDS["threshold_slider"],   "min"),
        Output(TRADE_GRAPH_IDS["threshold_slider"],   "max"),
        Output(TRADE_GRAPH_IDS["threshold_slider"],   "step"),
        Output(TRADE_GRAPH_IDS["threshold_slider"],   "marks"),
        Output(TRADE_GRAPH_IDS["threshold_slider"],   "value",
               allow_duplicate=True),
        Input(SHELL_IDS["url"],                       "pathname"),
        Input(SHELL_IDS["session_store"],             "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        cluster_id = ev.trade_graph_cluster_id or session.cluster_id

        layout_name = ev.trade_graph_layout or DEFAULT_TRADE_GRAPH_LAYOUT
        layout_cfg = _build_layout_cfg(layout_name)
        minimap_layout_cfg = {**layout_cfg, "padding": 4, "animate": False}

        empty_grid = _cluster_stats_grid(None)
        empty_slider = _slider_props_for_payload(None, threshold=0.0)

        if not cluster_id:
            return (
                [],
                build_stylesheet(ev.trade_graph_color_by, nodes_payload=[])[0],
                layout_cfg,
                [],
                minimap_layout_cfg,
                "No cluster selected.",
                empty_grid,
                {},
                *empty_slider,
            )

        res = backend.trade_graph(cluster_id=cluster_id)
        if not res.ok or res.data is None:
            logger.info(
                "trade-graph fetch failed for cluster %s: %s",
                cluster_id, res.error,
            )
            return (
                [],
                build_stylesheet(ev.trade_graph_color_by, nodes_payload=[])[0],
                layout_cfg,
                [],
                minimap_layout_cfg,
                f"Graph unavailable for {cluster_id}.",
                empty_grid,
                {},
                *empty_slider,
            )

        payload = res.data

        # Trade-level attribute lookup for color-by enrichment.  Per-
        # node residual lives on ``mean_residual``; categorical
        # attributes (asset_class / currency / product) live on the
        # cluster row, so every node in this cluster shares the same
        # value (the colour-by helper falls back gracefully when
        # values are missing).
        residuals_by_trade = _per_trade_residuals(backend, session.split, cluster_id)
        cluster_attrs = _cluster_attrs(backend, cluster_id)

        nodes_payload: List[Dict[str, Any]] = []
        for n in payload.nodes:
            data = {
                "id":         n.trade_id,
                "trade_type": n.trade_type,
                "cluster_id": n.cluster_id,
            }
            residual = residuals_by_trade.get(n.trade_id)
            if residual is not None and not _is_nan(residual):
                data["residual"] = float(residual)
            for k, v in cluster_attrs.items():
                data[k] = v
            nodes_payload.append({"data": data})

        # Threshold filter — drops weak edges before painting.  Cheap
        # enough to redo in this callback (we already have the
        # payload in hand) so the threshold slider's own render
        # callback only owns the label.
        thr = ev.trade_graph_weight_threshold
        edges_payload: List[Dict[str, Any]] = []
        for e in payload.edges:
            w = float(e.weight)
            if w < thr:
                continue
            edges_payload.append({"data": {"source": e.source, "target": e.target, "weight": w}})

        n_hidden = len(payload.edges) - len(edges_payload)
        status = (
            f"{payload.cluster_id} · "
            f"{payload.stats.n_nodes:,} nodes · "
            f"{len(edges_payload):,} edges"
        )
        if n_hidden > 0:
            status += f" · {n_hidden:,} hidden"

        stylesheet, _legend_pairs = build_stylesheet(
            ev.trade_graph_color_by, nodes_payload=nodes_payload,
        )

        elements = [*nodes_payload, *edges_payload]

        cluster_stats_children = _cluster_stats_grid(
            {
                "n_nodes":     payload.stats.n_nodes,
                "n_edges":     len(edges_payload),
                "density":     payload.stats.density,
                "mean_weight": payload.stats.mean_weight,
            }
        )

        # Store payload for downstream callbacks — keep the raw node
        # list, the filtered edge list, and a mapping of edges by
        # source so the neighbours popover can find top-k weights
        # without re-walking the whole edge list.
        store_payload: Dict[str, Any] = {
            "cluster_id":   payload.cluster_id,
            "nodes":        nodes_payload,
            "edges":        edges_payload,
            "n_target":     payload.n_target_trades,
            "n_elementary": payload.n_elementary_trades,
        }

        # Per-cluster slider params — min stays at 0, max is the
        # cluster's max edge weight, step is max/6 (six discrete
        # increments), marks label 0 / mean / max.
        slider_props = _slider_props_for_payload(
            payload, threshold=ev.trade_graph_weight_threshold,
        )

        return (
            elements,
            stylesheet,
            layout_cfg,
            elements,           # mini-map shares the same elements
            minimap_layout_cfg,
            status,
            cluster_stats_children,
            store_payload,
            *slider_props,
        )


def _slider_props_for_payload(
    payload: Optional[Any],
    *,
    threshold: float,
) -> Tuple[float, float, float, Dict[Any, Any], Any]:
    """Compute per-cluster slider props.

    Returns ``(min, max, step, marks, value)`` for the threshold
    slider, with the following semantics:

    * **min**  always ``0``.
    * **max**  the maximum edge weight in the cluster (or ``1.0`` as
      a safe fallback when the payload is empty / missing).
    * **step** ``max / 6`` so the user moves through six discrete
      increments from 0 → max.
    * **marks** three labelled tick stops at ``0`` / mean / max so the
      cluster's mean weight is visible without making the slider
      logarithmic.
    * **value** clamped to ``[0, max]`` so a stale threshold from
      the previous cluster never sits past the new max (which would
      hide every edge by accident).

    Round-trip: the returned ``value`` is fed back through the
    capture callback only when it actually differs from
    ``threshold`` — otherwise dash-cytoscape emits ``no_update``
    elsewhere and the slider doesn't visibly twitch.
    """
    if payload is None or not getattr(payload, "edges", None):
        # Sensible fallback so the slider isn't broken when no
        # cluster is selected or the payload came back empty.
        return 0.0, 1.0, 0.05, {0: "0", 1: "1"}, no_update

    weights = [float(e.weight) for e in payload.edges]
    max_w = max(weights) if weights else 1.0
    if max_w <= 0:
        max_w = 1.0
    step = round(max_w / 6.0, 4) or 0.01
    mean_w = float(payload.stats.mean_weight)
    if mean_w <= 0 or mean_w >= max_w:
        mean_w = max_w / 2.0

    # dcc.Slider's marks dict accepts numeric keys; strings render
    # as labels.  Three stops keep the slider readable on a 240-px
    # header column without crowding.
    marks: Dict[Any, Any] = {
        0: "0",
        round(mean_w, 3): {
            "label": f"μ {mean_w:.3g}",
            "style": {"color": "#a3a3a3", "fontSize": "10px"},
        },
        round(max_w, 3): {
            "label": f"max {max_w:.3g}",
            "style": {"color": "#cbd5e1", "fontSize": "10px"},
        },
    }

    # Clamp the slider only when the persisted threshold is
    # genuinely out of range for the new cluster (e.g. cluster_3
    # had max 0.95 but the user just switched to cluster_7 where
    # max is 0.4 — without clamping the slider would sit past max
    # and hide every edge).  Otherwise return ``no_update`` so the
    # user's drag gesture never snaps back mid-flight.
    if threshold > max_w:
        clamped: Any = 0.0
    else:
        clamped = no_update

    return 0.0, float(max_w), float(step), marks, clamped


def _cluster_stats_grid(stats: Optional[Dict[str, Any]]) -> List[Any]:
    """2×2 KPI grid children for the Cluster Stats card."""
    if stats is None:
        return [
            KpiCard(label="Nodes",       value=_PLACEHOLDER),
            KpiCard(label="Edges",       value=_PLACEHOLDER),
            KpiCard(label="Density",     value=_PLACEHOLDER),
            KpiCard(label="Mean weight", value=_PLACEHOLDER),
        ]
    return [
        KpiCard(label="Nodes",       value=_fmt_int(stats.get("n_nodes"))),
        KpiCard(label="Edges",       value=_fmt_int(stats.get("n_edges"))),
        KpiCard(label="Density",     value=_fmt_float(stats.get("density"))),
        KpiCard(label="Mean weight", value=_fmt_float(stats.get("mean_weight"))),
    ]


def _per_trade_residuals(
    backend: "RadeBackend", split: str, cluster_id: str,
) -> Dict[str, float]:
    """Return ``{trade_id: mean_residual}`` for every trade in the cluster.

    Returns an empty dict on any backend failure — the caller treats
    "no residuals available" as "skip residual colouring", which
    falls back to ``trade_type`` via the stylesheet helper.
    """
    res = backend.trades_df(split, cluster_id=cluster_id)
    if not res.ok or res.data is None or res.data.empty:
        return {}
    df = res.data
    if "trade_id" not in df.columns or "mean_residual" not in df.columns:
        return {}
    return dict(zip(df["trade_id"], df["mean_residual"]))


def _cluster_attrs(
    backend: "RadeBackend", cluster_id: str,
) -> Dict[str, Any]:
    """Return the per-cluster categorical attributes.

    Every node in a cluster shares the same asset_class / currency /
    product (clusters are *defined* by these attributes), so a single
    lookup feeds every node.  When a column is missing we omit the
    key — the stylesheet helper's fallback handles the missing case.

    The output keys (``asset_class`` / ``currency_code`` /
    ``product_code``) are the **canonical** names the stylesheet
    helper expects in
    :data:`figures.trade_graph_stylesheet._NODE_COLUMN_FOR_MODE`.
    The actual parquet column name might be an alias
    (e.g. ``assetclass`` / ``ccy`` / ``product``); we walk
    :data:`data.backend.FILTER_DIMENSION_COLUMN_CANDIDATES` and pick
    whichever is present, then re-key under the canonical name so
    the stylesheet doesn't need to know about parquet aliases.
    """
    from ..data.backend import FILTER_DIMENSION_COLUMN_CANDIDATES

    res = backend.clusters_df(cluster_id=cluster_id)
    if not res.ok or res.data is None or res.data.empty:
        return {}
    row = res.data.iloc[0]
    columns = res.data.columns

    canonical_to_candidates = {
        "asset_class":   FILTER_DIMENSION_COLUMN_CANDIDATES["asset_class"],
        "currency_code": FILTER_DIMENSION_COLUMN_CANDIDATES["currency"],
        "product_code":  FILTER_DIMENSION_COLUMN_CANDIDATES["product"],
    }

    out: Dict[str, Any] = {}
    for canonical, candidates in canonical_to_candidates.items():
        for col in candidates:
            if col in columns:
                value = row.get(col)
                if value is not None and not _is_nan(value):
                    out[canonical] = str(value)
                break
    return out


# ═════════════════════════════════════════════════════════════════════
# RENDER — Selected-Trade card
# ═════════════════════════════════════════════════════════════════════


def _register_render_selected_card(app: "Dash") -> None:
    """Reflect the active selection inside the Selected-Trade card.

    All inputs come from the local store + session — no backend hit
    so card paints synchronously after every node tap.
    """

    @app.callback(
        Output(TRADE_GRAPH_IDS["selected_trade_id"],       "children"),
        Output(TRADE_GRAPH_IDS["selected_chip_strip"],     "children"),
        Output(TRADE_GRAPH_IDS["selected_metrics"],        "children"),
        Output(TRADE_GRAPH_IDS["selected_copy_btn"],       "disabled"),
        Output(TRADE_GRAPH_IDS["selected_deep_dive_btn"],  "disabled"),
        Input(SHELL_IDS["url"],                            "pathname"),
        Input(SHELL_IDS["session_store"],                  "data"),
        Input(TRADE_GRAPH_IDS["store_graph"],              "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
        store_data:   Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any, Any, bool, bool]:
        # NOTE: the neighbours button is *not* output here — it stays
        # always-enabled so its dmc.Popover trigger always fires (a
        # disabled HTML button has ``pointer-events: none`` which
        # blocks the popover click event entirely).  The dropdown
        # body shows "Pick a node…" while no selection exists, so a
        # stray click without a selection is harmless.
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        selected_id = session.evaluation.trade_graph_selected_trade_id

        if not selected_id or not store_data or not store_data.get("nodes"):
            return (
                _PLACEHOLDER,
                [],
                "Click a node in the graph to inspect the trade.",
                True,    # copy disabled
                True,    # deep-dive disabled
            )

        node = _find_node(store_data["nodes"], selected_id)
        if node is None:
            return (
                _PLACEHOLDER,
                [],
                f"Trade '{selected_id}' is not in the active cluster.",
                True, True,
            )

        chips = _build_chips(node)
        metrics = _build_metrics_row(node)

        return (
            selected_id,
            chips,
            metrics,
            False,    # copy enabled
            False,    # deep-dive enabled
        )


def _find_node(
    nodes_payload: Sequence[Dict[str, Any]], trade_id: str,
) -> Optional[Dict[str, Any]]:
    for n in nodes_payload:
        if n.get("data", {}).get("id") == trade_id:
            return n.get("data", {})
    return None


def _build_chips(node_data: Dict[str, Any]) -> List[Any]:
    """Return the chip strip for the Selected-Trade card.

    Renders one chip per categorical attribute we know about.  The
    chips share the ``rade-filter-chip`` style from rade.css so they
    visually match the global filter bar's chips.
    """
    rows: List[Tuple[str, str]] = []
    if (tt := node_data.get("trade_type")):
        rows.append(("Type", tt.capitalize()))
    if (ac := node_data.get("asset_class")):
        rows.append(("Asset", ac))
    if (cc := node_data.get("currency_code")):
        rows.append(("CCY", cc))
    if (pc := node_data.get("product_code")):
        rows.append(("Product", pc))

    return [
        html.Span(
            f"{label}: {value}",
            className=(
                "px-2 py-0.5 rounded-md text-[11px] "
                "bg-slate-800 text-slate-200 border border-slate-700"
            ),
        )
        for label, value in rows
    ]


def _build_metrics_row(node_data: Dict[str, Any]) -> Any:
    residual = node_data.get("residual")
    if residual is None or _is_nan(residual):
        return html.Span(
            "No metrics available for this trade.",
            className="text-[11px] text-slate-500 italic",
        )
    return html.Div(
        className="text-xs text-slate-300 flex items-center gap-2",
        children=[
            html.Span("Mean residual", className="text-slate-500"),
            html.Span(_fmt_float(residual), className="font-mono text-slate-100"),
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# RENDER — Neighbours popover list
# ═════════════════════════════════════════════════════════════════════


def _register_render_neighbours_list(app: "Dash") -> None:
    """Selected trade + k + cached graph → top-k neighbour rows."""

    @app.callback(
        Output(TRADE_GRAPH_IDS["selected_neighbours_list"],     "children"),
        Output(TRADE_GRAPH_IDS["selected_neighbours_btn_label"], "children"),
        Input(SHELL_IDS["url"],                                  "pathname"),
        Input(SHELL_IDS["session_store"],                        "data"),
        Input(TRADE_GRAPH_IDS["store_graph"],                    "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
        store_data:   Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any]:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        selected_id = ev.trade_graph_selected_trade_id
        k = max(1, min(20, ev.trade_graph_neighbour_k))
        button_label = f"Nearest {k}"

        if not selected_id or not store_data or not store_data.get("edges"):
            return (
                [
                    html.Span(
                        "Pick a node to load its neighbours.",
                        className="text-xs text-slate-500 italic",
                    ),
                ],
                button_label,
            )

        neighbours = _top_k_neighbours(store_data["edges"], selected_id, k=k)
        if not neighbours:
            return (
                [
                    html.Span(
                        "This trade has no neighbours under the active threshold.",
                        className="text-xs text-slate-500 italic",
                    ),
                ],
                button_label,
            )

        rows = [
            html.Div(
                id={"type": "tg-neighbour-row", "trade_id": tid},
                className=(
                    "flex items-center justify-between gap-2 "
                    "px-2 py-1 rounded-md cursor-pointer "
                    "hover:bg-slate-800 transition-colors"
                ),
                # ``n_clicks`` initialised so the pattern-matching
                # callback's ``any(n_clicks_list)`` filter works.
                n_clicks=0,
                children=[
                    html.Code(
                        tid,
                        className="font-mono text-xs text-slate-200 truncate",
                    ),
                    html.Span(
                        f"ρ {weight:.2f}",
                        className="text-[11px] font-mono text-violet-300",
                    ),
                ],
            )
            for tid, weight in neighbours
        ]

        return rows, button_label


def _top_k_neighbours(
    edges_payload: Sequence[Dict[str, Any]],
    trade_id: str,
    *,
    k: int,
) -> List[Tuple[str, float]]:
    """Walk the edge list, return the k strongest connections to ``trade_id``.

    Edges are undirected — we look at both ``source`` and ``target``
    sides.  Self-loops are dropped server-side already, but we
    defensively skip them here too.
    """
    pairs: List[Tuple[str, float]] = []
    for e in edges_payload:
        data = e.get("data", {})
        src, tgt = data.get("source"), data.get("target")
        if src == trade_id and tgt and tgt != trade_id:
            pairs.append((str(tgt), float(data.get("weight", 0.0))))
        elif tgt == trade_id and src and src != trade_id:
            pairs.append((str(src), float(data.get("weight", 0.0))))

    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[:k]


# ═════════════════════════════════════════════════════════════════════
# RENDER — Node Legend
# ═════════════════════════════════════════════════════════════════════


def _register_render_legend(app: "Dash") -> None:
    """Color-by + cached payload → legend body children."""

    @app.callback(
        Output(TRADE_GRAPH_IDS["legend_body"], "children"),
        Input(SHELL_IDS["url"],                "pathname"),
        Input(SHELL_IDS["session_store"],      "data"),
        Input(TRADE_GRAPH_IDS["store_graph"],  "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
        store_data:   Optional[Dict[str, Any]],
    ) -> Any:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        color_by = session.evaluation.trade_graph_color_by
        nodes_payload = (store_data or {}).get("nodes") or []

        # Run the stylesheet helper just for its (mode, pairs) tuple
        # — we discard the rules and use the legend pairs to render
        # the legend card body.  Both helpers stay in lockstep this
        # way (no chance of legend / stylesheet drift).
        _rules, pairs = build_stylesheet(color_by, nodes_payload=nodes_payload)
        return build_legend_body(color_by, pairs=pairs)


# ═════════════════════════════════════════════════════════════════════
# RENDER — Threshold label
# ═════════════════════════════════════════════════════════════════════


def _register_render_threshold_label(app: "Dash") -> None:
    """Slider value → "Min weight" pill label.

    Lives separately from the session-sync because the label is a
    pure local mirror of the slider — keeping it in its own callback
    means dragging the slider doesn't churn unrelated session
    consumers.
    """

    @app.callback(
        Output(TRADE_GRAPH_IDS["threshold_value_label"], "children"),
        Input(TRADE_GRAPH_IDS["threshold_slider"],       "value"),
    )
    def _render(threshold: Optional[float]) -> str:
        # Slider max is now per-cluster (driven by the render
        # callback) so we no longer hard-clamp at 1.0 here — just
        # format whatever value the slider emits.  ``max(0, …)``
        # still guards against the (rare) negative-value edge case.
        try:
            return f"{max(0.0, float(threshold)):.3g}"
        except (TypeError, ValueError):
            return "0.00"


# ═════════════════════════════════════════════════════════════════════
# RENDER — Density distribution + Edges-vs-nodes scatter
# ═════════════════════════════════════════════════════════════════════


def _register_render_density_chart(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(TRADE_GRAPH_IDS["density_chart"], "figure"),
        Input(SHELL_IDS["url"],                  "pathname"),
        Input(SHELL_IDS["session_store"],        "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Any:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        selected_cluster = (
            session.evaluation.trade_graph_cluster_id or session.cluster_id
        )

        res = backend.graph_stats_df()
        if not res.ok or res.data is None or res.data.empty:
            return empty_figure("No graph stats available.")

        return density_distribution(
            res.data, selected_cluster_id=selected_cluster,
        )


def _register_render_edges_vs_nodes_chart(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(TRADE_GRAPH_IDS["edges_vs_nodes_chart"], "figure"),
        Input(SHELL_IDS["url"],                         "pathname"),
        Input(SHELL_IDS["session_store"],               "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Any:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        selected_cluster = (
            session.evaluation.trade_graph_cluster_id or session.cluster_id
        )

        res = backend.graph_stats_df()
        if not res.ok or res.data is None or res.data.empty:
            return empty_figure("No graph stats available.")

        return edges_vs_nodes_scatter(
            res.data, selected_cluster_id=selected_cluster,
        )


# ═════════════════════════════════════════════════════════════════════
# CLIENTSIDE — Fit view + Export PNG
# ═════════════════════════════════════════════════════════════════════


def _register_clientside(app: "Dash") -> None:
    """Wire the two clientside actions to their JS implementations.

    Both use the existing ``window.dash_clientside.trade_graph``
    namespace from ``assets/js/trade_graph.js``.  Output goes to a
    placeholder Div property to satisfy Dash's "every callback must
    have an Output" rule — neither action needs to round-trip server
    state.
    """
    app.clientside_callback(
        ClientsideFunction(namespace="trade_graph", function_name="fit_view"),
        # The button itself is the receiver of the no-op write; we
        # never read this property so it's a safe sink.
        Output(TRADE_GRAPH_IDS["fit_btn"], "n_clicks"),
        Input(TRADE_GRAPH_IDS["fit_btn"],  "n_clicks"),
        State(TRADE_GRAPH_IDS["cytoscape"], "id"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction(namespace="trade_graph", function_name="export_png"),
        Output(TRADE_GRAPH_IDS["export_btn"], "n_clicks"),
        Input(TRADE_GRAPH_IDS["export_btn"],  "n_clicks"),
        State(TRADE_GRAPH_IDS["cytoscape"],   "id"),
        prevent_initial_call=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Layout-config helper
# ─────────────────────────────────────────────────────────────────────


def _build_layout_cfg(layout_name: str) -> Dict[str, Any]:
    """Cytoscape ``layout`` prop dict with per-algorithm tuning.

    The previous Dash UI passed cose-specific tuning options
    (``nodeRepulsion``, ``idealEdgeLength``, ``nodeOverlap``); without
    them the default cose run produces a tight cluster that visually
    reads like a grid for densely connected graphs.  We layer the
    same options here.

    ``randomize: True`` is set on every cose render so dash-cytoscape
    re-runs the layout when ``elements`` change but ``name`` stays the
    same.  Without this flag, switching cluster (same layout name,
    new elements) leaves nodes in their old positions overlaid onto
    the new edge set.
    """
    cfg: Dict[str, Any] = {
        "name":    layout_name,
        "fit":     True,
        "padding": 30,
        "animate": True,
    }

    # Layout-specific tuning.  Only **primitive** options are safe here
    # — dash-cytoscape JSON-serialises this dict over the wire, so any
    # JavaScript function string (e.g. ``"function(node) { return
    # node.degree(); }"``) arrives at cytoscape as a literal string and
    # blows up with ``options.<key> is not a function``.  If we ever
    # need a function-based option (concentric ring assignment, sort
    # comparator, …) it has to be wired through a clientside callback,
    # not embedded in the layout dict.
    if layout_name == "cose":
        cfg.update({
            # Higher repulsion = nodes spread further apart so the
            # network doesn't collapse into a tight ball.  Tuned on
            # 25 → 800 node clusters; same magnitudes the previous UI
            # used in `ensemble_analytics/callbacks/trade_graph_cb.py`.
            "nodeRepulsion":    8000,
            "idealEdgeLength":  80,
            "nodeOverlap":      20,
            "edgeElasticity":   100,
            # Ensure the layout actually re-runs whenever the render
            # callback ships a new layout dict — see docstring.
            "randomize":        True,
            # Cap iterations so very large clusters (~1k nodes) don't
            # block the browser while we wait for the layout to
            # converge.
            "numIter":          1000,
        })
    elif layout_name == "concentric":
        cfg.update({
            "minNodeSpacing": 30,
            # Cytoscape defaults to ranking by node degree (most
            # connected → centre) and uniform level widths, which is
            # the behaviour we want.  Explicit ``concentric`` /
            # ``levelWidth`` overrides require JS functions and so
            # cannot live in this dict.
        })
    elif layout_name == "breadthfirst":
        cfg.update({
            "directed":        False,
            "spacingFactor":   1.25,
        })

    return cfg


# ─────────────────────────────────────────────────────────────────────
# Formatters / helpers
# ─────────────────────────────────────────────────────────────────────


def _fmt_int(x: Optional[float]) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return _PLACEHOLDER


def _fmt_float(x: Optional[float], *, precision: int = 3) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        return f"{float(x):.{precision}g}"
    except (TypeError, ValueError):
        return _PLACEHOLDER


def _is_nan(value: Any) -> bool:
    """``True`` for ``NaN`` / ``None``; survives non-numeric inputs."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


__all__ = ["register"]
