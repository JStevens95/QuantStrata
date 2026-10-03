"""Evaluation page shell — filter bar + sub-tab nav + content slot.

The shell is the single ``build()`` target registered in the router
for every ``/evaluation/*`` path.  It composes three regions top-to-
bottom:

1. Global filter bar (:mod:`..components.evaluation_filter_bar`).
   One canonical WHERE clause applied to every sub-tab.
2. ``dmc.Tabs`` row — 4 primary perspectives: Portfolio, Cross-Cluster,
   Trade-Graph, Cluster Deep-Dive.  Which one is active is derived
   from ``dcc.Location.pathname`` (not session state) so the URL is
   always the source of truth, and bookmarks / deep links work.
3. Content slot — a div the router callback fills with the active
   sub-tab's ``build_*()`` output.

URL contract
------------
* ``/evaluation``               → redirects to ``/evaluation/portfolio``.
* ``/evaluation/portfolio``     → Portfolio sub-tab (default).
* ``/evaluation/cross-cluster`` → Cross-Cluster sub-tab.
* ``/evaluation/trade-graph``   → Trade-Graph sub-tab.
* ``/evaluation/cluster``       → Cluster Deep-Dive sub-tab.

Anything else under ``/evaluation/`` falls through to the 404 page at
the top-level router; we deliberately don't swallow unknown sub-paths
here so typos surface immediately instead of silently redirecting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify

from ...components.evaluation_filter_bar import build_evaluation_filter_bar
from ...data.session import (
    DEFAULT_EVALUATION_SUBTAB,
    EVALUATION_SUBTABS,
    Session,
)
from .cluster_deep_dive import build_cluster_deep_dive
from .cross_cluster import build_cross_cluster
from .portfolio import build_portfolio
from .trade_graph import build_trade_graph


# Every dynamic id in the shell lives here so the Phase E.0d callbacks
# never hardcode strings.
EVALUATION_IDS = {
    "root":    "evaluation-root",
    "tabs":    "evaluation-tabs",
    "content": "evaluation-content",
}


# ─────────────────────────────────────────────────────────────────────
# Sub-tab registry
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _EvaluationSubtab:
    """Declarative description of one sub-tab.

    ``slug`` is both the URL token (``/evaluation/{slug}``) and the
    ``value`` prop on the matching ``dmc.TabsTab`` — keeping them
    identical means the tab → URL sync is an O(1) lookup both ways.

    ``build`` is a kwargs-only callable accepting ``session`` so every
    sub-tab can seed initial UI state from session at build time
    (Page Contract §3 Rule L1).  Each sub-tab builder has the
    matching signature ``build_<name>(*, session: Optional[Session] = None)``;
    the dispatch goes through :func:`build_subtab_content` which
    forwards ``session`` uniformly.
    """

    slug: str
    label: str
    icon: str
    build: Callable[..., Any]


# Ordered — this is the order the tabs render left-to-right in the
# ``dmc.Tabs`` row.  The slugs must be a subset of ``EVALUATION_SUBTABS``
# so the router + session agree on what's valid.
EVALUATION_SUBTAB_SPECS: Tuple[_EvaluationSubtab, ...] = (
    _EvaluationSubtab(
        slug="portfolio",
        label="Portfolio",
        icon="tabler:chart-line",
        build=build_portfolio,
    ),
    _EvaluationSubtab(
        slug="cross-cluster",
        label="Cross-Cluster",
        icon="tabler:arrows-shuffle",
        build=build_cross_cluster,
    ),
    _EvaluationSubtab(
        slug="trade-graph",
        label="Trade-Graph",
        icon="tabler:affiliate",
        build=build_trade_graph,
    ),
    _EvaluationSubtab(
        slug="cluster",
        label="Cluster",
        icon="tabler:microscope",
        build=build_cluster_deep_dive,
    ),
)

_SPEC_BY_SLUG: Dict[str, _EvaluationSubtab] = {
    spec.slug: spec for spec in EVALUATION_SUBTAB_SPECS
}

# Defensive: the tabs registry must match the session's canonical list.
# Surfacing this as an import-time assert means an accidental divergence
# fails CI instead of corrupting session state at runtime.
assert set(_SPEC_BY_SLUG.keys()) == set(EVALUATION_SUBTABS), (
    "EVALUATION_SUBTAB_SPECS out of sync with session.EVALUATION_SUBTABS"
)


# ─────────────────────────────────────────────────────────────────────
# URL ↔ slug helpers
# ─────────────────────────────────────────────────────────────────────


def active_subtab_from_path(pathname: Optional[str]) -> str:
    """Return the canonical sub-tab slug for a given URL pathname.

    Unknown / missing paths fall back to :data:`DEFAULT_EVALUATION_SUBTAB`
    (Portfolio), which is what the bare ``/evaluation`` URL resolves to.
    """
    if not pathname:
        return DEFAULT_EVALUATION_SUBTAB
    prefix = "/evaluation/"
    if pathname == "/evaluation":
        return DEFAULT_EVALUATION_SUBTAB
    if not pathname.startswith(prefix):
        return DEFAULT_EVALUATION_SUBTAB
    slug = pathname[len(prefix):].strip("/").split("/", 1)[0]
    return slug if slug in _SPEC_BY_SLUG else DEFAULT_EVALUATION_SUBTAB


def path_for_subtab(slug: str) -> str:
    """Inverse of :func:`active_subtab_from_path`."""
    if slug not in _SPEC_BY_SLUG:
        slug = DEFAULT_EVALUATION_SUBTAB
    return f"/evaluation/{slug}"


def build_subtab_content(
    slug: str,
    *,
    session: Optional[Session] = None,
) -> Any:
    """Build the body of the given sub-tab.

    Falls back to the default sub-tab's builder on an unknown slug so
    the UI never hits an ``AttributeError`` / ``KeyError`` mid-render.

    Parameters
    ----------
    slug
        Sub-tab token (the value of the ``dmc.TabsTab`` and the URL
        suffix under ``/evaluation/``).
    session
        Live :class:`Session` forwarded to the sub-tab's
        ``build_<name>(*, session=...)`` builder so initial UI state
        (group-by select, cluster-pin, etc.) is seeded from session
        at build time.  Page Contract §3 Rule L1 — eliminates the
        need for hydration callbacks (and the URL-trigger races they
        create) by sourcing initial values from session, not from a
        post-mount callback round-trip.
    """
    spec = _SPEC_BY_SLUG.get(slug) or _SPEC_BY_SLUG[DEFAULT_EVALUATION_SUBTAB]
    return spec.build(session=session)


# ─────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────


def build_evaluation(
    pathname: Optional[str] = None,
    *,
    session: Optional[Session] = None,
) -> html.Div:
    """Build the Evaluation page tree for a given ``pathname``.

    Parameters
    ----------
    pathname
        Current ``dcc.Location.pathname``.  When ``None`` (e.g. the
        preview script that doesn't run the router) the default
        Portfolio sub-tab is rendered.
    session
        Live :class:`Session` whose :attr:`Session.evaluation` slice
        seeds the filter bar's initial values (Page Contract §3
        Rule L1 — initial state from session at build time, no
        hydration callback).  ``None`` falls back to a fresh
        :class:`Session` so unit tests / preview scripts that don't
        thread the session through still render a sensible default.
    """
    active = active_subtab_from_path(pathname)
    sess = session or Session()
    eval_state = sess.evaluation

    return html.Div(
        id=EVALUATION_IDS["root"],
        className="rade-page rade-evaluation",
        children=[
            build_evaluation_filter_bar(
                initial_filters=eval_state.filters,
                initial_open=eval_state.filter_bar_open,
            ),
            _tabs_row(active_slug=active),
            html.Div(
                id=EVALUATION_IDS["content"],
                className="rade-evaluation-content",
                children=build_subtab_content(active, session=sess),
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Internal builders
# ─────────────────────────────────────────────────────────────────────


def _tabs_row(*, active_slug: str) -> dmc.Tabs:
    """Sub-tab navigation.

    Using ``dmc.Tabs`` with ``value``/``onTabChange`` gives us the DMC
    pill styling for free; the E.0d callback listens on this
    component's ``value`` and pushes matching URLs via
    ``dcc.Location.pathname``.
    """
    tabs_list = dmc.TabsList(
        children=[
            dmc.TabsTab(
                value=spec.slug,
                leftSection=DashIconify(icon=spec.icon, width=14),
                children=spec.label,
            )
            for spec in EVALUATION_SUBTAB_SPECS
        ],
        grow=False,
    )

    return dmc.Tabs(
        id=EVALUATION_IDS["tabs"],
        value=active_slug,
        variant="pills",
        color="violet",
        radius="sm",
        className="rade-evaluation-tabs",
        children=[tabs_list],
    )


__all__ = [
    "EVALUATION_IDS",
    "EVALUATION_SUBTAB_SPECS",
    "active_subtab_from_path",
    "build_evaluation",
    "build_subtab_content",
    "path_for_subtab",
]
