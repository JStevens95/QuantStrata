"""URL router — maps ``dcc.Location.pathname`` to a page layout.

Owns the single authoritative callback that swaps the content slot,
updates the topbar breadcrumb page segment and flips the active class
on the matching sidebar nav item whenever the URL changes.

Pages are registered declaratively via the :data:`ROUTES` table so
adding a new page in Phases C–F is a three-line change:

.. code-block:: python

    from .pages import monitoring
    ROUTES["/monitoring"] = PageSpec(
        path="/monitoring",
        title="Monitoring",
        build=monitoring.build,
    )

Sub-routes
----------
Pages with internal sub-tabs register *each* sub-path as its own
:class:`PageSpec` and set ``sidebar_path`` to the parent route so the
sidebar still highlights the correct top-level entry and, crucially,
so the router can detect a within-page navigation and **skip the
``page_content`` rebuild** — preserving any ephemeral component state
(open drawers, scroll position, half-filled form fields) that would
otherwise be clobbered by a remount.

Design-spec anchors
-------------------
* §9  — state handling (placeholders use :class:`Empty`).
* §11 — nav taxonomy (must mirror ``NAV_ITEMS``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from dash import Input, Output, State, no_update
from dash import html

from .components.state_wrappers import Empty
from .data.session import Session
from .layouts.data_quality import build_data_quality
from .layouts.evaluation import build_evaluation
from .layouts.governance import build_governance
from .layouts.inference import build_inference
from .layouts.monitoring import build_monitoring
from .layouts.overview import build_overview
from .layouts.risk_management import build_risk_management
from .layouts.shell import SHELL_IDS, build_chrome
from .layouts.splash import build_splash

if TYPE_CHECKING:
    from dash import Dash

    from .data.backend import RadeBackend


# ─────────────────────────────────────────────────────────────────────
# Page spec + registry
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PageSpec:
    """Declarative description of a top-level page.

    Attributes
    ----------
    path
        URL path that triggers this page.  Must match exactly
        (``dcc.Location.pathname``).
    title
        Breadcrumb-second-segment display string when this page is
        active.
    build
        Factory returning the page layout.  Called on every *top-level*
        route change so pages don't have to be careful about singleton
        vs per-request state.

        Signature: ``build(*, session: Optional[Session] = None) -> Any``.
        Every builder accepts ``session`` for uniformity (Page Contract
        §3 Rule L1 — initial UI state from session at build time);
        builders that don't yet need it ignore the kwarg.
    sidebar_path
        Which sidebar nav item to highlight for this route.  Defaults
        to :attr:`path` — sub-routes like ``/evaluation/portfolio``
        set this to ``/evaluation`` so the parent nav item stays
        active and the router treats the sub-paths as within-page
        navigation (skipping page-content rebuilds).
    """

    path: str
    title: str
    build: Callable[..., Any]
    sidebar_path: Optional[str] = None

    @property
    def top_level(self) -> str:
        """Effective sidebar path — falls back to :attr:`path`."""
        return self.sidebar_path if self.sidebar_path is not None else self.path


def _placeholder(page_title: str, phase: str) -> Callable[..., Any]:
    """Return a ``build()`` function that renders an :class:`Empty` card.

    Used for every page whose real implementation lives in a later
    phase.  The message tells the reviewer exactly which phase ships
    the real content so demo sessions aren't confusing.
    """
    message = f"Real implementation ships in {phase}."

    def _build(*, session: Optional[Session] = None) -> Any:
        del session  # unused on placeholder pages
        return html.Div(
            className="p-8",
            children=Empty(
                title=page_title,
                message=message,
                icon="tabler:layout-board",
            ),
        )

    return _build


def _build_evaluation_for(pathname: str) -> Callable[..., Any]:
    """Thin factory so each Evaluation sub-path captures its own pathname.

    ``build_evaluation`` needs the pathname to pick the initial active
    sub-tab and the live session to seed the filter bar's initial
    values.  Wrapping it here lets :class:`PageSpec` keep a single
    uniform ``build(session=...)`` signature across every page.
    """

    def _build(*, session: Optional[Session] = None) -> Any:
        return build_evaluation(pathname, session=session)

    return _build


# Registry — keyed by URL pathname, mirrors ``NAV_ITEMS`` plus the
# evaluation sub-paths and the terminal 404 case.  Keep this in sync
# with the sidebar: every entry in ``NAV_ITEMS`` must have at least
# one matching ROUTES entry, and vice-versa.
ROUTES: Dict[str, PageSpec] = {
    # ``/`` is the splash / "front-door" page: it picks the artefact
    # version, populates ``session.active_version`` and then pushes the
    # browser to ``/overview``.  Keeping splash at ``/`` means a fresh
    # tab or a hand-typed ``http://host:port/`` always lands on the
    # version-picker first, which is the only place that hydrates the
    # session — without it every downstream page renders empty.
    "/": PageSpec(
        path="/",
        title="Welcome",
        build=build_splash,
    ),
    # Legacy alias — keeps any existing ``/splash`` bookmarks /
    # docs / deeplinks working after the route swap.
    "/splash": PageSpec(
        path="/splash",
        title="Welcome",
        build=build_splash,
    ),
    "/overview": PageSpec(
        path="/overview",
        title="Overview",
        build=build_overview,
    ),

    # ── Evaluation — parent + four sub-tabs ─────────────────────
    #
    # Bare ``/evaluation`` resolves to the default sub-tab (Portfolio)
    # so the sidebar link doesn't land on an "empty shell"; the actual
    # sidebar href points at ``/evaluation/portfolio`` to keep URLs
    # canonical.  We keep the bare entry for hand-typed URLs / old
    # bookmarks.
    "/evaluation": PageSpec(
        path="/evaluation",
        title="Evaluation",
        build=_build_evaluation_for("/evaluation/portfolio"),
        sidebar_path="/evaluation",
    ),
    "/evaluation/portfolio": PageSpec(
        path="/evaluation/portfolio",
        title="Evaluation · Portfolio",
        build=_build_evaluation_for("/evaluation/portfolio"),
        sidebar_path="/evaluation",
    ),
    "/evaluation/cross-cluster": PageSpec(
        path="/evaluation/cross-cluster",
        title="Evaluation · Cross-Cluster",
        build=_build_evaluation_for("/evaluation/cross-cluster"),
        sidebar_path="/evaluation",
    ),
    "/evaluation/trade-graph": PageSpec(
        path="/evaluation/trade-graph",
        title="Evaluation · Trade-Graph",
        build=_build_evaluation_for("/evaluation/trade-graph"),
        sidebar_path="/evaluation",
    ),
    "/evaluation/cluster": PageSpec(
        path="/evaluation/cluster",
        title="Evaluation · Cluster Deep-Dive",
        build=_build_evaluation_for("/evaluation/cluster"),
        sidebar_path="/evaluation",
    ),

    # ── Remaining top-level pages (stubs until Phase F) ─────────
    "/monitoring": PageSpec(
        path="/monitoring",
        title="Monitoring",
        build=build_monitoring,
    ),
    "/governance": PageSpec(
        path="/governance",
        title="Governance",
        build=build_governance,
    ),
    "/inference": PageSpec(
        path="/inference",
        title="Inference Console",
        build=build_inference,
    ),
    "/risk-management": PageSpec(
        path="/risk-management",
        title="Risk Management",
        build=build_risk_management,
    ),
    "/scenario-lab": PageSpec(
        path="/scenario-lab",
        title="Scenario Lab",
        build=_placeholder("Scenario Lab", "Phase F"),
    ),
    "/report-builder": PageSpec(
        path="/report-builder",
        title="Report Builder",
        build=_placeholder("Report Builder", "Phase F"),
    ),
    "/data-quality": PageSpec(
        path="/data-quality",
        title="Data Quality",
        build=build_data_quality,
    ),
    "/assistant": PageSpec(
        path="/assistant",
        title="AI Assistant",
        build=_placeholder("AI Assistant", "Phase F"),
    ),
}


def _build_not_found(*, session: Optional[Session] = None) -> Any:
    """Terminal 404 layout.  ``session`` is accepted for signature
    parity with every other ``PageSpec.build`` and unused."""
    del session
    return html.Div(
        className="p-8",
        children=Empty(
            title="Page not found",
            message="The URL you requested doesn't exist in Rade.",
            icon="tabler:map-pin-off",
        ),
    )


NOT_FOUND = PageSpec(
    path="__not_found__",
    title="Not Found",
    build=_build_not_found,
)


_SPLASH_TOP_LEVEL = "/"


def _is_splash_route(top_level: str) -> bool:
    """``/`` and ``/splash`` both map to splash; everything else is chrome."""
    return top_level in ("/", "/splash")


def resolve_page(pathname: str | None) -> PageSpec:
    """Return the :class:`PageSpec` for a given URL pathname.

    Falls back to :data:`NOT_FOUND` for unknown routes.  Empty / None
    pathnames (the initial tick before ``dcc.Location`` resolves) are
    treated as ``/``.
    """
    if not pathname:
        return ROUTES["/"]
    return ROUTES.get(pathname, NOT_FOUND)


# ─────────────────────────────────────────────────────────────────────
# Callback registration
# ─────────────────────────────────────────────────────────────────────


def register_router(app: "Dash", backend: "RadeBackend") -> None:
    """Register the URL → page-tree callback.

    Writes the entire visible page tree into ``app-host``.  For splash
    routes this is the splash layout itself (no chrome); for every
    other route it's :func:`..layouts.shell.build_chrome` wrapping the
    page builder's output.

    Within-page navigation (e.g. Evaluation sub-tab clicks) is detected
    via the ``top-level-store`` and skipped — the chrome stays mounted
    and the page's own callbacks update the inner sub-tab content.
    """
    del backend  # unused today; reserved for per-page server render

    @app.callback(
        Output(SHELL_IDS["app_host"],        "children"),
        Output(SHELL_IDS["top_level_store"], "data"),
        Input(SHELL_IDS["url"],              "pathname"),
        State(SHELL_IDS["top_level_store"],  "data"),
        State(SHELL_IDS["session_store"],    "data"),
        # Page Contract §4 Rule C5 — explicit opt-in.  The router is
        # the single source of truth for page rendering; on app boot
        # we *need* the implicit initial pathname callback to fire so
        # that ``app-host`` is filled with the resolved page tree
        # (otherwise the user sees an empty viewport until they
        # interact with the URL).  Within-page nav is cheap because
        # ``_route`` returns ``no_update`` for ``app-host`` when the
        # top-level segment is unchanged.
        prevent_initial_call=False,
    )
    def _route(
        pathname: str | None,
        prev_top_level: str | None,
        session_data: Dict[str, Any] | None,
    ) -> List[Any]:
        spec = resolve_page(pathname)
        new_top_level = spec.top_level

        # Within-page navigation (e.g. Evaluation sub-tab clicks):
        # the chrome stays mounted, sidebar/topbar remain in place,
        # and the page's own callbacks (e.g. ``evaluation_cb._sync_from_url``)
        # update the inner sub-tab content slot.  Returning ``no_update``
        # for ``app-host`` is what keeps the chrome alive across these
        # within-page transitions.
        if prev_top_level == new_top_level and prev_top_level is not None:
            return [no_update, new_top_level]

        # Cross-top-level navigation: rebuild the visible tree from
        # scratch.  Splash gets the full-viewport treatment with no
        # chrome at all; every other page wraps its content in
        # ``build_chrome`` so the sidebar + topbar reflect the new
        # active page.
        # Build the live session once and thread it into every page
        # builder.  Page Contract §3 Rule L1: initial UI state is
        # seeded from session at build time, eliminating hydration
        # callbacks and the race conditions they create.
        session = Session.from_store(session_data)

        if _is_splash_route(new_top_level):
            return [build_splash(session=session), new_top_level]

        # Source the topbar split + PnL-space toggles from the session
        # so the user's current choices survive the chrome rebuild.
        # ``current_pnl_space`` was added in Phase 3.3 alongside the
        # API ``?space=`` query param introduced in Phase 3.2.
        page_tree = build_chrome(
            page_content=spec.build(session=session),
            breadcrumb_title=spec.title,
            active_path=new_top_level,
            current_split=session.split,
            current_pnl_space=session.pnl_space,
        )
        return [page_tree, new_top_level]


__all__ = [
    "NOT_FOUND",
    "PageSpec",
    "ROUTES",
    "register_router",
    "resolve_page",
]
