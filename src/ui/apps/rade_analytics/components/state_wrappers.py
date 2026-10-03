"""State-wrapper primitives: ``Loading``, ``Empty``, ``Error``, ``AuthGate``.

Every page in the Rade UI routes through these four components to show
consistent feedback during the four states of a data fetch:

* **Loading**   — the request is in flight; render a skeleton matching
  the final layout so the page doesn't reflow on arrival.
* **Empty**     — the request succeeded but the dataset is empty; show
  a low-key placeholder with a hint at what would populate it.
* **Error**     — the request failed; show a boxed error with the
  server's message and an optional retry action.
* **AuthGate**  — wrap any sensitive content so the same gate logic
  (sign-in required / role missing) applies everywhere.

All four are pure presentational functions — zero callbacks, zero
state — so they drop inline into any layout:

.. code-block:: python

    result = backend.portfolio_timeseries(...)
    if result is None:                   # still fetching
        return Loading(variant="chart")
    if not result.ok:
        return Error(result=result, on_retry_id="retry-portfolio")
    if not len(result.data):
        return Empty(title="No portfolio data for this version")
    return chart_container("Portfolio", plot_portfolio(result.data))

Design-spec anchors
-------------------
* §6  — component palette.
* §9  — state handling contract (loading / empty / error).
* §10 — accessibility (every icon has a text label).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal, Optional

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify

if TYPE_CHECKING:
    # Typed in signatures only; keep the runtime dependency soft so
    # this module imports even if a downstream refactor temporarily
    # breaks ``..data.backend``.
    from ..data.backend import BackendResult


# ─────────────────────────────────────────────────────────────────────
# Internal skeleton helpers
# ─────────────────────────────────────────────────────────────────────


def _shimmer(size_classes: str) -> html.Div:
    """Base shimmer primitive; caller controls size with Tailwind classes."""
    return html.Div(className=f"rade-skeleton-shimmer {size_classes}")


def _kpi_card_skeleton() -> html.Div:
    return html.Div(
        className="rade-card-compact flex flex-col gap-2",
        children=[
            _shimmer("h-3 w-20"),   # label
            _shimmer("h-7 w-32"),   # value
            _shimmer("h-2 w-16"),   # delta / sparkline
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────

LoadingVariant = Literal["card", "kpi_strip", "chart", "table", "text"]


def Loading(
    variant: LoadingVariant = "card",
    *,
    count: int = 1,
    height: Optional[str] = None,
) -> html.Div:
    """Skeleton placeholder matching the expected final shape.

    Parameters
    ----------
    variant
        Which shape to render.  Match the real component that will
        replace this skeleton so the page doesn't reflow on arrival:

        * ``card``       a single card-sized rectangle (default).
        * ``kpi_strip``  a row of ``count`` KPI cards.
        * ``chart``      a tall card, for Plotly / network slots.
        * ``table``      a card with ``count`` row-shaped bars.
        * ``text``       ``count`` text-line shimmers, no card chrome.
    count
        Number of sub-skeletons for ``kpi_strip`` / ``table`` /
        ``text``.  Ignored for ``card`` / ``chart``.
    height
        Override Tailwind height class for ``chart`` / ``card`` (e.g.
        ``"h-96"``).  Defaults to ``"h-80"`` for ``chart`` and
        ``"h-32"`` for ``card``.
    """
    if variant == "card":
        return html.Div(
            className=f"rade-card rade-skeleton-shimmer {height or 'h-32'}",
        )
    if variant == "chart":
        return html.Div(
            className=f"rade-card rade-skeleton-shimmer {height or 'h-80'}",
        )
    if variant == "kpi_strip":
        # Map a handful of common layouts to Tailwind grid classes;
        # fall through to 4-column as a safe default.
        grid_class = {
            2: "grid-cols-2",
            3: "grid-cols-3",
            4: "grid-cols-4",
            5: "grid-cols-5",
            6: "grid-cols-6",
        }.get(count, "grid-cols-4")
        return html.Div(
            className=f"grid {grid_class} gap-4",
            children=[_kpi_card_skeleton() for _ in range(count)],
        )
    if variant == "table":
        return html.Div(
            className="rade-card flex flex-col gap-3",
            children=[_shimmer("h-4") for _ in range(max(count, 1))],
        )
    if variant == "text":
        return html.Div(
            className="flex flex-col gap-2",
            children=[_shimmer("h-3") for _ in range(max(count, 1))],
        )
    raise ValueError(f"Unknown Loading variant: {variant!r}")


# ─────────────────────────────────────────────────────────────────────
# Empty
# ─────────────────────────────────────────────────────────────────────


def Empty(
    title: str = "Nothing to show",
    *,
    message: Optional[str] = None,
    icon: str = "tabler:inbox",
    action: Optional[Any] = None,
) -> html.Div:
    """Placeholder for "fetch succeeded, dataset is empty".

    Parameters
    ----------
    title
        Short headline (e.g. ``"No trades in this cluster"``).
    message
        Optional secondary line nudging next steps (e.g. "Try clearing
        the desk filter").
    icon
        Iconify identifier.  Defaults to ``tabler:inbox``; use
        ``tabler:search-off`` for filter-driven empties.
    action
        Optional Dash component rendered underneath — typically a
        ``dmc.Button`` that clears a filter or opens the picker.
    """
    body: List[Any] = [
        DashIconify(icon=icon, width=32, className="text-slate-600"),
        html.Div(
            title,
            className="text-sm font-semibold text-slate-300",
        ),
    ]
    if message:
        body.append(
            html.Div(
                message,
                className="text-xs text-slate-500 max-w-md",
            )
        )
    if action is not None:
        body.append(html.Div(action, className="mt-2"))

    return html.Div(
        className=(
            "rade-card flex flex-col items-center justify-center "
            "gap-3 py-10 text-center"
        ),
        children=body,
    )


# ─────────────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────────────


def Error(
    *,
    result: "Optional[BackendResult]" = None,
    title: str = "Something went wrong",
    message: Optional[str] = None,
    icon: str = "tabler:alert-triangle",
    on_retry_id: Optional[str] = None,
) -> html.Div:
    """Placeholder for "fetch failed".

    Parameters
    ----------
    result
        A :class:`BackendResult` whose ``ok`` is ``False``.  When
        given, ``message`` is sourced from ``result.error`` and
        ``title`` is annotated with ``result.status_code`` if present.
    title, message
        Explicit alternatives if no BackendResult is handy.
    icon
        Iconify identifier (default ``tabler:alert-triangle``).
    on_retry_id
        If set, renders a Retry button with this DOM id so a callback
        can listen to its ``n_clicks`` and re-issue the query.
    """
    effective_title = title
    effective_message = message

    if result is not None and not result.ok:
        if result.status_code is not None:
            effective_title = f"{title} ({result.status_code})"
        effective_message = result.error or effective_message

    body: List[Any] = [
        DashIconify(icon=icon, width=32, className="text-rose-400"),
        html.Div(
            effective_title,
            className="text-sm font-semibold text-rose-300",
        ),
    ]
    if effective_message:
        body.append(
            html.Div(
                effective_message,
                className=(
                    "text-xs text-slate-400 max-w-md whitespace-pre-wrap"
                ),
            )
        )
    if on_retry_id:
        body.append(
            dmc.Button(
                id=on_retry_id,
                children="Retry",
                size="xs",
                variant="light",
                color="red",
                leftSection=DashIconify(icon="tabler:refresh", width=14),
                mt="sm",
            )
        )

    return html.Div(
        className=(
            "rade-card flex flex-col items-center justify-center "
            "gap-3 py-10 text-center border-rose-500/40"
        ),
        children=body,
    )


# ─────────────────────────────────────────────────────────────────────
# AuthGate
# ─────────────────────────────────────────────────────────────────────


def AuthGate(
    children: Any,
    *,
    authenticated: bool = True,
    required_role: Optional[str] = None,
    current_roles: Optional[List[str]] = None,
) -> Any:
    """Conditional wrapper that hides children until auth is satisfied.

    The Rade UI does not yet have an auth layer, so ``authenticated``
    defaults to ``True`` and every call is a pass-through.  When we add
    a session cookie / SSO integration, the app factory will inject
    ``authenticated`` / ``current_roles`` from the request context and
    existing call-sites will transparently start gating content — no
    page refactor required.

    Parameters
    ----------
    children
        The Dash tree to render once the gate opens.
    authenticated
        Overall signed-in flag.  ``False`` -> sign-in prompt.
    required_role
        Optional role the user must hold (e.g. ``"governance"``).
    current_roles
        Roles the current user holds.  Ignored unless
        ``required_role`` is set.
    """
    if not authenticated:
        return _sign_in_prompt()
    if required_role is not None:
        roles = current_roles or []
        if required_role not in roles:
            return _forbidden_prompt(required_role)
    return children


def _sign_in_prompt() -> html.Div:
    return html.Div(
        className=(
            "rade-card flex flex-col items-center justify-center "
            "gap-3 py-10 text-center"
        ),
        children=[
            DashIconify(icon="tabler:lock", width=32, className="text-slate-500"),
            html.Div(
                "Sign in required",
                className="text-sm font-semibold text-slate-300",
            ),
            html.Div(
                "Please sign in to view this content.",
                className="text-xs text-slate-500",
            ),
        ],
    )


def _forbidden_prompt(required_role: str) -> html.Div:
    return html.Div(
        className=(
            "rade-card flex flex-col items-center justify-center "
            "gap-3 py-10 text-center"
        ),
        children=[
            DashIconify(icon="tabler:shield-off", width=32, className="text-amber-400"),
            html.Div(
                "Not authorised",
                className="text-sm font-semibold text-amber-300",
            ),
            html.Div(
                f"This page requires the '{required_role}' role.",
                className="text-xs text-slate-500",
            ),
        ],
    )


__all__ = [
    "AuthGate",
    "Empty",
    "Error",
    "Loading",
    "LoadingVariant",
]
