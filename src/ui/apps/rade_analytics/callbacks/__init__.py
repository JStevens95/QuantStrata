"""Callback orchestration — one public entry point: :func:`register_all`.

The app factory calls :func:`register_all` exactly once, which in turn
registers every callback the app needs.  Subsequent phases add
page-local callback modules here (``splash_cb`` in Phase C,
``landing_cb`` in Phase D, etc.).

Keeping registration centralised means the factory stays small and
dependency wiring (``app`` + ``backend``) happens in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..router import register_router
from . import (
    cluster_deep_dive_cb,
    evaluation_cb,
    inference_cb,
    overview_cb,
    portfolio_cb,
    risk_management_cb,
    splash_cb,
    trade_graph_cb,
)

# governance_cb intentionally NOT imported / registered while the
# /governance route is parked on the placeholder layout.  The module
# stays in the tree so re-enabling is a one-line change once we come
# back to wire it up against fresh evaluation artefacts:
#
#     from . import governance_cb
#     ...
#     governance_cb.register(app, backend)
#
# Leaving the registration off keeps the callback graph from emitting
# `/prism/v1/governance/registry` requests while the page is parked.

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


def register_all(app: "Dash", backend: "RadeBackend") -> None:
    """Register every callback on ``app``.

    Parameters
    ----------
    app
        The Dash app returned by :func:`rade_analytics.app.create_app`.
    backend
        Shared :class:`RadeBackend` instance so callbacks can fetch
        data without rebuilding the HTTP client each tick.
    """
    register_router(app, backend)
    splash_cb.register(app, backend)
    overview_cb.register(app, backend)
    evaluation_cb.register(app, backend)
    portfolio_cb.register(app, backend)
    trade_graph_cb.register(app, backend)
    cluster_deep_dive_cb.register(app, backend)
    inference_cb.register(app, backend)
    risk_management_cb.register(app, backend)
    # governance_cb intentionally not registered — see import block.


__all__ = ["register_all"]
