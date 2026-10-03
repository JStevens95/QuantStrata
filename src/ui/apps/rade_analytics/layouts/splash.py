"""Splash page — the "front door" before users enter the main app.

Layout
------
Full-viewport flex column:

    top-right          : active-version pill
    centre (flex-grow) : hero (logo + RADE + tagline) stacked over card
    bottom-left        : small copyright / build line

Only interactive elements live in the card (status strip, Enter CTA,
error banner); the identity block (logo + wordmark + tagline)
deliberately sits above the card so it reads against the gradient
backdrop rather than being boxed in.

Purpose
-------
* Announce the product identity.
* Show API liveness at a glance (status dot + URL).
* Confirm which ensemble version the API is bound to (read-only pill,
  top-right).  There is no version *switcher* here on purpose: the
  FastAPI server is bound to a single ``PRISM_ARTIFACTS_DIR`` /
  ``PRISM_REGISTRY_DIR`` at startup, so any in-UI "switch" would be a
  lie.  Multi-version UX, if/when it lands, belongs in the topbar of
  the main app shell, not on a one-shot splash page.
* One CTA: "Enter Rade" -> stamps the API's active version into the
  session store and navigates to ``/overview``.

This module only owns layout.  Data-fetching and navigation are wired
by :mod:`..callbacks.splash_cb`, which targets the DOM ids exported in
:data:`SPLASH_IDS` so callbacks never hardcode strings.

Styling contract
----------------
All visual treatment comes from the ``rade-splash-*`` classes defined
in ``assets/rade.css``.  The layout deliberately avoids Tailwind
arbitrary values because the repository ships a pre-compiled
``rade.css`` — new arbitrary values would silently no-op without a
Tailwind rebuild.  The custom classes give us stable visuals with
zero build step.

Design spec anchors
-------------------
* §2 Palette — gradient glows mirror the logo gradient
  (violet-500 -> cyan-400).
* §3 Typography — Inter wordmark, JetBrains Mono for version ids.
* §6 Components — DMC Button.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify

if TYPE_CHECKING:
    from ..data.session import Session


# Every dynamic id on the splash lives here so callbacks in
# splash_cb.py never hardcode strings.  Also used by tests to target
# individual elements.
#
# ``active_version`` targets the top-right read-only pill.  The
# in-card version dropdown was removed: the API is bound to one
# version at server-startup time, so a switcher would have been
# decorative-only and misleading.
SPLASH_IDS = {
    "root":              "splash-root",
    "status_dot":        "splash-status-dot",
    "status_label":      "splash-status-label",
    "api_url":           "splash-api-url",
    "active_version":    "splash-active-version",
    "enter_btn":         "splash-enter-btn",
    "error_banner":      "splash-error-banner",
}


def build_splash(*, session: Optional["Session"] = None) -> html.Div:
    """Build the splash page layout.

    The ``session`` kwarg is accepted for uniformity with every other
    page builder (Page Contract §2.1) but is currently unused — the
    splash callback hydrates dynamic state itself on first pathname
    tick, and there's nothing on this page to seed from session.
    """
    del session  # unused today; reserved for forward-compat
    return html.Div(
        id=SPLASH_IDS["root"],
        className="rade-splash-root",
        children=[
            _backdrop(),
            html.Div(
                className="rade-splash-layout",
                children=[
                    _topbar(),
                    _center(),
                ],
            ),
            _copyright(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Backdrop — soft gradient glow behind everything
# ─────────────────────────────────────────────────────────────────────


def _backdrop() -> html.Div:
    """Decorative blurred gradient so the page reads against depth."""
    return html.Div(
        className="rade-splash-backdrop",
        children=[
            html.Div(className="rade-splash-glow-violet"),
            html.Div(className="rade-splash-glow-cyan"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Top bar — active version pill (flush right)
# ─────────────────────────────────────────────────────────────────────


def _topbar() -> html.Div:
    return html.Div(
        className="rade-splash-topbar",
        children=[
            html.Span(
                "-",
                id=SPLASH_IDS["active_version"],
                className="rade-splash-version-pill",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Centre column — hero stacked on top of interactive card
# ─────────────────────────────────────────────────────────────────────


def _center() -> html.Div:
    return html.Div(
        className="rade-splash-center",
        children=[
            _hero(),
            _card(),
        ],
    )


def _hero() -> html.Div:
    """Identity block: logo + RADE wordmark + small uppercase tagline."""
    return html.Div(
        className="rade-splash-hero",
        children=[
            html.Img(
                src="/assets/logo.svg",
                className="rade-splash-hero-logo",
                alt="Rade logo",
            ),
            html.H1("RADE", className="rade-splash-hero-title"),
            html.P(
                "Quantitative Model Intelligence",
                className="rade-splash-hero-subtitle",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Card — interactive surface (status, version switcher, CTA)
# ─────────────────────────────────────────────────────────────────────


def _card() -> html.Div:
    return html.Div(
        className="rade-splash-card",
        children=[
            _status_strip(),
            _cta_block(),
        ],
    )


def _status_strip() -> html.Div:
    """Live / offline indicator + the API URL the app is pointed at."""
    return html.Div(
        className="rade-splash-status-strip",
        children=[
            html.Div(
                className="rade-splash-status-inner",
                children=[
                    html.Div(
                        id=SPLASH_IDS["status_dot"],
                        className="rade-status-dot rade-status-dot--booting",
                    ),
                    html.Span(
                        "Connecting...",
                        id=SPLASH_IDS["status_label"],
                        className="rade-splash-status-label",
                    ),
                ],
            ),
            html.Span(
                "-",
                id=SPLASH_IDS["api_url"],
                className="rade-splash-api-url",
            ),
        ],
    )


def _cta_block() -> html.Div:
    """Primary Enter button + fallback error banner for bootstrap errors."""
    return html.Div(
        className="rade-splash-cta-block",
        children=[
            html.Div(
                id=SPLASH_IDS["error_banner"],
                className="rade-splash-error-banner",
                children="",
            ),
            dmc.Button(
                id=SPLASH_IDS["enter_btn"],
                children="Enter Rade",
                rightSection=DashIconify(icon="tabler:arrow-right", width=16),
                size="md",
                fullWidth=True,
                variant="gradient",
                gradient={"from": "violet", "to": "cyan", "deg": 135},
                disabled=True,
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Footer — bottom-left build / copyright line
# ─────────────────────────────────────────────────────────────────────


def _copyright() -> html.Div:
    return html.Div(
        "(c) Rade Platform - v0.1",
        className="rade-splash-copyright",
    )


__all__ = ["SPLASH_IDS", "build_splash"]
