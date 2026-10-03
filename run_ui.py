"""Launch the Rade Analytics Dash UI against a running PRISM API.

This script assumes the PRISM FastAPI server is already running — use
``run_server.py`` (or a dedicated ``uvicorn`` invocation) to start it
first.  The typical workflow is:

1.  Start the API in the morning::

        python run_server.py \
            --artifacts-dir /data/rade/artifacts \
            --registry-dir  /data/rade/registry \
            --version       latest

2.  Launch the dashboard whenever it's needed::

        python run_ui.py --api-url http://localhost:8000

    (or, with ``RADE_UI_API_URL=http://localhost:8000`` exported, just
    ``python run_ui.py``.)

What this script does
---------------------
1.  Builds a :class:`RadeUiSettings` — explicit CLI args win over the
    ``RADE_UI_*`` environment variables, which in turn override the
    built-in defaults.
2.  Probes ``{api_url}/health`` before boot so configuration mistakes
    surface as a one-line error instead of as error banners on every
    UI tile.
3.  Calls :func:`src.ui.apps.rade_analytics.app.create_app` — this
    instantiates the typed ``RadeApiClient``, the cache, the layout
    shell and every callback module.
4.  Hands off to the Dash dev server on ``{ui_host}:{ui_port}``
    (default ``0.0.0.0:8050``).

Programmatic use
----------------
Every CLI flag has a keyword-argument equivalent on :func:`run`::

    from run_ui import run

    run(api_url="http://api.internal:8000", ui_port=9000)

Returns only when the Dash dev server exits.  For production, point a
WSGI container at ``src.ui.apps.rade_analytics.app:create_app`` rather
than using this dev-server launcher.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import httpx

from src.ui.apps.rade_analytics.app import create_app
from src.ui.apps.rade_analytics.config import RadeUiSettings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rade.run_ui")


# ─────────────────────────────────────────────────────────────────────
# API health probe
# ─────────────────────────────────────────────────────────────────────


def _probe_api(api_url: str, *, timeout_s: float) -> Optional[dict]:
    """Ping ``{api_url}/health`` and return the parsed JSON body.

    Returns ``None`` on any connection / HTTP error so the caller can
    decide whether to abort or warn-and-continue.  We never raise from
    here — a misconfigured URL should produce a clear log line, not a
    stack trace.
    """
    url = f"{api_url.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        log.error("API health check failed: %s (url=%s)", exc, url)
        return None


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def run(
    *,
    api_url: Optional[str] = None,
    ui_host: Optional[str] = None,
    ui_port: Optional[int] = None,
    debug:   Optional[bool] = None,
) -> int:
    """Boot the Rade UI.

    Explicit arguments override ``RADE_UI_*`` env vars, which in turn
    override the :class:`RadeUiSettings` defaults.  Any argument left
    as ``None`` falls through to the env var / default chain.

    Returns a Unix-style exit code (0 on clean shutdown, 1 when the
    API health probe fails).
    """
    overrides: dict = {}
    if api_url is not None:
        overrides["api_url"] = api_url
    if ui_host is not None:
        overrides["host"] = ui_host
    if ui_port is not None:
        overrides["port"] = ui_port
    if debug is not None:
        overrides["debug"] = debug
    settings = RadeUiSettings(**overrides)

    log.info(
        "Rade UI starting — api=%s cache=%s host=%s:%s debug=%s",
        settings.api_url,
        settings.cache_type,
        settings.host,
        settings.port,
        settings.debug,
    )

    # Pre-flight probe — catches the classic "API isn't up yet" failure
    # before every callback on the dashboard error-banners at once.
    health = _probe_api(settings.api_url, timeout_s=settings.api_timeout_s)
    if health is None:
        log.error(
            "Cannot reach the PRISM API at %s — start it via "
            "`python run_server.py` first, or point --api-url at the "
            "running instance.",
            settings.api_url,
        )
        return 1
    log.info(
        "API reachable — status=%s version=%s artifacts_dir=%s",
        health.get("status"),
        health.get("version"),
        health.get("artifacts_dir"),
    )

    app = create_app(settings)

    display_host = (
        "localhost" if settings.host in ("0.0.0.0", "::") else settings.host
    )
    base_url = f"http://{display_host}:{settings.port}"
    log.info("Rade UI ready — open %s", base_url)
    log.info("  · Overview    %s/overview", base_url)
    log.info("  · Evaluation  %s/evaluation/portfolio", base_url)
    log.info("  · Trade-Graph %s/evaluation/trade-graph", base_url)
    log.info("  · Deep-Dive   %s/evaluation/cluster", base_url)

    try:
        app.run(
            host=settings.host,
            port=settings.port,
            debug=settings.debug,
        )
    except KeyboardInterrupt:
        log.info("Received Ctrl-C — shutting down.")
    return 0


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ui.py",
        description=(
            "Launch the Rade Analytics Dash UI.  Requires the PRISM "
            "API to already be running (see run_server.py)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="URL of the running PRISM API "
             "(defaults to $RADE_UI_API_URL or http://localhost:8000).",
    )
    parser.add_argument(
        "--ui-host",
        type=str,
        default=None,
        help="Dash bind host (defaults to $RADE_UI_HOST or 0.0.0.0).",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=None,
        help="Dash bind port (defaults to $RADE_UI_PORT or 8050).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Enable Dash debug mode (overrides $RADE_UI_DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint — kept available for future ``python run_ui.py --flag`` use."""
    args = _parse_args(argv)
    return run(
        api_url=args.api_url,
        ui_host=args.ui_host,
        ui_port=args.ui_port,
        debug=args.debug,
    )


# ─────────────────────────────────────────────────────────────────────
# Programmatic launch (current default)
# ─────────────────────────────────────────────────────────────────────
# Edit the values below if your local setup deviates from the defaults,
# then run::
#
#     python run_ui.py
#
# To switch to CLI mode later, replace the body of the ``__main__``
# block with ``sys.exit(main())`` — :func:`main` + :func:`_parse_args`
# are already wired.

API_URL = "http://localhost:8000"
UI_HOST = "0.0.0.0"
UI_PORT = 8050
DEBUG   = False


if __name__ == "__main__":
    sys.exit(
        run(
            api_url=API_URL,
            ui_host=UI_HOST,
            ui_port=UI_PORT,
            debug=DEBUG,
        )
    )
