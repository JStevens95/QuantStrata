"""Launch the PRISM FastAPI server against a trained ensemble bundle.

What this script does
---------------------
1.  Builds a :class:`src.rade_ml_pt.ensemble.api.config.Settings` from
    the supplied ``artifacts_dir`` / ``registry_dir`` / ``version``
    (falling back to the ``PRISM_*`` env vars for anything not passed
    explicitly on the CLI).
2.  Resolves ``version`` — passing ``latest`` lets the server pick up
    the newest run under the registry on every start.
3.  Boots uvicorn against :func:`create_app`, which registers every
    PRISM router (overview / portfolio / clusters / trade-graph /
    training-curves / quality / predictions / …) and wires the
    :class:`ArtifactReader` lifespan hook.

Intended usage
--------------
Run first thing in the morning, leave it up all day, and launch
``run_ui.py`` whenever a teammate wants the dashboard.  The UI is a
thin client — everything it renders is served by this process.

CLI
---
From the project root::

    python run_server.py \
        --artifacts-dir /data/rade/artifacts \
        --registry-dir  /data/rade/registry \
        --version       latest

Env-var equivalent (useful in systemd / Docker)::

    export PRISM_ARTIFACTS_DIR=/data/rade/artifacts
    export PRISM_REGISTRY_DIR=/data/rade/registry
    export PRISM_ENSEMBLE_VERSION=latest
    python run_server.py

Programmatic use
----------------
Every CLI flag has a keyword-argument equivalent on :func:`run`::

    from run_server import run

    run(
        artifacts_dir="/data/rade/artifacts",
        registry_dir="/data/rade/registry",
        version="v2026.04.17-a1b2c",
        port=8000,
    )

Returns only when uvicorn exits (Ctrl-C, SIGTERM, unhandled error).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from src.rade_ml_pt.ensemble.api.app import create_app
from src.rade_ml_pt.ensemble.api.config import Settings, set_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rade.run_server")


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def run(
    *,
    artifacts_dir: Optional[str] = None,
    registry_dir:  Optional[str] = None,
    version:       str = "latest",
    host:          str = "0.0.0.0",
    port:          int = 8000,
    log_level:     str = "info",
    reload:        bool = False,
) -> int:
    """Boot the PRISM API server.

    *artifacts_dir* / *registry_dir* are required — they can be passed
    explicitly, or omitted in favour of the ``PRISM_ARTIFACTS_DIR`` /
    ``PRISM_REGISTRY_DIR`` environment variables.  If neither source
    provides them the call aborts with a clear error.

    Returns a Unix-style exit code (0 on clean shutdown, 1 on
    misconfiguration).
    """
    settings = _build_settings(
        artifacts_dir=artifacts_dir,
        registry_dir=registry_dir,
        version=version,
        host=host,
        port=port,
        log_level=log_level,
    )
    if settings is None:
        return 1

    # Publish settings before create_app() reads them.  The API's
    # lifespan hook uses the same singleton so every router sees a
    # consistent view.
    set_settings(settings)

    log.info(
        "PRISM API starting — artifacts=%s registry=%s version=%s host=%s:%s",
        settings.artifacts_dir,
        settings.registry_dir,
        settings.ensemble_version,
        settings.host,
        settings.port,
    )

    # ``reload=True`` needs an import-string app (uvicorn forks), so we
    # branch: dev-reload mode relies on env vars propagating to the
    # child; one-shot mode passes the already-built app object.
    if reload:
        uvicorn.run(
            "src.rade_ml_pt.ensemble.api.app:get_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
            reload=True,
        )
    else:
        app = create_app()
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
        )
    return 0


# ─────────────────────────────────────────────────────────────────────
# Settings resolution
# ─────────────────────────────────────────────────────────────────────


def _build_settings(
    *,
    artifacts_dir: Optional[str],
    registry_dir:  Optional[str],
    version:       str,
    host:          str,
    port:          int,
    log_level:     str,
) -> Optional[Settings]:
    """Construct a validated :class:`Settings` or log a clear error.

    Explicit CLI args win over env vars.  Anything not passed falls
    through to pydantic-settings' env loader (``PRISM_*``).  Missing
    mandatory paths surface as a one-line error rather than a
    ValidationError traceback — easier to read in a terminal.
    """
    try:
        if artifacts_dir is not None and registry_dir is not None:
            settings = Settings(
                artifacts_dir=str(Path(artifacts_dir).resolve()),
                registry_dir=str(Path(registry_dir).resolve()),
                ensemble_version=version,
                host=host,
                port=port,
                log_level=log_level,
            )
        else:
            # pydantic-settings reads PRISM_* env vars.  type: ignore
            # because the stub expects both required fields as kwargs.
            settings = Settings(  # type: ignore[call-arg]
                ensemble_version=version,
                host=host,
                port=port,
                log_level=log_level,
            )
    except Exception as exc:         # noqa: BLE001 — single-line error banner
        log.error(
            "Cannot build PRISM settings: %s.  Supply both --artifacts-dir "
            "and --registry-dir, or set PRISM_ARTIFACTS_DIR / "
            "PRISM_REGISTRY_DIR in the environment.",
            exc,
        )
        return None

    # Warn loudly before startup if the paths don't exist — otherwise
    # the eval pipeline error surfaces only at first request.
    for label, path in (
        ("artifacts_dir", settings.artifacts_path),
        ("registry_dir",  settings.registry_path),
    ):
        if not path.exists():
            log.warning("%s does not exist on disk: %s", label, path)

    return settings


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_server.py",
        description=(
            "Launch the PRISM FastAPI server.  Serves ensemble "
            "evaluation artefacts that the Rade Analytics Dash UI "
            "(run_ui.py) reads.  Intended to run persistently for a "
            "day's work."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group(
        "Artefact locations",
        "Required unless PRISM_ARTIFACTS_DIR / PRISM_REGISTRY_DIR are "
        "set in the environment.",
    )
    paths.add_argument("--artifacts-dir", type=str, default=None,
                       help="Root of the ensemble artefacts tree.")
    paths.add_argument("--registry-dir",  type=str, default=None,
                       help="Root of the training registry (per-member bundles).")
    paths.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Ensemble version to serve — 'latest' or a concrete version tag.",
    )

    server = parser.add_argument_group("Server")
    server.add_argument("--host", type=str, default="0.0.0.0",
                        help="Uvicorn bind host.")
    server.add_argument("--port", type=int, default=8000,
                        help="Uvicorn bind port.")
    server.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="Uvicorn log level.",
    )
    server.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn's auto-reloader (dev only — watches src/).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint — kept available for future ``python run_server.py --flag`` use."""
    args = _parse_args(argv)
    return run(
        artifacts_dir=args.artifacts_dir,
        registry_dir=args.registry_dir,
        version=args.version,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


# ─────────────────────────────────────────────────────────────────────
# Programmatic launch (current default)
# ─────────────────────────────────────────────────────────────────────
# Edit the values below to match your local paths, then run::
#
#     python run_server.py
#
# To switch to CLI mode later, replace the body of the ``__main__``
# block with ``sys.exit(main())`` — :func:`main` + :func:`_parse_args`
# are already wired.

# TODO: point these at your local artefact tree before running.
ARTIFACTS_DIR = "artifacts/ensemble"
REGISTRY_DIR  = "artifacts/registry"
VERSION       = "latest"

HOST          = "0.0.0.0"
PORT          = 8000
LOG_LEVEL     = "info"
RELOAD        = False


if __name__ == "__main__":
    sys.exit(
        run(
            artifacts_dir=ARTIFACTS_DIR,
            registry_dir=REGISTRY_DIR,
            version=VERSION,
            host=HOST,
            port=PORT,
            log_level=LOG_LEVEL,
            reload=RELOAD,
        )
    )
