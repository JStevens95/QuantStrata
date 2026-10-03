"""Launch the PRISM API server programmatically.

Usage
-----
Edit the three paths below to point at your artifacts / registry tree
and the ensemble version you want to serve, then run::

    python examples/rade_ml_pt/hybrid_gnn_rnn/12_run_prism_api.py

OpenAPI docs: http://localhost:8000/docs
Health:       http://localhost:8000/health
Versions:     http://localhost:8000/versions
"""
from __future__ import annotations

import uvicorn

from src.rade_ml_pt.ensemble.api.app import create_app
from src.rade_ml_pt.ensemble.api.config import Settings, set_settings


# ── Configure ─────────────────────────────────────────────────────
ARTIFACTS_DIR = "/path/to/artifacts"
REGISTRY_DIR = "/path/to/registry"
ENSEMBLE_VERSION = "latest"

HOST = "0.0.0.0"
PORT = 8000


def main() -> None:
    set_settings(Settings(
        artifacts_dir=ARTIFACTS_DIR,
        registry_dir=REGISTRY_DIR,
        ensemble_version=ENSEMBLE_VERSION,
        host=HOST,
        port=PORT,
    ))

    app = create_app()
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
