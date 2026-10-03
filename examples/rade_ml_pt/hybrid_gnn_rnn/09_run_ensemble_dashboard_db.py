"""
Launch the DB-ready Ensemble Analytics dashboard.

This is the fast-loading version of the ensemble dashboard that reads
from pre-computed ``db_ready/`` files instead of building the full
``GlobalPredictionStore``.  Supports both ``file`` and ``sqlite``
backends.

Prerequisites
-------------
Run the ensemble evaluation pipeline with ``save_db_artifacts=True``
(the default) to generate the ``db_ready/`` directory.

Optionally, publish to SQLite for query-based access::

    from src.rade_ml_pt.ensemble.publish_to_db import publish_to_sqlite
    publish_to_sqlite("/path/to/evaluation/db_ready")
"""
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Configuration ─────────────────────────────────────────────────
REGISTRY_DIR = "/path/to/your/registry"
ARTIFACTS_DIR = "/path/to/your/artifacts"
VERSION = "latest"

# Backend: "cache"  loads everything into RAM at startup (fastest, recommended)
#          "file"   reads from evaluation/ JSON+NPZ with per-request caching
#          "sqlite" reads from a local .db file (requires publish_to_sqlite first)
BACKEND = "cache"

# Only needed for sqlite backend — path to the .db file.
# If None, defaults to db_ready/ensemble.db
DB_PATH = None


def main():
    t_start = time.perf_counter()
    from src.ui.apps.ensemble_analytics_db import create_app

    app = create_app(
        registry_dir=REGISTRY_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        version=VERSION,
        backend=BACKEND,
        db_path=DB_PATH,
        debug=False,
    )

    elapsed = time.perf_counter() - t_start
    print(f"\nStarting DB-ready Ensemble Analytics Dashboard")
    print(f"  Backend:      {BACKEND}")
    print(f"  Registry:     {REGISTRY_DIR}")
    print(f"  Artifacts:    {ARTIFACTS_DIR}")
    print(f"  Version:      {VERSION}")
    print(f"  Startup time: {elapsed:.1f}s")
    print(f"  URL:          http://127.0.0.1:8052\n")

    app.run(host="127.0.0.1", port=8052, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
