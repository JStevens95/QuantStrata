"""
Launch the Dash UI in API mode (connects to a running FastAPI server).

This starts INSTANTLY — no data loading.  All data is fetched over HTTP
from the FastAPI server started by script 10.

Prerequisites
-------------
1. Start the FastAPI server first::

       python 10_run_fastapi_server.py

2. Wait until you see "PRISM API ready" in that terminal.
3. Then run this script.

The FastAPI server can be on the same machine (localhost) or a remote
host.  Just change API_URL below.
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Configuration ─────────────────────────────────────────────────
# Point this to wherever the FastAPI server is running.
# Same machine: "http://localhost:8000"
# Remote server: "http://192.168.1.50:8000"
API_URL = "http://localhost:8000"

DASH_HOST = "127.0.0.1"
DASH_PORT = 8052


def main():
    import requests

    print(f"\nPRISM Dash UI (API mode)")
    print(f"  Connecting to FastAPI at {API_URL} ...")

    try:
        resp = requests.get(f"{API_URL}/api/v1/admin/health", timeout=5)
        resp.raise_for_status()
        health = resp.json()
        print(f"  Server status: {health.get('status', '?')}")
        print(f"  Version:       {health.get('version', '?')}")
        print(f"  Splits:        {health.get('cached_splits', [])}")
    except Exception as exc:
        print(f"\n  ERROR: Cannot reach FastAPI server at {API_URL}")
        print(f"  Detail: {exc}")
        print(f"\n  Make sure script 10_run_fastapi_server.py is running first.\n")
        sys.exit(1)

    from src.ui.apps.ensemble_analytics_db import create_app

    app = create_app(
        registry_dir="",
        artifacts_dir="",
        version="latest",
        backend="api",
        api_url=API_URL,
        debug=False,
    )

    print(f"  Dashboard:     http://{DASH_HOST}:{DASH_PORT}\n")
    app.run(host=DASH_HOST, port=DASH_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
