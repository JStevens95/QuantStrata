"""
Launch the PRISM FastAPI server.

This loads all ensemble artifacts into RAM (takes 5-10 minutes) and then
serves them over HTTP.  Once running, the Dash UI or Retool can connect
instantly via ``ApiBackend``.

Run this FIRST, then start the Dash UI with script 11.
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
REGISTRY_DIR = "/path/to/your/registry"
ARTIFACTS_DIR = "/path/to/your/artifacts"
VERSION = "latest"

HOST = "0.0.0.0"
PORT = 8000


def main():
    from src.ui.apps.ensemble_analytics_db.api.main import create_app
    from src.ui.apps.ensemble_analytics_db.api.config import Settings, set_settings

    set_settings(Settings(
        artifacts_dir=ARTIFACTS_DIR,
        registry_dir=REGISTRY_DIR,
        ensemble_version=VERSION,
        host=HOST,
        port=PORT,
    ))

    app = create_app()

    import uvicorn
    print(f"\nStarting PRISM FastAPI Server")
    print(f"  Registry:  {REGISTRY_DIR}")
    print(f"  Artifacts: {ARTIFACTS_DIR}")
    print(f"  Version:   {VERSION}")
    print(f"  URL:       http://{HOST}:{PORT}")
    print(f"  Docs:      http://{HOST}:{PORT}/docs\n")

    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
