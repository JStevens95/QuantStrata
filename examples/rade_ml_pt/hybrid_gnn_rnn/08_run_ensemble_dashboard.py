#!/usr/bin/env python3
"""
Launch the Ensemble Analytics Dashboard.

Requires ensemble training and evaluation to have been run first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ui.apps.ensemble_analytics import create_app

# -- Inputs (update these to match your environment) ----------------------
REGISTRY_DIR = "/path/to/your/registry"
ARTIFACTS_DIR = "/path/to/your/artifacts"
# -------------------------------------------------------------------------

app = create_app(
    registry_dir=REGISTRY_DIR,
    artifacts_dir=ARTIFACTS_DIR,
    version="latest",
    debug=True,
)

app.run(host="127.0.0.1", port=8051, debug=True)
