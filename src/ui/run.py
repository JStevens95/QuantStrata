"""
Entry point to run a QuantStrata Dash app by name.

From the project root:

    python -m src.ui.run pricing_calculator

Runs the FX Vanilla Pricing Calculator on http://127.0.0.1:8050 by default.
Add new apps under src/ui/apps/<name>/ with create_app() and register in APP_REGISTRY below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when run as __main__
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

APP_REGISTRY = {
    "pricing_calculator": ("src.ui.apps.pricing_calculator.app", "create_app"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a QuantStrata Dash app by name.",
        epilog="Example: python -m src.ui.run pricing_calculator",
    )
    parser.add_argument(
        "app",
        choices=list(APP_REGISTRY),
        help="App name to run",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the Dash server (default: 8050)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    args = parser.parse_args()

    mod_path, attr = APP_REGISTRY[args.app]
    mod = __import__(mod_path, fromlist=[attr])
    create_app = getattr(mod, attr)
    app = create_app()
    app.run_server(debug=args.debug, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
