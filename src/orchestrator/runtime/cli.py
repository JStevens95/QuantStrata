"""
Optional CLI stub.

We deliberately keep CLI minimal in V1 to avoid premature complexity.
Add argparse/typer later once pipeline set expands.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "CLI is not implemented yet.\n"
        "Use programmatic entrypoints instead:\n"
        "  from src.orchestrator import run_pipeline_by_name\n"
        "  run_pipeline_by_name('configs/run_marketdata.json')\n"
    )