"""
Pipeline discovery / registration.

Design goals
------------
- Keep this file *boringly simple* and easy to read.
- Avoid import side-effects elsewhere by importing pipeline modules here only.
- Keep pipeline names Vn-proof by registering a *versioned* name.
- Optionally provide an *alias* for convenience in examples.

Add new pipelines
-----------------
1) Import the new pipeline builder inside `register_builtin_pipelines`.
2) Register it under a versioned name: "<domain>.<pipeline>.vN".
3) (Optional) Register an alias without ".vN" pointing to the current default.
"""

from __future__ import annotations

from src.orchestrator.core.registry import PipelineRegistry


def register_builtin_pipelines(registry: PipelineRegistry) -> None:
    """
    Register all built-in pipelines into the provided registry.

    Parameters
    ----------
    registry:
        Registry instance that maps pipeline names -> builder functions.

    Notes
    -----
    - We import pipeline builders inside this function to avoid import-time side effects.
    - Prefer *versioned* names as the stable configuration contract (Vn-proof).
    """

    # ---------------------------------------------------------------------
    # Marketdata: build timeseries dataset (SyntheticProvider for now)
    # ---------------------------------------------------------------------
    # Import the builder function from the pipeline module.
    # We keep the import local to prevent side effects at module import time.
    from src.orchestrator.pipelines.marketdata.build_timeseries import build_pipeline

    # Register the stable, versioned pipeline name.
    registry.register("marketdata.build_timeseries", build_pipeline)