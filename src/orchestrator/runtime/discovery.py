"""
Pipeline discovery / registration.

Key design rule:
- This is the only place where we import pipeline modules for registration.
- That avoids import side effects across the rest of the codebase.
"""

from __future__ import annotations

from src.orchestrator.core.registry import PipelineRegistry


def register_builtin_pipelines(registry: PipelineRegistry) -> None:
    """
    Register built-in pipelines into the registry.

    Add new pipelines here over time.
    """
    # Import inside the function to avoid side-effects on module import.
    from src.orchestrator.pipeline.marketdata.replay_static import build_pipeline as md_replay_static

    registry.register("marketdata.replay_static_v1", md_replay_static)