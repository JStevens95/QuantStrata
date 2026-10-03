"""FastAPI dependency injection.

Process-wide singletons constructed during the app's lifespan startup
and exposed here as ``Depends`` providers.  Five collaborators live here:

* :class:`ArtifactReader` — serves on-disk **evaluation** artifacts
  (parquet + JSON) to the existing PRISM routers.
* :class:`InferenceStateManager` — holds the live
  :class:`EnsembleInferencePipeline` for the staged inference workflow
  exposed by the ``/inference`` router (Stage 8).
* :class:`InferenceResultReader` — serves on-disk **inference run**
  artifacts (manifest + parquets) to the historical-run endpoints
  added in Stage 10.
* :class:`MonitoringStateManager` — holds the live
  :class:`EnsembleMonitoringPipeline` for the staged monitoring
  workflow exposed by the ``/monitoring`` router (M.4).
* :class:`MonitoringResultReader` — serves on-disk **monitoring run**
  artifacts (manifest + drift summary + per-cluster drift tables)
  to the historical-run endpoints under ``/monitoring/runs/...``.

The injection points are re-exported from their respective service
modules; this file is the canonical FastAPI surface so routers can
``from .dependencies import …`` consistently.
"""
from __future__ import annotations

from src.rade_ml_pt.ensemble.api.services.inference_state import (
    InferenceStateManager,
    get_inference_state_manager,
    set_inference_state_manager,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_reader import (
    MonitoringResultReader,
    get_monitoring_result_reader,
    set_monitoring_result_reader,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_state import (
    MonitoringStateManager,
    get_monitoring_state_manager,
    set_monitoring_state_manager,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader
from src.rade_ml_pt.ensemble.api.services.result_reader import (
    InferenceResultReader,
    get_result_reader,
    set_result_reader,
)

_reader: ArtifactReader | None = None


def set_reader(reader: ArtifactReader) -> None:
    """Called once during the lifespan startup event."""
    global _reader
    _reader = reader


def get_reader() -> ArtifactReader:
    """FastAPI ``Depends`` — returns the singleton reader."""
    if _reader is None:
        raise RuntimeError("ArtifactReader not initialised. Server not ready.")
    return _reader


__all__ = [
    "ArtifactReader",
    "InferenceResultReader",
    "InferenceStateManager",
    "MonitoringResultReader",
    "MonitoringStateManager",
    "get_inference_state_manager",
    "get_monitoring_result_reader",
    "get_monitoring_state_manager",
    "get_reader",
    "get_result_reader",
    "set_inference_state_manager",
    "set_monitoring_result_reader",
    "set_monitoring_state_manager",
    "set_reader",
    "set_result_reader",
]
