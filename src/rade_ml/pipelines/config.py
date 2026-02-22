"""
Pipeline configuration.

``PipelineConfig`` aggregates references to the training config, data config,
and infrastructure paths (registry, experiment tracker, artifacts).  It is
the single object passed into every pipeline's ``run()`` method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PipelineConfig:
    """
    Top-level configuration for an end-to-end pipeline run.

    Attributes
    ----------
    training_config : dict or None
        Serialised ``TrainingConfig`` (or the dataclass itself -- pipelines
        should accept either).
    data_config : dict or None
        Serialised ``DataPipelineConfig`` or model-specific data config.
    model_config : dict or None
        Model-specific architecture configuration.
    registry_dir : str or None
        Root directory for the model registry.
    tracking_dir : str or None
        Root directory for experiment tracking.
    artifacts_dir : str or None
        Root directory for pipeline-produced artifacts (plots, reports, etc.).
    version_or_tag : str
        Registry version or tag to load (used by eval / inference pipelines).
    metadata : dict
        Arbitrary key-value pairs forwarded into run records.
    """

    training_config: Optional[Dict[str, Any]] = None
    data_config: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None
    registry_dir: Optional[str] = None
    tracking_dir: Optional[str] = None
    artifacts_dir: Optional[str] = None
    version_or_tag: str = "latest"
    metadata: Dict[str, Any] = field(default_factory=dict)
