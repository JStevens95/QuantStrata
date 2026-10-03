"""
rade_static_replication
========================

Preprocessing library that turns a raw derivatives portfolio (trade attributes +
historical scenario PnL) into the elementary-basis PnL tensors consumed by the
``rade_ml`` hybrid GNN-RNN model, using the theory of *static replication*.

Public entry points::

    from src.rade_static_replication import run, Orchestrator, OrchestratorConfig
    from src.rade_static_replication import load_config, default_registry

See :func:`run` / :mod:`api` for the one-call facade, or drive ``Orchestrator``
stage-by-stage for debugging. Architecture and module map are documented in
``README.md``.
"""
from __future__ import annotations

__version__ = "0.2.0"

from src.rade_static_replication.api import run
from src.rade_static_replication.assets.registry import default_registry
from src.rade_static_replication.config.loader import config_from_dict, load_config
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.pipeline.context import RunContext
from src.rade_static_replication.pipeline.orchestrator import Orchestrator

__all__ = [
    "__version__",
    "run",
    "Orchestrator",
    "RunContext",
    "OrchestratorConfig",
    "load_config",
    "config_from_dict",
    "default_registry",
]
