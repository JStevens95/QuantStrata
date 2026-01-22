"""
Pipeline: marketdata.replay_static_v1

Purpose
-------
Load a MarketDataset from disk, build a StaticProvider around it,
and optionally save a copy under the run artifacts folder.

Expected cfg.params
-------------------
dataset_path : str (required)
    Path to the MarketDataset artifact directory.
provider_name : str (optional)
    Provider display name (default "StaticProvider").
save_enabled : bool (optional)
    If True, save dataset into run artifacts (default True).
save_name : str (optional)
    Artifacts subfolder name (default "replayed_dataset").
overwrite : bool (optional)
    If True, overwrite saved artifact if it exists (default False).
static_config : dict (optional)
    Overrides for StaticProviderConfig fields.
"""

from __future__ import annotations

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.steps.marketdata.build_static_provider import BuildStaticProviderStep
from src.orchestrator.steps.marketdata.load_dataset import LoadDatasetStep
from src.orchestrator.steps.marketdata.save_dataset import SaveDatasetStep


def build_pipeline(cfg: RunConfig) -> Pipeline:
    """
    Build the replay_static_v1 pipeline.

    Notes
    -----
    Pipeline is intentionally simple in V1:
    - load dataset
    - create provider
    - save dataset (optional)
    """
    steps = [
        LoadDatasetStep(name="marketdata.load_dataset"),
        BuildStaticProviderStep(name="marketdata.build_static_provider"),
        SaveDatasetStep(name="marketdata.save_dataset"),
    ]
    return Pipeline(name="marketdata.replay_static_v1", steps=steps)