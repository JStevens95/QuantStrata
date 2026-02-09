"""
Step: SaveDatasetStep

Saves the loaded dataset into the run ArtifactStore, unless disabled.

This step is optional because some users may only want replay without copying.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.orchestrator.core.context import Context
from src.orchestrator.core.errors import ConfigError
from src.orchestrator.core.step import Step


@dataclass(slots=True)
class SaveDatasetStep(Step):
    """
    Save the dataset currently in ctx.state["dataset"] into run artifacts.

    Inputs
    ------
    ctx.state["dataset"] : MarketDataset

    Config params
    -------------
    save_enabled : bool (default True)
    save_name : str (default "replayed_dataset")
    overwrite : bool (default False)

    Outputs
    -------
    ctx.state["saved_dataset_path"] : str
        Path to saved dataset artifact directory.
    """

    def run(self, ctx: Context) -> Context:
        # Pull dataset from context.
        dataset = ctx.get("dataset")
        if dataset is None:
            raise ConfigError("SaveDatasetStep requires ctx.state['dataset'] to be set.")

        # Read pipeline params.
        params = dict(getattr(ctx.cfg, "params", {}) or {})

        # Allow disabling saves for fast iterations.
        save_enabled = bool(params.get("save_enabled", True))
        if not save_enabled:
            ctx.logger.info("SaveDatasetStep disabled | save_enabled=False")
            return ctx

        # Determine artifact folder name inside this run.
        save_name = str(params.get("save_name", "replayed_dataset")).strip() or "replayed_dataset"
        overwrite = bool(params.get("overwrite", False))

        # Artifact store is required to persist outputs.
        store = getattr(ctx, "artifact_store", None)
        if store is None:
            raise ConfigError("Context.artifact_store is missing; cannot save artifacts.")

        # Save dataset via ArtifactStore helper.
        saved_path = store.save_market_dataset(dataset, name=save_name, overwrite=overwrite)

        # Store path for downstream usage (and manifest).
        ctx.put("saved_dataset_path", str(saved_path))

        ctx.logger.info("Saved MarketDataset | name=%s | path=%s", save_name, saved_path)
        return ctx