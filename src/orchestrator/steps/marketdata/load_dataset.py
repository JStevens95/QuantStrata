"""
Step: LoadDatasetStep

Loads a MarketDataset from cfg.params["dataset_path"] and stores it in ctx.state.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.orchestrator.artifacts.serializers import load_dataset
from src.orchestrator.core.context import Context
from src.orchestrator.core.errors import ConfigError
from src.orchestrator.core.step import Step


@dataclass(slots=True)
class LoadDatasetStep(Step):
    """
    Load a MarketDataset from disk.

    Outputs
    -------
    ctx.state["dataset"] : MarketDataset
        The loaded dataset.
    """

    def run(self, ctx: Context) -> Context:
        # Read pipeline parameters from config.
        params = dict(getattr(ctx.cfg, "params", {}) or {})

        # Extract the required dataset_path.
        dataset_path = str(params.get("dataset_path", "")).strip()
        if not dataset_path:
            raise ConfigError("LoadDatasetStep requires cfg.params['dataset_path'] (non-empty).")

        # Load dataset through orchestrator serializer wrapper.
        dataset = load_dataset(dataset_path)

        # Store dataset in context state for subsequent steps.
        ctx.put("dataset", dataset)

        # Log a compact summary (avoid printing huge content).
        ctx.logger.info(
            "Loaded MarketDataset | path=%s | n_dates=%d | n_scenarios=%d | n_panels=%d",
            dataset_path,
            len(dataset.dates),
            dataset.n_scenarios,
            len(dataset.panels),
        )

        return ctx