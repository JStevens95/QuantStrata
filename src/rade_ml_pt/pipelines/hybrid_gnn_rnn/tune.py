"""
Hyperparameter tuning pipeline for the Hybrid GNN-RNN model (PyTorch).

Defines the search space over GNN, RNN, fusion, and training hyperparameters
and wires into the generic TunePipeline orchestration.

Usage::

    config = PipelineConfig(
        data_config=HybridGnnRnnDataConfig(...).to_dict(),
        artifacts_dir="./rade_ml_output/artifacts",
        metadata={"run_name": "gnn_rnn_tune"},
    )
    pipeline = HybridGnnRnnTunePipeline(
        config,
        tuner_kwargs={"n_trials": 100, "direction": "minimize", "pruner": "hyperband"},
    )
    result = pipeline.run()
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch.nn as nn

from src.rade_ml_pt.pipelines.base import TunePipeline
from src.rade_ml_pt.pipelines.config import PipelineConfig
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml_pt.data.hybrid_gnn_rnn.build import build_dataset

if TYPE_CHECKING:
    import optuna
    from src.rade_ml_pt.data.result import DataBuildResult

logger = logging.getLogger(__name__)


class HybridGnnRnnTunePipeline(TunePipeline):
    """
    Concrete tuning pipeline for Hybrid GNN-RNN.

    Search space covers:
        - GNN: units, depth, dropout, aggregation
        - RNN: units, depth, dropout, recurrent_dropout
        - Fusion: units, dropout
        - Training: learning_rate, batch_size
    """

    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = HybridGnnRnnDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = HybridGnnRnnDataConfig()

        job = config.metadata.get("job", {})
        return build_dataset(config=data_config, job=job)

    def search_space(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """
        Sample hyperparameters for one trial.

        Override this method to customise the search space for your
        specific use case (e.g. narrower ranges after initial exploration).
        """
        return {
            "gnn_units": trial.suggest_int("gnn_units", 32, 256, step=32),
            "gnn_depth": trial.suggest_int("gnn_depth", 1, 4),
            "gnn_dropout": trial.suggest_float("gnn_dropout", 0.0, 0.5, step=0.05),
            "gnn_aggregation": trial.suggest_categorical(
                "gnn_aggregation", ["mean", "sum", "max"],
            ),
            "rnn_units": trial.suggest_int("rnn_units", 32, 256, step=32),
            "rnn_depth": trial.suggest_int("rnn_depth", 1, 3),
            "rnn_dropout": trial.suggest_float("rnn_dropout", 0.0, 0.5, step=0.05),
            "rnn_recurrent_dropout": trial.suggest_float(
                "rnn_recurrent_dropout", 0.0, 0.3, step=0.05,
            ),
            "fusion_units": trial.suggest_int("fusion_units", 32, 256, step=32),
            "fusion_dropout": trial.suggest_float("fusion_dropout", 0.0, 0.5, step=0.05),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

    def build_trial_model(
        self,
        params: Dict[str, Any],
        data_result: "DataBuildResult",
    ) -> nn.Module:
        """Build a HybridGnnRnn model from the sampled hyperparameters."""
        from src.rade_ml_pt.models.hybrid_gnn_rnn.model import HybridGnnRnn
        from src.rade_ml_pt.models.hybrid_gnn_rnn.config import HybridGnnRnnModelConfig

        base_config = self.config.model_config or {}
        if hasattr(base_config, "to_dict"):
            base_config = base_config.to_dict()

        config_dict = HybridGnnRnnModelConfig.from_dict(base_config).to_dict()

        gnn = config_dict["gnn_layer"]
        gnn["parameters"]["units"] = params["gnn_units"]
        gnn["parameters"]["dropout_rate"] = params["gnn_dropout"]
        gnn["general"]["depth"] = params["gnn_depth"]
        gnn["general"]["aggregation"] = params["gnn_aggregation"]

        rnn = config_dict["rnn_layer"]
        rnn["parameters"]["units"] = params["rnn_units"]
        rnn["parameters"]["dropout"] = params["rnn_dropout"]
        rnn["parameters"]["recurrent_dropout"] = params["rnn_recurrent_dropout"]
        rnn["general"]["depth"] = params["rnn_depth"]

        fusion = config_dict["fusion_layer"]
        fusion["parameters"]["units"] = params["fusion_units"]
        fusion["parameters"]["dropout_rate"] = params["fusion_dropout"]

        model = HybridGnnRnn(config=config_dict)
        logger.debug(
            f"Trial model: gnn={params['gnn_units']}x{params['gnn_depth']}, "
            f"rnn={params['rnn_units']}x{params['rnn_depth']}, "
            f"fusion={params['fusion_units']}, lr={params['learning_rate']:.2e}"
        )
        return model

    def post_tune(
        self,
        result: Any,
        config: PipelineConfig,
        data_result: Optional["DataBuildResult"] = None,
    ) -> None:
        """Log best params and optionally train a final model with them."""
        logger.info(
            f"Best trial #{result.best_trial_number}: "
            f"val_loss={result.best_value:.6f} | "
            f"params={result.best_params}"
        )
        logger.info(
            f"Trials: {result.n_completed} completed, {result.n_pruned} pruned, "
            f"{result.elapsed_seconds:.1f}s total"
        )
