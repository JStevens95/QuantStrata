"""
Training pipeline for the Hybrid GNN-RNN model.

Wires model-specific build_data and build_model hooks into the generic
TrainPipeline orchestration (data -> model -> Trainer.fit -> register -> track).
"""
from __future__ import annotations

import logging

import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.rade_ml.pipelines.base import TrainPipeline
from src.rade_ml.pipelines.config import PipelineConfig
from src.rade_ml.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml.data.hybrid_gnn_rnn.build import build_dataset

if TYPE_CHECKING:
    import tensorflow as tf
    from src.rade_ml.data.result import DataBuildResult
    from src.rade_ml.registry.store import ModelRegistry
    from src.rade_ml.tracking.run import Run
    from src.rade_ml.tracking.tracker import ExperimentTracker
    from src.rade_ml.core.types import TrainingResult

logger = logging.getLogger(__name__)


class HybridGnnRnnTrainPipeline(TrainPipeline):
    """
    Concrete training pipeline for Hybrid GNN-RNN.

    Implements the two required abstract hooks:
        - build_data:  load trade PnL, encode attributes, build graph, construct tf.data.Datasets.
        - build_model: instantiate HybridGnnRnn and compile with MSE loss.
    """

    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = HybridGnnRnnDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = HybridGnnRnnDataConfig()

        job = config.metadata.get("job", {})
        result = build_dataset(config=data_config, job=job)

        if data_config.plot_trade_graph:
            self._plot_graph(result, data_config)

        return result

    def _plot_graph(self, result: "DataBuildResult", config: HybridGnnRnnDataConfig) -> None:
        """Visualise the trade graph after the data build."""
        from src.rade_ml.data.hybrid_gnn_rnn.plots import plot_trade_graph
        from src.rade_ml.data.hybrid_gnn_rnn.build import HybridGnnRnnResult

        if not isinstance(result, HybridGnnRnnResult) or result.graph_builder is None:
            logger.warning("Cannot plot trade graph: missing graph_builder on result.")
            return

        builder = result.graph_builder
        n = builder.sparse_shape[0]
        is_target = np.zeros(n, dtype=bool)
        if result.target_idx is not None:
            is_target[result.target_idx] = True

        trade_ids = None
        if result.elementary_ids and result.target_ids:
            all_ids = [""] * n
            if result.elementary_idx is not None:
                for i, tid in zip(result.elementary_idx, result.elementary_ids):
                    all_ids[i] = tid
            if result.target_idx is not None:
                for i, tid in zip(result.target_idx, result.target_ids):
                    all_ids[i] = tid
            trade_ids = all_ids

        save_dir = Path(config.folders.root_folder) / "plots"
        save_path = save_dir / "trade_graph.png"

        plot_trade_graph(
            adjacency_indices=builder.sparse_indices,
            adjacency_values=builder.sparse_values,
            adjacency_dense_shape=np.array(builder.sparse_shape, dtype=np.int64),
            is_target=is_target,
            trade_ids=trade_ids,
            features=builder.features,
            title="Trade Relationship Graph — Training",
            save_path=save_path,
        )
        logger.info(f"Trade graph visualisation saved to {save_path}")

    def build_model(
        self,
        config: PipelineConfig,
        data_result: "DataBuildResult",
    ) -> "tf.keras.Model":
        from src.rade_ml.models.hybrid_gnn_rnn.model import HybridGnnRnn
        from src.rade_ml.models.hybrid_gnn_rnn.config import (
            HybridGnnRnnModelConfig,
            default_model_config,
        )

        raw = config.model_config or default_model_config()
        if hasattr(raw, "to_dict"):
            model_config = raw.to_dict()
        else:
            model_config = HybridGnnRnnModelConfig.from_dict(raw).to_dict()
        model = HybridGnnRnn(config=model_config)
        logger.info("Hybrid GNN-RNN model built (compile deferred to Trainer via TrainingConfig)")
        return model

    def post_train(
        self,
        result: "TrainingResult",
        model: "tf.keras.Model",
        registry: Optional["ModelRegistry"] = None,
        tracker: Optional["ExperimentTracker"] = None,
        run: Optional["Run"] = None,
        data_result: Optional["DataBuildResult"] = None,
    ) -> None:
        """
        Register model, log run, then run custom post-training plotting and reports.
        The base run() will still call _generate_training_report after this.
        """
        # 1. Default: register model and log to tracker.
        super().post_train(
            result, model, registry=registry, tracker=tracker, run=run,
            data_result=data_result,
        )

        # 2. Custom: model-specific post-training plots/reports (using trained model + data).
        if self.config.artifacts_dir and data_result is not None:
            self._post_train_plots(result, model, data_result)

    def _post_train_plots(
        self,
        result: "TrainingResult",
        model: "tf.keras.Model",
        data_result: "DataBuildResult",
    ) -> None:
        """Run GNN-RNN-specific plots after training (e.g. prediction scatter, attention)."""
        # Example: add custom plots that need the trained model.
        # The standard training report (loss curve, config) is generated by the base
        # run() via _generate_training_report after post_train.
        pass
