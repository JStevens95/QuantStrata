"""
Training pipeline for the Hybrid GNN-RNN model.

Wires model-specific build_data and build_model hooks into the generic
TrainPipeline orchestration (data -> model -> Trainer.fit -> register -> track).
"""
from __future__ import annotations

import logging

import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING

from src.rade_ml.pipelines.base import TrainPipeline
from src.rade_ml.pipelines.config import PipelineConfig
from src.rade_ml.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml.data.hybrid_gnn_rnn.build import build_dataset

if TYPE_CHECKING:
    import tensorflow as tf
    from src.rade_ml.data.result import DataBuildResult

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

        n = result.graph_builder.adjacency_dense.shape[0]
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

        features = result.graph_builder.features

        save_dir = Path(config.folders.root_folder) / "plots"
        save_path = save_dir / "trade_graph.png"

        plot_trade_graph(
            adjacency=result.graph_builder.adjacency_dense,
            is_target=is_target,
            trade_ids=trade_ids,
            features=features,
            title="Trade Relationship Graph — Training",
            save_path=save_path,
        )
        logger.info(f"Trade graph visualisation saved to {save_path}")

    def build_model(
        self,
        config: PipelineConfig,
        data_result: "DataBuildResult",
    ) -> "tf.keras.Model":
        import tensorflow as tf
        from src.rade_ml.models.hybrid_gnn_rnn.model import HybridGnnRnn
        from src.rade_ml.models.hybrid_gnn_rnn.config import default_model_config

        model_config = config.model_config or default_model_config()
        model = HybridGnnRnn(config=model_config)

        loss_name = model_config.get("general", {}).get("loss", "mse")
        lr = model_config.get("general", {}).get("learning_rate", 1e-3)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=loss_name,
        )
        logger.info(f"Hybrid GNN-RNN model compiled with loss={loss_name}, lr={lr}")
        return model
