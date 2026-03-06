"""
Training pipeline for the Hybrid GNN-RNN model (PyTorch).

Wires model-specific build_data and build_model hooks into the generic
TrainPipeline orchestration (data -> model -> Trainer.fit -> register -> track).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import torch

from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

from src.rade_ml_pt.pipelines.base import TrainPipeline
from src.rade_ml_pt.pipelines.config import PipelineConfig
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml_pt.data.hybrid_gnn_rnn.build import build_dataset

if TYPE_CHECKING:
    import torch.nn as nn
    from src.rade_ml_pt.data.result import DataBuildResult
    from src.rade_ml_pt.registry.store import ModelRegistry
    from src.rade_ml_pt.tracking.run import Run
    from src.rade_ml_pt.tracking.tracker import ExperimentTracker
    from src.rade_ml_pt.core.types import TrainingResult

logger = logging.getLogger(__name__)


class HybridGnnRnnTrainPipeline(TrainPipeline):
    """
    Concrete training pipeline for Hybrid GNN-RNN.

    Implements the two required abstract hooks:
        - build_data:  load trade PnL, encode attributes, build graph, construct DataLoaders.
        - build_model: instantiate HybridGnnRnn model.
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
        from src.rade_ml_pt.data.hybrid_gnn_rnn.plots import plot_trade_graph
        from src.rade_ml_pt.data.hybrid_gnn_rnn.build import HybridGnnRnnResult

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
    ) -> "nn.Module":
        from src.rade_ml_pt.models.hybrid_gnn_rnn.model import HybridGnnRnn
        from src.rade_ml_pt.models.hybrid_gnn_rnn.config import (
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
        model: "nn.Module",
        registry: Optional["ModelRegistry"] = None,
        tracker: Optional["ExperimentTracker"] = None,
        run: Optional["Run"] = None,
        data_result: Optional["DataBuildResult"] = None,
    ) -> None:
        """
        Register model, log run, save inference artifacts alongside the model,
        then run custom post-training plotting.
        """
        super().post_train(
            result, model, registry=registry, tracker=tracker, run=run,
            data_result=data_result,
        )

        if self._registered_entry is not None and data_result is not None:
            self._save_inference_artifacts(self._registered_entry, data_result)

        if self.config.artifacts_dir and data_result is not None:
            self._post_train_plots(result, model, data_result)

    def _save_inference_artifacts(
        self,
        entry: Any,
        data_result: "DataBuildResult",
    ) -> None:
        """
        Save artifacts needed for cold-start inference alongside the registered model.

        Persists: graph builder, encoder, scalers, data config, and trade universe
        into the registry version directory so a single registry.load() provides
        the path to everything needed for inference and UI analytics.
        """
        import joblib
        from src.rade_ml_pt.data.hybrid_gnn_rnn.build import HybridGnnRnnResult

        if not isinstance(data_result, HybridGnnRnnResult):
            logger.warning("Cannot save inference artifacts: data_result is not HybridGnnRnnResult")
            return

        version_dir = Path(entry.model_dir)

        if data_result.builder is not None:
            data_result.builder.save(version_dir / "graph_builder.pkl")
            logger.info(f"Saved graph_builder.pkl to {version_dir}")

        if data_result.encoder is not None:
            joblib.dump(data_result.encoder, version_dir / "encoder.pkl")
            logger.info(f"Saved encoder.pkl to {version_dir}")

        target_scaler = data_result.metadata.get("target_pnl_transformer")
        if target_scaler is not None:
            joblib.dump(target_scaler, version_dir / "target_scaler.pkl")

        elementary_scaler = data_result.metadata.get("elementary_pnl_transformer")
        if elementary_scaler is not None:
            joblib.dump(elementary_scaler, version_dir / "elementary_scaler.pkl")

        data_config = self.config.data_config
        if data_config is not None:
            if hasattr(data_config, "to_json"):
                data_config.to_json(version_dir / "data_config.json")
            elif isinstance(data_config, dict):
                with open(version_dir / "data_config.json", "w") as f:
                    json.dump(data_config, f, indent=2)

        universe = {
            "elementary_ids": data_result.metadata.get("elementary_ids", []),
            "target_ids": data_result.metadata.get("target_ids", []),
            "elementary_idx": [int(x) for x in data_result.metadata.get("elementary_idx", [])],
            "target_idx": [int(x) for x in data_result.metadata.get("target_idx", [])],
            "selected_trades": data_result.metadata.get("selected_trades", []),
            "removed_trades": data_result.metadata.get("removed_trades", []),
        }
        with open(version_dir / "trade_universe.json", "w") as f:
            json.dump(universe, f, indent=2)

        if data_result.target_pnl is not None:
            data_result.target_pnl.to_parquet(version_dir / "target_pnl.parquet")
        if data_result.elementary_pnl is not None:
            data_result.elementary_pnl.to_parquet(version_dir / "elementary_pnl.parquet")

        if data_result.target_attributes is not None:
            with open(version_dir / "target_attributes.json", "w") as f:
                json.dump(data_result.target_attributes, f, indent=2)
        if data_result.elementary_attributes is not None:
            with open(version_dir / "elementary_attributes.json", "w") as f:
                json.dump(data_result.elementary_attributes, f, indent=2)

        self._save_datasets(version_dir, data_result)

        logger.info(f"Inference artifacts saved to {version_dir}")

    def _save_datasets(
        self,
        version_dir: Path,
        data_result: "DataBuildResult",
    ) -> None:
        """
        Persist DataLoader backing data to the registry version directory.

        Saves the underlying dataset tensors via ``torch.save`` so the eval
        pipeline can reconstruct DataLoaders without re-running the full data build.
        Each split is saved to its own file under datasets/.
        """
        ds_dir = version_dir / "datasets"
        ds_dir.mkdir(exist_ok=True)

        for name, loader in [("train", data_result.train_ds),
                             ("val", data_result.val_ds),
                             ("test", data_result.test_ds)]:
            if loader is not None and hasattr(loader, "dataset"):
                save_path = ds_dir / f"{name}.pt"
                torch.save(loader.dataset, str(save_path))
                logger.info(f"Saved {name} dataset to {save_path}")

    def _post_train_plots(
        self,
        result: "TrainingResult",
        model: "nn.Module",
        data_result: "DataBuildResult",
    ) -> None:
        """Run GNN-RNN-specific plots after training (e.g. prediction scatter, attention)."""
        pass
