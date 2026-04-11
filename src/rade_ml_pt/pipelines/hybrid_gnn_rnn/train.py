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
from src.rade_ml_pt.data.hybrid_gnn_rnn.plots import plot_pnl_distribution, plot_trade_graph

if TYPE_CHECKING:
    import torch.nn as nn
    from src.rade_ml_pt.data.result import DataBuildResult
    from src.rade_ml_pt.registry.store import ModelRegistry
    from src.rade_ml_pt.tracking.run import Run
    from src.rade_ml_pt.tracking.tracker import ExperimentTracker
    from src.rade_ml_pt.core.types import TrainingResult

# define module level logging.
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
        from src.rade_ml_pt.core.config import sanitize_yaml_values

        raw = config.model_config or default_model_config()
        if hasattr(raw, "to_dict"):
            model_config = raw.to_dict()
        elif isinstance(raw, dict):
            model_config = HybridGnnRnnModelConfig.from_dict(
                sanitize_yaml_values(raw)
            ).to_dict()
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
        then run custom post-training plotting and inline evaluation.
        """
        # default: register model and log to tracker
        super().post_train(result, model, registry=registry, tracker=tracker, run=run, data_result=data_result)

        # custom: save input artifacts.
        if self._registered_entry is not None and data_result is not None:
            self._save_training_artifacts(self._registered_entry, data_result)

        if self.config.artifacts_dir and data_result is not None:
            self._post_train_plots(result, model, data_result)

    def _save_training_artifacts(self, entry: Any, data_result: "DataBuildResult") -> None:
        """
        Save artifacts needed for cold-start inference alongside the registered model.

        Persists: graph builder, encoder, scalers, data config, and trade universe
        into the registry version directory so a single registry.load() provides
        the path to everything needed for inference and UI analytics.
        """
        import joblib
        from src.rade_ml_pt.data.hybrid_gnn_rnn.build import HybridGnnRnnResult

        # check data results are of correct type.
        assert isinstance(data_result, HybridGnnRnnResult), \
            "Cannot save artifacts: data_result isn't HybridGnnRnnResult"

        # create dir for saving.
        version_dir = Path(entry.model_dir)

        # ---- saving graph builder object & results. ----
        if data_result.graph_builder is not None:
            data_result.graph_builder.save(version_dir / "graph_builder.pkl")
            logger.info(f"Saved graph_builder.pkl to {version_dir}")
        if data_result.graph_results is not None:
            joblib.dump(value=data_result.graph_results, filename=version_dir / "graph_results.joblib", compress=5)
            logger.info(f"Saved graph_results.joblib to {version_dir}")

        # ---- saving attribute encoder object & results ----
        if data_result.encoder is not None:
            data_result.encoder.save(version_dir / "encoder.pkl")
            logger.info(f"Saved encoder.pkl to {version_dir}")
        if data_result.encoder_results is not None:
            joblib.dump(value=data_result.encoder_results, filename=version_dir / "encoder_results.joblib", compress=5)
            logger.info(f"Saved encoder_results.joblib to {version_dir}")

        # ---- saving elementary and target transformer objects. ----
        target_scaler = data_result.metadata["target_pnl_transformer"]
        if target_scaler is not None:
            joblib.dump(value=target_scaler, filename=version_dir / "target_scaler.pkl")
            logger.info(f"Saved target_scaler.pkl to {version_dir}")
        elementary_scaler = data_result.metadata["elementary_pnl_transformer"]
        if elementary_scaler is not None:
            joblib.dump(value=elementary_scaler, filename=version_dir / "elementary_scaler.pkl")
            logger.info(f"Saved elementary_scaler.pkl to {version_dir}")

        # ---- saving data pipeline configuration. ----
        data_config = data_result.data_config
        if data_config is not None:
            if hasattr(data_config, "to_json"):
                data_config.to_json(path=version_dir / "data_config.json")
                logger.info(f"Saved data_config.json to {version_dir}")
            elif isinstance(data_config, dict):
                with open(version_dir / "data_config.json", "w") as f:
                    json.dump(data_config, f, indent=2)
                logger.info(f"Saved data_config.json to {version_dir}")

        # ---- saving trade universe. ----
        def _to_serializable(val):
            """Convert numpy arrays/scalars to JSON-serializable Python types."""
            if hasattr(val, "tolist"):
                return val.tolist()
            return val

        universe = {
            # scenario / split tracking.
            "scenarios": data_result.metadata.get("scenarios", []),
            "sequence_length": data_result.metadata.get("sequence_length", 1),
            "scenario_idx": data_result.metadata.get("scenario_idx", []),

            # training periods
            "train_indices": _to_serializable(data_result.metadata.get("train_indices", [])),
            "train_starts": _to_serializable(data_result.metadata.get("train_starts", [])),
            "train_ends": _to_serializable(data_result.metadata.get("train_ends", [])),
            "train_size": _to_serializable(data_result.metadata.get("train_size", 0)),
            "train_scenarios": _to_serializable(data_result.metadata.get("train_scenarios", [])),
            "train_end_scenarios": _to_serializable(data_result.metadata.get("train_end_scenarios", [])),

            # validation periods.
            "val_indices": _to_serializable(data_result.metadata.get("val_indices", [])),
            "val_starts": _to_serializable(data_result.metadata.get("val_starts", [])),
            "val_ends": _to_serializable(data_result.metadata.get("val_ends", [])),
            "val_size": _to_serializable(data_result.metadata.get("val_size", 0)),
            "val_scenarios": _to_serializable(data_result.metadata.get("val_scenarios", [])),
            "val_end_scenarios": _to_serializable(data_result.metadata.get("val_end_scenarios", [])),

            # test periods.
            "test_indices": _to_serializable(data_result.metadata.get("test_indices", [])),
            "test_starts": _to_serializable(data_result.metadata.get("test_starts", [])),
            "test_ends": _to_serializable(data_result.metadata.get("test_ends", [])),
            "test_size": _to_serializable(data_result.metadata.get("test_size", 0)),
            "test_scenarios": _to_serializable(data_result.metadata.get("test_scenarios", [])),
            "test_end_scenarios": _to_serializable(data_result.metadata.get("test_end_scenarios", [])),

            # trade universe
            "elementary_ids": data_result.metadata.get("elementary_ids", []),
            "target_ids": data_result.metadata.get("target_ids", []),
            "elementary_idx": data_result.metadata.get("elementary_idx", []),
            "target_idx": data_result.metadata.get("target_idx", []),
            "selected_trades": data_result.metadata.get("selected_trades", []),
            "removed_trades": data_result.metadata.get("removed_trades", []),
        }
        with open(version_dir / "trade_universe.json", "w") as f:
            json.dump(universe, f, indent=2)
        logger.info(f"Saved trade_universe.json to {version_dir}")

        # ---- saving scaled and reduced elementary and target pnl. ----
        if data_result.target_pnl is not None:
            data_result.target_pnl.to_parquet(path=version_dir / "target_pnl.parquet")
            logger.info(f"Saved target_pnl.parquet to {version_dir}")
        if data_result.elementary_pnl is not None:
            data_result.elementary_pnl.to_parquet(path=version_dir / "elementary_pnl.parquet")
            logger.info(f"Saved elementary_pnl.parquet to {version_dir}")

        # ---- saving elementary and target attributes ----
        if data_result.target_attributes is not None:
            with open(version_dir / "target_attributes.json", "w") as f:
                json.dump(data_result.target_attributes, f, indent=2)
            logger.info(f"Saved target_attributes.json to {version_dir}")
        if data_result.elementary_attributes is not None:
            with open(version_dir / "elementary_attributes.json", "w") as f:
                json.dump(data_result.elementary_attributes, f, indent=2)
            logger.info(f"Saved elementary_attributes.json to {version_dir}")

        # ---- save datasets ----
        self._save_datasets(version_dir, data_result)

        # ---- save input job object as pkl file. ----
        job = self.config.metadata.get("job", {})
        for key, value in job.items():
            if key not in ["name", "request_log"]:
                joblib.dump(value, version_dir / f"{key}.joblib", compress=5)
            logger.info(f"Saved {key} to {version_dir}")
        logger.info(f"Artifacts saved to {version_dir}")

    @staticmethod
    def _save_datasets(version_dir: Path, data_result: "DataBuildResult") -> None:
        """
        Persist DataLoader backing data to the registry version directory.

        Saves the underlying dataset tensors via ``torch.save`` so the eval
        pipeline can reconstruct DataLoaders without re-running the full data build.
        Each split is saved to its own file under datasets/.
        """
        ds_dir = version_dir / "datasets"
        ds_dir.mkdir(exist_ok=True)

        for name, loader in [
            ("train", data_result.train_ds), ("val", data_result.val_ds), ("test", data_result.test_ds)
        ]:
            if loader is not None and hasattr(loader, "dataset"):
                save_path = ds_dir / f"{name}.pt"
                torch.save(loader.dataset, str(save_path))
                logger.info(f"Saved {name} dataset to {save_path}")

    def _post_train_plots(self, result: "TrainingResult", model: "nn.Module", data_result: "DataBuildResult") -> None:
        """Run GNN-RNN-specific plots after training (e.g. prediction scatter, attention)."""
        from src.rade_ml_pt.training.plots import plot_training_analytics

        # define training artifacts path.
        save_path = Path(self.config.artifacts_dir, "training", self._registered_entry.version)
        save_path.mkdir(parents=True, exist_ok=True)

        # plotting training analytics.
        plot_training_analytics(result=result, save_dir=save_path)

        # plotting pnl distribution, if specified in configuration.
        dc = self.config.data_config
        plot_pnl = dc.get("plot_pnl_distribution", False) if isinstance(dc, dict) else getattr(dc, "plot_pnl_distribution", False)
        if plot_pnl and self.config.artifacts_dir:
            plot_pnl_distribution(
                elementary_pnl=data_result.elementary_pnl, target_pnl=data_result.target_pnl,
                train_indices=data_result.metadata["train_indices"], save_path=save_path
            )

        # plotting trade graph
        plot_graph = dc.get("plot_trade_graph", False) if isinstance(dc, dict) else getattr(dc, "plot_trade_graph", False)
        if plot_graph:
            # plot trade graph analytics.
            plot_trade_graph(
                adjacency_indices=data_result.graph_results["sparse_indices"],
                adjacency_values=data_result.graph_results["sparse_values"],
                adjacency_dense_shape=data_result.graph_results["sparse_shape"],
                is_target=data_result.graph_results["is_target"],
                features=data_result.encoder_results["combined_features"],
                trade_ids=data_result.metadata["elementary_ids"] + data_result.metadata["target_ids"],
                save_path=save_path
            )
