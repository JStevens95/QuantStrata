"""
Training pipeline for the Hybrid GNN-RNN model (PyTorch).

Wires model-specific build_data and build_model hooks into the generic
TrainPipeline orchestration (data -> model -> Trainer.fit -> register -> track).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
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
            self._save_training_artifacts(self._registered_entry, data_result, result)

        if self.config.artifacts_dir and data_result is not None:
            self._post_train_plots(result, model, data_result)

    def _save_training_artifacts(
        self,
        entry: Any,
        data_result: "DataBuildResult",
        result: "TrainingResult",
    ) -> None:
        """
        Save artifacts needed for cold-start inference alongside the registered model.

        Persists: graph builder, encoder, scalers, data config, trade universe,
        and per-epoch training curves into the registry version directory so
        a single registry.load() provides the path to everything needed for
        inference and UI analytics.

        Per the PRISM pipeline output contract (see
        ``docs/platform_designs/prism_retool_migration.md`` §11.15.1) this
        method is the single producer of all per-member registry files.
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

        # ---- save lightweight RF-keys sidecar (avoids loading
        #      cluster_assets.joblib at inference time just to decide
        #      which clusters are affected by new scenarios; see
        #      ``intersecting_risk_factors`` and the lazy-load in
        #      ``_build_affected_inputs``). ----
        cluster_assets = job.get("cluster_assets")
        if cluster_assets is not None:
            rf_keys = {
                asset_name: list(asset.risk_factor_shocks.keys())
                for asset_name, asset in cluster_assets.items()
            }
            with open(version_dir / "cluster_rf_keys.json", "w") as f:
                json.dump(rf_keys, f, indent=2)
            logger.info(f"Saved cluster_rf_keys.json to {version_dir}")

        # ---- save per-epoch training curves (PRISM contract §11.15.1). ----
        self._save_training_curves(version_dir, result)

        # ---- save monitoring + DQ baselines (PRISM contract §11.15.1). ----
        # Residual-based baselines (which need a forward pass) are emitted
        # by the eval pipeline per §11.15.2 — eval already runs inference
        # on the train split, so we avoid a duplicate forward pass here.
        self._save_baselines(version_dir, data_result)

        logger.info(f"Artifacts saved to {version_dir}")

    def _save_baselines(
        self,
        version_dir: Path,
        data_result: "DataBuildResult",
    ) -> None:
        """
        Write per-member feature baselines for drift monitoring and DQ.

        Three parquets land under ``version_dir``:

        - ``monitoring/baseline_feature_stats.parquet``        (Phase 4a)
        - ``quality/completeness_train_baseline.parquet``      (Phase 5a)
        - ``quality/feature_summary_train_baseline.parquet``   (Phase 5a)

        All three are computed purely from the scaled training slice of
        ``data_result.elementary_pnl``.  They do **not** require a model
        forward pass.  Residual baselines, which do, are produced by the
        eval pipeline per the PRISM contract §11.15.2.

        Raises
        ------
        RuntimeError
            If cluster id cannot be resolved, or if ``elementary_pnl`` /
            ``train_indices`` are absent.  The contract demands these
            files exist after every training run, so missing inputs
            surface loudly rather than silently producing empty files.
        """
        from src.rade_ml_pt.monitoring.baselines import save_feature_baseline
        from src.rade_ml_pt.quality.completeness import save_completeness
        from src.rade_ml_pt.quality.summary import save_feature_summary

        cluster_id = self.config.metadata.get("cluster_id")
        if not cluster_id:
            # Fall back to the member registry version name for standalone
            # runs that bypass EnsembleConfig.  Log loudly — standalone
            # runs still produce the file, but the cluster label will be
            # the opaque version string.
            cluster_id = getattr(self._registered_entry, "version", None) or "standalone"
            logger.warning(
                "cluster_id missing from pipeline metadata; falling back to "
                "'%s' for baseline labelling.  Set metadata['cluster_id'] "
                "via EnsembleConfig.get_member_pipeline_config to suppress.",
                cluster_id,
            )

        if data_result.elementary_pnl is None:
            raise RuntimeError(
                f"data_result.elementary_pnl is None for '{cluster_id}' — "
                "cannot emit training baselines."
            )

        train_indices = data_result.metadata.get("train_indices")
        if train_indices is None or len(train_indices) == 0:
            raise RuntimeError(
                f"train_indices empty for '{cluster_id}' — cannot emit "
                "training baselines."
            )

        train_features = data_result.elementary_pnl.iloc[np.asarray(train_indices)]

        save_feature_baseline(
            version_dir / "monitoring" / "baseline_feature_stats.parquet",
            features=train_features,
            cluster_id=cluster_id,
        )
        save_completeness(
            version_dir / "quality" / "completeness_train_baseline.parquet",
            features=train_features,
            cluster_id=cluster_id,
        )
        save_feature_summary(
            version_dir / "quality" / "feature_summary_train_baseline.parquet",
            features=train_features,
            cluster_id=cluster_id,
        )

    @staticmethod
    def _save_training_curves(version_dir: Path, result: "TrainingResult") -> None:
        """
        Persist per-epoch training curves as a parquet alongside the
        registered model.

        This is a required output under the PRISM pipeline contract
        (``docs/platform_designs/prism_retool_migration.md`` §11.15.1).
        Consumed by ``ArtifactCache`` and the PRISM training-curve
        endpoint; no configuration flag gates emission.

        Schema
        ------
        ``epoch``       int32    (required)
        ``train_loss``  float32  (required — sourced from ``history["loss"]``)
        *other columns* float32  (one per extra key in ``result.history``)

        Every per-epoch series the trainer emits is persisted; this
        writer is metric-agnostic by design so new metrics added to
        the trainer automatically flow through to Retool/PRISM
        without any change here or in ``ArtifactCache``.

        A series is rejected (and raises) if its length differs from
        ``len(history["loss"])``, since a mismatched curve cannot be
        joined to the ``epoch`` axis and would mislead downstream
        consumers.  Non-numeric or non-length-aware values are not
        silently dropped — they raise ``RuntimeError``.

        File-level pyarrow metadata stores ``_schema_version=1``.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        history = result.history or {}
        loss = history.get("loss")
        if not loss:
            raise RuntimeError(
                f"Training for '{version_dir.name}' completed without a "
                "populated loss history; cannot emit training_curves.parquet. "
                "This indicates a Trainer bug, not a legitimate off-mode."
            )
        n_epochs = len(loss)
        columns: dict = {
            "epoch": np.arange(n_epochs, dtype=np.int32),
            "train_loss": np.asarray(loss, dtype=np.float32),
        }
        # Persist every other per-epoch series the trainer emitted.
        # We refuse to silently drop or truncate: any key whose length
        # does not match ``n_epochs`` is a trainer-side contract break.
        for key, vals in history.items():
            if key == "loss" or vals is None:
                continue
            try:
                arr = np.asarray(vals, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"history['{key}'] for '{version_dir.name}' is not a "
                    f"numeric per-epoch series: {exc}"
                ) from exc
            if arr.ndim != 1 or arr.shape[0] != n_epochs:
                raise RuntimeError(
                    f"history['{key}'] for '{version_dir.name}' has shape "
                    f"{arr.shape}, expected ({n_epochs},).  Every metric in "
                    "history must be a flat per-epoch series aligned with 'loss'."
                )
            columns[key] = arr

        df = pd.DataFrame(columns)
        # ``pa.table`` (module-level factory) avoids the spurious
        # "Parameter 'cls_1' unfilled" warning PyCharm's pyarrow stub
        # produces against the ``Table.from_pandas`` classmethod.
        table = pa.table(df)
        schema_metadata = {
            **(table.schema.metadata or {}),
            b"_schema_version": b"1",
        }
        table = table.replace_schema_metadata(schema_metadata)
        out_path = version_dir / "training_curves.parquet"
        pq.write_table(table, str(out_path))
        logger.info(f"Saved training_curves.parquet to {version_dir}")

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
