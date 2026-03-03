"""
Evaluation pipeline for the Hybrid GNN-RNN model.

Loads a registered model, builds test data, and runs evaluation through the
generic Evaluator. The post_eval hook adds GNN-RNN-specific diagnostics
and persists evaluation analytics for the UI dashboard.

When datasets were saved during training (registry/{version}/datasets/),
build_data() loads them directly instead of re-running the full data pipeline.
This enables true cold-start evaluation from the registry alone.

In addition to test-set evaluation, this pipeline collects predicted vs actual
PnL for all available splits (train, val, test) so the UI can show how model
output compares to actual PnL across the full timeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING, Optional, Tuple

import numpy as np

from src.rade_ml.pipelines.base import EvalPipeline
from src.rade_ml.pipelines.config import PipelineConfig
from src.rade_ml.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml.data.hybrid_gnn_rnn.build import build_dataset, HybridGnnRnnResult

if TYPE_CHECKING:
    import tensorflow as tf
    from src.rade_ml.core.types import EvaluationResult
    from src.rade_ml.data.result import DataBuildResult

logger = logging.getLogger(__name__)


class HybridGnnRnnEvalPipeline(EvalPipeline):
    """
    Concrete evaluation pipeline for Hybrid GNN-RNN.

    Overrides the base ``run()`` to collect predictions on all splits
    (train, val, test) — not just test — so the UI can show model PnL
    vs actual PnL across the full training/evaluation timeline.

    If cached datasets exist in the registry version directory, they are loaded
    directly — no data pipeline is re-run.
    """

    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        cached = self._load_cached_data()
        if cached is not None:
            logger.info("EvalPipeline: loaded cached datasets from registry (skipped data build)")
            return cached

        logger.info("EvalPipeline: no cached datasets found, running full data build")
        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = HybridGnnRnnDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = HybridGnnRnnDataConfig()

        job = config.metadata.get("job", {})
        return build_dataset(config=data_config, job=job)

    def run(self) -> "EvaluationResult":
        """
        Execute evaluation on the test set, then collect predictions on
        all available splits (train, val, test) for PnL comparison analytics.

        The per-split predictions are saved alongside the standard evaluation
        artifacts so the UI can render predicted vs actual PnL timelines
        across the full data period.
        """
        from src.rade_ml.evaluation.evaluator import Evaluator

        logger.info("EvalPipeline: starting")

        model, entry = self.load_model(self.config)
        self._loaded_entry = entry
        logger.info(f"EvalPipeline: loaded model version '{entry.version}'")

        data_result = self.build_data(self.config)
        logger.info("EvalPipeline: data built")

        target_scaler = self.get_target_scaler(data_result)
        evaluator = Evaluator(model=model)

        eval_result = evaluator.run(
            data_result.test_ds,
            target_scaler=target_scaler,
        )
        logger.info("EvalPipeline: test evaluation complete")

        split_predictions = self._collect_all_split_predictions(
            model, data_result, target_scaler,
        )

        self.post_eval(
            eval_result, self.config,
            data_result=data_result,
            split_predictions=split_predictions,
        )

        logger.info("EvalPipeline: done")
        return eval_result

    def _collect_all_split_predictions(
        self,
        model: "tf.keras.Model",
        data_result: "DataBuildResult",
        target_scaler: Optional[Any],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run the forward pass on each available split and collect
        predictions vs targets (in original PnL units if scaler provided).

        Returns a dict keyed by split name, each containing
        'predictions' and 'targets' arrays.
        """
        from src.rade_ml.evaluation.evaluator import Evaluator

        evaluator = Evaluator(model=model)
        results = {}

        for name, ds in [("train", data_result.train_ds),
                         ("val", data_result.val_ds),
                         ("test", data_result.test_ds)]:
            if ds is None:
                continue

            preds, targets = evaluator._collect_predictions(ds)

            if target_scaler is not None:
                preds, targets = evaluator._inverse_transform(target_scaler, preds, targets)

            results[name] = {"predictions": preds, "targets": targets}
            logger.info(
                f"Collected {name} split predictions: {preds.shape[0]} samples, "
                f"MAE={np.mean(np.abs(preds - targets)):.6f}"
            )

        return results

    def _load_cached_data(self) -> Optional[HybridGnnRnnResult]:
        """
        Attempt to load datasets and artifacts from the registry version directory.

        Returns a fully populated HybridGnnRnnResult if cached datasets exist,
        otherwise None (triggering a fallback to full data build).
        """
        import tensorflow as tf
        import pandas as pd
        import joblib

        if self._loaded_entry is None:
            return None

        version_dir = Path(self._loaded_entry.model_dir)
        ds_dir = version_dir / "datasets"

        if not (ds_dir / "test").exists():
            return None

        test_ds = tf.data.Dataset.load(str(ds_dir / "test"))
        train_ds = (
            tf.data.Dataset.load(str(ds_dir / "train"))
            if (ds_dir / "train").exists() else None
        )
        val_ds = (
            tf.data.Dataset.load(str(ds_dir / "val"))
            if (ds_dir / "val").exists() else None
        )

        metadata = {}
        scaler_path = version_dir / "target_scaler.pkl"
        if scaler_path.exists():
            metadata["target_pnl_transformer"] = joblib.load(scaler_path)

        elem_scaler_path = version_dir / "elementary_scaler.pkl"
        if elem_scaler_path.exists():
            metadata["elementary_pnl_transformer"] = joblib.load(elem_scaler_path)

        target_pnl = None
        target_path = version_dir / "target_pnl.parquet"
        if target_path.exists():
            target_pnl = pd.read_parquet(target_path)

        elementary_pnl = None
        elem_path = version_dir / "elementary_pnl.parquet"
        if elem_path.exists():
            elementary_pnl = pd.read_parquet(elem_path)

        target_attributes = None
        tgt_attr_path = version_dir / "target_attributes.json"
        if tgt_attr_path.exists():
            import json
            with open(tgt_attr_path) as f:
                target_attributes = json.load(f)

        elementary_attributes = None
        elem_attr_path = version_dir / "elementary_attributes.json"
        if elem_attr_path.exists():
            import json
            with open(elem_attr_path) as f:
                elementary_attributes = json.load(f)

        return HybridGnnRnnResult(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            metadata=metadata,
            target_pnl=target_pnl,
            elementary_pnl=elementary_pnl,
            target_attributes=target_attributes,
            elementary_attributes=elementary_attributes,
        )

    def get_target_scaler(self, data_result: "DataBuildResult") -> Optional[Any]:
        """Return target PnL scaler for inverse transform to original units."""
        return data_result.metadata.get("target_pnl_transformer")

    def post_eval(
        self,
        eval_result: "EvaluationResult",
        config: PipelineConfig,
        data_result: Optional["DataBuildResult"] = None,
        split_predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> None:
        if eval_result.metrics:
            logger.info(
                f"Hybrid GNN-RNN evaluation | "
                f"loss={eval_result.loss:.6f} | "
                f"residual_mae={eval_result.metrics.get('residual_mae', 'N/A')} | "
                f"residual_p95={eval_result.metrics.get('residual_p95', 'N/A')}"
            )

        if eval_result.residuals is not None and eval_result.residuals.ndim == 2:
            per_trade_mae = np.mean(np.abs(eval_result.residuals), axis=0)
            worst_idx = int(np.argmax(per_trade_mae))
            logger.info(
                f"Per-target MAE: mean={np.mean(per_trade_mae):.6f}, "
                f"worst_trade_idx={worst_idx} (mae={per_trade_mae[worst_idx]:.6f})"
            )

        if config.artifacts_dir and self._loaded_entry is not None:
            self._save_evaluation_data(eval_result, config, data_result, split_predictions)

    def _save_evaluation_data(
        self,
        eval_result: "EvaluationResult",
        config: PipelineConfig,
        data_result: Optional["DataBuildResult"] = None,
        split_predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> None:
        """
        Persist evaluation analytics to artifacts_dir/evaluation/{version}/.

        Saves per-split predictions (train/val/test) so the UI can render
        model PnL vs actual PnL across the entire timeline, not just test.
        """
        version = self._loaded_entry.version
        eval_dir = Path(config.artifacts_dir) / "evaluation" / version
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_result.to_json(eval_dir / "eval_results.json")
        logger.info(f"Saved eval_results.json to {eval_dir}")

        if eval_result.predictions is not None and eval_result.targets is not None:
            np.savez_compressed(
                eval_dir / "predictions.npz",
                predictions=eval_result.predictions,
                targets=eval_result.targets,
            )

        if eval_result.residuals is not None:
            np.savez_compressed(eval_dir / "residuals.npz", residuals=eval_result.residuals)

        if split_predictions:
            splits_dir = eval_dir / "splits"
            splits_dir.mkdir(exist_ok=True)
            for split_name, arrays in split_predictions.items():
                np.savez_compressed(
                    splits_dir / f"{split_name}.npz",
                    predictions=arrays["predictions"],
                    targets=arrays["targets"],
                )
            logger.info(
                f"Saved per-split predictions for: {list(split_predictions.keys())}"
            )

        if isinstance(data_result, HybridGnnRnnResult):
            if data_result.target_pnl is not None:
                data_result.target_pnl.to_parquet(eval_dir / "target_pnl.parquet")
            if data_result.elementary_pnl is not None:
                data_result.elementary_pnl.to_parquet(eval_dir / "elementary_pnl.parquet")

        logger.info(f"Evaluation analytics saved to {eval_dir}")
