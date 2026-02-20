"""
Model evaluation module for the rade ML framework.

This module provides an Evaluator class that takes a trained Keras model and
a test dataset, runs inference, and populates a structured EvaluationResult.

The Evaluator computes:
    - Keras-compiled metrics (loss, MAE, MSE, etc.)
    - Predictions, targets, and residuals (for downstream analysis / plotting)
    - Aggregate residual statistics (MAE, max, percentiles)

Usage:
    evaluator = Evaluator(model)
    result = evaluator.run(test_ds)

    print(result.summary())
    result.to_json("eval_results.json")
"""
from __future__ import annotations

import time
import logging

import numpy as np
import tensorflow as tf

from typing import Any, Dict, List, Optional, Tuple

from src.rade_ml.core.types import EvaluationResult

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate a trained TensorFlow model on a held-out dataset.

    Produces an EvaluationResult containing compiled metrics, raw predictions,
    targets, and residuals for downstream reporting.

    Parameters
    ----------
    model : tf.keras.Model
        A compiled (and trained) Keras model.
    """

    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        test_data: tf.data.Dataset,
        return_predictions: bool = True,
        additional_metrics: Optional[Dict[str, callable]] = None,
    ) -> EvaluationResult:
        """
        Run full evaluation on test data.

        Parameters
        ----------
        test_data : tf.data.Dataset
            Batched test dataset yielding (inputs, targets) tuples.
        return_predictions : bool
            If True, store raw predictions/targets/residuals in the result.
            Set to False for very large datasets to avoid memory pressure.
        additional_metrics : dict, optional
            Extra metric functions ``{name: fn(y_true, y_pred) -> float}``.

        Returns
        -------
        EvaluationResult
        """
        start = time.time()

        # 1. Keras compiled metrics (loss + whatever was passed to model.compile)
        raw_metrics = self.model.evaluate(test_data, verbose=0, return_dict=True)
        metrics = {k: float(v) for k, v in raw_metrics.items()}
        loss_val = metrics.get("loss")

        # 2. Collect predictions and ground-truth targets
        predictions, targets = self._collect_predictions(test_data)

        # 3. Residuals
        residuals = predictions - targets

        # 4. Additional user-supplied metrics
        if additional_metrics:
            for name, fn in additional_metrics.items():
                try:
                    metrics[name] = float(fn(targets, predictions))
                except Exception as exc:
                    logger.warning(f"Additional metric '{name}' failed: {exc}")

        # 5. Aggregate residual statistics
        metrics.update(self._aggregate_stats(residuals))

        eval_time = time.time() - start
        logger.info(
            f"Evaluation complete in {eval_time:.2f}s | "
            f"loss={loss_val:.6f} | samples={len(targets)}"
        )

        return EvaluationResult(
            metrics=metrics,
            loss=loss_val,
            predictions=predictions if return_predictions else None,
            targets=targets if return_predictions else None,
            residuals=residuals if return_predictions else None,
            dataset_info={
                "samples": int(len(targets)),
                "output_shape": list(predictions.shape),
                "eval_time_seconds": eval_time,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_predictions(
        self,
        dataset: tf.data.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model.predict for predictions; iterate dataset for targets.

        model.predict handles batching internally and is more memory-efficient
        than manual per-batch inference.
        """
        all_targets: List[np.ndarray] = []
        for batch in dataset:
            if isinstance(batch, (tuple, list)):
                y_batch = batch[1]
            elif isinstance(batch, dict):
                raise ValueError("Dict-style datasets must include a 'y' key or use (x, y) tuples.")
            else:
                raise ValueError("Dataset must yield (inputs, targets) tuples for evaluation.")
            all_targets.append(
                y_batch.numpy() if isinstance(y_batch, tf.Tensor) else np.asarray(y_batch)
            )

        targets_arr = np.concatenate(all_targets, axis=0)
        preds_arr = self.model.predict(dataset, verbose=0)

        if isinstance(preds_arr, tf.Tensor):
            preds_arr = preds_arr.numpy()
        preds_arr = np.asarray(preds_arr)

        return preds_arr, targets_arr

    @staticmethod
    def _aggregate_stats(residuals: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics on residuals."""
        abs_res = np.abs(residuals)
        return {
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_mae": float(np.mean(abs_res)),
            "residual_max": float(np.max(abs_res)),
            "residual_p95": float(np.percentile(abs_res, 95)),
            "residual_p99": float(np.percentile(abs_res, 99)),
        }
