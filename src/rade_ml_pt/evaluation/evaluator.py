"""
Model evaluation module for the rade_ml_pt (PyTorch) framework.

This module provides an Evaluator class that takes a trained PyTorch model and
a test DataLoader, runs inference, and populates a structured EvaluationResult.

The Evaluator computes:
    - Loss and any user-supplied metrics
    - Predictions, targets, and residuals (for downstream analysis / plotting)
    - Aggregate residual statistics (MAE, max, percentiles)

Usage:
    evaluator = Evaluator(model, loss_fn=nn.MSELoss())
    result = evaluator.run(test_loader)

    print(result.summary())
    result.to_json("eval_results.json")
"""
from __future__ import annotations

import time
import logging

import numpy as np
import torch
import torch.nn as nn

from typing import Any, Callable, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from src.rade_ml_pt.core.types import EvaluationResult

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate a trained PyTorch model on a held-out dataset.

    Produces an EvaluationResult containing loss, metrics, raw predictions,
    targets, and residuals for downstream reporting.

    Parameters
    ----------
    model : nn.Module
        A trained PyTorch model.
    loss_fn : nn.Module or callable, optional
        Loss function used to compute the evaluation loss. If None, loss is
        not computed (metrics dict will not contain 'loss').
    device : torch.device, optional
        Device to run evaluation on. Defaults to the device of the model's
        first parameter.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn

        # resolve device from model parameters if not explicitly specified
        if device is not None:
            self.device = device
        else:
            params = list(model.parameters())
            self.device = params[0].device if params else torch.device("cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        test_data: DataLoader,
        return_predictions: bool = True,
        additional_metrics: Optional[Dict[str, Callable]] = None,
    ) -> EvaluationResult:
        """
        Run full evaluation on test data.

        Parameters
        ----------
        test_data : DataLoader
            Batched test DataLoader yielding (inputs, targets) tuples.
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

        # collect predictions, targets, and per-batch loss in a single pass
        predictions, targets, avg_loss = self._collect_predictions(test_data)

        metrics: Dict[str, float] = {}
        loss_val: Optional[float] = None

        if avg_loss is not None:
            loss_val = avg_loss
            metrics["loss"] = avg_loss

        # residuals
        residuals = predictions - targets

        # user-supplied additional metrics
        if additional_metrics:
            for name, fn in additional_metrics.items():
                try:
                    metrics[name] = float(fn(targets, predictions))
                except Exception as exc:
                    logger.warning(f"Additional metric '{name}' failed: {exc}")

        # aggregate residual statistics
        metrics.update(self._aggregate_stats(residuals))

        eval_time = time.time() - start
        loss_str = f"{loss_val:.6f}" if loss_val is not None else "N/A"
        logger.info(
            f"Evaluation complete in {eval_time:.2f}s | "
            f"loss={loss_str} | samples={len(targets)}"
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
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """
        Collect predictions and targets in a single pass over the DataLoader.

        Uses one iteration to guarantee targets and predictions align. If the
        DataLoader shuffles each epoch, separate iterations would yield
        different orderings and corrupt residual metrics.

        Returns
        -------
        tuple of (predictions_np, targets_np, avg_loss_or_None)
        """
        self.model.eval()
        all_targets: List[np.ndarray] = []
        all_preds: List[np.ndarray] = []
        running_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    x_batch, y_batch = batch[0], batch[1]
                else:
                    raise ValueError(
                        "DataLoader must yield (inputs, targets) tuples for evaluation."
                    )

                # move inputs to the evaluation device
                if isinstance(x_batch, torch.Tensor):
                    x_batch = x_batch.to(self.device)
                elif isinstance(x_batch, dict):
                    x_batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in x_batch.items()
                    }

                if isinstance(y_batch, torch.Tensor):
                    y_batch = y_batch.to(self.device)

                pred = self.model(x_batch)

                # accumulate batch loss if a loss function is provided
                if self.loss_fn is not None:
                    batch_loss = self.loss_fn(pred, y_batch)
                    running_loss += batch_loss.item()
                    n_batches += 1

                # convert to numpy for aggregation
                y_np = y_batch.cpu().numpy() if isinstance(y_batch, torch.Tensor) else np.asarray(y_batch)
                pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
                all_targets.append(y_np)
                all_preds.append(pred_np)

        targets_arr = np.concatenate(all_targets, axis=0)
        preds_arr = np.concatenate(all_preds, axis=0)
        avg_loss = (running_loss / n_batches) if n_batches > 0 else None
        return preds_arr, targets_arr, avg_loss

    @staticmethod
    def _inverse_transform(
        scaler: Any,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply inverse_transform to predictions and targets.

        Expects scaler.inverse_transform(X) with X shape (n_samples, n_features).
        Handles squeezed/expanded shapes for compatibility.
        """
        if not hasattr(scaler, "inverse_transform"):
            raise TypeError("target_scaler must have inverse_transform method")
        preds = np.asarray(predictions, dtype=np.float64)
        tgts = np.asarray(targets, dtype=np.float64)
        original_shape_pred = preds.shape
        original_shape_tgt = tgts.shape
        preds_2d = preds.reshape(-1, preds.shape[-1]) if preds.ndim > 2 else preds
        tgts_2d = tgts.reshape(-1, tgts.shape[-1]) if tgts.ndim > 2 else tgts
        preds_inv = scaler.inverse_transform(preds_2d)
        tgts_inv = scaler.inverse_transform(tgts_2d)
        preds_inv = preds_inv.reshape(original_shape_pred)
        tgts_inv = tgts_inv.reshape(original_shape_tgt)
        return preds_inv.astype(np.float32), tgts_inv.astype(np.float32)

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
