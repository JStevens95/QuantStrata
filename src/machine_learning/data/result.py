"""
Base result type for data builders.

All model-specific data builders (pricing, calibration, gnn_rnn_hybrid) return
a result that inherits from ``DataBuildResult``. The training pipeline can
type-hint ``DataBuildResult`` and accept any concrete subclass.

Usage:
    def train_and_evaluate(result: DataBuildResult) -> None:
        trainer.fit(result.train_ds, result.val_ds)
        if result.holdout_ds is not None:
            evaluator.evaluate(result.holdout_ds)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DataBuildResult:
    """
    Base result from any data builder.

    Guarantees ``train_ds`` and ``val_ds`` for the training pipeline.
    Subclasses add model-specific fields (scalers, holdout set, etc.).

    Attributes
    ----------
    train_ds : tf.data.Dataset
        Training dataset (batched, shuffled).
    val_ds : tf.data.Dataset
        Validation dataset (batched, no shuffle).
    metadata : dict
        Build metadata (splits, config, etc.).

    Properties
    ----------
    holdout_ds : tf.data.Dataset or None
        Test / projection / holdout dataset for evaluation. Returns ``None``
        if the model does not expose one. Override in subclasses.
    """

    train_ds: Any  # tf.data.Dataset
    val_ds: Any    # tf.data.Dataset
    metadata: dict = field(default_factory=dict)

    @property
    def holdout_ds(self) -> Optional[Any]:
        """Test or projection dataset for evaluation. None if not available."""
        return None
