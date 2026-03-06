"""
PyTorch training loop (replaces Keras ``model.fit()``).

The :class:`Trainer` owns the optimiser, loss function, LR scheduler and
callback lifecycle.  It iterates over PyTorch ``DataLoader`` (or any iterable
of ``(inputs, targets)`` batches), handles gradient clipping, optional mixed
precision, and constructs a :class:`~src.rade_ml_pt.core.types.TrainingResult`
at the end of training.

Usage::

    trainer = Trainer(model, config)
    trainer.compile(loss="mse")
    result = trainer.fit(train_loader, val_loader)
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.rade_ml_pt.core.types import TrainingResult
from src.rade_ml_pt.training.callbacks import Callback, get_standard_callbacks

if TYPE_CHECKING:
    from src.rade_ml_pt.core.config import TrainingConfig

logger = logging.getLogger(__name__)

# Mapping of short loss names to PyTorch loss classes
_LOSS_MAP: Dict[str, type] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
    "smooth_l1": nn.SmoothL1Loss,
}


def setup_training_environment(config: Optional["TrainingConfig"] = None, seed: int = 42) -> None:
    """Set global seeds and deterministic flags for reproducibility.

    :param config: optional training configuration (reserved for future use).
    :param seed: random seed applied to PyTorch and NumPy.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


class Trainer:
    """Encapsulates the compile → fit → evaluate lifecycle for a PyTorch model.

    :param model: the ``nn.Module`` to train.
    :param config: :class:`TrainingConfig` driving epochs, optimizer, callbacks, etc.
    :param seed: reproducibility seed passed to :func:`setup_training_environment`.
    :param custom_callbacks: extra callbacks appended after standard ones.
    """

    def __init__(
        self,
        model: nn.Module,
        config: "TrainingConfig",
        seed: int = 42,
        custom_callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.seed = seed
        self.custom_callbacks = custom_callbacks or []

        self._is_compiled = False
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.scheduler: Optional[Any] = None
        self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # compile
    # ------------------------------------------------------------------

    def compile(
        self,
        loss: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[List[str]] = None,
    ) -> "Trainer":
        """Build the loss function, optimizer and (optional) LR scheduler.

        :param loss: loss name (key in ``_LOSS_MAP``); defaults to ``config.loss``.
        :param optimizer: pre-built optimizer; otherwise built from config.
        :param metrics: reserved for future per-batch metric tracking.
        :returns: ``self`` for fluent chaining.
        """
        loss_name = loss or self.config.loss
        loss_cls = _LOSS_MAP.get(loss_name.lower())
        if loss_cls is None:
            raise ValueError(f"Unknown loss: {loss_name!r}. Choose from {list(_LOSS_MAP)}")
        self.loss_fn = loss_cls()

        if optimizer is None:
            self.optimizer = self.config.optimizer.build(self.model.parameters())
        else:
            self.optimizer = optimizer

        if self.config.lr_schedule is not None:
            # Per-epoch scheduling: total_steps == epochs
            total_steps = self.config.epochs
            self.scheduler = self.config.lr_schedule.build(self.optimizer, total_steps=total_steps)

        self._is_compiled = True
        return self

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: Any,
        val_data: Any = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Run the full training loop.

        :param train_data: iterable yielding ``(inputs, targets)`` batches.
        :param val_data: optional validation iterable with same format.
        :returns: :class:`TrainingResult` summarising the run.
        """
        if not self._is_compiled:
            self.compile()

        callbacks = get_standard_callbacks(self.config) + list(self.custom_callbacks)

        # Inject model & optimizer references into callbacks
        for cb in callbacks:
            cb._model = self.model
            if hasattr(cb, "set_optimizer") and self.optimizer is not None:
                cb.set_optimizer(self.optimizer)

        history: Dict[str, List[float]] = {}
        start_time = time.time()

        for cb in callbacks:
            cb.on_train_begin()

        for epoch in range(self.config.epochs):
            for cb in callbacks:
                cb.on_epoch_begin(epoch)

            # ---- training pass ----
            self.model.train()
            train_losses: List[float] = []

            for batch in train_data:
                inputs, targets = self._unpack_batch(batch)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, targets) if targets is not None else outputs.mean()
                loss.backward()

                self._clip_gradients()
                self.optimizer.step()
                train_losses.append(loss.item())

            epoch_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            logs: Dict[str, float] = {"loss": epoch_train_loss}

            # ---- validation pass ----
            if val_data is not None:
                val_loss = self._run_validation(val_data)
                if val_loss is not None:
                    logs["val_loss"] = val_loss

            # Step LR scheduler (per-epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate history
            for key, value in logs.items():
                history.setdefault(key, []).append(value)

            for cb in callbacks:
                cb.on_epoch_end(epoch, logs)

            # Early stopping check
            if any(getattr(cb, "stop_training", False) for cb in callbacks):
                break

        for cb in callbacks:
            cb.on_train_end()

        total_time = time.time() - start_time
        return self._build_result(history, total_time)

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """Run a single evaluation pass and return aggregated loss.

        :param test_data: iterable yielding ``(inputs, targets)`` batches.
        :returns: dict with ``"loss"`` key.
        """
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch in test_data:
                inputs, targets = self._unpack_batch(batch)
                outputs = self.model(inputs)
                if targets is not None and self.loss_fn is not None:
                    losses.append(self.loss_fn(outputs, targets).item())
        return {"loss": float(np.mean(losses)) if losses else 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch: Any):
        """Extract ``(inputs, targets)`` from a batch that may be a tuple/list."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        return batch, None


    def _clip_gradients(self) -> None:
        """Apply gradient clipping based on optimizer config."""
        if self.config.optimizer.clipnorm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimizer.clipnorm)
        if self.config.optimizer.clipvalue:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.optimizer.clipvalue)

    def _run_validation(self, val_data: Any) -> Optional[float]:
        """Compute mean validation loss over *val_data*."""
        self.model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_data:
                inputs, targets = self._unpack_batch(batch)
                outputs = self.model(inputs)
                if targets is not None:
                    val_losses.append(self.loss_fn(outputs, targets).item())
        return float(np.mean(val_losses)) if val_losses else None

    def _build_result(self, history: Dict[str, List[float]], total_time: float) -> TrainingResult:
        """Construct the canonical :class:`TrainingResult` from accumulated data."""
        val_losses = history.get("val_loss", history.get("loss", []))
        best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else 1
        best_val_loss = float(min(val_losses)) if val_losses else 0.0
        best_train_loss = (
            float(history["loss"][best_epoch - 1]) if "loss" in history and best_epoch <= len(history["loss"]) else 0.0
        )
        final_epoch = len(history.get("loss", []))
        stopped_early = final_epoch < self.config.epochs

        return TrainingResult(
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_train_loss=best_train_loss,
            final_epoch=final_epoch,
            training_time_seconds=total_time,
            stopped_early=stopped_early,
            config=self.config.to_dict(),
            model_summary=self._get_model_summary(),
        )

    def _get_model_summary(self) -> Dict[str, Any]:
        """Return a compact dict describing the model architecture."""
        try:
            return {
                "name": getattr(self.model, "_model_name", self.model.__class__.__name__),
                "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "non_trainable_params": sum(p.numel() for p in self.model.parameters() if not p.requires_grad),
                "modules": len(list(self.model.modules())),
            }
        except Exception:
            return {}
