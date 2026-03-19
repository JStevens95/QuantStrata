"""
Custom loss functions for PyTorch training.

Provides tail-aware losses for models predicting heavy-tailed targets
(e.g. exotic trade PnL with barrier knock-in spikes).

Available losses:

- ``QuantileLoss``         : pinball loss at a single quantile tau.
- ``HuberQuantileLoss``    : Huber (smooth baseline) + quantile (tail awareness).
- ``CompositeMAEQuantileLoss`` : MAE (median-optimal baseline) + quantile (tail penalty).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Pinball (quantile) loss at a single quantile level.

    For tau > 0.5 the loss penalises under-prediction more heavily,
    forcing the model to capture upper-tail events (e.g. PnL spikes).

    L(y, y_hat) = max(tau * (y - y_hat), (tau - 1) * (y - y_hat))

    :param tau: quantile level in (0, 1). Default 0.95 targets the 95th percentile.
    """

    def __init__(self, tau: float = 0.95) -> None:
        super().__init__()
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.tau = tau

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        error = targets - predictions
        return torch.max(self.tau * error, (self.tau - 1.0) * error).mean()


class HuberQuantileLoss(nn.Module):
    """Huber loss + quantile tail penalty.

    Combines Huber loss (MSE-like near zero, MAE-like for outliers) with a
    quantile penalty that forces the model to respect upper-tail magnitude.
    Good default pairing with RobustScaler-transformed targets.

    L = HuberLoss(delta) + alpha * QuantileLoss(tau)

    :param delta: Huber transition threshold. Values below delta get squared
        loss (precise gradients); above get linear loss (stable for spikes).
        Default 2.0 works well when targets are RobustScaler'd.
    :param tau: quantile level for the tail penalty. Default 0.95.
    :param alpha: weight of the quantile term relative to Huber. Default 0.3.
    """

    def __init__(
        self,
        delta: float = 2.0,
        tau: float = 0.95,
        alpha: float = 0.3,
    ) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(reduction="mean", delta=delta)
        self.quantile = QuantileLoss(tau=tau)
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.huber(predictions, targets) + self.alpha * self.quantile(predictions, targets)


class CompositeMAEQuantileLoss(nn.Module):
    """MAE + quantile tail penalty.

    The MAE term gives a median-optimal baseline prediction. The quantile
    term at tau > 0.5 adds an asymmetric penalty that forces the model to
    predict closer to the true magnitude during spike scenarios rather than
    averaging them down.

    L = MAE + alpha * QuantileLoss(tau)

    :param tau: quantile level for the tail penalty. Default 0.95.
    :param alpha: weight of the quantile term relative to MAE. Default 0.5.
    """

    def __init__(self, tau: float = 0.95, alpha: float = 0.5) -> None:
        super().__init__()
        self.mae = nn.L1Loss()
        self.quantile = QuantileLoss(tau=tau)
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mae(predictions, targets) + self.alpha * self.quantile(predictions, targets)
