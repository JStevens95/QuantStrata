"""
Warmup-cosine learning rate schedule for PyTorch.

Ports TF ``tf.keras.optimizers.schedules.LearningRateSchedule`` to a
``torch.optim.lr_scheduler.LambdaLR`` subclass so the schedule integrates
natively with any PyTorch optimizer.

Usage::

    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, total_steps=10_000, min_lr=1e-6)
    for step in range(total_steps):
        optimizer.step()
        scheduler.step()
"""
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup followed by cosine annealing to *min_lr*.

    During the first *warmup_steps* the learning rate ramps linearly from 0 to
    the optimizer's base LR.  After warmup it decays following a half-cosine
    from base LR down to *min_lr* over the remaining steps.

    :param optimizer: wrapped PyTorch optimizer.
    :param warmup_steps: number of linear warmup steps.
    :param total_steps: total training steps (warmup + cosine decay).
    :param min_lr: floor learning rate at the end of cosine decay.
    :param last_epoch: index of last epoch (default ``-1``).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        # Snapshot base LRs before LambdaLR.__init__ (which calls step())
        self.base_lrs_copy = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        """Return multiplicative factor for the base LR at *step*."""
        if step < self.warmup_steps:
            # Linear ramp: 0 → 1 over warmup_steps
            return step / max(self.warmup_steps, 1)

        # Cosine decay fraction in [0, 1]
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)

        # LambdaLR multiplies base_lr by the returned value, so express the
        # target lr = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi*progress))
        # as a multiplier of base_lr.
        base_lr = self.base_lrs_copy[0]
        min_ratio = self.min_lr / max(base_lr, 1e-12)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    def get_config(self) -> dict:
        """Serialise schedule hyperparameters."""
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }
