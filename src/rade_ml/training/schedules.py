"""
Custom learning rate schedules for TensorFlow training.

This module contains TF-native learning rate schedule classes that are used by LrScheduleConfig in core/config.py.
Keeping them here avoids forcing a TensorFlow import at config.py import time, which means the pure-Python
configuration dataclasses can be loaded, serialised, and transmitted without TensorFlow installed.

Usage:
    schedule = WarmupCosineSchedule(
        initial_lr=1e-3,
        warmup_steps=500,
        decay_steps=4500,
        min_lr=1e-6,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
"""
from __future__ import annotations

import tensorflow as tf

from typing import Dict, Any


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup followed by cosine decay.

    During the warmup phase (steps 0 … warmup_steps) the learning rate
    increases linearly from 0 to *initial_lr*.  After warmup, cosine
    decay reduces the rate smoothly down to *min_lr*.

    Parameters
    ----------
    initial_lr : float
        Peak learning rate reached after warmup.
    warmup_steps : int
        Number of warmup steps.
    decay_steps : int
        Number of cosine-decay steps after warmup.
    min_lr : float
        Minimum learning rate at the end of decay.
    """

    def __init__(self, initial_lr: float, warmup_steps: int, decay_steps: int, min_lr: float = 1e-6) -> None:
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        warmup_lr = self.initial_lr * (step / tf.maximum(warmup_steps, 1.0))

        progress = (step - warmup_steps) / tf.maximum(decay_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = (
            self.min_lr
            + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + tf.cos(3.14159265 * progress))
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self) -> Dict[str, Any]:
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "min_lr": self.min_lr,
        }
