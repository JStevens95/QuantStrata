"""Unit tests for rade_ml.training.schedules -- WarmupCosineSchedule."""
import pytest
import tensorflow as tf

from src.rade_ml.training.schedules import WarmupCosineSchedule


class TestWarmupCosineSchedule:
    def test_warmup_phase_increases(self):
        sched = WarmupCosineSchedule(initial_lr=1e-3, warmup_steps=100, decay_steps=900, min_lr=1e-6)
        lr_0 = float(sched(tf.constant(0)))
        lr_50 = float(sched(tf.constant(50)))
        lr_99 = float(sched(tf.constant(99)))
        assert lr_50 > lr_0
        assert lr_99 > lr_50

    def test_decay_phase_decreases(self):
        sched = WarmupCosineSchedule(initial_lr=1e-3, warmup_steps=10, decay_steps=90, min_lr=1e-6)
        lr_start = float(sched(tf.constant(10)))
        lr_mid = float(sched(tf.constant(50)))
        lr_end = float(sched(tf.constant(100)))
        assert lr_start > lr_mid
        assert lr_mid > lr_end

    def test_min_lr_floor(self):
        sched = WarmupCosineSchedule(initial_lr=1e-3, warmup_steps=10, decay_steps=90, min_lr=1e-5)
        lr_final = float(sched(tf.constant(10000)))
        assert lr_final >= 1e-5

    def test_get_config_roundtrip(self):
        sched = WarmupCosineSchedule(initial_lr=2e-3, warmup_steps=50, decay_steps=450, min_lr=1e-6)
        cfg = sched.get_config()
        assert cfg["initial_lr"] == 2e-3
        assert cfg["warmup_steps"] == 50
