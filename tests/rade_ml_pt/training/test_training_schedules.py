import pytest
import torch

from src.rade_ml_pt.training.schedules import WarmupCosineSchedule


class TestWarmupCosineSchedule:
    def _make_optimizer(self, lr=1e-3):
        model = torch.nn.Linear(5, 1)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def test_warmup_phase_increases(self):
        optimizer = self._make_optimizer(lr=1e-3)
        sched = WarmupCosineSchedule(
            optimizer, warmup_steps=100, total_steps=1000, min_lr=1e-6
        )

        lrs = []
        for step in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            sched.step()

        assert lrs[50] > lrs[0]

    def test_decay_phase_decreases(self):
        optimizer = self._make_optimizer(lr=1e-3)
        sched = WarmupCosineSchedule(
            optimizer, warmup_steps=10, total_steps=100, min_lr=1e-6
        )

        for _ in range(10):
            sched.step()
        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        for _ in range(45):
            sched.step()
        lr_mid = optimizer.param_groups[0]["lr"]

        for _ in range(45):
            sched.step()
        lr_end = optimizer.param_groups[0]["lr"]

        assert lr_at_warmup_end > lr_mid
        assert lr_mid > lr_end

    def test_min_lr_floor(self):
        optimizer = self._make_optimizer(lr=1e-3)
        sched = WarmupCosineSchedule(
            optimizer, warmup_steps=10, total_steps=100, min_lr=1e-5
        )

        for _ in range(10000):
            sched.step()

        lr_final = optimizer.param_groups[0]["lr"]
        assert lr_final >= 1e-5 - 1e-8

    def test_get_config_roundtrip(self):
        optimizer = self._make_optimizer()
        sched = WarmupCosineSchedule(
            optimizer, warmup_steps=50, total_steps=500, min_lr=1e-6
        )

        cfg = sched.get_config()
        assert cfg["warmup_steps"] == 50
        assert cfg["total_steps"] == 500
