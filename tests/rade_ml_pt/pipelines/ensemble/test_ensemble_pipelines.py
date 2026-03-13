"""Unit tests for ensemble pipelines (eval and infer).

Training pipeline tests are kept lightweight since the full train pipeline
depends on the model-specific TrainPipeline subclass running end-to-end.
The eval and infer pipelines are tested with pre-registered member models.
"""
from __future__ import annotations

import json
import pytest
import numpy as np

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
from src.rade_ml_pt.pipelines.ensemble.eval import EnsembleEvalPipeline
from src.rade_ml_pt.pipelines.ensemble.infer import EnsembleInferencePipeline

from tests.rade_ml_pt.pipelines.ensemble.conftest import N_TARGETS_0, N_TARGETS_1


class TestEnsembleEvalPipeline:
    def _register_ensemble(self, registry_with_members):
        registry, versions, config = registry_with_members
        ens_reg = EnsembleRegistry(config.registry_dir)
        ens_version = ens_reg.register(config, versions, tags=["test"])
        return config, ens_version

    def test_eval_runs_and_returns_metrics(self, registry_with_members):
        config, ens_version = self._register_ensemble(registry_with_members)
        pipeline = EnsembleEvalPipeline(config, ensemble_version="test")
        result = pipeline.run()

        assert "ensemble_metrics" in result
        assert "per_member_metrics" in result
        assert "cluster_0" in result["per_member_metrics"]
        assert "cluster_1" in result["per_member_metrics"]

    def test_eval_saves_artifacts(self, registry_with_members):
        config, ens_version = self._register_ensemble(registry_with_members)
        pipeline = EnsembleEvalPipeline(config, ensemble_version="test")
        result = pipeline.run()

        from pathlib import Path
        eval_dir = Path(config.artifacts_dir) / "ensemble" / result["ensemble_version"] / "evaluation"
        assert (eval_dir / "ensemble_metrics.json").exists()
        assert (eval_dir / "per_member_metrics.json").exists()

    def test_eval_per_member_metrics_have_expected_keys(self, registry_with_members):
        config, ens_version = self._register_ensemble(registry_with_members)
        pipeline = EnsembleEvalPipeline(config, ensemble_version="test")
        result = pipeline.run()

        for cid in ["cluster_0", "cluster_1"]:
            m = result["per_member_metrics"][cid]
            assert "mae" in m
            assert "mse" in m
            assert "rmse" in m


class TestEnsembleInferencePipeline:
    def _setup_inference(self, registry_with_members):
        """Register ensemble and prepare member_inputs for inference."""
        registry, versions, config = registry_with_members
        ens_reg = EnsembleRegistry(config.registry_dir)
        ens_reg.register(config, versions, tags=["infer_test"])

        import torch
        member_inputs = {
            "cluster_0": {"features": torch.randn(4, 4)},
            "cluster_1": {"features": torch.randn(4, 4)},
        }

        config.metadata["inference"] = {
            "input_mode": "new_scenarios",
            "member_inputs": member_inputs,
        }
        return config

    def test_infer_new_scenarios(self, registry_with_members):
        config = self._setup_inference(registry_with_members)
        pipeline = EnsembleInferencePipeline(config, ensemble_version="infer_test")
        result = pipeline.run()

        assert result.predictions is not None
        assert result.predictions.shape[0] == 4
        assert result.n_samples == 4

    def test_infer_saves_csv(self, registry_with_members):
        config = self._setup_inference(registry_with_members)
        pipeline = EnsembleInferencePipeline(config, ensemble_version="infer_test")
        result = pipeline.run()

        from pathlib import Path
        csv_path = Path(config.artifacts_dir) / "inference" / "predictions.csv"
        assert csv_path.exists()

    def test_infer_new_trades_mode(self, registry_with_members):
        registry, versions, config = registry_with_members
        ens_reg = EnsembleRegistry(config.registry_dir)
        ens_reg.register(config, versions, tags=["trades_test"])

        import torch
        config.metadata["inference"] = {
            "input_mode": "new_trades",
            "member_inputs": {
                "cluster_0": {"features": torch.randn(2, 4)},
                "cluster_1": {"features": torch.randn(2, 4)},
            },
            "new_trade_assignments": {
                "cluster_0": ["new_trade_X"],
            },
        }

        pipeline = EnsembleInferencePipeline(config, ensemble_version="trades_test")
        result = pipeline.run()

        assert result.predictions is not None
        assert result.metadata.get("new_trade_assignments") is not None

    def test_infer_unknown_mode_raises(self, registry_with_members):
        config = self._setup_inference(registry_with_members)
        config.metadata["inference"]["input_mode"] = "unknown_mode"

        pipeline = EnsembleInferencePipeline(config, ensemble_version="infer_test")
        with pytest.raises(ValueError, match="Unknown input_mode"):
            pipeline.run()

    def test_infer_missing_member_inputs_raises(self, registry_with_members):
        registry, versions, config = registry_with_members
        ens_reg = EnsembleRegistry(config.registry_dir)
        ens_reg.register(config, versions, tags=["empty_test"])

        config.metadata["inference"] = {
            "input_mode": "new_scenarios",
        }

        pipeline = EnsembleInferencePipeline(config, ensemble_version="empty_test")
        with pytest.raises(ValueError, match="member_inputs"):
            pipeline.run()
