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
    """Integration tests against the staged pipeline.

    The previous pre-built ``member_inputs`` short-circuit was
    removed from :class:`EnsembleInferencePipeline`.  Fresh tests
    that drive the staged path
    (``load → load_scenarios → validate_scenarios → run_inference``)
    will be added once the synthetic-staged-path fixture lands.

    For now :meth:`test_infer_unknown_mode_raises` covers the
    pipeline's input_mode validation; the rest of the contract is
    exercised by the eval suite above and by the event-protocol
    unit tests in ``test_infer_events.py``.
    """

    def test_infer_unknown_mode_raises(self, registry_with_members):
        """An unsupported ``input_mode`` raises before any work is done."""
        registry, versions, config = registry_with_members
        ens_reg = EnsembleRegistry(config.registry_dir)
        ens_reg.register(config, versions, tags=["unknown_mode_test"])

        config.metadata["inference"] = {"input_mode": "unknown_mode"}

        pipeline = EnsembleInferencePipeline(
            config, ensemble_version="unknown_mode_test",
        )
        with pytest.raises(ValueError, match="Unknown input_mode"):
            pipeline.run()
