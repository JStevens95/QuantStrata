"""Unit tests for HybridGnnRnnEvalPipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.rade_ml_pt.core.types import EvaluationResult
from src.rade_ml_pt.data.result import DataBuildResult
from src.rade_ml_pt.pipelines.config import PipelineConfig
from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.eval import HybridGnnRnnEvalPipeline

from tests.rade_ml_pt.pipelines.hybrid_gnn_rnn.conftest import (
    DummyModel,
    make_loaders,
    BATCH,
    SEQ,
    ELEM,
    TARG,
)


def _make_data_result_with_test():
    """DataBuildResult with all three splits."""
    train_ds, val_ds = make_loaders()
    np.random.seed(1)
    X_test = np.random.randn(BATCH, SEQ, ELEM).astype(np.float32)
    y_test = np.random.randn(BATCH, TARG).astype(np.float32)
    test_ds = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH,
    )
    return DataBuildResult(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds)


def _mock_registry_entry(tmp_path=None):
    """Create a mock registry entry with version and model_dir."""
    entry = MagicMock()
    entry.version = "v1"
    entry.model_dir = str(tmp_path / "registry" / "v1") if tmp_path else "/tmp/registry/v1"
    return entry


# ---------------------------------------------------------------------------
# Tests: instantiation
# ---------------------------------------------------------------------------


class TestEvalPipelineInstantiation:
    def test_instantiates_with_config(self, pipeline_config):
        pipeline = HybridGnnRnnEvalPipeline(pipeline_config)
        assert pipeline.config is pipeline_config

    def test_inherits_eval_pipeline(self, pipeline_config):
        from src.rade_ml_pt.pipelines.base import EvalPipeline

        pipeline = HybridGnnRnnEvalPipeline(pipeline_config)
        assert isinstance(pipeline, EvalPipeline)


# ---------------------------------------------------------------------------
# Tests: build_data
# ---------------------------------------------------------------------------


class TestBuildData:
    @patch(
        "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.eval.build_dataset",
        return_value=_make_data_result_with_test(),
    )
    def test_build_data_falls_back_to_full_build(self, mock_build):
        """When no cached data exists, build_data runs the full data pipeline."""
        cfg = PipelineConfig(data_config={"seed": 42})
        pipeline = HybridGnnRnnEvalPipeline(cfg)

        result = pipeline.build_data(cfg)
        mock_build.assert_called_once()
        assert result.test_ds is not None

    @patch(
        "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.eval.build_dataset",
        return_value=_make_data_result_with_test(),
    )
    def test_build_data_resolves_dict_config(self, mock_build):
        """build_data should convert dict data_config to HybridGnnRnnDataConfig."""
        cfg = PipelineConfig(data_config={"seed": 99, "batch_size": 16})
        pipeline = HybridGnnRnnEvalPipeline(cfg)

        pipeline.build_data(cfg)
        call_kwargs = mock_build.call_args
        from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig

        config_arg = call_kwargs.kwargs.get("config", call_kwargs[1].get("config"))
        assert isinstance(config_arg, HybridGnnRnnDataConfig)


# ---------------------------------------------------------------------------
# Tests: get_target_scaler
# ---------------------------------------------------------------------------


class TestGetTargetScaler:
    def test_returns_none_when_no_scaler(self, pipeline_config):
        """get_target_scaler returns None when metadata has no scaler."""
        pipeline = HybridGnnRnnEvalPipeline(pipeline_config)
        data_result = DataBuildResult(metadata={})
        assert pipeline.get_target_scaler(data_result) is None

    def test_returns_scaler_when_present(self, pipeline_config):
        """get_target_scaler returns the transformer from metadata."""
        pipeline = HybridGnnRnnEvalPipeline(pipeline_config)
        mock_scaler = MagicMock()
        data_result = DataBuildResult(
            metadata={"target_pnl_transformer": mock_scaler}
        )
        assert pipeline.get_target_scaler(data_result) is mock_scaler


# ---------------------------------------------------------------------------
# Tests: _collect_all_split_predictions
# ---------------------------------------------------------------------------


class TestCollectSplitPredictions:
    def test_collects_all_available_splits(self, pipeline_config):
        """Should collect predictions for every non-None split."""
        pipeline = HybridGnnRnnEvalPipeline(pipeline_config)
        model = DummyModel()
        model.eval()
        data_result = _make_data_result_with_test()

        results = pipeline._collect_all_split_predictions(model, data_result, None)

        assert "train" in results
        assert "val" in results
        assert "test" in results
        for split in results.values():
            assert "predictions" in split
            assert "targets" in split
            assert isinstance(split["predictions"], np.ndarray)

    def test_skips_none_splits(self, pipeline_config):
        """Should skip splits where the DataLoader is None."""
        pipeline = HybridGnnRnnEvalPipeline(pipeline_config)
        model = DummyModel()
        model.eval()
        data_result = DataBuildResult(
            train_ds=None,
            val_ds=None,
            test_ds=_make_data_result_with_test().test_ds,
        )

        results = pipeline._collect_all_split_predictions(model, data_result, None)

        assert "test" in results
        assert "train" not in results
        assert "val" not in results

    def test_prediction_shapes_match(self, pipeline_config):
        """Predictions and targets should have matching first dimension."""
        pipeline = HybridGnnRnnEvalPipeline(pipeline_config)
        model = DummyModel()
        model.eval()
        data_result = _make_data_result_with_test()

        results = pipeline._collect_all_split_predictions(model, data_result, None)

        for split_name, arrays in results.items():
            assert arrays["predictions"].shape[0] == arrays["targets"].shape[0], (
                f"{split_name}: prediction/target sample count mismatch"
            )


# ---------------------------------------------------------------------------
# Tests: post_eval
# ---------------------------------------------------------------------------


class TestPostEval:
    def test_post_eval_logs_without_artifacts_dir(self, pipeline_config):
        """post_eval should not fail when artifacts_dir is None."""
        pipeline = HybridGnnRnnEvalPipeline(pipeline_config)

        eval_result = EvaluationResult(
            metrics={"loss": 0.5, "residual_mae": 0.3, "residual_p95": 0.8},
            loss=0.5,
        )

        pipeline.post_eval(eval_result, pipeline_config)

    def test_post_eval_saves_artifacts(self, tmp_path):
        """post_eval should save eval_results.json when artifacts_dir is set."""
        cfg = PipelineConfig(artifacts_dir=str(tmp_path / "artifacts"))
        pipeline = HybridGnnRnnEvalPipeline(cfg)
        pipeline._loaded_entry = _mock_registry_entry(tmp_path)

        preds = np.random.randn(BATCH, TARG).astype(np.float32)
        targets = np.random.randn(BATCH, TARG).astype(np.float32)

        eval_result = EvaluationResult(
            metrics={"loss": 0.5, "residual_mae": 0.3},
            loss=0.5,
            predictions=preds,
            targets=targets,
            residuals=preds - targets,
        )

        pipeline.post_eval(eval_result, cfg, data_result=DataBuildResult())
        eval_dir = tmp_path / "artifacts" / "evaluation" / "v1"
        assert (eval_dir / "eval_results.json").exists()

    def test_post_eval_saves_split_predictions(self, tmp_path):
        """post_eval should save per-split .npz files when provided."""
        cfg = PipelineConfig(artifacts_dir=str(tmp_path / "artifacts"))
        pipeline = HybridGnnRnnEvalPipeline(cfg)
        pipeline._loaded_entry = _mock_registry_entry(tmp_path)

        eval_result = EvaluationResult(metrics={"loss": 0.5}, loss=0.5)
        split_predictions = {
            "train": {
                "predictions": np.zeros((BATCH, TARG)),
                "targets": np.ones((BATCH, TARG)),
            },
            "test": {
                "predictions": np.zeros((BATCH, TARG)),
                "targets": np.ones((BATCH, TARG)),
            },
        }

        pipeline.post_eval(
            eval_result, cfg,
            data_result=DataBuildResult(),
            split_predictions=split_predictions,
        )

        splits_dir = tmp_path / "artifacts" / "evaluation" / "v1" / "splits"
        assert (splits_dir / "train.npz").exists()
        assert (splits_dir / "test.npz").exists()


# ---------------------------------------------------------------------------
# Tests: full run (mocked model load + data build)
# ---------------------------------------------------------------------------


class TestEvalPipelineRun:
    @patch("src.rade_ml_pt.pipelines.hybrid_gnn_rnn.eval.build_dataset")
    def test_full_run_returns_evaluation_result(self, mock_build):
        """run() should return an EvaluationResult with metrics."""
        mock_build.return_value = _make_data_result_with_test()

        cfg = PipelineConfig(
            data_config={"seed": 42},
            metadata={"run_name": "eval_test"},
        )
        pipeline = HybridGnnRnnEvalPipeline(cfg)

        mock_entry = MagicMock()
        mock_entry.version = "v1"
        mock_entry.model_dir = "/tmp/fake"

        model = DummyModel()
        with patch.object(pipeline, "load_model", return_value=(model, mock_entry)):
            result = pipeline.run()

        assert isinstance(result, EvaluationResult)
        assert result.metrics is not None
        assert "residual_mae" in result.metrics

    @patch("src.rade_ml_pt.pipelines.hybrid_gnn_rnn.eval.build_dataset")
    def test_run_calls_build_data(self, mock_build):
        """run() should call build_data to produce test data."""
        mock_build.return_value = _make_data_result_with_test()

        cfg = PipelineConfig(data_config={"seed": 42})
        pipeline = HybridGnnRnnEvalPipeline(cfg)

        mock_entry = MagicMock()
        mock_entry.version = "v1"
        mock_entry.model_dir = "/tmp/fake"

        with patch.object(pipeline, "load_model", return_value=(DummyModel(), mock_entry)):
            pipeline.run()

        mock_build.assert_called_once()

    @patch("src.rade_ml_pt.pipelines.hybrid_gnn_rnn.eval.build_dataset")
    def test_run_collects_split_predictions(self, mock_build):
        """run() should collect predictions on all splits."""
        mock_build.return_value = _make_data_result_with_test()

        cfg = PipelineConfig(data_config={"seed": 42})
        pipeline = HybridGnnRnnEvalPipeline(cfg)

        mock_entry = MagicMock()
        mock_entry.version = "v1"
        mock_entry.model_dir = "/tmp/fake"

        with patch.object(pipeline, "load_model", return_value=(DummyModel(), mock_entry)):
            with patch.object(pipeline, "post_eval") as mock_post_eval:
                pipeline.run()

                mock_post_eval.assert_called_once()
                call_kwargs = mock_post_eval.call_args
                split_preds = call_kwargs.kwargs.get(
                    "split_predictions",
                    call_kwargs[1].get("split_predictions") if len(call_kwargs) > 1 else None,
                )
                assert split_preds is not None
                assert "test" in split_preds
