"""Unit tests for HybridGnnRnnTrainPipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from src.rade_ml_pt.core.config import TrainingConfig
from src.rade_ml_pt.core.types import TrainingResult
from src.rade_ml_pt.data.result import DataBuildResult
from src.rade_ml_pt.pipelines.config import PipelineConfig
from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline

from tests.rade_ml_pt.pipelines.hybrid_gnn_rnn.conftest import DummyModel, make_loaders


def _make_data_result():
    train_ds, val_ds = make_loaders()
    return DataBuildResult(train_ds=train_ds, val_ds=val_ds)


# ---------------------------------------------------------------------------
# Tests: instantiation
# ---------------------------------------------------------------------------


class TestTrainPipelineInstantiation:
    def test_instantiates_with_config(self, pipeline_config):
        pipeline = HybridGnnRnnTrainPipeline(pipeline_config)
        assert pipeline.config is pipeline_config

    def test_inherits_train_pipeline(self, pipeline_config):
        from src.rade_ml_pt.pipelines.base import TrainPipeline

        pipeline = HybridGnnRnnTrainPipeline(pipeline_config)
        assert isinstance(pipeline, TrainPipeline)


# ---------------------------------------------------------------------------
# Tests: build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_build_model_returns_nn_module(self, pipeline_config, data_result):
        """build_model should return an nn.Module using default config."""
        pipeline = HybridGnnRnnTrainPipeline(pipeline_config)
        model = pipeline.build_model(pipeline_config, data_result)
        assert isinstance(model, nn.Module)

    def test_build_model_with_explicit_config(self, data_result):
        """build_model should accept model_config dict from PipelineConfig."""
        from src.rade_ml_pt.models.hybrid_gnn_rnn.config import default_model_config

        model_cfg = default_model_config()
        cfg = PipelineConfig(model_config=model_cfg)
        pipeline = HybridGnnRnnTrainPipeline(cfg)
        model = pipeline.build_model(cfg, data_result)
        assert isinstance(model, nn.Module)

    def test_build_model_with_dataclass_config(self, data_result):
        """build_model should accept a HybridGnnRnnModelConfig dataclass."""
        from src.rade_ml_pt.models.hybrid_gnn_rnn.config import HybridGnnRnnModelConfig

        model_cfg = HybridGnnRnnModelConfig()
        cfg = PipelineConfig(model_config=model_cfg)
        pipeline = HybridGnnRnnTrainPipeline(cfg)
        model = pipeline.build_model(cfg, data_result)
        assert isinstance(model, nn.Module)


# ---------------------------------------------------------------------------
# Tests: build_data (mocked -- avoids needing real data files)
# ---------------------------------------------------------------------------


class TestBuildData:
    @patch(
        "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train.build_dataset",
        return_value=_make_data_result(),
    )
    def test_build_data_calls_build_dataset(self, mock_build, pipeline_config):
        """build_data should delegate to build_dataset with resolved config."""
        cfg = PipelineConfig(data_config={"seed": 42})
        pipeline = HybridGnnRnnTrainPipeline(cfg)

        result = pipeline.build_data(cfg)
        mock_build.assert_called_once()
        assert result.train_ds is not None

    @patch(
        "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train.build_dataset",
        return_value=_make_data_result(),
    )
    @patch("src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train.HybridGnnRnnDataConfig")
    def test_build_data_creates_default_config_when_none(self, mock_config_cls, mock_build):
        """build_data should instantiate HybridGnnRnnDataConfig when data_config is None."""
        cfg = PipelineConfig(data_config=None)
        pipeline = HybridGnnRnnTrainPipeline(cfg)
        pipeline.build_data(cfg)

        mock_config_cls.assert_called_once()
        mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: post_train
# ---------------------------------------------------------------------------


class TestPostTrain:
    def test_post_train_without_registry(self, pipeline_config, dummy_model):
        """post_train should work when no registry/tracker is provided."""
        pipeline = HybridGnnRnnTrainPipeline(pipeline_config)
        result = TrainingResult(history={"loss": [0.5]}, final_epoch=1)

        pipeline.post_train(result, dummy_model)
        assert pipeline._registered_entry is None

    def test_post_train_with_registry(self, pipeline_config_with_dirs, dummy_model):
        """post_train should register model when registry is provided."""
        pipeline = HybridGnnRnnTrainPipeline(pipeline_config_with_dirs)
        result = TrainingResult(history={"loss": [0.5]}, final_epoch=1)

        from src.rade_ml_pt.registry.store import ModelRegistry

        registry = ModelRegistry(pipeline_config_with_dirs.registry_dir)
        pipeline.post_train(result, dummy_model, registry=registry)
        assert pipeline._registered_entry is not None


# ---------------------------------------------------------------------------
# Tests: full run (mocked data build, real training loop)
# ---------------------------------------------------------------------------


class TestTrainPipelineRun:
    @patch("src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train.build_dataset")
    def test_full_run_returns_training_result(self, mock_build):
        """run() should orchestrate data->model->train and return TrainingResult."""
        mock_build.return_value = _make_data_result()

        cfg = PipelineConfig(
            training_config=TrainingConfig(
                epochs=2, loss="mse", early_stopping=None, log_dir=None,
            ).to_dict(),
            data_config={"seed": 42},
            metadata={"run_name": "test", "generate_training_report": False},
        )

        pipeline = HybridGnnRnnTrainPipeline(cfg)
        with patch.object(pipeline, "build_model", return_value=DummyModel()):
            result = pipeline.run()

        assert isinstance(result, TrainingResult)
        assert result.final_epoch == 2
        assert "loss" in result.history

    @patch("src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train.build_dataset")
    def test_run_invokes_build_data_and_build_model(self, mock_build):
        """run() should call both build_data and build_model exactly once."""
        mock_build.return_value = _make_data_result()

        cfg = PipelineConfig(
            training_config=TrainingConfig(
                epochs=1, loss="mse", early_stopping=None, log_dir=None,
            ).to_dict(),
            data_config={"seed": 42},
            metadata={"generate_training_report": False},
        )

        pipeline = HybridGnnRnnTrainPipeline(cfg)
        mock_build_model = MagicMock(return_value=DummyModel())
        pipeline.build_model = mock_build_model

        pipeline.run()

        mock_build.assert_called_once()
        mock_build_model.assert_called_once()
