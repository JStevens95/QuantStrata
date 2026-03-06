"""Unit tests for rade_ml_pt.pipelines.base -- abstract pipeline classes."""
import pytest

from src.rade_ml_pt.pipelines.base import TrainPipeline, EvalPipeline, InferencePipeline
from src.rade_ml_pt.pipelines.config import PipelineConfig


class TestTrainPipelineAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            TrainPipeline(PipelineConfig())

    def test_subclass_must_implement_build_data(self):
        class IncompleteTrainPipeline(TrainPipeline):
            def build_model(self, config, data_result):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteTrainPipeline(PipelineConfig())

    def test_subclass_must_implement_build_model(self):
        class IncompleteTrainPipeline(TrainPipeline):
            def build_data(self, config):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteTrainPipeline(PipelineConfig())

    def test_concrete_subclass_instantiates(self):
        class ConcreteTrainPipeline(TrainPipeline):
            def build_data(self, config):
                return None
            def build_model(self, config, data_result):
                return None

        pipeline = ConcreteTrainPipeline(PipelineConfig())
        assert pipeline.config is not None


class TestEvalPipelineAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            EvalPipeline(PipelineConfig())

    def test_concrete_subclass_instantiates(self):
        class ConcreteEvalPipeline(EvalPipeline):
            def build_data(self, config):
                return None

        pipeline = ConcreteEvalPipeline(PipelineConfig())
        assert pipeline.config is not None


class TestInferencePipelineAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            InferencePipeline(PipelineConfig())

    def test_concrete_subclass_instantiates(self):
        class ConcreteInferencePipeline(InferencePipeline):
            def prepare_inputs(self, config):
                return {"inputs": None}

        pipeline = ConcreteInferencePipeline(PipelineConfig())
        assert pipeline.config is not None
