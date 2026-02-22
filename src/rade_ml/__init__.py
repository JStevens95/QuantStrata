"""
rade ML framework -- machine learning library for quantitative finance.

Top-level re-exports for convenience:
    from rade_ml import BaseModel, TrainingConfig, Trainer
"""
from src.rade_ml.core.base import BaseModel
from src.rade_ml.core.config import TrainingConfig, DataPipelineConfig
from src.rade_ml.registry import ModelRegistry, RegistryEntry
from src.rade_ml.tracking import ExperimentTracker, Run
from src.rade_ml.inference import InferenceRunner
from src.rade_ml.tuning import Tuner, TuningResult
from src.rade_ml.pipelines import PipelineConfig, TrainPipeline, EvalPipeline, InferencePipeline

__all__ = [
    "BaseModel",
    "TrainingConfig",
    "DataPipelineConfig",
    "ModelRegistry",
    "RegistryEntry",
    "ExperimentTracker",
    "Run",
    "InferenceRunner",
    "Tuner",
    "TuningResult",
    "PipelineConfig",
    "TrainPipeline",
    "EvalPipeline",
    "InferencePipeline",
]
