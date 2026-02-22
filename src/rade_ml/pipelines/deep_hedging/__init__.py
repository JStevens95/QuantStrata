"""
Pipeline wiring for the Deep Hedging model.
"""
from src.rade_ml.pipelines.deep_hedging.train import DeepHedgingTrainPipeline
from src.rade_ml.pipelines.deep_hedging.eval import DeepHedgingEvalPipeline
from src.rade_ml.pipelines.deep_hedging.infer import DeepHedgingInferencePipeline

__all__ = [
    "DeepHedgingTrainPipeline",
    "DeepHedgingEvalPipeline",
    "DeepHedgingInferencePipeline",
]
