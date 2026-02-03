"""
Generic ML pipeline for QuantStrata.

- training: run_training() for any Trainable model
- evaluation: evaluate_model() with standardised outputs
- inference: load_model(), predict() for deployment
"""

from src.m_learning.pipeline.training import run_training, TrainingLoop
from src.m_learning.pipeline.evaluation import evaluate_model
from src.m_learning.pipeline.inference import load_model, predict, save_model
from src.m_learning.pipeline.tuning import run_tuning

__all__ = [
    "run_training",
    "TrainingLoop",
    "evaluate_model",
    "load_model",
    "predict",
    "save_model",
    "run_tuning",
]
