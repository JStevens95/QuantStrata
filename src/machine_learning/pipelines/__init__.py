"""
Generic ML pipeline for QuantStrata.

- training: run_training() for any Trainable model
- evaluation: evaluate_model() with standardised outputs
- inference: load_model(), predict() for deployment
"""

from src.machine_learning.pipelines.training import run_training, TrainingLoop
from src.machine_learning.pipelines.evaluation import evaluate_model
from src.machine_learning.pipelines.inference import load_model, predict, save_model
from src.machine_learning.pipelines.tuning import run_tuning

__all__ = [
    "run_training",
    "TrainingLoop",
    "evaluate_model",
    "load_model",
    "predict",
    "save_model",
    "run_tuning",
]
