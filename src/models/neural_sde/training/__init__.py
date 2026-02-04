"""
Training pipelines for Neural SDEs.

Provides:
- Score matching training
- Maximum likelihood calibration
- Moment matching
- Historical data fitting

Example:
    from src.models.neural_sde.training import ScoreMatchingTrainer
    
    trainer = ScoreMatchingTrainer(learning_rate=1e-3)
    trainer.fit(sde, historical_data)
"""

from src.models.neural_sde.training.trainer import (
    NeuralSDETrainer,
    TrainingConfig,
    TrainingResult,
)
from src.models.neural_sde.training.losses import (
    ScoreMatchingLoss,
    MomentMatchingLoss,
    PathwiseLoss,
)

__all__ = [
    "NeuralSDETrainer",
    "TrainingConfig",
    "TrainingResult",
    "ScoreMatchingLoss",
    "MomentMatchingLoss",
    "PathwiseLoss",
]
