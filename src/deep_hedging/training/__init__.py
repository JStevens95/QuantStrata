"""
Deep Hedging Training Infrastructure

Training utilities for deep hedging agents:
- HedgingTrainer: Main training loop with risk measure optimisation
- Batch simulation utilities
- Gradient computation via finite differences or autodiff
"""

from src.deep_hedging.training.trainer import (
    HedgingTrainer,
    simulate_hedging_batch,
    train_deep_hedging,
)

__all__ = [
    "HedgingTrainer",
    "simulate_hedging_batch",
    "train_deep_hedging",
]
