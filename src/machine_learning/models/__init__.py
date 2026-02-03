"""
Machine Learning Models.

This module provides TensorFlow-based ML models for:
    - Option pricing
    - Model calibration
    - Portfolio risk prediction

Available Models:
    - MLPPricer: Multi-layer perceptron for option pricing
    - AttentionPricer: Attention-based pricing model (planned)
    - CalibrationNet: Neural network for model calibration
    - HybridGnnRnn: Graph + RNN model for portfolio P&L prediction

Usage:
    from src.machine_learning.models import MLPPricer
    
    model = MLPPricer(hidden_units=[128, 64, 32])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_ds, validation_data=val_ds, epochs=100)
"""
from src.machine_learning.models.pricing.model import (
    MLPPricer,
    create_mlp_pricer,
)

__all__ = [
    "MLPPricer",
    "create_mlp_pricer",
]
