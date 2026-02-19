"""
Machine Learning Models.

This module provides the HybridGnnRnn model for portfolio P&L prediction
from elementary trade embeddings (structural GNN + temporal RNN).

Usage:
    from src.machine_learning.models import HybridGnnRnn, default_hybrid_model_config

    model_config = default_hybrid_model_config(gnn_units=32, rnn_units=32)
    model = HybridGnnRnn(model_config=model_config)
"""
from src.machine_learning.models.gnn_rnn_hybrid import (
    HybridGnnRnn,
    default_hybrid_model_config,
)

__all__ = [
    "HybridGnnRnn",
    "default_hybrid_model_config",
]
