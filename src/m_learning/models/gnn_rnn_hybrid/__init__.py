"""
GNN-RNN Hybrid Model for Portfolio P&L Prediction.

This module provides a hybrid architecture combining:
- Graph Neural Networks (GNN) for trade structure modelling
- Recurrent Neural Networks (RNN/LSTM) for temporal P&L patterns
- Cross-attention fusion and target-specific projections

Usage:
    from src.m_learning.models.gnn_rnn_hybrid import HybridGnnRnn, default_hybrid_model_config
    from src.m_learning.data.gnn_rnn_hybrid import build_gnn_data

    data = build_gnn_data(use_synthetic=True, n_samples=500)
    model_config = default_hybrid_model_config(n_targets=10)
    model = HybridGnnRnn(model_config, name="hybrid_pnl")
    model.fit(data.train_ds, validation_data=data.val_ds, epochs=10)
"""
from src.m_learning.models.gnn_rnn_hybrid.hybrid_model import HybridGnnRnn
from src.m_learning.models.gnn_rnn_hybrid.config import default_hybrid_model_config
from src.m_learning.models.gnn_rnn_hybrid.layers import (
    GnnBlock,
    GraphSage,
    MixedGraphSage,
    RnnBlock,
    FusionLayer,
    TargetAttentionLayer,
    TargetPnlOutput,
)

__all__ = [
    "HybridGnnRnn",
    "default_hybrid_model_config",
    "GnnBlock",
    "GraphSage",
    "MixedGraphSage",
    "RnnBlock",
    "FusionLayer",
    "TargetAttentionLayer",
    "TargetPnlOutput",
]
