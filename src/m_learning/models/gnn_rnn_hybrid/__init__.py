"""
GNN-RNN Hybrid Model for Portfolio P&L Prediction.

This module provides a hybrid architecture combining:
- Graph Neural Networks (GNN) for trade structure modelling
- Recurrent Neural Networks (RNN/LSTM) for temporal P&L patterns
- Cross-attention fusion and target-specific projections

Usage:
    from src.m_learning.models.gnn_rnn_hybrid import (
        HybridGnnRnn,
        BatchedHybridGnnRnn,
        create_gnn_tf_dataset,
    )
    from src.m_learning.data.gnn_synthetic import default_hybrid_model_config

    model_config = default_hybrid_model_config(n_targets=10)
    model = BatchedHybridGnnRnn(model_config, name="hybrid_pnl")
"""
from src.m_learning.models.gnn_rnn_hybrid.hybrid_model import HybridGnnRnn
from src.m_learning.models.gnn_rnn_hybrid.config import default_hybrid_model_config
from src.m_learning.models.gnn_rnn_hybrid.wrapper import (
    BatchedHybridGnnRnn,
    create_gnn_tf_dataset,
    train_val_split_gnn,
)
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
    "BatchedHybridGnnRnn",
    "default_hybrid_model_config",
    "create_gnn_tf_dataset",
    "train_val_split_gnn",
    "GnnBlock",
    "GraphSage",
    "MixedGraphSage",
    "RnnBlock",
    "FusionLayer",
    "TargetAttentionLayer",
    "TargetPnlOutput",
]
