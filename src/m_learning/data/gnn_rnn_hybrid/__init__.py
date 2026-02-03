"""
Data building and preprocessing for models/gnn_rnn_hybrid.

Output: tf.data.Dataset(s) (train_ds, val_ds, proj_ds) so the generic
pipeline has a single, repeatable interface.
"""

from src.m_learning.data.gnn_rnn_hybrid.build import (
    GnnDataResult,
    build_gnn_data,
)

__all__ = [
    "GnnDataResult",
    "build_gnn_data",
]
