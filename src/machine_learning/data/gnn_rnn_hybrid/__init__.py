"""
Data building and preprocessing for models/gnn_rnn_hybrid.

Output: tf.data.Dataset(s) (train_ds, val_ds, proj_ds) via build_tf_dataset
with static graph inputs injected per batch.
"""

from src.machine_learning.data.gnn_rnn_hybrid.config import HybridGnnRnnDataConfig
from src.machine_learning.data.gnn_rnn_hybrid.build import (
    GnnDataResult,
    build_gnn_data,
)

__all__ = [
    "HybridGnnRnnDataConfig",
    "GnnDataResult",
    "build_gnn_data",
]
