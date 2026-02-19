"""
Data preparation and dataset utilities for GNN-RNN hybrid model.

Structure:
    - data/gnn_rnn_hybrid/ → build_gnn_data() → train_ds, val_ds, proj_ds

Normalisation uses sklearn.preprocessing.StandardScaler (fit on train only).
Splitting uses sklearn.model_selection.train_test_split.
Datasets are built via build_tf_dataset() wrapping tf.data.Dataset.from_tensor_slices.

Usage:
    from src.machine_learning.data import build_gnn_data
    data = build_gnn_data(config)
    train_ds, val_ds, proj_ds = data.train_ds, data.val_ds, data.proj_ds
"""
from src.machine_learning.data.dataset import build_tf_dataset, SyntheticData
from src.machine_learning.utilities.trade_attribute_encoder import TradeAttributeEncoder
from src.machine_learning.utilities.trade_graph_builder import TradeGraphBuilder
from src.machine_learning.data.gnn_rnn_hybrid import GnnDataResult, build_gnn_data
from src.machine_learning.data.manifest import DatasetManifest

__all__ = [
    "build_tf_dataset",
    "SyntheticData",
    "TradeAttributeEncoder",
    "TradeGraphBuilder",
    "GnnDataResult",
    "build_gnn_data",
    "DatasetManifest",
]
