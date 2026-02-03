"""
Data preparation and dataset utilities.

Structure aligns with NOTEBOOK_PLAN: data/<model>/ builders output tf.data.Dataset(s).
- data/pricing/ → build_pricing_data() → train_ds, val_ds, test_ds
- data/gnn_rnn_hybrid/ → build_gnn_data() → train_ds, val_ds, proj_ds

Usage:
    from src.m_learning.data import build_pricing_data, build_gnn_data
    data = build_pricing_data(n_samples=2000, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_ds, val_ds, test_ds = data.train_ds, data.val_ds, data.test_ds
"""
from src.m_learning.data.dataset import (
    TFDataset,
    NormalizationStats,
    create_pricing_dataset,
    create_calibration_dataset,
)
from src.m_learning.data.types import (
    MLDataset,
    PricingFeatures,
    CalibrationFeatures,
)
from src.m_learning.data.common import TradeAttributeEncoder, TradeGraphBuilder
from src.m_learning.data.pricing import PricingDataResult, build_pricing_data
from src.m_learning.data.gnn_rnn_hybrid import GnnDataResult, build_gnn_data

__all__ = [
    "TFDataset",
    "NormalizationStats",
    "create_pricing_dataset",
    "create_calibration_dataset",
    "MLDataset",
    "PricingFeatures",
    "CalibrationFeatures",
    "TradeAttributeEncoder",
    "TradeGraphBuilder",
    "PricingDataResult",
    "build_pricing_data",
    "GnnDataResult",
    "build_gnn_data",
]
