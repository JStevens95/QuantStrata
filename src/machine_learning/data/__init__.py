"""
Data preparation and dataset utilities.

Structure: data/<model>/ builders output tf.data.Dataset(s).
    - data/pricing/      → build_pricing_data()  → train_ds, val_ds, test_ds
    - data/calibration/  → build_calibration_data() → train_ds, val_ds, test_ds
    - data/gnn_rnn_hybrid/ → build_gnn_data()    → train_ds, val_ds, proj_ds

Normalisation uses sklearn.preprocessing.StandardScaler (fit on train only).
Splitting uses sklearn.model_selection.train_test_split.
Datasets are built via build_tf_dataset() wrapping tf.data.Dataset.from_tensor_slices.

Usage:
    from src.machine_learning.data import build_pricing_data, build_gnn_data
    data = build_pricing_data(n_samples=2000, seed=42)
    train_ds, val_ds, test_ds = data.train_ds, data.val_ds, data.test_ds
"""
from src.machine_learning.data.dataset import (
    build_tf_dataset,
    SyntheticData,
    create_pricing_dataset,
    create_calibration_dataset,
)
from src.machine_learning.data.common import TradeAttributeEncoder, TradeGraphBuilder
from src.machine_learning.data.pricing import (
    PricingDataResult,
    build_pricing_data,
)
from src.machine_learning.data.calibration import (
    CalibrationDataResult,
    build_calibration_data,
)
from src.machine_learning.data.portfolio import (
    build_gnn_dataset_from_portfolio,
    gnn_inputs_to_tf_dataset,
)
from src.machine_learning.data.gnn_rnn_hybrid import GnnDataResult, build_gnn_data
from src.machine_learning.data.manifest import DatasetManifest

__all__ = [
    "build_tf_dataset",
    "SyntheticData",
    "create_pricing_dataset",
    "create_calibration_dataset",
    "TradeAttributeEncoder",
    "TradeGraphBuilder",
    "PricingDataResult",
    "build_pricing_data",
    "CalibrationDataResult",
    "build_calibration_data",
    "build_gnn_dataset_from_portfolio",
    "gnn_inputs_to_tf_dataset",
    "GnnDataResult",
    "build_gnn_data",
    "DatasetManifest",
]
