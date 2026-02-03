"""
Data preparation and dataset utilities.

This module provides:
    - TFDataset: TensorFlow-native dataset wrapper
    - NormalizationStats: Feature/target normalization
    - Data generation utilities for pricing and calibration

Usage:
    from src.m_learning.data import (
        TFDataset,
        create_pricing_dataset,
        create_calibration_dataset,
    )
    
    # Generate synthetic pricing data
    dataset = create_pricing_dataset(n_samples=10000, seed=42)
    
    # Normalize
    dataset.normalize_features()
    dataset.normalize_targets()
    
    # Split
    train, val, test = dataset.split(train=0.7, val=0.15, test=0.15)
    
    # Create tf.data.Dataset
    train_ds = train.to_tf_dataset(batch_size=256, shuffle=True)
"""
from src.m_learning.data.dataset import (
    TFDataset,
    NormalizationStats,
    create_pricing_dataset,
    create_calibration_dataset,
)

# Legacy exports for backward compatibility
from src.m_learning.data.types import (
    MLDataset,
    PricingFeatures,
    CalibrationFeatures,
)
from src.m_learning.data.pricing import (
    build_pricing_dataset_from_mc,
    build_pricing_dataset_from_analytic,
)
from src.m_learning.data.calibration import (
    build_calibration_dataset,
)
from src.m_learning.data.portfolio import (
    build_gnn_dataset_from_portfolio,
    gnn_inputs_to_tf_dataset,
)

__all__ = [
    # New TF-native API
    "TFDataset",
    "NormalizationStats",
    "create_pricing_dataset",
    "create_calibration_dataset",
    # Legacy API (still supported)
    "MLDataset",
    "PricingFeatures",
    "CalibrationFeatures",
    "build_pricing_dataset_from_mc",
    "build_pricing_dataset_from_analytic",
    "build_calibration_dataset",
    "build_gnn_dataset_from_portfolio",
    "gnn_inputs_to_tf_dataset",
]
