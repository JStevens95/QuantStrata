"""
Data preparation module for QuantStrata ML.

Provides:
- types: MLDataset, feature/target schemas
- pricing: Adapters for pricing data (MC paths, analytic)
- calibration: Adapters for calibration data
- portfolio: Adapters for portfolio/GNN inputs
"""

from src.m_learning.data.types import MLDataset, PricingFeatures, CalibrationFeatures
from src.m_learning.data.pricing import (
    build_pricing_dataset_from_mc,
    build_pricing_dataset_from_analytic,
)
from src.m_learning.data.calibration import build_calibration_dataset
from src.m_learning.data.portfolio import (
    build_gnn_dataset_from_portfolio,
    gnn_inputs_to_tf_dataset,
)

__all__ = [
    "MLDataset",
    "PricingFeatures",
    "CalibrationFeatures",
    "build_pricing_dataset_from_mc",
    "build_pricing_dataset_from_analytic",
    "build_calibration_dataset",
    "build_gnn_dataset_from_portfolio",
    "gnn_inputs_to_tf_dataset",
]
