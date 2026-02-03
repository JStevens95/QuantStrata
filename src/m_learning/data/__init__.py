"""
Data preparation and dataset utilities.

This module provides:
    - TFDataset: TensorFlow-native dataset wrapper
    - NormalizationStats: Feature/target normalization
    - Data generation utilities for pricing, calibration, and GNN models

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
from src.m_learning.data.delta_hedging import (
    HedgingPath,
    build_delta_hedging_dataset,
    generate_gbm_path,
    simulate_hedging_path,
    simulate_hedging_paths,
)
from src.m_learning.data.gnn_synthetic import (
    SyntheticGnnData,
    generate_synthetic_gnn_data,
    generate_synthetic_trade_features,
    build_knn_adjacency,
    generate_pnl_history,
    generate_targets,
    default_hybrid_model_config,
)
from src.m_learning.data.portfolio_builder import (
    # Market
    build_fx_market,
    DEFAULT_FX_SPOT_ID,
    DEFAULT_FX_VOL_ID,
    DEFAULT_DOM_CURVE_ID,
    DEFAULT_FOR_CURVE_ID,
    # Portfolio builders
    build_elementary_portfolio,
    build_target_portfolio,
    # Feature extraction
    TradeFeatures,
    extract_trade_features,
    # GNN data
    GnnPortfolioData,
    build_gnn_portfolio_data,
    train_val_projection_split,
    # Convenience
    build_fx_gnn_data,
)
# Per-model data builders (output tf.data.Dataset)
from src.m_learning.data.common import TradeAttributeEncoder, TradeGraphBuilder
from src.m_learning.data.pricing import PricingDataResult, build_pricing_data
from src.m_learning.data.gnn_rnn_hybrid import GnnDataResult, build_gnn_data

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
    "HedgingPath",
    "build_delta_hedging_dataset",
    "generate_gbm_path",
    "simulate_hedging_path",
    "simulate_hedging_paths",
    # GNN synthetic data
    "SyntheticGnnData",
    "generate_synthetic_gnn_data",
    "generate_synthetic_trade_features",
    "build_knn_adjacency",
    "generate_pnl_history",
    "generate_targets",
    "default_hybrid_model_config",
    # Portfolio builder (new)
    "build_fx_market",
    "DEFAULT_FX_SPOT_ID",
    "DEFAULT_FX_VOL_ID",
    "DEFAULT_DOM_CURVE_ID",
    "DEFAULT_FOR_CURVE_ID",
    "build_elementary_portfolio",
    "build_target_portfolio",
    "TradeFeatures",
    "extract_trade_features",
    "GnnPortfolioData",
    "build_gnn_portfolio_data",
    "train_val_projection_split",
    "build_fx_gnn_data",
    # Per-model builders
    "TradeAttributeEncoder",
    "TradeGraphBuilder",
    "PricingDataResult",
    "build_pricing_data",
    "GnnDataResult",
    "build_gnn_data",
]
