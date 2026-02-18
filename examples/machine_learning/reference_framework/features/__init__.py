"""
Central feature definitions and transforms.

Eliminates training-serving skew by defining feature logic once and reusing
in both training and inference. Supports pricing and GNN-RNN feature sets.

Public API:
    - FeatureSchema: Schema for a feature set (names, dtypes, transform ids)
    - FeatureRegistry: Registry mapping feature/transform names to functions
    - Standardiser: Z-score or min-max normalization (fittable, serialisable)
    - compute_features: Apply schema + transforms to raw data
"""

from reference_framework.features.schema import (
    FeatureSchema,
    PricingFeatureSchema,
    GnnFeatureSchema,
)
from reference_framework.features.registry import FeatureRegistry, get_registry
from reference_framework.features.transforms.standardiser import Standardiser

__all__ = [
    "FeatureSchema",
    "PricingFeatureSchema",
    "GnnFeatureSchema",
    "FeatureRegistry",
    "get_registry",
    "Standardiser",
]
