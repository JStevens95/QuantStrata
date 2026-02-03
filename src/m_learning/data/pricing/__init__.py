"""
Data building and preprocessing for models/pricing.

Output: tf.data.Dataset(s) (train_ds, val_ds, test_ds) so the generic
pipeline has a single, repeatable interface.
"""

from src.m_learning.data.pricing.build import (
    PricingDataResult,
    build_pricing_data,
)

__all__ = [
    "PricingDataResult",
    "build_pricing_data",
]
