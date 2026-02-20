"""
Base data result types for rade ML framework.

Provides a generic DataBuildResult that every model-specific data builder should inherit from.
This lets the Trainer and evaluation pipelines accept any model's data result without coupling
to a specific model architecture.

Usage:
    @dataclass
    class HybridGnnRnnResult(DataBuildResult):
        # model-specific fields ...
        graph_adjacency: Any = None
"""
from __future__ import annotations

import tensorflow as tf

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DataBuildResult:
    """
    Base data result returned by any model-specific data builder.

    Every model-specific result (e.g. HybridGnnRnnResult) should inherit from this so
    that the Trainer can accept any model's output generically.
    """

    # core tf.data.Dataset splits.
    train_ds: Optional[tf.data.Dataset] = None
    val_ds: Optional[tf.data.Dataset] = None
    test_ds: Optional[tf.data.Dataset] = None

    # metadata dictionary for tracking pipeline provenance.
    metadata: Dict[str, Any] = field(default_factory=dict)
