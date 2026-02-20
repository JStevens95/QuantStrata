"""
Base data build result type for rade ML framework.

All model-specific data builders should return a subclass of DataBuildResult so that the Trainer can accept any
model's data result generically.

Usage:
    @dataclass
    class MyModelResult(DataBuildResult):
        extra_field: np.ndarray = None
"""
from __future__ import annotations

import tensorflow as tf

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DataBuildResult:
    """
    Base data result returned by any data builder.

    Provides the minimum contract that the Trainer expects:
        - train_ds: training tf.data.Dataset
        - val_ds: optional validation tf.data.Dataset
        - test_ds: optional test tf.data.Dataset
        - metadata: dictionary of provenance / diagnostic info
    """

    train_ds: tf.data.Dataset = None
    val_ds: Optional[tf.data.Dataset] = None
    test_ds: Optional[tf.data.Dataset] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
