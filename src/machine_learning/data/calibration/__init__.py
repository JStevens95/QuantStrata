"""
Calibration data preparation for ML-based model calibration.

Output: tf.data.Dataset(s) (train_ds, val_ds, test_ds) via sklearn + build_tf_dataset.
"""

from src.machine_learning.data.calibration.build import (
    build_calibration_data,
    CalibrationDataResult,
)

__all__ = [
    "build_calibration_data",
    "CalibrationDataResult",
]
