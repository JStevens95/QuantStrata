"""
Calibration data preparation for ML-based model calibration.

Output: MLDataset or TFDataset (features = market quotes / IV surface,
targets = model parameters) for the generic training pipeline.
"""

from src.machine_learning.data.calibration.build import (
    build_calibration_dataset,
    CalibrationDataResult,
)

__all__ = [
    "build_calibration_dataset",
    "CalibrationDataResult",
]
