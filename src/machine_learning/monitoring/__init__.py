"""
Production monitoring for deployed ML models.

Provides real-time and batch-mode drift detection for features and
predictions, enabling automated alerts when model behaviour deviates
from its training baseline.  This is essential for front-office ML
systems where stale models can generate significant P&L risk.

Modules:
    - drift  : Feature and prediction drift detectors

Usage:
    from src.machine_learning.monitoring import DriftDetector, DriftReport

    detector = DriftDetector.from_baseline(train_features, train_predictions)

    # In production — check incoming batch
    report = detector.check(live_features, live_predictions)
    if report.drifted:
        trigger_retraining_alert(report)
"""
from src.machine_learning.monitoring.drift import (
    DriftDetector,
    DriftReport,
    DriftCheckResult,
)

__all__ = [
    "DriftDetector",
    "DriftReport",
    "DriftCheckResult",
]
