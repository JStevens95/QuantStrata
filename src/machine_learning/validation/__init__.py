"""
Pre-deployment validation gates for ML models.

Provides automated quality checks that a model must pass before it can be
promoted to staging or production.  This is a critical component of any
hedge-fund / investment-bank ML governance framework.

Modules:
    - gates : Configurable validation gate runner
    - checks : Individual check functions (metric thresholds, stability, etc.)

Usage:
    from src.machine_learning.validation import ValidationGate, run_validation

    gate = ValidationGate(
        min_r2=0.90,
        max_mae=0.05,
        max_mape=2.0,
        max_max_error=0.50,
        min_samples=1000,
    )
    report = gate.validate(eval_result, dataset_manifest)
    if report.passed:
        promote_to_staging(model)
"""
from src.machine_learning.validation.gates import (
    ValidationGate,
    ValidationReport,
    ValidationCheck,
    run_validation,
)

__all__ = [
    "ValidationGate",
    "ValidationReport",
    "ValidationCheck",
    "run_validation",
]
