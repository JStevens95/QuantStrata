"""
Configurable pre-deployment validation gates.

A ``ValidationGate`` defines a set of quality thresholds that a model must
satisfy before it is eligible for deployment.  The ``validate()`` method
runs all checks against an ``EvaluationResult`` and returns a structured
``ValidationReport`` with pass/fail status for each check.

This pattern is standard in production ML systems at hedge funds and
investment banks, where model risk management requires auditable evidence
that a model meets quantitative performance and data-quality criteria.

Usage:
    gate = ValidationGate(
        min_r2=0.90,
        max_mae=0.05,
        max_mape=2.0,
        max_max_error=0.50,
        min_samples=1000,
    )
    report = gate.validate(eval_result, dataset_manifest)
    print(report.summary())
    assert report.passed, "Model failed validation gates"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.machine_learning.core.types import EvaluationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    """
    Result of a single validation check.

    Parameters
    ----------
    name : str
        Human-readable check name (e.g. ``"R² ≥ 0.90"``).
    passed : bool
        Whether the check passed.
    actual : float
        Actual value observed.
    threshold : float
        Threshold the value was compared against.
    operator : str
        Comparison operator used (``">="`` , ``"<="`` , ``">"`` , ``"<"``).
    message : str
        Descriptive message.
    """

    name: str
    passed: bool
    actual: float
    threshold: float
    operator: str = ">="
    message: str = ""

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.actual:.4f} {self.operator} {self.threshold:.4f}"


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Aggregated result of all validation checks.

    Parameters
    ----------
    checks : list of ValidationCheck
        Individual check results.
    model_name : str
        Name of the model being validated.
    timestamp : str
        ISO timestamp of the validation run.
    metadata : dict
        Additional metadata (e.g. dataset hash, run ID).
    """

    checks: List[ValidationCheck] = field(default_factory=list)
    model_name: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """``True`` if every check passed."""
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if not c.passed)

    @property
    def failed_checks(self) -> List[ValidationCheck]:
        """List of failed checks."""
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        """Formatted summary string for logging / reporting."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            "=" * 60,
            f"VALIDATION REPORT — {status}",
            "=" * 60,
            f"Model     : {self.model_name}",
            f"Timestamp : {self.timestamp}",
            f"Checks    : {self.n_passed} passed, {self.n_failed} failed "
            f"(of {len(self.checks)} total)",
            "-" * 60,
        ]
        for check in self.checks:
            tag = "  PASS" if check.passed else "**FAIL"
            lines.append(
                f"  {tag}  {check.name:25s}  "
                f"actual={check.actual:10.4f}  {check.operator}  {check.threshold:10.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "passed": self.passed,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "n_checks": len(self.checks),
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "actual": c.actual,
                    "threshold": c.threshold,
                    "operator": c.operator,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

class ValidationGate:
    """
    Configurable set of pre-deployment quality gates.

    Each threshold attribute defines a metric constraint.  Setting a
    threshold to ``None`` disables that check.

    Parameters
    ----------
    min_r2 : float, optional
        Minimum acceptable R² score.
    max_mae : float, optional
        Maximum acceptable Mean Absolute Error.
    max_mape : float, optional
        Maximum acceptable Mean Absolute Percentage Error (%).
    max_rmse : float, optional
        Maximum acceptable Root Mean Squared Error.
    max_max_error : float, optional
        Maximum acceptable worst-case absolute error.
    max_p95_error : float, optional
        Maximum acceptable 95th-percentile error.
    min_samples : int, optional
        Minimum number of evaluation samples required.
    custom_checks : list of callable, optional
        Additional check functions with signature
        ``(eval_result, **kwargs) -> ValidationCheck``.

    Example
    -------
    >>> gate = ValidationGate(min_r2=0.95, max_mae=0.02, min_samples=5000)
    >>> report = gate.validate(eval_result)
    >>> assert report.passed
    """

    def __init__(
        self,
        min_r2: Optional[float] = None,
        max_mae: Optional[float] = None,
        max_mape: Optional[float] = None,
        max_rmse: Optional[float] = None,
        max_max_error: Optional[float] = None,
        max_p95_error: Optional[float] = None,
        min_samples: Optional[int] = None,
        custom_checks: Optional[List[Any]] = None,
    ):
        self.min_r2 = min_r2
        self.max_mae = max_mae
        self.max_mape = max_mape
        self.max_rmse = max_rmse
        self.max_max_error = max_max_error
        self.max_p95_error = max_p95_error
        self.min_samples = min_samples
        self.custom_checks = custom_checks or []

    def validate(
        self,
        eval_result: EvaluationResult,
        model_name: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        """
        Run all configured checks against an evaluation result.

        Parameters
        ----------
        eval_result : EvaluationResult
            Model evaluation output containing metrics and dataset info.
        model_name : str
            Model identifier for the report.
        metadata : dict, optional
            Additional metadata to attach to the report.

        Returns
        -------
        ValidationReport
        """
        checks: List[ValidationCheck] = []
        metrics = eval_result.metrics

        # --- Metric-based checks ---
        if self.min_r2 is not None and "r2" in metrics:
            checks.append(self._check_min("R²", metrics["r2"], self.min_r2))

        if self.max_mae is not None and "mae" in metrics:
            checks.append(self._check_max("MAE", metrics["mae"], self.max_mae))

        if self.max_mape is not None and "mape" in metrics:
            checks.append(self._check_max("MAPE (%)", metrics["mape"], self.max_mape))

        if self.max_rmse is not None and "rmse" in metrics:
            checks.append(self._check_max("RMSE", metrics["rmse"], self.max_rmse))

        if self.max_max_error is not None and "max_error" in metrics:
            checks.append(self._check_max("Max Error", metrics["max_error"], self.max_max_error))

        if self.max_p95_error is not None and "p95_error" in metrics:
            checks.append(self._check_max("P95 Error", metrics["p95_error"], self.max_p95_error))

        # --- Data-quality checks ---
        if self.min_samples is not None:
            n_samples = eval_result.dataset_info.get("n_samples", 0)
            checks.append(self._check_min(
                "Sample count", float(n_samples), float(self.min_samples),
            ))

        # --- Custom checks ---
        for check_fn in self.custom_checks:
            try:
                result = check_fn(eval_result)
                if isinstance(result, ValidationCheck):
                    checks.append(result)
            except Exception as exc:
                logger.warning("Custom check failed with error: %s", exc)
                checks.append(ValidationCheck(
                    name="Custom check (error)",
                    passed=False,
                    actual=0.0,
                    threshold=0.0,
                    message=str(exc),
                ))

        report = ValidationReport(
            checks=checks,
            model_name=model_name,
            metadata=metadata or {},
        )

        # Log outcome
        if report.passed:
            logger.info("Validation PASSED for model '%s' (%d checks)", model_name, len(checks))
        else:
            logger.warning(
                "Validation FAILED for model '%s': %d/%d checks failed",
                model_name, report.n_failed, len(checks),
            )

        return report

    # --- helpers ---

    @staticmethod
    def _check_min(name: str, actual: float, threshold: float) -> ValidationCheck:
        """Check that ``actual >= threshold``."""
        return ValidationCheck(
            name=name,
            passed=actual >= threshold,
            actual=actual,
            threshold=threshold,
            operator=">=",
            message=f"{name}: {actual:.6f} >= {threshold:.6f}",
        )

    @staticmethod
    def _check_max(name: str, actual: float, threshold: float) -> ValidationCheck:
        """Check that ``actual <= threshold``."""
        return ValidationCheck(
            name=name,
            passed=actual <= threshold,
            actual=actual,
            threshold=threshold,
            operator="<=",
            message=f"{name}: {actual:.6f} <= {threshold:.6f}",
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_validation(
    eval_result: EvaluationResult,
    gate: Optional[ValidationGate] = None,
    model_name: str = "unknown",
    **thresholds: float,
) -> ValidationReport:
    """
    Run validation gates against an evaluation result.

    Convenience wrapper around ``ValidationGate.validate()``.

    Parameters
    ----------
    eval_result : EvaluationResult
        Model evaluation output.
    gate : ValidationGate, optional
        Pre-configured gate (overrides ``**thresholds``).
    model_name : str
        Model identifier.
    **thresholds
        Keyword arguments passed to ``ValidationGate()`` constructor
        (e.g. ``min_r2=0.95``, ``max_mae=0.02``).

    Returns
    -------
    ValidationReport
    """
    if gate is None:
        gate = ValidationGate(**thresholds)
    return gate.validate(eval_result, model_name=model_name)


__all__ = [
    "ValidationGate",
    "ValidationReport",
    "ValidationCheck",
    "run_validation",
]
