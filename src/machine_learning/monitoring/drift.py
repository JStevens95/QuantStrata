"""
Feature and prediction drift detection for production ML models.

Drift detection compares the statistical properties of live data against
a stored baseline (typically the training distribution).  When
significant drift is detected, the system can trigger alerts, shadow
evaluation, or automatic retraining.

Supported detection methods:
    - **Population Stability Index (PSI)** — standard industry metric for
      distribution shift.  PSI > 0.1 indicates moderate drift, > 0.25
      indicates significant drift.
    - **Kolmogorov–Smirnov (KS) test** — non-parametric two-sample test
      that detects any distributional change.
    - **Mean / Std shift** — simple z-score test on per-feature statistics.

Usage:
    # Build baseline from training data
    detector = DriftDetector.from_baseline(
        features=train_features,
        predictions=train_predictions,
        feature_names=["spot", "strike", "vol", "rate", "ttm"],
    )

    # Check live batch
    report = detector.check(live_features, live_predictions)
    print(report.summary())
    if report.drifted:
        send_alert(report)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DriftCheckResult:
    """
    Result of a single drift check on one feature or the prediction column.

    Parameters
    ----------
    name : str
        Feature or column name being checked.
    method : str
        Detection method used (``"psi"``, ``"ks"``, ``"mean_shift"``).
    statistic : float
        Test statistic value.
    threshold : float
        Threshold above which drift is flagged.
    drifted : bool
        Whether this check indicates drift.
    p_value : float, optional
        p-value for statistical tests (KS).
    baseline_mean : float
        Mean of the baseline distribution.
    baseline_std : float
        Std of the baseline distribution.
    live_mean : float
        Mean of the live distribution.
    live_std : float
        Std of the live distribution.
    """

    name: str
    method: str
    statistic: float
    threshold: float
    drifted: bool
    p_value: Optional[float] = None
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    live_mean: float = 0.0
    live_std: float = 0.0

    def __repr__(self) -> str:
        tag = "DRIFT" if self.drifted else "OK"
        return f"[{tag}] {self.name} ({self.method}): {self.statistic:.4f} vs {self.threshold:.4f}"


@dataclass
class DriftReport:
    """
    Aggregated drift report across all features and predictions.

    Parameters
    ----------
    feature_checks : list of DriftCheckResult
        Per-feature drift check results.
    prediction_check : DriftCheckResult, optional
        Drift check on the prediction distribution.
    timestamp : str
        ISO timestamp of the check.
    n_samples : int
        Number of live samples checked.
    metadata : dict
        Additional metadata.
    """

    feature_checks: List[DriftCheckResult] = field(default_factory=list)
    prediction_check: Optional[DriftCheckResult] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    n_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        """``True`` if any feature or prediction drift was detected."""
        checks = list(self.feature_checks)
        if self.prediction_check is not None:
            checks.append(self.prediction_check)
        return any(c.drifted for c in checks)

    @property
    def drifted_features(self) -> List[str]:
        """Names of features that drifted."""
        return [c.name for c in self.feature_checks if c.drifted]

    @property
    def n_drifted(self) -> int:
        """Total number of drifted checks."""
        n = sum(1 for c in self.feature_checks if c.drifted)
        if self.prediction_check is not None and self.prediction_check.drifted:
            n += 1
        return n

    def summary(self) -> str:
        """Formatted summary for logging."""
        status = "DRIFT DETECTED" if self.drifted else "NO DRIFT"
        lines = [
            "=" * 60,
            f"DRIFT REPORT — {status}",
            "=" * 60,
            f"Timestamp  : {self.timestamp}",
            f"Samples    : {self.n_samples:,}",
            f"Drifted    : {self.n_drifted} / {len(self.feature_checks) + (1 if self.prediction_check else 0)}",
            "-" * 60,
        ]
        for check in self.feature_checks:
            tag = "**DRIFT" if check.drifted else "  OK   "
            lines.append(
                f"  {tag}  {check.name:20s}  {check.method:10s}  "
                f"stat={check.statistic:.4f}  thr={check.threshold:.4f}"
            )
        if self.prediction_check is not None:
            pc = self.prediction_check
            tag = "**DRIFT" if pc.drifted else "  OK   "
            lines.append(
                f"  {tag}  {'predictions':20s}  {pc.method:10s}  "
                f"stat={pc.statistic:.4f}  thr={pc.threshold:.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "drifted": self.drifted,
            "n_drifted": self.n_drifted,
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "feature_checks": [
                {
                    "name": c.name,
                    "method": c.method,
                    "statistic": c.statistic,
                    "threshold": c.threshold,
                    "drifted": c.drifted,
                    "p_value": c.p_value,
                }
                for c in self.feature_checks
            ],
            "prediction_check": (
                {
                    "method": self.prediction_check.method,
                    "statistic": self.prediction_check.statistic,
                    "threshold": self.prediction_check.threshold,
                    "drifted": self.prediction_check.drifted,
                }
                if self.prediction_check
                else None
            ),
            "metadata": self.metadata,
        }

    def to_json(self, path: Union[str, Path]) -> None:
        """Save drift report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Feature and prediction drift detector.

    Stores a baseline distribution (from training data) and provides a
    ``check()`` method that compares live data against it.

    Parameters
    ----------
    baseline_stats : dict
        Per-feature statistics: ``{name: {"mean": ..., "std": ..., "hist": ...}}``.
    prediction_stats : dict, optional
        Prediction distribution statistics.
    feature_names : list of str
        Ordered list of feature names.
    psi_threshold : float
        PSI threshold for flagging drift (default 0.25).
    ks_alpha : float
        Significance level for KS test (default 0.05).
    n_bins : int
        Number of histogram bins for PSI computation.
    method : str
        Detection method: ``"psi"`` (default), ``"ks"``, or ``"mean_shift"``.

    Example
    -------
    >>> detector = DriftDetector.from_baseline(train_features, train_predictions)
    >>> report = detector.check(live_features, live_predictions)
    >>> if report.drifted:
    ...     print("Drift detected:", report.drifted_features)
    """

    def __init__(
        self,
        baseline_stats: Dict[str, Dict[str, Any]],
        prediction_stats: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        psi_threshold: float = 0.25,
        ks_alpha: float = 0.05,
        n_bins: int = 20,
        method: str = "psi",
    ):
        self.baseline_stats = baseline_stats
        self.prediction_stats = prediction_stats
        self.feature_names = feature_names or list(baseline_stats.keys())
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.n_bins = n_bins
        self.method = method

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_baseline(
        cls,
        features: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        n_bins: int = 20,
        method: str = "psi",
        psi_threshold: float = 0.25,
        ks_alpha: float = 0.05,
    ) -> "DriftDetector":
        """
        Build a detector from training-time arrays.

        Parameters
        ----------
        features : np.ndarray
            Training feature array of shape ``(n_samples, n_features)``.
        predictions : np.ndarray, optional
            Training prediction array of shape ``(n_samples,)``.
        feature_names : list of str, optional
            Feature column names.
        n_bins : int
            Histogram bins for PSI.
        method : str
            Detection method.
        psi_threshold : float
            PSI drift threshold.
        ks_alpha : float
            KS test significance level.

        Returns
        -------
        DriftDetector
        """
        features = np.asarray(features, dtype=np.float64)
        n_features = features.shape[1] if features.ndim >= 2 else 1

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Compute per-feature baseline statistics
        baseline_stats: Dict[str, Dict[str, Any]] = {}
        for i, name in enumerate(feature_names):
            col = features[:, i] if features.ndim >= 2 else features
            hist_counts, bin_edges = np.histogram(col, bins=n_bins)
            baseline_stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "hist_counts": hist_counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            }

        # Prediction baseline
        pred_stats = None
        if predictions is not None:
            predictions = np.asarray(predictions, dtype=np.float64).flatten()
            hist_counts, bin_edges = np.histogram(predictions, bins=n_bins)
            pred_stats = {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "hist_counts": hist_counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            }

        return cls(
            baseline_stats=baseline_stats,
            prediction_stats=pred_stats,
            feature_names=feature_names,
            psi_threshold=psi_threshold,
            ks_alpha=ks_alpha,
            n_bins=n_bins,
            method=method,
        )

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(
        self,
        features: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DriftReport:
        """
        Check live data for drift against the stored baseline.

        Parameters
        ----------
        features : np.ndarray
            Live feature array of shape ``(n_samples, n_features)``.
        predictions : np.ndarray, optional
            Live predictions array.
        metadata : dict, optional
            Additional metadata for the report.

        Returns
        -------
        DriftReport
        """
        features = np.asarray(features, dtype=np.float64)
        n_samples = len(features)

        feature_checks: List[DriftCheckResult] = []

        for i, name in enumerate(self.feature_names):
            if name not in self.baseline_stats:
                continue

            col = features[:, i] if features.ndim >= 2 else features
            baseline = self.baseline_stats[name]

            if self.method == "psi":
                check = self._check_psi(name, col, baseline)
            elif self.method == "ks":
                check = self._check_ks(name, col, baseline)
            elif self.method == "mean_shift":
                check = self._check_mean_shift(name, col, baseline)
            else:
                raise ValueError(f"Unknown drift method: {self.method}")

            feature_checks.append(check)

        # Prediction drift
        pred_check = None
        if predictions is not None and self.prediction_stats is not None:
            predictions = np.asarray(predictions, dtype=np.float64).flatten()
            if self.method == "psi":
                pred_check = self._check_psi("predictions", predictions, self.prediction_stats)
            elif self.method == "ks":
                pred_check = self._check_ks("predictions", predictions, self.prediction_stats)
            else:
                pred_check = self._check_mean_shift("predictions", predictions, self.prediction_stats)

        report = DriftReport(
            feature_checks=feature_checks,
            prediction_check=pred_check,
            n_samples=n_samples,
            metadata=metadata or {},
        )

        if report.drifted:
            logger.warning(
                "Drift detected in %d features: %s",
                report.n_drifted,
                report.drifted_features,
            )

        return report

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _check_psi(
        self,
        name: str,
        live_col: np.ndarray,
        baseline: Dict[str, Any],
    ) -> DriftCheckResult:
        """
        Population Stability Index (PSI) drift check.

        PSI measures the shift between two distributions using binned
        proportions.  Industry convention:
            - PSI < 0.1  : no drift
            - 0.1 ≤ PSI < 0.25 : moderate drift
            - PSI ≥ 0.25 : significant drift
        """
        bin_edges = np.array(baseline["bin_edges"])
        baseline_counts = np.array(baseline["hist_counts"], dtype=np.float64)

        # Compute live histogram with the same bin edges
        live_counts, _ = np.histogram(live_col, bins=bin_edges)
        live_counts = live_counts.astype(np.float64)

        # Convert to proportions (avoid zero division)
        eps = 1e-8
        baseline_prop = (baseline_counts + eps) / (baseline_counts.sum() + eps * len(baseline_counts))
        live_prop = (live_counts + eps) / (live_counts.sum() + eps * len(live_counts))

        # PSI = Σ (live_i - baseline_i) * ln(live_i / baseline_i)
        psi = float(np.sum((live_prop - baseline_prop) * np.log(live_prop / baseline_prop)))

        return DriftCheckResult(
            name=name,
            method="psi",
            statistic=psi,
            threshold=self.psi_threshold,
            drifted=psi >= self.psi_threshold,
            baseline_mean=baseline["mean"],
            baseline_std=baseline["std"],
            live_mean=float(np.mean(live_col)),
            live_std=float(np.std(live_col)),
        )

    def _check_ks(
        self,
        name: str,
        live_col: np.ndarray,
        baseline: Dict[str, Any],
    ) -> DriftCheckResult:
        """
        Kolmogorov–Smirnov two-sample test.

        Uses the approximate critical value for the given significance level
        rather than requiring scipy (optional dependency).
        """
        try:
            from scipy.stats import ks_2samp

            # Reconstruct baseline samples from histogram (approximation)
            baseline_samples = self._reconstruct_samples(baseline)
            stat, p_value = ks_2samp(baseline_samples, live_col)

            return DriftCheckResult(
                name=name,
                method="ks",
                statistic=stat,
                threshold=self.ks_alpha,
                drifted=p_value < self.ks_alpha,
                p_value=p_value,
                baseline_mean=baseline["mean"],
                baseline_std=baseline["std"],
                live_mean=float(np.mean(live_col)),
                live_std=float(np.std(live_col)),
            )
        except ImportError:
            logger.warning("scipy not installed — falling back to mean_shift for KS check")
            return self._check_mean_shift(name, live_col, baseline)

    def _check_mean_shift(
        self,
        name: str,
        live_col: np.ndarray,
        baseline: Dict[str, Any],
    ) -> DriftCheckResult:
        """
        Simple z-score based mean shift test.

        Flags drift when the live mean deviates by more than 3 standard
        errors from the baseline mean.
        """
        baseline_mean = baseline["mean"]
        baseline_std = max(baseline["std"], 1e-8)
        live_mean = float(np.mean(live_col))

        # z-score of the shift
        se = baseline_std / np.sqrt(max(len(live_col), 1))
        z_score = abs(live_mean - baseline_mean) / se

        # Threshold: 3 standard errors (≈ 99.7% confidence)
        threshold = 3.0

        return DriftCheckResult(
            name=name,
            method="mean_shift",
            statistic=z_score,
            threshold=threshold,
            drifted=z_score > threshold,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            live_mean=live_mean,
            live_std=float(np.std(live_col)),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_samples(
        stats: Dict[str, Any],
        n_samples: int = 10000,
    ) -> np.ndarray:
        """
        Approximate reconstruction of samples from histogram statistics.

        Uses bin midpoints weighted by counts to generate a synthetic sample
        for statistical tests.
        """
        bin_edges = np.array(stats["bin_edges"])
        counts = np.array(stats["hist_counts"])

        midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = counts.sum()

        if total == 0:
            return np.random.normal(stats["mean"], max(stats["std"], 1e-8), size=n_samples)

        # Weighted sampling from bin midpoints
        probs = counts / total
        indices = np.random.choice(len(midpoints), size=n_samples, p=probs)
        bin_width = bin_edges[1] - bin_edges[0]

        # Add uniform jitter within each bin
        samples = midpoints[indices] + np.random.uniform(-bin_width / 2, bin_width / 2, size=n_samples)
        return samples

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, path: Union[str, Path]) -> None:
        """Save detector baseline to JSON."""
        data = {
            "baseline_stats": self.baseline_stats,
            "prediction_stats": self.prediction_stats,
            "feature_names": self.feature_names,
            "psi_threshold": self.psi_threshold,
            "ks_alpha": self.ks_alpha,
            "n_bins": self.n_bins,
            "method": self.method,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DriftDetector":
        """Load detector from saved JSON baseline."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


__all__ = [
    "DriftDetector",
    "DriftReport",
    "DriftCheckResult",
]
