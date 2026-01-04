from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from src.models.numeric.monte_carlo.base import MonteCarloEstimate


# -------------------------------------------------------------------------------------------------
# Basic batch estimators
# -------------------------------------------------------------------------------------------------

def mean_stderr(samples: np.ndarray) -> Tuple[float, float, int]:
    """
    Compute (mean, stderr, n) for 1D samples.

    stderr uses unbiased sample variance with ddof=1 when n>1:
        stderr = sqrt( Var_unbiased / n )
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("samples must be 1D.")

    n = int(x.size)
    if n <= 0:
        raise ValueError("samples must be non-empty.")

    mean = float(x.mean())
    if n == 1:
        return mean, 0.0, 1

    var = float(x.var(ddof=1))
    stderr = math.sqrt(max(0.0, var / n))
    return mean, stderr, n


def mean_confidence_interval(
    mean: float,
    stderr: float,
    *,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    """
    Normal-approx confidence interval for the mean.

    Default z ≈ 1.96 (95% CI).
    """
    half = float(z) * float(stderr)
    return float(mean - half), float(mean + half)


def estimate_from_samples(
    samples: np.ndarray,
    *,
    meta: Optional[Dict[str, Any]] = None,
    z: float = 1.959963984540054,
) -> MonteCarloEstimate:
    """
    Convert a 1D payoff sample vector into a MonteCarloEstimate.

    Returns
    -------
    MonteCarloEstimate with:
      - mean
      - stderr
      - n_paths
      - conf_int_95 (normal approximation)
      - meta passthrough
    """
    m, se, n = mean_stderr(samples)
    ci = mean_confidence_interval(m, se, z=z)
    return MonteCarloEstimate(
        mean=float(m),
        stderr=float(se),
        n_paths=int(n),
        conf_int_95=(float(ci[0]), float(ci[1])),
        meta=dict(meta or {}),
    )


# -------------------------------------------------------------------------------------------------
# Streaming stats (single-variable)
# -------------------------------------------------------------------------------------------------

@dataclass(slots=True)
class OnlineStats:
    """
    Online mean/variance accumulator (Welford).

    Use this when you don't want to store all payoffs in memory.

    Notes
    -----
    - Variance is unbiased when n>1 (ddof=1).
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations

    def update(self, x: float) -> None:
        """Update with one observation x."""
        self.n += 1
        delta = float(x) - self.mean
        self.mean += delta / self.n
        delta2 = float(x) - self.mean
        self.m2 += delta * delta2

    def update_many(self, xs: Iterable[float]) -> None:
        """Update from an iterable of observations."""
        for v in xs:
            self.update(float(v))

    @property
    def variance(self) -> float:
        """Unbiased sample variance (ddof=1) when n>1."""
        if self.n <= 1:
            return 0.0
        return float(self.m2 / (self.n - 1))

    @property
    def stderr(self) -> float:
        """Standard error of the sample mean."""
        if self.n <= 1:
            return 0.0
        return math.sqrt(max(0.0, self.variance / self.n))

    def conf_int_95(self, *, z: float = 1.959963984540054) -> Tuple[float, float]:
        """Normal-approx 95% confidence interval for the mean."""
        return mean_confidence_interval(self.mean, self.stderr, z=z)


def estimate_from_online_stats(
    stats: OnlineStats,
    *,
    meta: Optional[Dict[str, Any]] = None,
    z: float = 1.959963984540054,
) -> MonteCarloEstimate:
    """
    Convert OnlineStats into a MonteCarloEstimate.
    """
    if stats.n <= 0:
        raise ValueError("OnlineStats is empty (n=0).")
    ci = mean_confidence_interval(stats.mean, stats.stderr, z=z)
    return MonteCarloEstimate(
        mean=float(stats.mean),
        stderr=float(stats.stderr),
        n_paths=int(stats.n),
        conf_int_95=(float(ci[0]), float(ci[1])),
        meta=dict(meta or {}),
    )


# -------------------------------------------------------------------------------------------------
# Streaming stats (two-variable): for control variates without storing arrays
# -------------------------------------------------------------------------------------------------

@dataclass(slots=True)
class OnlineCovStats:
    """
    Online mean/variance/covariance accumulator for paired samples (y, c).

    This supports control variates in a single pass without storing arrays.

    Tracked quantities
    ------------------
    - mean_y, var_y
    - mean_c, var_c
    - cov_yc
    """
    n: int = 0

    mean_y: float = 0.0
    mean_c: float = 0.0

    m2_y: float = 0.0  # sum (y - mean_y)^2
    m2_c: float = 0.0  # sum (c - mean_c)^2
    c2_yc: float = 0.0  # sum (y - mean_y_prev) * (c - mean_c_new)  (Welford-style cross term)

    def update(self, y: float, c: float) -> None:
        """
        Update accumulator with one pair (y, c).

        Uses a numerically stable one-pass algorithm (Welford generalization).
        """
        y = float(y)
        c = float(c)

        self.n += 1
        n = self.n

        dy = y - self.mean_y
        self.mean_y += dy / n

        dc = c - self.mean_c
        self.mean_c += dc / n

        # Update second moments using the updated means
        self.m2_y += dy * (y - self.mean_y)
        self.m2_c += dc * (c - self.mean_c)

        # Cross term: using dy (based on old mean_y) and (c - new_mean_c)
        self.c2_yc += dy * (c - self.mean_c)

    def update_many(self, ys: Iterable[float], cs: Iterable[float]) -> None:
        """Update from paired iterables (ys, cs)."""
        for y, c in zip(ys, cs):
            self.update(float(y), float(c))

    @property
    def var_y(self) -> float:
        """Unbiased sample variance of y (ddof=1) when n>1."""
        if self.n <= 1:
            return 0.0
        return float(self.m2_y / (self.n - 1))

    @property
    def var_c(self) -> float:
        """Unbiased sample variance of c (ddof=1) when n>1."""
        if self.n <= 1:
            return 0.0
        return float(self.m2_c / (self.n - 1))

    @property
    def cov_yc(self) -> float:
        """Unbiased sample covariance Cov(y, c) (ddof=1) when n>1."""
        if self.n <= 1:
            return 0.0
        return float(self.c2_yc / (self.n - 1))


def estimate_with_control_variate(
    stats: OnlineCovStats,
    *,
    c_expectation: float,
    meta: Optional[Dict[str, Any]] = None,
    z: float = 1.959963984540054,
    var_floor: float = 0.0,
) -> MonteCarloEstimate:
    """
    Build a control-variate adjusted MonteCarloEstimate from OnlineCovStats.

    Control variate adjustment
    -------------------------
    Given payoff samples y and control samples c with known E[c],
    define adjusted sample:
        y_adj = y - beta * (c - E[c])

    The optimal beta (in least squares / variance-minimizing sense) is:
        beta = Cov(y, c) / Var(c)

    This function computes the adjusted mean and its stderr using only streaming stats.

    Notes
    -----
    - If Var(c) is ~0, we set beta=0 (no adjustment).
    - stderr uses the implied adjusted variance:
        Var(y_adj) = Var(y) + beta^2 Var(c) - 2 beta Cov(y,c)
      then stderr = sqrt( Var(y_adj) / n ).
    """
    if stats.n <= 0:
        raise ValueError("OnlineCovStats is empty (n=0).")

    n = int(stats.n)
    var_c = float(stats.var_c)
    cov_yc = float(stats.cov_yc)
    var_y = float(stats.var_y)

    # Guard degenerate control.
    if var_c <= max(float(var_floor), 0.0):
        beta = 0.0
    else:
        beta = cov_yc / var_c

    mean_adj = float(stats.mean_y - beta * (stats.mean_c - float(c_expectation)))

    # Adjusted variance (unbiased-ish at the sample level; then stderr uses /n).
    var_adj = float(var_y + (beta * beta) * var_c - 2.0 * beta * cov_yc)
    var_adj = max(0.0, var_adj)

    stderr = 0.0 if n <= 1 else math.sqrt(var_adj / n)

    ci = mean_confidence_interval(mean_adj, stderr, z=z)

    out_meta = dict(meta or {})
    out_meta.update(
        {
            "control_variate": True,
            "beta": float(beta),
            "c_expectation": float(c_expectation),
            "mean_y": float(stats.mean_y),
            "mean_c": float(stats.mean_c),
            "var_y": float(var_y),
            "var_c": float(var_c),
            "cov_yc": float(cov_yc),
        }
    )

    return MonteCarloEstimate(
        mean=float(mean_adj),
        stderr=float(stderr),
        n_paths=int(n),
        conf_int_95=(float(ci[0]), float(ci[1])),
        meta=out_meta,
    )