from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def mean_stderr(samples: np.ndarray) -> Tuple[float, float, int]:
    """
    Compute (mean, stderr, n) for 1D samples.

    stderr uses unbiased sample variance with ddof=1 when n>1.
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


def mean_confidence_interval(mean: float, stderr: float, *, z: float = 1.959963984540054) -> Tuple[float, float]:
    """
    Normal-approx confidence interval for the mean. Default z ~ 95%.
    """
    half = float(z) * float(stderr)
    return float(mean - half), float(mean + half)


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
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def update_many(self, xs: Iterable[float]) -> None:
        for v in xs:
            self.update(float(v))

    @property
    def variance(self) -> float:
        if self.n <= 1:
            return 0.0
        return float(self.m2 / (self.n - 1))

    @property
    def stderr(self) -> float:
        if self.n <= 1:
            return 0.0
        return math.sqrt(max(0.0, self.variance / self.n))

    def conf_int_95(self) -> Tuple[float, float]:
        return mean_confidence_interval(self.mean, self.stderr)