from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True, slots=True)
class MonteCarloEstimate:
    """
    Container for a Monte Carlo estimate.

    Attributes
    ----------
    mean:
        Sample mean.
    stderr:
        Standard error of the mean: sqrt(Var / n).
    n_paths:
        Number of paths/samples used.
    conf_int_95:
        Optional 95% confidence interval for the mean (normal approximation).
    meta:
        Optional metadata (e.g., seed, antithetic flag, chunk sizes).
    """

    mean: float
    stderr: float
    n_paths: int
    conf_int_95: Optional[Tuple[float, float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if self.stderr < 0.0:
            raise ValueError("stderr must be non-negative.")