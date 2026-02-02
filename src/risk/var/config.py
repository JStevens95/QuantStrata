"""VaR configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class VarConfig:
    """
    Configuration for Value-at-Risk computation.

    Attributes
    ----------
    confidence : float
        Confidence level (e.g. 0.99 for 99% VaR).
    horizon_days : int
        VaR horizon in days (used for scaling; e.g. sqrt(horizon_days) for i.i.d.).
    method : str
        "historical", "parametric", or "mc".
    """
    confidence: float = 0.99
    horizon_days: int = 1
    method: str = "historical"

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1).")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1.")
        if self.method not in ("historical", "parametric", "mc"):
            raise ValueError("method must be 'historical', 'parametric', or 'mc'.")


@dataclass(frozen=True, slots=True)
class VarResult:
    """
    Result of a VaR computation.

    Attributes
    ----------
    var : float
        Value-at-Risk (positive number = loss at confidence).
    cvar : float, optional
        Conditional VaR (expected shortfall) if computed.
    method : str
        Method used ("historical", "parametric", "mc").
    confidence : float
        Confidence level.
    horizon_days : int
        Horizon in days.
    metadata : dict
        Optional method-specific info (n_observations, n_simulations, etc.).
    """
    var: float
    method: str
    confidence: float
    horizon_days: int
    cvar: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
