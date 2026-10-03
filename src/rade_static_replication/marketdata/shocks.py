"""
Shock application — resolve raw shocks into absolute market states.

Your environment expresses some shocks **relatively** and others **absolutely**.
That convention is a property of *how the data was produced*, so it is declared
here (per field) and applied at the client/builder boundary. Everything downstream
then sees only resolved absolute levels.

Usage::

    conv = ShockConvention({"spot": ShockMode.LOG_RELATIVE, "rate": ShockMode.ADDITIVE})
    spot_states = conv.apply("spot", base_spot, raw_spot_shock)   # (n_scenarios, ...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from src.rade_static_replication.domain.enums import ShockMode


def apply_shock(base: np.ndarray | float, shock: np.ndarray, mode: ShockMode) -> np.ndarray:
    """Resolve a raw ``shock`` against ``base`` into absolute levels.

    ``base`` broadcasts against ``shock`` (e.g. base curve ``(n_pillars,)`` vs shock
    ``(n_scenarios, n_pillars)``).
    """
    base = np.asarray(base, dtype=np.float64)
    shock = np.asarray(shock, dtype=np.float64)
    if mode == ShockMode.ABSOLUTE:
        return shock
    if mode == ShockMode.ADDITIVE:
        return base + shock
    if mode == ShockMode.RELATIVE:
        return base * (1.0 + shock)
    if mode == ShockMode.LOG_RELATIVE:
        return base * np.exp(shock)
    raise ValueError(f"unknown shock mode {mode!r}")


@dataclass(frozen=True)
class ShockConvention:
    """Per-field shock modes with a default fallback."""
    modes: Mapping[str, ShockMode] = field(default_factory=dict)
    default: ShockMode = ShockMode.ABSOLUTE

    def mode_for(self, field_name: str) -> ShockMode:
        return self.modes.get(field_name, self.default)

    def apply(self, field_name: str, base: np.ndarray | float, shock: np.ndarray) -> np.ndarray:
        return apply_shock(base, shock, self.mode_for(field_name))
