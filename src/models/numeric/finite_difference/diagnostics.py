from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid


@dataclass(frozen=True, slots=True)
class FdDiagnostics:
    """
    Diagnostics container for a single FD run.

    Intended usage
    --------------
    - Examples / notebooks: plot V(S,0), error curves, and the full surface heatmap.
    - Regression tests: keep small surfaces or just values_t0 for sanity checks.

    Notes
    -----
    - values_t0_per_unit and surface_per_unit are *per 1 unit notional* (unscaled).
    - spot_grid is always in *linear* space (S), even if the PDE grid is log-space.
    """
    x_grid: SpatialGrid1D
    t_grid: TimeGrid

    # Linear-space spot coordinates aligned with the FD solution arrays.
    spot_grid: np.ndarray            # shape (n_x,)
    time_grid: np.ndarray            # shape (n_t,)

    values_t0_per_unit: np.ndarray   # shape (n_x,)
    surface_per_unit: Optional[np.ndarray] = None  # shape (n_t, n_x) if stored

    # Useful for annotations / reference curves.
    spot0: float = 0.0
    strike: float = 0.0
    expiry: float = 0.0
    r_d: float = 0.0
    r_f: float = 0.0
    sigma: float = 0.0
    x0: float = 0.0

    # Optional extra metadata.
    meta: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.spot_grid.shape != self.values_t0_per_unit.shape:
            raise ValueError("spot_grid and values_t0_per_unit must have the same shape.")
        if self.surface_per_unit is not None:
            if self.surface_per_unit.shape != (self.time_grid.size, self.spot_grid.size):
                raise ValueError("surface_per_unit must have shape (n_t, n_x).")

    def pv_per_unit_at_spot0(self) -> float:
        """
        Return PV per unit at S0 by picking the nearest node (fast diagnostic).
        Prefer pricer interpolation for production PV.
        """
        idx = int(np.argmin(np.abs(self.spot_grid - float(self.spot0))))
        return float(self.values_t0_per_unit[idx])