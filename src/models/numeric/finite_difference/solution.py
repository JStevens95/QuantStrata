from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid


@dataclass(frozen=True, slots=True)
class FdSolution:
    """
    Container for a 1D finite-difference PDE solution.

    Stores
    ------
    - `values_t0`: the solution slice at t=0 across all x nodes (including boundaries).
    - `surface` (optional): full surface V[t_index, x_index] if requested.

    Interpolation
    -------------
    `value_at(x0)` linearly interpolates on the t=0 slice.
    The caller must provide x0 in the same coordinate system as the grid
    (linear x or log-space x depending on x_grid.is_log_space).
    """
    x_grid: SpatialGrid1D
    t_grid: TimeGrid
    values_t0: np.ndarray
    surface: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        v0 = np.asarray(self.values_t0, dtype=np.float64)
        if v0.shape != (self.x_grid.n,):
            raise ValueError("values_t0 must have shape (n_x,).")
        object.__setattr__(self, "values_t0", v0)

        if self.surface is not None:
            surf = np.asarray(self.surface, dtype=np.float64)
            if surf.shape != (self.t_grid.n, self.x_grid.n):
                raise ValueError("surface must have shape (n_t, n_x).")
            object.__setattr__(self, "surface", surf)

    def value_at(self, x0: float) -> float:
        """
        Linear interpolation of V(t=0, x=x0), with clamping to grid bounds.
        """
        x = self.x_grid.x
        v = self.values_t0
        x0 = float(x0)

        # Clamp to grid to avoid extrapolation surprises in risk runs.
        if x0 <= float(x[0]):
            return float(v[0])
        if x0 >= float(x[-1]):
            return float(v[-1])

        # Find left index j such that x[j] <= x0 < x[j+1].
        j = int(np.searchsorted(x, x0, side="right") - 1)

        xL = float(x[j])
        xR = float(x[j + 1])
        vL = float(v[j])
        vR = float(v[j + 1])

        # Linear interpolation weight.
        w = (x0 - xL) / (xR - xL)
        return float((1.0 - w) * vL + w * vR)