from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpatialGrid1D:
    """
    1D spatial grid for finite-difference PDEs.

    Key conventions
    ---------------
    - `x` is the coordinate array used by the PDE engine (S, logS, r, etc.).
    - V1 assumes a *uniform* grid spacing for standard central-difference stencils.

    Notes
    -----
    - If `is_log_space=True`, `x` is interpreted as log-space coordinates.
      The product adapter (e.g., FX PDE pricer) is responsible for mapping PDE
      coefficients correctly for that coordinate choice.
    """
    x: np.ndarray
    is_log_space: bool = False
    name: str = "x"

    def __post_init__(self) -> None:
        # Validate immediately so hot-path code doesn't repeatedly re-check grids.
        self.validate()

    @property
    def n(self) -> int:
        """Number of grid nodes."""
        return int(self.x.size)

    @property
    def dx(self) -> float:
        """Uniform grid spacing."""
        return float(self.x[1] - self.x[0])

    def interior_x(self) -> np.ndarray:
        """Return interior coordinates x[1:-1] (excluding boundaries)."""
        return self.x[1:-1]

    def linear_space_values(self) -> np.ndarray:
        """
        Return the spatial coordinate in *linear* space:
        - exp(x) if log-space
        - x if linear-space
        """
        if self.is_log_space:
            return np.exp(self.x)
        return self.x.copy()

    def validate(self, *, tol: float = 1e-12) -> None:
        """
        Validate grid assumptions used by V1 stencils (uniform, strictly increasing).

        Parameters
        ----------
        tol:
            Tolerance for uniform spacing check.
        """
        if self.n < 3:
            raise ValueError("SpatialGrid1D requires at least 3 nodes (needs interior points).")

        # Ensure x is a 1D array of finite float64 values.
        x = np.asarray(self.x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("SpatialGrid1D.x must be a 1D array.")
        if not np.all(np.isfinite(x)):
            raise ValueError("SpatialGrid1D.x contains non-finite values.")

        # Strictly increasing.
        dxs = np.diff(x)
        if not np.all(dxs > 0.0):
            raise ValueError("Spatial grid must be strictly increasing.")

        # Uniform spacing (V1 assumption).
        if float(np.max(np.abs(dxs - dxs[0]))) > float(tol):
            raise ValueError("Spatial grid must be uniform in V1 (constant dx).")

        # Store canonical float64 array (defensive: keep consistent dtype/layout).
        object.__setattr__(self, "x", x)

    @staticmethod
    def uniform(*, x_min: float, x_max: float, n: int, name: str = "x") -> "SpatialGrid1D":
        """
        Create a uniform linear grid between [x_min, x_max] with n nodes.
        """
        if int(n) < 3:
            raise ValueError("n must be >= 3.")
        x = np.linspace(float(x_min), float(x_max), int(n), dtype=np.float64)
        return SpatialGrid1D(x=x, is_log_space=False, name=name)

    @staticmethod
    def log_uniform(*, x_min: float, x_max: float, n: int, name: str = "log_x") -> "SpatialGrid1D":
        """
        Create a uniform grid in log-space: y = log(x).

        Parameters
        ----------
        x_min, x_max:
            Bounds in *linear* space (must be positive).
        n:
            Number of grid nodes (>=3).
        """
        if float(x_min) <= 0.0 or float(x_max) <= 0.0:
            raise ValueError("log_uniform requires x_min>0 and x_max>0.")
        if int(n) < 3:
            raise ValueError("n must be >= 3.")

        y = np.linspace(np.log(float(x_min)), np.log(float(x_max)), int(n), dtype=np.float64)
        return SpatialGrid1D(x=y, is_log_space=True, name=name)


@dataclass(frozen=True, slots=True)
class TimeGrid:
    """
    1D time grid for backward PDE stepping.

    Conventions
    -----------
    - The solver expects time nodes increasing: t[0] >= 0, t[-1] = T.
    - The solver steps backward from T -> 0, but the grid is stored forward.
    """
    t: np.ndarray
    name: str = "t"

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n(self) -> int:
        """Number of time nodes."""
        return int(self.t.size)

    @property
    def dt(self) -> np.ndarray:
        """Per-step dt array: dt[k] = t[k+1] - t[k]."""
        return np.diff(self.t).astype(np.float64, copy=False)

    def validate(self) -> None:
        """
        Validate time grid monotonicity and finiteness.
        """
        if self.n < 2:
            raise ValueError("TimeGrid requires at least 2 nodes.")

        t = np.asarray(self.t, dtype=np.float64)
        if t.ndim != 1:
            raise ValueError("TimeGrid.t must be a 1D array.")
        if not np.all(np.isfinite(t)):
            raise ValueError("Time grid contains non-finite values.")
        if float(t[0]) < 0.0:
            raise ValueError("Time grid must start at t>=0.")
        if not np.all(np.diff(t) > 0.0):
            raise ValueError("Time grid must be strictly increasing.")

        object.__setattr__(self, "t", t)

    @staticmethod
    def uniform(*, t0: float, t1: float, n: int, name: str = "t") -> "TimeGrid":
        """
        Create a uniform time grid between [t0, t1] with n nodes.
        """
        if int(n) < 2:
            raise ValueError("n must be >= 2.")
        t = np.linspace(float(t0), float(t1), int(n), dtype=np.float64)
        return TimeGrid(t=t, name=name)