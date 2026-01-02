from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Union

from src.models.numeric.finite_difference.grids import SpatialGrid1D

ArrayLike = Union[np.ndarray, float]
CoeffFn = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True, slots=True)
class Operator1D:
    """
    Tridiagonal representation of the interior spatial operator L, plus boundary coupling.

    PDE form (generic)
    ------------------
      V_t + a(x,t) V_x + b(x,t) V_xx - c(x,t) V = 0

    Central differences (uniform dx)
    --------------------------------
      V_x  ≈ (V_{i+1} - V_{i-1}) / (2 dx)
      V_xx ≈ (V_{i+1} - 2V_i + V_{i-1}) / (dx^2)

    We construct L on interior unknowns V[1:-1] such that:
      (L V)_i = b V_xx + a V_x - c V

    Boundary coupling (IMPORTANT)
    -----------------------------
    When acting on the full grid, the first interior equation (node 1) has a term
    involving V_left (node 0), and the last interior equation (node N-2) has a term
    involving V_right (node N-1).

    We store those coefficients explicitly:
      left_bc  = coefficient multiplying V_left  in row 0 (interior index 0)
      right_bc = coefficient multiplying V_right in row -1 (interior index -1)
    """
    lower: np.ndarray   # (n_int-1,) sub-diagonal for interior-interior coupling
    diag: np.ndarray    # (n_int,)   main diagonal
    upper: np.ndarray   # (n_int-1,) super-diagonal for interior-interior coupling
    left_bc: float      # coefficient multiplying V_left in first interior equation
    right_bc: float     # coefficient multiplying V_right in last interior equation


def _eval_coeff(
    coeff: ArrayLike | CoeffFn,
    *,
    x_full: np.ndarray,
    x_int: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Evaluate a coefficient on the interior nodes.

    Supports:
    - scalar -> broadcast to interior
    - callable f(x, t) -> evaluated on interior x_int
    - array:
        * shape (n_full,) -> sliced to interior
        * shape (n_int,)  -> used as-is
    """
    if callable(coeff):
        out = coeff(x_int, float(t))
        arr = np.asarray(out, dtype=np.float64)
        if arr.shape != x_int.shape:
            raise ValueError("Coefficient function must return array of shape (n_int,).")
        return arr

    if np.isscalar(coeff):
        return np.full_like(x_int, float(coeff), dtype=np.float64)

    arr = np.asarray(coeff, dtype=np.float64)
    if arr.shape == x_full.shape:
        return arr[1:-1].copy()
    if arr.shape == x_int.shape:
        return arr.copy()

    raise ValueError("Coefficient array must have shape (n_full,) or (n_int,) matching the grid.")


def build_operator_1d(
    grid: SpatialGrid1D,
    *,
    a: ArrayLike | CoeffFn,
    b: ArrayLike | CoeffFn,
    c: ArrayLike | CoeffFn,
    t: float,
) -> Operator1D:
    """
    Build the interior tridiagonal operator L and explicit boundary-coupling coefficients.

    Parameters
    ----------
    grid:
        Spatial grid (validated at construction).
    a, b, c:
        PDE coefficients in:
          V_t + a V_x + b V_xx - c V = 0
    t:
        Time at which to evaluate time-dependent coefficients.

    Returns
    -------
    Operator1D
        Tridiagonal arrays for interior unknowns plus boundary coupling (left_bc/right_bc).
    """
    x_full = grid.x
    x_int = grid.interior_x()
    dx = grid.dx

    if x_int.size < 1:
        raise ValueError("Grid too small: no interior points.")

    # Evaluate coefficients on interior nodes.
    a_i = _eval_coeff(a, x_full=x_full, x_int=x_int, t=t)
    b_i = _eval_coeff(b, x_full=x_full, x_int=x_int, t=t)
    c_i = _eval_coeff(c, x_full=x_full, x_int=x_int, t=t)

    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx

    # Full stencil coefficients per interior row (length n_int):
    #   lower_full[i] multiplies V_{i-1} (which is boundary for i=0)
    #   upper_full[i] multiplies V_{i+1} (which is boundary for i=n_int-1)
    lower_full = (b_i * inv_dx2) - (a_i * 0.5 * inv_dx)
    diag = (-2.0 * b_i * inv_dx2) - c_i
    upper_full = (b_i * inv_dx2) + (a_i * 0.5 * inv_dx)

    # Extract boundary coupling terms (these were the original bug source).
    left_bc = float(lower_full[0])
    right_bc = float(upper_full[-1])

    # Interior-interior tridiagonal arrays:
    # - lower is length n_int-1 (couples interior i with i-1, excluding boundary)
    # - upper is length n_int-1 (couples interior i with i+1, excluding boundary)
    return Operator1D(
        lower=lower_full[1:].copy(),
        diag=diag.copy(),
        upper=upper_full[:-1].copy(),
        left_bc=left_bc,
        right_bc=right_bc,
    )