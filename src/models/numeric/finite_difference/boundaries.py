from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

from src.models.numeric.finite_difference.grids import SpatialGrid1D

BoundarySide = Literal["left", "right"]
ValueFn = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class DirichletBC:
    """
    Dirichlet boundary condition:
      V(x_boundary, t) = g(t)
    """
    side: BoundarySide
    value: ValueFn  # g(t)

    def eval(self, t: float) -> float:
        return float(self.value(float(t)))


@dataclass(frozen=True, slots=True)
class NeumannBC:
    """
    Neumann boundary condition:
      V_x(x_boundary, t) = q(t)

    V1 enforcement (simple, first-order)
    ------------------------------------
    We convert Neumann into an *effective Dirichlet* boundary value using a
    one-sided first-order relation:

      Left:  (V1 - V0)/dx = q  =>  V0 = V1 - q*dx
      Right: (VN-1 - VN-2)/dx = q => VN-1 = VN-2 + q*dx

    This requires an interior "guess" value (V1 or VN-2).
    """
    side: BoundarySide
    derivative: ValueFn  # q(t)

    def eval(self, t: float) -> float:
        return float(self.derivative(float(t)))


@dataclass(frozen=True, slots=True)
class BoundaryPair:
    """
    Convenience container holding left and right boundary conditions.
    """
    left: DirichletBC | NeumannBC
    right: DirichletBC | NeumannBC

    def has_neumann(self) -> bool:
        """True if either boundary is Neumann (requires interior_guess)."""
        return isinstance(self.left, NeumannBC) or isinstance(self.right, NeumannBC)


def boundary_values_for_time(
    grid: SpatialGrid1D,
    bc: BoundaryPair,
    t: float,
    *,
    interior_guess: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute (V_left, V_right) at time t.

    - Dirichlet: direct evaluation.
    - Neumann: converts derivative to a boundary value using the current interior_guess.

    Parameters
    ----------
    interior_guess:
        Interior values (shape (n_int,)) used to infer boundary values for Neumann BCs.

    Returns
    -------
    (v_left, v_right)
        Boundary values at the current time.
    """
    dx = grid.dx
    n_int = grid.n - 2

    if bc.has_neumann():
        if interior_guess is None:
            raise ValueError("Neumann BC requires interior_guess to infer boundary values.")
        interior_guess = np.asarray(interior_guess, dtype=np.float64)
        if interior_guess.shape != (n_int,):
            raise ValueError(f"interior_guess must have shape ({n_int},)")

    # Left boundary value.
    if isinstance(bc.left, DirichletBC):
        v_left = bc.left.eval(t)
    else:
        # V0 = V1 - q*dx, where V1 is the first interior node.
        q = bc.left.eval(t)
        v1 = float(interior_guess[0])
        v_left = v1 - q * dx

    # Right boundary value.
    if isinstance(bc.right, DirichletBC):
        v_right = bc.right.eval(t)
    else:
        # VN-1 = VN-2 + q*dx, where VN-2 is the last interior node.
        q = bc.right.eval(t)
        v_nm2 = float(interior_guess[-1])
        v_right = v_nm2 + q * dx

    return float(v_left), float(v_right)