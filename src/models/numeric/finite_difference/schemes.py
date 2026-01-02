from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Union

from src.models.numeric.finite_difference.boundaries import BoundaryPair, boundary_values_for_time
from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid
from src.models.numeric.finite_difference.operators import CoeffFn, build_operator_1d
from src.models.numeric.finite_difference.solution import FdSolution
from src.models.numeric.finite_difference.tridiagonal import solve_tridiagonal, solve_tridiagonal_psor


@dataclass(frozen=True, slots=True)
class ThetaScheme:
    """
    Theta-scheme for backward time-stepping.

    theta meanings
    --------------
    theta = 0.0  -> explicit (conditionally stable)
    theta = 0.5  -> Crank–Nicolson (common default)
    theta = 1.0  -> fully implicit (unconditionally stable for diffusion)
    """
    theta: float = 0.5

    def __post_init__(self) -> None:
        th = float(self.theta)
        if not (0.0 <= th <= 1.0):
            raise ValueError("theta must be in [0, 1].")


TerminalPayoffFn = Callable[[np.ndarray], np.ndarray]


def solve_pde_theta(
    *,
    x_grid: SpatialGrid1D,
    t_grid: TimeGrid,
    terminal_payoff: TerminalPayoffFn,
    a: Union[float, np.ndarray, CoeffFn],
    b: Union[float, np.ndarray, CoeffFn],
    c: Union[float, np.ndarray, CoeffFn],
    boundaries: BoundaryPair,
    scheme: ThetaScheme = ThetaScheme(theta=0.5),
    store_surface: bool = False,
) -> FdSolution:
    """
    Solve the 1D PDE backward in time using a theta-scheme:

      V_t + a(x,t) V_x + b(x,t) V_xx - c(x,t) V = 0
      V(x,T) = terminal_payoff(x)

    Returns
    -------
    FdSolution with:
      - values_t0: solution at t=0 for all x nodes (including boundaries)
      - optional surface: V[t_index, x_index] if store_surface=True

    Boundary handling
    -----------------
    - Dirichlet: direct.
    - Neumann: converted to an effective boundary value using a one-sided relation.
      This requires an interior guess. For theta>0 we apply a small fixed-point
      update if Neumann is present (keeps V1 simple but usable).
    """
    theta = float(scheme.theta)

    x = x_grid.x
    t = t_grid.t
    dt = t_grid.dt

    n_x = x_grid.n
    n_t = t_grid.n
    n_int = n_x - 2

    # Allocate current solution vector (includes boundaries).
    V = np.zeros(n_x, dtype=np.float64)

    # Terminal condition at T.
    V[:] = np.asarray(terminal_payoff(x), dtype=np.float64)

    # Optional surface storage.
    surface = None
    if store_surface:
        surface = np.empty((n_t, n_x), dtype=np.float64)
        surface[-1, :] = V

    # Backward stepping: k = n_t-2 ... 0
    for k in range(n_t - 2, -1, -1):
        t_n = float(t[k])         # target time
        t_np1 = float(t[k + 1])   # known time
        dt_k = float(dt[k])

        # Interior unknowns at t_{n+1} (known).
        V_np1_int = V[1:-1].copy()

        # Boundary values at t_{n+1} (use known interior for Neumann conversion).
        vL_np1, vR_np1 = boundary_values_for_time(x_grid, boundaries, t_np1, interior_guess=V_np1_int)

        # Build operators at t_n and t_{n+1}.
        L_n = build_operator_1d(x_grid, a=a, b=b, c=c, t=t_n)
        L_np1 = build_operator_1d(x_grid, a=a, b=b, c=c, t=t_np1)

        # ------------------------------------------------------------------
        # RHS = (I + (1-theta) dt L_{n+1}) V^{n+1}   including boundary terms
        # ------------------------------------------------------------------
        rhs = V_np1_int.copy()

        if (1.0 - theta) != 0.0:
            w = (1.0 - theta) * dt_k

            # Main diagonal contribution.
            rhs += w * (L_np1.diag * V_np1_int)

            # Off-diagonal interior-interior contributions.
            rhs[1:] += w * (L_np1.lower * V_np1_int[:-1])
            rhs[:-1] += w * (L_np1.upper * V_np1_int[1:])

            # Boundary contributions (CRITICAL FIX: use left_bc/right_bc).
            rhs[0] += w * (L_np1.left_bc * vL_np1)
            rhs[-1] += w * (L_np1.right_bc * vR_np1)

        # ------------------------------------------------------------------
        # Solve (I - theta dt L_n) V^n_int = rhs + theta dt * (boundary terms at t_n)
        # ------------------------------------------------------------------
        if theta == 0.0:
            # Pure explicit scheme: V^n_int is rhs already.
            V_n_int = rhs
        else:
            # Build tridiagonal matrix A = I - theta dt L_n for interior unknowns.
            w = theta * dt_k
            A_diag = (1.0 - w * L_n.diag).copy()
            A_lower = (-w * L_n.lower).copy()
            A_upper = (-w * L_n.upper).copy()

            # For Dirichlet boundaries, vL_n/vR_n are known.
            # For Neumann, boundary values depend on interior unknowns.
            # V1 approach: small fixed-point update (1–2 iterations) if Neumann exists.
            guess_int = V_np1_int.copy()

            def _solve_with_boundaries(guess: np.ndarray) -> np.ndarray:
                vL_n, vR_n = boundary_values_for_time(x_grid, boundaries, t_n, interior_guess=guess)

                # Add implicit boundary contribution to RHS (CRITICAL FIX).
                rhs_eff = rhs.copy()
                rhs_eff[0] += w * (L_n.left_bc * vL_n)
                rhs_eff[-1] += w * (L_n.right_bc * vR_n)

                # Solve for interior values.
                return solve_tridiagonal(A_lower, A_diag, A_upper, rhs_eff)

            # First solve using initial guess.
            V_n_int = _solve_with_boundaries(guess_int)

            # If Neumann BC present, do one refinement solve (cheap, improves consistency).
            if boundaries.has_neumann():
                V_n_int = _solve_with_boundaries(V_n_int)

        # ------------------------------------------------------------------
        # Write back full solution vector at t_n (including boundaries)
        # ------------------------------------------------------------------
        V[1:-1] = V_n_int

        # Recompute boundaries at t_n using the updated interior (especially important for Neumann).
        vL_n, vR_n = boundary_values_for_time(x_grid, boundaries, t_n, interior_guess=V_n_int)
        V[0] = vL_n
        V[-1] = vR_n

        if store_surface and surface is not None:
            surface[k, :] = V

    return FdSolution(
        x_grid=x_grid,
        t_grid=t_grid,
        values_t0=V.copy(),
        surface=surface,
    )


def solve_pde_theta_american(
    *,
    x_grid: SpatialGrid1D,
    t_grid: TimeGrid,
    terminal_payoff: TerminalPayoffFn,
    intrinsic_payoff: TerminalPayoffFn,
    a: Union[float, np.ndarray, CoeffFn],
    b: Union[float, np.ndarray, CoeffFn],
    c: Union[float, np.ndarray, CoeffFn],
    boundaries: BoundaryPair,
    scheme: ThetaScheme = ThetaScheme(theta=0.5),
    store_surface: bool = False,
    psor_omega: float = 1.2,
    psor_tol: float = 1e-10,
    psor_max_iter: int = 50_000,
) -> FdSolution:
    """
    American-style PDE solve using theta-scheme + PSOR constraint enforcement.

    PDE form:
      V_t + a(x,t) V_x + b(x,t) V_xx - c(x,t) V = 0

    Constraint (early exercise):
      V(x,t) >= intrinsic_payoff(x)

    Implementation outline
    ----------------------
    1) Terminal condition at T is terminal_payoff(x).
    2) At each backward step, solve:
         A V^n = rhs
       but enforce V^n >= intrinsic via PSOR on the tridiagonal system.

    Notes
    -----
    - This is the standard method used in FD pricing of American options.
    - For robust V1 behaviour, use theta=1.0 (implicit) or theta=0.5 (CN).
    """
    x_grid.validate()
    t_grid.validate()
    theta = float(scheme.theta)

    x = x_grid.x
    t = t_grid.t
    dt = t_grid.dt

    n_x = x_grid.n
    n_t = t_grid.n

    # Full grid values (including boundaries).
    V = np.zeros(n_x, dtype=np.float64)

    # Terminal condition at expiry.
    V[:] = np.asarray(terminal_payoff(x), dtype=np.float64)

    # Intrinsic payoff floor (time-independent for vanilla).
    intrinsic_all = np.asarray(intrinsic_payoff(x), dtype=np.float64)
    intrinsic_int = intrinsic_all[1:-1].copy()

    surface = None
    if store_surface:
        surface = np.empty((n_t, n_x), dtype=np.float64)
        surface[-1, :] = V

    for k in range(n_t - 2, -1, -1):
        t_n = float(t[k])
        t_np1 = float(t[k + 1])
        dt_k = float(dt[k])

        # Known interior at next time (starting guess for PSOR).
        V_np1_int = V[1:-1].copy()

        # Boundary values at t_{n+1}.
        vL_np1, vR_np1 = boundary_values_for_time(x_grid, boundaries, t_np1, interior_guess=V_np1_int)

        L_n = build_operator_1d(x_grid, a=a, b=b, c=c, t=t_n)
        L_np1 = build_operator_1d(x_grid, a=a, b=b, c=c, t=t_np1)

        rhs = V_np1_int.copy()

        # Explicit component (I + (1-theta) dt L_{n+1}) V^{n+1}
        if (1.0 - theta) != 0.0:
            rhs += (1.0 - theta) * dt_k * (L_np1.diag * V_np1_int)
            rhs[1:] += (1.0 - theta) * dt_k * (L_np1.lower * V_np1_int[:-1])
            rhs[:-1] += (1.0 - theta) * dt_k * (L_np1.upper * V_np1_int[1:])

            rhs[0] += (1.0 - theta) * dt_k * (L_np1.lower[0] * vL_np1)
            rhs[-1] += (1.0 - theta) * dt_k * (L_np1.upper[-1] * vR_np1)

        if theta == 0.0:
            # Explicit step + projection (not recommended for production, but supported).
            V_n_int = np.maximum(rhs, intrinsic_int)
        else:
            # Build A = I - theta dt L_n
            A_diag = (1.0 - theta * dt_k * L_n.diag).copy()
            A_lower = (-theta * dt_k * L_n.lower).copy()
            A_upper = (-theta * dt_k * L_n.upper).copy()

            # Boundary values at t_n (Dirichlet typical; Neumann supported via interior_guess).
            vL_n, vR_n = boundary_values_for_time(x_grid, boundaries, t_n, interior_guess=V_np1_int)

            # Move boundary terms to RHS.
            rhs[0] -= A_lower[0] * vL_n
            rhs[-1] -= A_upper[-1] * vR_n

            # PSOR solve enforcing V >= intrinsic.
            V_n_int = solve_tridiagonal_psor(
                A_lower,
                A_diag,
                A_upper,
                rhs,
                payoff_floor=intrinsic_int,
                x0=V_np1_int,
                omega=psor_omega,
                tol=psor_tol,
                max_iter=psor_max_iter,
            )

        # Write back and update boundaries.
        V[1:-1] = V_n_int

        vL_n, vR_n = boundary_values_for_time(x_grid, boundaries, t_n, interior_guess=V_n_int)
        V[0] = vL_n
        V[-1] = vR_n

        if store_surface and surface is not None:
            surface[k, :] = V

    return FdSolution(x_grid=x_grid, t_grid=t_grid, values_t0=V.copy(), surface=surface)