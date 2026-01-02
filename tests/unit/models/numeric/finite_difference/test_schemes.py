from __future__ import annotations

import math
import numpy as np

from src.models.numeric.finite_difference.boundaries import BoundaryPair, DirichletBC
from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid
from src.models.numeric.finite_difference.schemes import ThetaScheme, solve_pde_theta


def test_theta_scheme_preserves_constant_solution_when_operator_is_zero() -> None:
    """
    If a=b=c=0, the PDE is:

        V_t = 0

    With a constant terminal payoff and constant Dirichlet boundaries,
    the solution must remain constant for all times (including at t=0).

    This is a *very* strong invariant test: any unintended operator contribution,
    wrong sign, or stepping bug will typically break it.
    """
    # -----------------------------
    # Build spatial and time grids
    # -----------------------------
    x_grid = SpatialGrid1D.uniform(x_min=-1.0, x_max=+1.0, n=101, name="x")
    t_grid = TimeGrid.uniform(t0=0.0, t1=1.0, n=51, name="t")

    # -----------------------------
    # Define constant terminal payoff
    # -----------------------------
    const_val = 7.25

    def terminal_payoff(x: np.ndarray) -> np.ndarray:
        # Terminal payoff at T is constant everywhere
        return np.full_like(x, const_val, dtype=np.float64)

    # -----------------------------
    # Dirichlet boundaries consistent with the constant solution
    # -----------------------------
    boundaries = BoundaryPair(
        left=DirichletBC(side="left", value=lambda _t: const_val),
        right=DirichletBC(side="right", value=lambda _t: const_val),
    )

    # -----------------------------
    # Zero operator coefficients -> V_t = 0
    # -----------------------------
    a = 0.0  # convection coefficient
    b = 0.0  # diffusion coefficient
    c = 0.0  # reaction/discount coefficient

    # -----------------------------
    # Solve for a couple theta values
    # -----------------------------
    for theta in (0.0, 0.5, 1.0):
        sol = solve_pde_theta(
            x_grid=x_grid,
            t_grid=t_grid,
            terminal_payoff=terminal_payoff,
            a=a,
            b=b,
            c=c,
            boundaries=boundaries,
            scheme=ThetaScheme(theta=theta),
            store_surface=True,  # storing surface lets us check more than just t=0
        )

        # -----------------------------
        # Check t=0 slice is constant
        # -----------------------------
        assert np.allclose(sol.values_t0, const_val, rtol=0.0, atol=1e-14)

        # -----------------------------
        # If surface is stored, check every (t,x) node is constant
        # -----------------------------
        assert sol.surface is not None
        assert np.allclose(sol.surface, const_val, rtol=0.0, atol=1e-14)


def test_theta_scheme_matches_closed_form_solution_for_backward_diffusion_pde() -> None:
    """
    End-to-end PDE regression test for the generic FD engine.

    We choose a PDE with a known closed-form solution that matches the solver's
    backward-stepping convention with a terminal condition at T.

    PDE solved by the engine:
        V_t + b * V_xx = 0     (a=0, c=0, b=nu)

    Closed-form solution on x in [0, 1] with Dirichlet boundaries V(0,t)=V(1,t)=0:
        V(x, t) = exp(-pi^2 * nu * (T - t)) * sin(pi * x)

    Terminal condition at t=T:
        V(x, T) = sin(pi * x)

    Therefore the expected slice at t=0 is:
        V(x, 0) = exp(-pi^2 * nu * T) * sin(pi * x)

    Why this is a good test:
    - It directly checks correctness of time-stepping + operator assembly.
    - It uses nontrivial interior curvature (sin), so V_xx matters.
    - Boundaries are clean (0), avoiding boundary ambiguities.
    """
    # -----------------------------
    # Choose PDE parameters
    # -----------------------------
    nu = 0.40  # diffusion coefficient b in: V_t + b V_xx = 0
    T = 0.50   # terminal time (maturity)

    # -----------------------------
    # Build grids (moderate sizes to keep CI fast but accurate)
    # -----------------------------
    # Spatial domain [0, 1] with uniform spacing.
    x_grid = SpatialGrid1D.uniform(x_min=0.0, x_max=1.0, n=121, name="x")

    # Time grid from 0 to T with uniform spacing.
    t_grid = TimeGrid.uniform(t0=0.0, t1=T, n=241, name="t")

    # -----------------------------
    # Define terminal payoff V(x, T)
    # -----------------------------
    # This is the terminal condition the solver starts from at time T.
    def terminal_payoff(x: np.ndarray) -> np.ndarray:
        # Use sin(pi x) which is 0 at both boundaries (Dirichlet-consistent).
        return np.sin(math.pi * x)

    # -----------------------------
    # Define boundaries V(0,t)=0 and V(1,t)=0
    # -----------------------------
    boundaries = BoundaryPair(
        left=DirichletBC(side="left", value=lambda _t: 0.0),
        right=DirichletBC(side="right", value=lambda _t: 0.0),
    )

    # -----------------------------
    # Solve PDE using Crank–Nicolson (theta=0.5)
    # -----------------------------
    # a=0 (no convection), b=nu (diffusion), c=0 (no discount/decay term).
    solution = solve_pde_theta(
        x_grid=x_grid,
        t_grid=t_grid,
        terminal_payoff=terminal_payoff,
        a=0.0,
        b=nu,
        c=0.0,
        boundaries=boundaries,
        scheme=ThetaScheme(theta=0.5),
        store_surface=False,  # keep memory small for unit tests
    )

    # -----------------------------
    # Compute analytic solution slice at t=0
    # -----------------------------
    # Expected V(x,0) = exp(-pi^2 * nu * T) * sin(pi x)
    x = x_grid.x
    expected_t0 = math.exp(-(math.pi**2) * nu * T) * np.sin(math.pi * x)

    # -----------------------------
    # Compare FD vs analytic
    # -----------------------------
    # Use an error metric that is stable in CI.
    err = np.max(np.abs(solution.values_t0 - expected_t0))

    # This tolerance is conservative for CN on this grid.
    # If you later increase n_x/n_t, you can tighten it.
    assert err < 2.0e-3