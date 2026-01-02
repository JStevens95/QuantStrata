from __future__ import annotations

import pytest
import numpy as np

from src.models.numeric.finite_difference.boundaries import BoundaryPair, DirichletBC
from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid
from src.models.numeric.finite_difference.schemes import ThetaScheme, solve_pde_theta


def test_dirichlet_boundaries_are_enforced_at_t0() -> None:
    """
    Ensure that Dirichlet boundary conditions are respected by the solver.

    We don’t require a particular closed-form interior solution here; we only assert
    the boundary values are exactly what the boundary functions prescribe at t=0.

    This catches:
      - boundary update omissions
      - wrong boundary side wiring
      - stepping direction mistakes (T->0 vs 0->T)
    """
    # -----------------------------
    # Grids
    # -----------------------------
    x_grid = SpatialGrid1D.uniform(x_min=0.0, x_max=1.0, n=101, name="x")
    t_grid = TimeGrid.uniform(t0=0.0, t1=2.0, n=41, name="t")

    # -----------------------------
    # Time-dependent Dirichlet boundaries
    # -----------------------------
    # Left boundary increases with time; right decreases.
    gL = lambda t: 1.0 + 0.1 * float(t)
    gR = lambda t: 2.0 - 0.2 * float(t)

    boundaries = BoundaryPair(
        left=DirichletBC(side="left", value=gL),
        right=DirichletBC(side="right", value=gR),
    )

    # -----------------------------
    # Terminal payoff at T (arbitrary but finite and smooth-ish)
    # -----------------------------
    def terminal_payoff(x: np.ndarray) -> np.ndarray:
        # Something non-trivial in the interior to ensure boundary influence exists
        return 0.5 + x * (1.0 - x)

    # -----------------------------
    # Choose a simple diffusion PDE:
    #   V_t + b V_xx = 0
    # which is stable with theta >= 0.5
    # -----------------------------
    a = 0.0
    b = 0.25
    c = 0.0

    sol = solve_pde_theta(
        x_grid=x_grid,
        t_grid=t_grid,
        terminal_payoff=terminal_payoff,
        a=a,
        b=b,
        c=c,
        boundaries=boundaries,
        scheme=ThetaScheme(theta=0.5),
        store_surface=False,
    )

    # -----------------------------
    # At t=0, the boundary nodes should match gL(0) and gR(0)
    # -----------------------------
    assert float(sol.values_t0[0]) == pytest.approx(float(gL(0.0)), rel=0.0, abs=1e-14)
    assert float(sol.values_t0[-1]) == pytest.approx(float(gR(0.0)), rel=0.0, abs=1e-14)


def test_boundary_coupling_terms_influence_first_and_last_interior_nodes() -> None:
    """
    Regression test for the boundary-coupling bug.

    What this test catches
    ----------------------
    The original bug pattern is:
      - building L on interior nodes only,
      - but then mistakenly using L.lower[0] / L.upper[-1] as boundary coefficients,
        which is wrong because those arrays *exclude* the boundary-coupled entries.

    This test forces the boundaries to be nonzero at intermediate times so that:
      - the first and last interior nodes MUST be pushed away from zero.

    Setup
    -----
    PDE:
        V_t + b V_xx = 0  (a=0, c=0)
    Terminal:
        V(x,T) = 0 everywhere (so without boundary influence, interior remains 0)
    Boundaries:
        V(left,t)  = 1
        V(right,t) = 2
    Expectation:
        At t=0, interior near boundaries must be > 0 due to diffusion of boundary values.
    """
    # -----------------------------
    # Grid + parameters
    # -----------------------------
    T = 0.10
    b = 1.0

    # Small grids are enough to expose the bug and keep the test very fast.
    x_grid = SpatialGrid1D.uniform(x_min=0.0, x_max=1.0, n=41, name="x")
    t_grid = TimeGrid.uniform(t0=0.0, t1=T, n=51, name="t")

    # -----------------------------
    # Terminal payoff: identically zero
    # -----------------------------
    def terminal_payoff(x: np.ndarray) -> np.ndarray:  # noqa: ARG001
        # Start from zero everywhere at t=T.
        return np.zeros_like(x, dtype=np.float64)

    # -----------------------------
    # Nonzero Dirichlet boundaries
    # -----------------------------
    boundaries = BoundaryPair(
        left=DirichletBC(side="left", value=lambda _t: 1.0),
        right=DirichletBC(side="right", value=lambda _t: 2.0),
    )

    # -----------------------------
    # Solve with fully implicit scheme (very stable)
    # -----------------------------
    sol = solve_pde_theta(
        x_grid=x_grid,
        t_grid=t_grid,
        terminal_payoff=terminal_payoff,
        a=0.0,
        b=b,
        c=0.0,
        boundaries=boundaries,
        scheme=ThetaScheme(theta=1.0),
        store_surface=False,
    )

    # -----------------------------
    # Validate interior is influenced by boundaries
    # -----------------------------
    v0 = sol.values_t0

    # First interior node (index 1) should move toward the left boundary value.
    # If boundary coupling is broken, this tends to remain ~0.
    assert float(v0[1]) > 1e-6

    # Last interior node (index -2) should move toward the right boundary value.
    assert float(v0[-2]) > 1e-6

    # Also sanity: solution should be between boundary min/max for pure diffusion in this setup.
    assert float(v0[1]) < 2.0 + 1e-12
    assert float(v0[-2]) < 2.0 + 1e-12