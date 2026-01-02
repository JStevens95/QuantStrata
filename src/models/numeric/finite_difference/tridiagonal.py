from __future__ import annotations

import numpy as np


def solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve a tridiagonal linear system A x = rhs using the Thomas algorithm.

    A is defined by:
      - lower: sub-diagonal (length n-1)
      - diag : main diagonal (length n)
      - upper: super-diagonal (length n-1)

    rhs can be:
      - shape (n,) for a single RHS
      - shape (n, m) for multiple RHS columns (solved in-place columnwise)

    Returns
    -------
    x : ndarray with same shape as rhs

    Notes
    -----
    - This function allocates working buffers but remains O(n).
    - We use float64 to reduce numerical issues in PDE solves.
    """
    lower = np.asarray(lower, dtype=np.float64)
    diag = np.asarray(diag, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)

    n = int(diag.size)
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValueError("Invalid tridiagonal sizes.")
    if rhs.shape[0] != n:
        raise ValueError("rhs first dimension must match diag length.")

    # Copy to avoid mutating user arrays (production-safe behaviour).
    c_prime = np.empty(n - 1, dtype=np.float64)
    d_prime = rhs.copy()

    # Forward elimination
    denom0 = diag[0]
    if denom0 == 0.0:
        raise ValueError("Zero pivot encountered in tridiagonal solver.")
    c_prime[0] = upper[0] / denom0
    d_prime[0] = d_prime[0] / denom0

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if denom == 0.0:
            raise ValueError("Zero pivot encountered in tridiagonal solver.")

        if i < n - 1:
            c_prime[i] = upper[i] / denom

        d_prime[i] = (d_prime[i] - lower[i - 1] * d_prime[i - 1]) / denom

    # Back substitution
    x = d_prime
    for i in range(n - 2, -1, -1):
        x[i] = x[i] - c_prime[i] * x[i + 1]

    return x


def solve_tridiagonal_psor(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    payoff_floor: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    omega: float = 1.2,
    max_iter: int = 50_000,
    tol: float = 1e-10,
    pivot_eps: float = 1e-14,
) -> np.ndarray:
    """
    Solve the tridiagonal system A x = rhs under an inequality constraint:
        x >= payoff_floor
    using Projected Successive Over-Relaxation (PSOR).

    This is the standard workhorse for American-style early exercise in FD/PDE.

    Parameters
    ----------
    lower, diag, upper:
        Tridiagonal coefficients defining A (lower/upper length n-1, diag length n).
    rhs:
        Right-hand side vector (shape (n,)).
    payoff_floor:
        Constraint vector (shape (n,)) representing intrinsic value at this time slice.
    x0:
        Initial guess. If provided, should be shape (n,). In PDE stepping, using the
        previous time slice is typically excellent.
    omega:
        Relaxation parameter in (0, 2). 1.0 is Gauss-Seidel; 1.1-1.5 often speeds up.
    max_iter:
        Maximum PSOR iterations.
    tol:
        Convergence tolerance on max-norm of iterate difference.
    pivot_eps:
        Small threshold to treat a pivot as numerically zero.

    Returns
    -------
    x:
        Solution satisfying x >= payoff_floor.

    Notes
    -----
    - Assumes A is suitable for SOR convergence (typical for implicit FD matrices).
    - O(n * iters). For vanilla grids (n~400) this is fast in practice.
    """
    lower = np.asarray(lower, dtype=np.float64)
    diag = np.asarray(diag, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    payoff_floor = np.asarray(payoff_floor, dtype=np.float64)

    n = int(diag.size)
    if lower.shape != (n - 1,) or upper.shape != (n - 1,):
        raise ValueError("Invalid tridiagonal sizes: lower/upper must be length n-1.")
    if rhs.shape != (n,):
        raise ValueError("rhs must have shape (n,).")
    if payoff_floor.shape != (n,):
        raise ValueError("payoff_floor must have shape (n,).")

    w = float(omega)
    if not (0.0 < w < 2.0):
        raise ValueError("omega must be in (0, 2).")

    if np.any(np.abs(diag) < pivot_eps):
        raise ValueError("Near-zero pivot encountered in PSOR solver (diag too small).")

    # Initial guess: use provided, else a safe floor-respecting guess.
    if x0 is None:
        x = np.maximum(payoff_floor, rhs / diag)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()
        if x.shape != (n,):
            raise ValueError("x0 must have shape (n,).")
        # Ensure feasibility from the start.
        x = np.maximum(x, payoff_floor)

    x_old = x.copy()

    for _ in range(int(max_iter)):
        x_old[:] = x

        # Gauss-Seidel sweep with projection
        for i in range(n):
            left = lower[i - 1] * x[i - 1] if i > 0 else 0.0
            right = upper[i] * x_old[i + 1] if i < n - 1 else 0.0

            # Unrelaxed GS update
            x_gs = (rhs[i] - left - right) / diag[i]

            # SOR relaxation
            x_i = (1.0 - w) * x_old[i] + w * x_gs

            # Project onto constraint set (early exercise)
            if x_i < payoff_floor[i]:
                x_i = payoff_floor[i]

            x[i] = x_i

        # Convergence check (max norm)
        err = float(np.max(np.abs(x - x_old)))
        if err < tol:
            break

    return x