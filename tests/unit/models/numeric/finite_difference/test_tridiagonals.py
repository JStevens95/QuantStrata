from __future__ import annotations

import pytest
import numpy as np

from src.models.numeric.finite_difference.tridiagonal import solve_tridiagonal


def _build_diagonally_dominant_tridiagonal(n: int, seed: int = 0):
    """
    Build a random *strictly* diagonally dominant tridiagonal system so:
      - Thomas solver has stable pivots
      - numpy solve is well-conditioned
    """
    rng = np.random.default_rng(seed)

    # Random off-diagonals (sub/super)
    lower = rng.uniform(-0.5, 0.5, size=n - 1).astype(np.float64)
    upper = rng.uniform(-0.5, 0.5, size=n - 1).astype(np.float64)

    # Make diagonal dominant: |diag_i| > |lower_i| + |upper_i| + margin
    diag = rng.uniform(1.0, 2.0, size=n).astype(np.float64)
    diag[0] += abs(upper[0]) + 1.0
    diag[-1] += abs(lower[-1]) + 1.0
    for i in range(1, n - 1):
        diag[i] += abs(lower[i - 1]) + abs(upper[i]) + 1.0

    return lower, diag, upper


def _to_full_matrix(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Convert tridiagonal representation into a full dense matrix (for test verification)."""
    n = int(diag.size)
    A = np.zeros((n, n), dtype=np.float64)

    # Fill main diagonal
    np.fill_diagonal(A, diag)

    # Fill sub-diagonal (lower)
    for i in range(n - 1):
        A[i + 1, i] = lower[i]

    # Fill super-diagonal (upper)
    for i in range(n - 1):
        A[i, i + 1] = upper[i]

    return A


@pytest.mark.parametrize("n", [3, 8, 25, 100])
def test_solve_tridiagonal_matches_numpy_solve(n: int) -> None:
    """
    Thomas solver should match numpy.linalg.solve on well-conditioned tridiagonal systems.
    """
    lower, diag, upper = _build_diagonally_dominant_tridiagonal(n=n, seed=42)

    # Build a random RHS vector
    rng = np.random.default_rng(123)
    rhs = rng.normal(size=n).astype(np.float64)

    # Solve using our Thomas solver
    x_thomas = solve_tridiagonal(lower=lower, diag=diag, upper=upper, rhs=rhs)

    # Solve using dense numpy solve (test baseline)
    A = _to_full_matrix(lower, diag, upper)
    x_np = np.linalg.solve(A, rhs)

    # They should be extremely close for diagonally-dominant systems
    assert x_thomas == pytest.approx(x_np, rel=1e-12, abs=1e-12)


def test_tridiagonal_solver_matches_dense_solve_for_random_system() -> None:
    """
    Sanity test for the Thomas tridiagonal solver.

    We build a strictly diagonally dominant tridiagonal matrix A (size n x n),
    solve A x = rhs using:
      1) our Thomas solver (tridiagonal)
      2) numpy.linalg.solve (dense)
    and compare results.

    This test protects against indexing mistakes in forward elimination /
    back substitution, and also checks our input-shape conventions.
    """
    # -----------------------------
    # Deterministic random generator
    # -----------------------------
    rng = np.random.default_rng(123)

    # -----------------------------
    # Problem size (moderate so CI is fast but meaningful)
    # -----------------------------
    n = 50

    # -----------------------------
    # Random sub- and super-diagonals (length n-1)
    # -----------------------------
    lower = rng.normal(size=n - 1)  # sub-diagonal: A[i, i-1] for i=1..n-1
    upper = rng.normal(size=n - 1)  # super-diagonal: A[i, i+1] for i=0..n-2

    # -----------------------------
    # Build a MAIN diagonal of length n (IMPORTANT!)
    #
    # For strict diagonal dominance we need:
    #   |diag[i]| > |lower[i-1]| + |upper[i]|   for i=1..n-2
    #   |diag[0]| > |upper[0]|
    #   |diag[n-1]| > |lower[n-2]|
    #
    # Note the alignment:
    #   diag[1:-1] couples with lower[0:n-2] and upper[1:n-1]
    # -----------------------------
    diag = 2.0 + rng.random(n)  # start with a positive baseline (length n)

    # First row dominance: only has an upper neighbor
    diag[0] += abs(upper[0]) + 1e-3

    # Last row dominance: only has a lower neighbor
    diag[-1] += abs(lower[-1]) + 1e-3

    # Interior rows dominance: has both lower and upper neighbors
    diag[1:-1] += np.abs(lower[:-1]) + np.abs(upper[1:]) + 1e-3

    # -----------------------------
    # Random RHS vector
    # -----------------------------
    rhs = rng.normal(size=n)

    # -----------------------------
    # Quick shape asserts (makes future debugging trivial)
    # -----------------------------
    assert lower.shape == (n - 1,)
    assert diag.shape == (n,)
    assert upper.shape == (n - 1,)
    assert rhs.shape == (n,)

    # -----------------------------
    # Solve using Thomas algorithm
    # -----------------------------
    x_thomas = solve_tridiagonal(lower=lower, diag=diag, upper=upper, rhs=rhs)

    # -----------------------------
    # Build dense matrix A for cross-checking
    # -----------------------------
    A = np.zeros((n, n), dtype=np.float64)
    A[np.arange(n), np.arange(n)] = diag
    A[np.arange(1, n), np.arange(n - 1)] = lower
    A[np.arange(n - 1), np.arange(1, n)] = upper

    # -----------------------------
    # Solve using dense solver
    # -----------------------------
    x_dense = np.linalg.solve(A, rhs)

    # -----------------------------
    # Compare
    # -----------------------------
    assert x_thomas == pytest.approx(x_dense, rel=1e-10, abs=1e-12)