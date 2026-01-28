"""
Tests for finite difference performance kernels.

Author: QuantStrata Team
"""
import pytest
import numpy as np

from src.core.performance.backend import numba_available
from src.core.performance.fd_kernels import (
    solve_tridiagonal,
    solve_tridiagonal_psor,
    solve_tridiagonal_batch,
    solve_tridiagonal_numpy,
    solve_tridiagonal_psor_numpy,
    solve_tridiagonal_batch_numpy,
)


class TestTridiagonalSolverNumpy:
    """Tests for NumPy tridiagonal solver."""
    
    def test_simple_system(self):
        """Test simple 3x3 tridiagonal system."""
        # System: 2x_0 - x_1 = 1
        #         -x_0 + 2x_1 - x_2 = 0
        #         -x_1 + 2x_2 = 1
        lower = np.array([-1.0, -1.0])
        diag = np.array([2.0, 2.0, 2.0])
        upper = np.array([-1.0, -1.0])
        rhs = np.array([1.0, 0.0, 1.0])
        
        x = solve_tridiagonal_numpy(lower, diag, upper, rhs)
        
        # Verify solution
        expected = np.array([1.0, 1.0, 1.0])
        assert np.allclose(x, expected)
    
    def test_solution_satisfies_equation(self):
        """Solution should satisfy Ax = b."""
        np.random.seed(42)
        n = 100
        
        # Generate diagonally dominant system (guaranteed solvable)
        lower = -np.random.rand(n - 1)
        upper = -np.random.rand(n - 1)
        diag = 3.0 * np.ones(n)  # Diagonally dominant
        rhs = np.random.randn(n)
        
        x = solve_tridiagonal_numpy(lower, diag, upper, rhs)
        
        # Compute Ax
        Ax = np.zeros(n)
        Ax[0] = diag[0] * x[0] + upper[0] * x[1]
        for i in range(1, n - 1):
            Ax[i] = lower[i - 1] * x[i - 1] + diag[i] * x[i] + upper[i] * x[i + 1]
        Ax[n - 1] = lower[n - 2] * x[n - 2] + diag[n - 1] * x[n - 1]
        
        assert np.allclose(Ax, rhs)
    
    def test_identity_like_system(self):
        """Near-identity should give x ≈ b."""
        n = 50
        lower = np.zeros(n - 1)
        diag = np.ones(n)
        upper = np.zeros(n - 1)
        rhs = np.random.randn(n)
        
        x = solve_tridiagonal_numpy(lower, diag, upper, rhs)
        assert np.allclose(x, rhs)


class TestPsorSolverNumpy:
    """Tests for NumPy PSOR solver."""
    
    def test_unconstrained_equals_thomas(self):
        """PSOR with low floor should equal Thomas solution."""
        np.random.seed(42)
        n = 50
        
        lower = -0.5 * np.ones(n - 1)
        diag = 2.0 * np.ones(n)
        upper = -0.5 * np.ones(n - 1)
        rhs = np.random.randn(n)
        floor = -1000 * np.ones(n)  # Very low, never binding
        
        x_thomas = solve_tridiagonal_numpy(lower, diag, upper, rhs)
        x_psor, _ = solve_tridiagonal_psor_numpy(
            lower, diag, upper, rhs, floor, omega=1.0, max_iter=1000
        )
        
        assert np.allclose(x_thomas, x_psor, rtol=1e-6)
    
    def test_constraint_binding(self):
        """PSOR should respect floor constraint."""
        n = 20
        lower = -0.5 * np.ones(n - 1)
        diag = 2.0 * np.ones(n)
        upper = -0.5 * np.ones(n - 1)
        rhs = np.zeros(n)  # Would give x = 0 without constraint
        floor = np.ones(n)  # Floor at 1
        
        x, _ = solve_tridiagonal_psor_numpy(
            lower, diag, upper, rhs, floor, omega=1.0, max_iter=1000
        )
        
        assert np.all(x >= floor - 1e-10)
    
    def test_convergence(self):
        """PSOR should converge in reasonable iterations."""
        n = 100
        lower = -0.5 * np.ones(n - 1)
        diag = 2.0 * np.ones(n)
        upper = -0.5 * np.ones(n - 1)
        rhs = np.random.randn(n)
        floor = -100 * np.ones(n)
        
        _, iters = solve_tridiagonal_psor_numpy(
            lower, diag, upper, rhs, floor, omega=1.2, max_iter=1000
        )
        
        assert iters < 1000  # Should converge before max


class TestBatchSolverNumpy:
    """Tests for NumPy batch tridiagonal solver."""
    
    def test_batch_equals_individual(self):
        """Batch solve should equal individual solves."""
        np.random.seed(42)
        n = 50
        m = 5  # Batch size
        
        lower = -np.random.rand(n - 1)
        diag = 3.0 * np.ones(n)
        upper = -np.random.rand(n - 1)
        rhs_batch = np.random.randn(n, m)
        
        # Batch solve
        x_batch = solve_tridiagonal_batch_numpy(lower, diag, upper, rhs_batch)
        
        # Individual solves
        for j in range(m):
            x_individual = solve_tridiagonal_numpy(lower, diag, upper, rhs_batch[:, j])
            assert np.allclose(x_batch[:, j], x_individual)
    
    def test_batch_shape(self):
        """Output shape should match input."""
        n = 30
        m = 10
        
        lower = -np.ones(n - 1)
        diag = 3.0 * np.ones(n)
        upper = -np.ones(n - 1)
        rhs_batch = np.random.randn(n, m)
        
        x_batch = solve_tridiagonal_batch_numpy(lower, diag, upper, rhs_batch)
        assert x_batch.shape == (n, m)


class TestUnifiedApi:
    """Tests for unified solver APIs."""
    
    def test_tridiagonal_numpy_backend(self):
        """Tridiagonal solve should work with NumPy backend."""
        lower = np.array([-1.0, -1.0])
        diag = np.array([2.0, 2.0, 2.0])
        upper = np.array([-1.0, -1.0])
        rhs = np.array([1.0, 0.0, 1.0])
        
        x = solve_tridiagonal(lower, diag, upper, rhs, backend="numpy")
        expected = np.array([1.0, 1.0, 1.0])
        assert np.allclose(x, expected)
    
    @pytest.mark.skipif(not numba_available(), reason="Numba not installed")
    def test_tridiagonal_numba_backend(self):
        """Tridiagonal solve should work with Numba backend."""
        lower = np.array([-1.0, -1.0])
        diag = np.array([2.0, 2.0, 2.0])
        upper = np.array([-1.0, -1.0])
        rhs = np.array([1.0, 0.0, 1.0])
        
        x = solve_tridiagonal(lower, diag, upper, rhs, backend="numba")
        expected = np.array([1.0, 1.0, 1.0])
        assert np.allclose(x, expected)
    
    @pytest.mark.skipif(not numba_available(), reason="Numba not installed")
    def test_backends_agree_tridiagonal(self):
        """NumPy and Numba should produce same tridiagonal results."""
        np.random.seed(42)
        n = 200
        
        lower = -np.random.rand(n - 1)
        diag = 3.0 * np.ones(n)
        upper = -np.random.rand(n - 1)
        rhs = np.random.randn(n)
        
        numpy_result = solve_tridiagonal(lower, diag, upper, rhs, backend="numpy")
        numba_result = solve_tridiagonal(lower, diag, upper, rhs, backend="numba")
        
        assert np.allclose(numpy_result, numba_result, rtol=1e-10)
    
    @pytest.mark.skipif(not numba_available(), reason="Numba not installed")
    def test_backends_agree_psor(self):
        """NumPy and Numba should produce same PSOR results."""
        np.random.seed(42)
        n = 100
        
        lower = -0.5 * np.ones(n - 1)
        diag = 2.0 * np.ones(n)
        upper = -0.5 * np.ones(n - 1)
        rhs = np.random.randn(n)
        floor = -10 * np.ones(n)
        
        numpy_result, numpy_iters = solve_tridiagonal_psor(
            lower, diag, upper, rhs, floor, backend="numpy"
        )
        numba_result, numba_iters = solve_tridiagonal_psor(
            lower, diag, upper, rhs, floor, backend="numba"
        )
        
        assert np.allclose(numpy_result, numba_result, rtol=1e-6)
    
    @pytest.mark.skipif(not numba_available(), reason="Numba not installed")
    def test_backends_agree_batch(self):
        """NumPy and Numba should produce same batch results."""
        np.random.seed(42)
        n = 100
        m = 5
        
        lower = -np.random.rand(n - 1)
        diag = 3.0 * np.ones(n)
        upper = -np.random.rand(n - 1)
        rhs_batch = np.random.randn(n, m)
        
        numpy_result = solve_tridiagonal_batch(lower, diag, upper, rhs_batch, backend="numpy")
        numba_result = solve_tridiagonal_batch(lower, diag, upper, rhs_batch, backend="numba")
        
        assert np.allclose(numpy_result, numba_result, rtol=1e-10)
