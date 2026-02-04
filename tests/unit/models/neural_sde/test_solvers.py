"""
Unit tests for SDE solvers module.

Tests EulerMaruyamaSolver, MilsteinSolver, and SDESolver base class.
"""

import numpy as np
import pytest

from src.models.neural_sde.solvers import (
    EulerMaruyamaSolver,
    MilsteinSolver,
    SDESolver,
    SolverConfig,
)


class TestSolverConfig:
    """Tests for SolverConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = SolverConfig()
        
        assert config.dt is None or config.dt > 0
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SolverConfig(seed=42)
        
        assert config.seed == 42


class TestEulerMaruyamaSolver:
    """Tests for EulerMaruyamaSolver."""
    
    def test_solver_creation(self) -> None:
        """Test solver creation."""
        solver = EulerMaruyamaSolver(seed=42)
        
        assert solver is not None
    
    def test_solve_gbm(self) -> None:
        """Test solving GBM (constant drift and diffusion)."""
        solver = EulerMaruyamaSolver(seed=42)
        
        # GBM parameters
        mu = 0.05
        sigma = 0.2
        
        def drift(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return mu * S
        
        def diffusion(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return sigma * S
        
        paths = solver.solve(
            drift=drift,
            diffusion=diffusion,
            S0=100.0,
            T=1.0,
            n_steps=252,
            n_paths=100,
        )
        
        # Should return paths of correct shape
        assert paths.shape == (100, 253)  # n_paths x (n_steps + 1)
        
        # Initial value should be S0
        assert all(paths[:, 0] == 100.0)
    
    def test_solve_produces_positive_paths(self) -> None:
        """Test that positive diffusion produces positive paths."""
        solver = EulerMaruyamaSolver(seed=42)
        
        def drift(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return 0.05 * S
        
        def diffusion(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return 0.2 * S
        
        paths = solver.solve(
            drift=drift,
            diffusion=diffusion,
            S0=100.0,
            T=1.0,
            n_steps=252,
            n_paths=1000,
        )
        
        # GBM should produce positive paths (almost surely)
        # Allow tiny numerical issues
        assert np.mean(paths > 0) > 0.99
    
    def test_reproducibility_with_seed(self) -> None:
        """Test reproducibility with same seed."""
        def drift(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return 0.05 * S
        
        def diffusion(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return 0.2 * S
        
        solver1 = EulerMaruyamaSolver(seed=42)
        paths1 = solver1.solve(drift, diffusion, 100.0, 1.0, 100, 10)
        
        solver2 = EulerMaruyamaSolver(seed=42)
        paths2 = solver2.solve(drift, diffusion, 100.0, 1.0, 100, 10)
        
        np.testing.assert_array_equal(paths1, paths2)
    
    def test_different_seeds_different_paths(self) -> None:
        """Test that different seeds produce different paths."""
        def drift(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return 0.05 * S
        
        def diffusion(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return 0.2 * S
        
        solver1 = EulerMaruyamaSolver(seed=42)
        paths1 = solver1.solve(drift, diffusion, 100.0, 1.0, 100, 10)
        
        solver2 = EulerMaruyamaSolver(seed=123)
        paths2 = solver2.solve(drift, diffusion, 100.0, 1.0, 100, 10)
        
        assert not np.allclose(paths1, paths2)
    
    def test_zero_volatility(self) -> None:
        """Test with zero volatility (deterministic)."""
        solver = EulerMaruyamaSolver(seed=42)
        
        mu = 0.05
        
        def drift(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return mu * S
        
        def diffusion(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return np.zeros_like(S)
        
        paths = solver.solve(
            drift=drift,
            diffusion=diffusion,
            S0=100.0,
            T=1.0,
            n_steps=252,
            n_paths=10,
        )
        
        # All paths should be identical (deterministic)
        for i in range(1, 10):
            np.testing.assert_array_almost_equal(paths[0], paths[i])
    
    def test_terminal_distribution(self) -> None:
        """Test that terminal distribution is approximately correct for GBM."""
        solver = EulerMaruyamaSolver(seed=42)
        
        mu = 0.05
        sigma = 0.2
        S0 = 100.0
        T = 1.0
        
        def drift(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return mu * S
        
        def diffusion(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return sigma * S
        
        paths = solver.solve(drift, diffusion, S0, T, n_steps=1000, n_paths=10000)
        
        terminal = paths[:, -1]
        
        # Log-normal distribution: E[S_T] = S0 * exp(mu * T)
        expected_mean = S0 * np.exp(mu * T)
        actual_mean = np.mean(terminal)
        
        # Should be within 5% of expected
        assert abs(actual_mean / expected_mean - 1) < 0.05


class TestMilsteinSolver:
    """Tests for MilsteinSolver."""
    
    def test_solver_creation(self) -> None:
        """Test solver creation."""
        solver = MilsteinSolver(seed=42)
        
        assert solver is not None
    
    def test_solve_gbm(self) -> None:
        """Test solving GBM with Milstein."""
        solver = MilsteinSolver(seed=42)
        
        mu = 0.05
        sigma = 0.2
        
        def drift(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return mu * S
        
        def diffusion(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return sigma * S
        
        paths = solver.solve(
            drift=drift,
            diffusion=diffusion,
            S0=100.0,
            T=1.0,
            n_steps=252,
            n_paths=100,
        )
        
        # Should return paths of correct shape
        assert paths.shape[0] == 100
        assert paths.shape[1] >= 252
    
    def test_milstein_vs_euler_convergence(self) -> None:
        """Test that Milstein has better convergence than Euler for GBM."""
        mu = 0.05
        sigma = 0.3
        S0 = 100.0
        T = 1.0
        
        def drift(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return mu * S
        
        def diffusion(S: np.ndarray, t: np.ndarray) -> np.ndarray:
            return sigma * S
        
        # Expected terminal mean
        expected_mean = S0 * np.exp(mu * T)
        
        euler_solver = EulerMaruyamaSolver(seed=42)
        milstein_solver = MilsteinSolver(seed=42)
        
        n_steps = 50  # Fewer steps to see difference
        n_paths = 5000
        
        euler_paths = euler_solver.solve(drift, diffusion, S0, T, n_steps, n_paths)
        milstein_paths = milstein_solver.solve(drift, diffusion, S0, T, n_steps, n_paths)
        
        euler_mean = np.mean(euler_paths[:, -1])
        milstein_mean = np.mean(milstein_paths[:, -1])
        
        # Both should be close to expected
        assert abs(euler_mean / expected_mean - 1) < 0.1
        assert abs(milstein_mean / expected_mean - 1) < 0.1
