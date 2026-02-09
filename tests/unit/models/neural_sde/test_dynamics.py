"""
Unit tests for Neural SDE dynamics module.

Tests NeuralSDEDynamics and NeuralSDEConfig.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.neural_sde.dynamics import (
    NeuralSDEConfig,
    NeuralSDEDynamics,
)
from src.models.neural_sde.networks import (
    NeuralDiffusionNetwork,
    NeuralDriftNetwork,
)


class TestNeuralSDEConfig:
    """Tests for NeuralSDEConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = NeuralSDEConfig()
        
        assert config.solver_type in ["euler", "milstein"]
        assert len(config.drift_hidden_dims) > 0
        assert len(config.diffusion_hidden_dims) > 0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = NeuralSDEConfig(
            solver_type="milstein",
            drift_hidden_dims=[64, 32],
            diffusion_hidden_dims=[64, 32],
            activation="tanh",
        )
        
        assert config.solver_type == "milstein"
        assert config.drift_hidden_dims == [64, 32]
        assert config.diffusion_hidden_dims == [64, 32]


class TestNeuralSDEDynamics:
    """Tests for NeuralSDEDynamics."""
    
    def test_dynamics_creation(self) -> None:
        """Test dynamics creation."""
        dynamics = NeuralSDEDynamics(seed=42)
        
        assert dynamics is not None
    
    def test_dynamics_with_config(self) -> None:
        """Test dynamics with custom config."""
        config = NeuralSDEConfig(
            drift_hidden_dims=[32, 16],
            diffusion_hidden_dims=[32, 16],
            solver_type="euler",
        )
        
        dynamics = NeuralSDEDynamics(config=config, seed=42)
        
        assert dynamics is not None
    
    def test_dynamics_with_custom_networks(self) -> None:
        """Test dynamics with custom networks."""
        drift_net = NeuralDriftNetwork(hidden_dims=[16], seed=42)
        diff_net = NeuralDiffusionNetwork(hidden_dims=[16], seed=42)
        
        dynamics = NeuralSDEDynamics(
            drift_network=drift_net,
            diffusion_network=diff_net,
            seed=42,
        )
        
        assert dynamics is not None
    
    def test_simulate(self) -> None:
        """Test path simulation."""
        dynamics = NeuralSDEDynamics(seed=42)
        
        paths = dynamics.simulate(
            S0=100.0,
            T=1.0,
            n_steps=100,
            n_paths=50,
        )
        
        # Should return correct shape
        assert paths.shape == (50, 101)
        
        # Initial value should be S0
        assert all(paths[:, 0] == 100.0)
    
    def test_simulate_positive_paths(self) -> None:
        """Test that simulation produces mostly positive paths."""
        dynamics = NeuralSDEDynamics(seed=42)
        
        paths = dynamics.simulate(
            S0=100.0,
            T=1.0,
            n_steps=252,
            n_paths=1000,
        )
        
        # Most paths should stay positive
        positive_ratio = np.mean(paths > 0)
        assert positive_ratio > 0.95
    
    def test_simulate_reproducibility(self) -> None:
        """Test simulation reproducibility."""
        dynamics1 = NeuralSDEDynamics(seed=42)
        paths1 = dynamics1.simulate(100.0, 1.0, 50, 10)
        
        dynamics2 = NeuralSDEDynamics(seed=42)
        paths2 = dynamics2.simulate(100.0, 1.0, 50, 10)
        
        np.testing.assert_array_equal(paths1, paths2)
    
    def test_compute_statistics(self) -> None:
        """Test statistics computation."""
        dynamics = NeuralSDEDynamics(seed=42)
        
        stats = dynamics.compute_statistics(
            S0=100.0, T=1.0, n_steps=100, n_paths=1000
        )
        
        # Should have basic statistics
        assert "mean_final" in stats
        assert "std_final" in stats
        assert "mean_return" in stats
        assert "std_return" in stats
    
    def test_drift_function(self) -> None:
        """Test drift function evaluation."""
        dynamics = NeuralSDEDynamics(seed=42)
        
        S = np.array([100.0])
        t = np.array([0.5])
        
        drift_val = dynamics.drift(S, t)
        
        assert drift_val is not None
        assert not np.isnan(drift_val).any()
    
    def test_diffusion_function(self) -> None:
        """Test diffusion function evaluation."""
        dynamics = NeuralSDEDynamics(seed=42)
        
        S = np.array([100.0])
        t = np.array([0.5])
        
        diff_val = dynamics.diffusion(S, t)
        
        assert diff_val is not None
        diff_val = np.atleast_1d(diff_val)
        assert not np.isnan(diff_val).any()
        assert np.all(diff_val > 0)  # Diffusion should be positive
    
    def test_save_and_load(self) -> None:
        """Test saving and loading dynamics (Option A: shapes and summary stats only; no path equality)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model")
            S0, T, n_steps, n_paths = 100.0, 0.5, 50, 10

            # Create and save (implementation appends .npy)
            dynamics1 = NeuralSDEDynamics(seed=42)
            dynamics1.simulate(S0, T, n_steps, n_paths)
            dynamics1.save(save_path)

            # Load and simulate (RNG state is not restored, so paths will differ from first run)
            dynamics2 = NeuralSDEDynamics.load(save_path)
            paths = dynamics2.simulate(S0, T, n_steps, n_paths)

            # Assert shape only
            assert paths.shape == (n_paths, n_steps + 1)

            # Assert initial condition
            assert np.all(paths[:, 0] == S0)

            # Assert summary stats: finite and reasonable (no path equality)
            assert np.all(np.isfinite(paths))
            terminal = paths[:, -1]
            assert np.isfinite(terminal.mean()) and np.isfinite(terminal.std())
            assert terminal.mean() > 0
            assert np.mean(paths > 0) > 0.95
    
    def test_different_solver_types(self) -> None:
        """Test with different solver types."""
        config_euler = NeuralSDEConfig(solver_type="euler")
        config_milstein = NeuralSDEConfig(solver_type="milstein")
        
        dynamics_euler = NeuralSDEDynamics(config=config_euler, seed=42)
        dynamics_milstein = NeuralSDEDynamics(config=config_milstein, seed=42)
        
        # Both should work
        paths_euler = dynamics_euler.simulate(100.0, 0.5, 50, 10)
        paths_milstein = dynamics_milstein.simulate(100.0, 0.5, 50, 10)
        
        assert paths_euler.shape == paths_milstein.shape
        # Results will differ due to different algorithms
    
    def test_long_simulation(self) -> None:
        """Test longer time horizon simulation."""
        dynamics = NeuralSDEDynamics(seed=42)
        
        paths = dynamics.simulate(
            S0=100.0,
            T=5.0,  # 5 years
            n_steps=1260,  # 5 * 252
            n_paths=100,
        )
        
        assert paths.shape == (100, 1261)
        
        # Paths should still be reasonable
        assert np.mean(paths > 0) > 0.9
        assert np.mean(paths < 10000) > 0.99
