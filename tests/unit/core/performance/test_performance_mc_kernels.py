"""
Tests for Monte Carlo performance kernels.

Author: QuantStrata Team
"""
import pytest
import numpy as np

from src.core.performance.backend import numba_available
from src.core.performance.mc_kernels import (
    simulate_gbm_paths,
    gbm_step,
    _gbm_step_exact_numpy,
    _gbm_step_euler_numpy,
    _gbm_step_milstein_numpy,
    _simulate_gbm_paths_numpy,
)


class TestGbmStepNumpy:
    """Tests for NumPy GBM step functions."""
    
    @pytest.fixture
    def setup_data(self):
        """Common test data."""
        np.random.seed(42)
        n_paths = 1000
        spot = np.full(n_paths, 100.0)
        z = np.random.randn(n_paths)
        dt = 1.0 / 252
        sqrt_dt = np.sqrt(dt)
        drift = 0.05
        vol = 0.20
        return spot, z, dt, sqrt_dt, drift, vol
    
    def test_exact_step_positive(self, setup_data):
        """Exact step should produce positive values."""
        spot, z, dt, sqrt_dt, drift, vol = setup_data
        result = _gbm_step_exact_numpy(spot, drift, vol, dt, sqrt_dt, z)
        assert np.all(result > 0)
    
    def test_exact_step_shape(self, setup_data):
        """Output shape should match input."""
        spot, z, dt, sqrt_dt, drift, vol = setup_data
        result = _gbm_step_exact_numpy(spot, drift, vol, dt, sqrt_dt, z)
        assert result.shape == spot.shape
    
    def test_exact_step_mean(self, setup_data):
        """Mean should be close to theoretical expectation."""
        spot, z, dt, sqrt_dt, drift, vol = setup_data
        # Use many paths for statistical accuracy
        n_large = 100000
        spot_large = np.full(n_large, 100.0)
        z_large = np.random.randn(n_large)
        
        result = _gbm_step_exact_numpy(spot_large, drift, vol, dt, sqrt_dt, z_large)
        
        # E[S_{t+dt}] = S_t * exp(μ * dt) for exact GBM
        expected_mean = 100.0 * np.exp(drift * dt)
        assert np.abs(result.mean() - expected_mean) < 0.5  # Within tolerance
    
    def test_euler_step_shape(self, setup_data):
        """Euler step should have correct shape."""
        spot, z, dt, sqrt_dt, drift, vol = setup_data
        result = _gbm_step_euler_numpy(spot, drift, vol, dt, sqrt_dt, z)
        assert result.shape == spot.shape
    
    def test_milstein_step_shape(self, setup_data):
        """Milstein step should have correct shape."""
        spot, z, dt, sqrt_dt, drift, vol = setup_data
        result = _gbm_step_milstein_numpy(spot, drift, vol, dt, sqrt_dt, z)
        assert result.shape == spot.shape


class TestSimulateGbmPathsNumpy:
    """Tests for NumPy GBM path simulation."""
    
    def test_terminal_shape(self):
        """Terminal spots should have correct shape."""
        np.random.seed(42)
        n_paths = 500
        n_steps = 100
        z = np.random.randn(n_steps, n_paths)
        
        terminal, paths = _simulate_gbm_paths_numpy(
            spot0=100.0, drift=0.05, vol=0.20,
            T=1.0, n_steps=n_steps, n_paths=n_paths,
            z=z, scheme="exact", store_paths=False
        )
        
        assert terminal.shape == (n_paths,)
        assert paths is None
    
    def test_paths_shape(self):
        """Full paths should have correct shape."""
        np.random.seed(42)
        n_paths = 100
        n_steps = 50
        z = np.random.randn(n_steps, n_paths)
        
        terminal, paths = _simulate_gbm_paths_numpy(
            spot0=100.0, drift=0.05, vol=0.20,
            T=1.0, n_steps=n_steps, n_paths=n_paths,
            z=z, scheme="exact", store_paths=True
        )
        
        assert terminal.shape == (n_paths,)
        assert paths.shape == (n_steps + 1, n_paths)
    
    def test_paths_start_at_spot0(self):
        """Paths should start at initial spot."""
        np.random.seed(42)
        n_paths = 100
        n_steps = 50
        z = np.random.randn(n_steps, n_paths)
        spot0 = 100.0
        
        _, paths = _simulate_gbm_paths_numpy(
            spot0=spot0, drift=0.05, vol=0.20,
            T=1.0, n_steps=n_steps, n_paths=n_paths,
            z=z, scheme="exact", store_paths=True
        )
        
        assert np.allclose(paths[0, :], spot0)
    
    def test_terminal_equals_last_path(self):
        """Terminal should equal last path point."""
        np.random.seed(42)
        n_paths = 100
        n_steps = 50
        z = np.random.randn(n_steps, n_paths)
        
        terminal, paths = _simulate_gbm_paths_numpy(
            spot0=100.0, drift=0.05, vol=0.20,
            T=1.0, n_steps=n_steps, n_paths=n_paths,
            z=z, scheme="exact", store_paths=True
        )
        
        assert np.allclose(terminal, paths[-1, :])
    
    @pytest.mark.parametrize("scheme", ["exact", "euler", "milstein"])
    def test_all_schemes_run(self, scheme):
        """All schemes should run without error."""
        np.random.seed(42)
        n_paths = 100
        n_steps = 50
        z = np.random.randn(n_steps, n_paths)
        
        terminal, _ = _simulate_gbm_paths_numpy(
            spot0=100.0, drift=0.05, vol=0.20,
            T=1.0, n_steps=n_steps, n_paths=n_paths,
            z=z, scheme=scheme, store_paths=False
        )
        
        assert terminal.shape == (n_paths,)


class TestUnifiedApi:
    """Tests for unified simulate_gbm_paths API."""
    
    def test_numpy_backend(self):
        """NumPy backend should work."""
        np.random.seed(42)
        n_paths = 100
        n_steps = 50
        z = np.random.randn(n_steps, n_paths)
        
        terminal, _ = simulate_gbm_paths(
            spot0=100.0, drift=0.05, vol=0.20,
            T=1.0, n_steps=n_steps, n_paths=n_paths,
            z=z, backend="numpy"
        )
        
        assert terminal.shape == (n_paths,)
    
    @pytest.mark.skipif(not numba_available(), reason="Numba not installed")
    def test_numba_backend(self):
        """Numba backend should work if available."""
        np.random.seed(42)
        n_paths = 100
        n_steps = 50
        z = np.random.randn(n_steps, n_paths)
        
        terminal, _ = simulate_gbm_paths(
            spot0=100.0, drift=0.05, vol=0.20,
            T=1.0, n_steps=n_steps, n_paths=n_paths,
            z=z, backend="numba"
        )
        
        assert terminal.shape == (n_paths,)
    
    @pytest.mark.skipif(not numba_available(), reason="Numba not installed")
    def test_backends_agree(self):
        """NumPy and Numba should produce same results."""
        np.random.seed(42)
        n_paths = 1000
        n_steps = 50
        z = np.random.randn(n_steps, n_paths)
        
        terminal_numpy, _ = simulate_gbm_paths(
            spot0=100.0, drift=0.05, vol=0.20,
            T=1.0, n_steps=n_steps, n_paths=n_paths,
            z=z, backend="numpy"
        )
        
        terminal_numba, _ = simulate_gbm_paths(
            spot0=100.0, drift=0.05, vol=0.20,
            T=1.0, n_steps=n_steps, n_paths=n_paths,
            z=z, backend="numba"
        )
        
        # Should be numerically identical (same algorithm)
        assert np.allclose(terminal_numpy, terminal_numba, rtol=1e-10)
