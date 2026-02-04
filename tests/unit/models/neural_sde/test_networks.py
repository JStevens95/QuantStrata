"""
Unit tests for Neural SDE networks module.

Tests NeuralDriftNetwork, NeuralDiffusionNetwork, and NetworkConfig.
"""

import numpy as np
import pytest

from src.models.neural_sde.networks import (
    NetworkConfig,
    NeuralDiffusionNetwork,
    NeuralDriftNetwork,
    NeuralNetwork,
)


class TestNetworkConfig:
    """Tests for NetworkConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = NetworkConfig()
        
        assert len(config.hidden_dims) > 0
        assert config.activation in ["relu", "tanh", "gelu"]
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = NetworkConfig(
            hidden_dims=[64, 32],
            activation="tanh",
        )
        
        assert config.hidden_dims == [64, 32]
        assert config.activation == "tanh"


class TestNeuralNetwork:
    """Tests for base NeuralNetwork class."""
    
    def test_network_creation(self) -> None:
        """Test network creation."""
        network = NeuralNetwork(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 16],
        )
        
        assert network is not None
    
    def test_forward_pass(self) -> None:
        """Test forward pass."""
        network = NeuralNetwork(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16, 8],
        )
        
        x = np.array([[1.0, 0.5], [2.0, 0.3]])
        output = network(x)
        
        assert output.shape == (2, 1)


class TestNeuralDriftNetwork:
    """Tests for NeuralDriftNetwork."""
    
    def test_network_creation(self) -> None:
        """Test drift network creation."""
        network = NeuralDriftNetwork(
            hidden_dims=[32, 16],
            activation="tanh",
        )
        
        assert network is not None
    
    def test_forward_pass_scalar(self) -> None:
        """Test forward pass with scalar inputs."""
        network = NeuralDriftNetwork(hidden_dims=[16, 8])
        
        S = 100.0
        t = 0.5
        
        output = network(S, t)
        
        assert isinstance(output, np.ndarray)
        assert output.shape == (1,) or output.shape == ()
    
    def test_forward_pass_array(self) -> None:
        """Test forward pass with array inputs."""
        network = NeuralDriftNetwork(hidden_dims=[16, 8])
        
        S = np.array([100.0, 101.0, 99.0])
        t = np.array([0.5, 0.5, 0.5])
        
        output = network(S, t)
        
        assert output.shape == (3,)
    
    def test_drift_is_real_valued(self) -> None:
        """Test that drift output is real-valued."""
        network = NeuralDriftNetwork()
        
        S = np.linspace(80, 120, 10)
        t = np.ones(10) * 0.5
        
        output = network(S, t)
        
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
    
    def test_input_normalization(self) -> None:
        """Test input normalization."""
        network = NeuralDriftNetwork(
            normalize_inputs=True,
            S_mean=100.0,
            S_std=20.0,
        )
        
        # Large input values should still work
        S = np.array([50.0, 100.0, 150.0])
        t = np.array([0.1, 0.5, 0.9])
        
        output = network(S, t)
        
        assert not np.any(np.isnan(output))
    
    def test_reproducibility_with_seed(self) -> None:
        """Test reproducibility with seed."""
        network1 = NeuralDriftNetwork(hidden_dims=[16], seed=42)
        network2 = NeuralDriftNetwork(hidden_dims=[16], seed=42)
        
        S = np.array([100.0])
        t = np.array([0.5])
        
        out1 = network1(S, t)
        out2 = network2(S, t)
        
        np.testing.assert_array_almost_equal(out1, out2)


class TestNeuralDiffusionNetwork:
    """Tests for NeuralDiffusionNetwork."""
    
    def test_network_creation(self) -> None:
        """Test diffusion network creation."""
        network = NeuralDiffusionNetwork(
            hidden_dims=[32, 16],
            activation="tanh",
        )
        
        assert network is not None
    
    def test_forward_pass_scalar(self) -> None:
        """Test forward pass with scalar inputs."""
        network = NeuralDiffusionNetwork(hidden_dims=[16, 8])
        
        S = 100.0
        t = 0.5
        
        output = network(S, t)
        
        assert isinstance(output, np.ndarray)
    
    def test_forward_pass_array(self) -> None:
        """Test forward pass with array inputs."""
        network = NeuralDiffusionNetwork(hidden_dims=[16, 8])
        
        S = np.array([100.0, 101.0, 99.0])
        t = np.array([0.5, 0.5, 0.5])
        
        output = network(S, t)
        
        assert output.shape == (3,)
    
    def test_diffusion_is_positive(self) -> None:
        """Test that diffusion output is positive."""
        network = NeuralDiffusionNetwork()
        
        S = np.linspace(80, 120, 10)
        t = np.ones(10) * 0.5
        
        output = network(S, t)
        
        # Diffusion coefficient should be positive
        assert all(output > 0)
    
    def test_diffusion_positivity_constraint(self) -> None:
        """Test that positivity constraint is enforced."""
        network = NeuralDiffusionNetwork(hidden_dims=[16, 8])
        
        # Test across many values
        S = np.random.uniform(50, 150, 100)
        t = np.random.uniform(0, 1, 100)
        
        output = network(S, t)
        
        # All outputs should be strictly positive
        assert all(output > 0)
        assert not np.any(np.isnan(output))
    
    def test_diffusion_bounded(self) -> None:
        """Test that diffusion is reasonably bounded."""
        network = NeuralDiffusionNetwork()
        
        S = np.array([100.0])
        t = np.array([0.5])
        
        output = network(S, t)
        
        # Should be reasonable (not exploding)
        assert output[0] < 10.0
