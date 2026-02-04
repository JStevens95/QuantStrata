"""
Neural networks for learning drift and diffusion functions.

Implements neural network architectures for:
- μ_θ(S, t): Learned drift function
- σ_θ(S, t): Learned diffusion function

Example:
    drift_net = NeuralDriftNetwork(hidden_dims=[64, 64])
    diffusion_net = NeuralDiffusionNetwork(hidden_dims=[64, 64])
    
    # Forward pass
    S = np.array([100.0, 101.0, 99.0])
    t = np.array([0.0, 0.1, 0.2])
    
    drift = drift_net(S, t)      # Learned drift
    vol = diffusion_net(S, t)    # Learned volatility (positive)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NetworkConfig:
    """Configuration for neural SDE networks."""
    
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"  # "relu", "tanh", "gelu"
    output_activation: Optional[str] = None
    dropout: float = 0.0
    use_batch_norm: bool = False
    
    # Input normalization
    normalize_inputs: bool = True
    S_mean: float = 100.0
    S_std: float = 20.0


# =============================================================================
# Base Network (NumPy implementation for portability)
# =============================================================================


class NeuralNetwork:
    """
    Simple MLP implementation using NumPy.
    
    Can be extended to use TensorFlow/PyTorch for GPU acceleration.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize network.
        
        Parameters
        ----------
        input_dim : int
            Input dimension.
        hidden_dims : list of int
            Hidden layer dimensions.
        output_dim : int
            Output dimension.
        activation : str
            Hidden layer activation.
        output_activation : str, optional
            Output activation (None for linear).
        seed : int, optional
            Random seed for weight initialization.
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        
        rng = np.random.default_rng(seed)
        
        # Initialize weights
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            W = rng.standard_normal((dims[i], dims[i + 1])) * scale
            b = np.zeros(dims[i + 1])
            
            self.weights.append(W)
            self.biases.append(b)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return self.forward(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through network.
        
        Parameters
        ----------
        x : ndarray
            Input of shape (batch_size, input_dim) or (input_dim,).
        
        Returns
        -------
        ndarray
            Output of shape (batch_size, output_dim).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = self._activate(x, self.activation)
        
        # Output layer
        x = x @ self.weights[-1] + self.biases[-1]
        
        if self.output_activation:
            x = self._activate(x, self.output_activation)
        
        return x
    
    def _activate(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == "relu":
            return np.maximum(0, x)
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "gelu":
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "softplus":
            return np.log(1 + np.exp(np.clip(x, -20, 20)))
        else:
            return x
    
    def get_parameters(self) -> dict:
        """Get network parameters."""
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set network parameters."""
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]


# =============================================================================
# Neural Drift Network
# =============================================================================


class NeuralDriftNetwork:
    """
    Neural network for learned drift function μ_θ(S, t).
    
    The drift function maps (spot, time) to instantaneous drift.
    
    Example:
        drift_net = NeuralDriftNetwork(hidden_dims=[64, 64])
        
        S = np.array([100.0, 101.0, 99.0])  # Spot prices
        t = np.array([0.0, 0.1, 0.2])       # Times
        
        drift = drift_net(S, t)  # Learned drift values
    """
    
    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "tanh",
        normalize_inputs: bool = True,
        S_mean: float = 100.0,
        S_std: float = 20.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize drift network.
        
        Parameters
        ----------
        hidden_dims : list of int
            Hidden layer sizes.
        activation : str
            Activation function.
        normalize_inputs : bool
            Whether to normalize inputs.
        S_mean, S_std : float
            Normalization parameters for spot.
        seed : int, optional
            Random seed.
        """
        hidden_dims = hidden_dims or [64, 64]
        
        self.normalize_inputs = normalize_inputs
        self.S_mean = S_mean
        self.S_std = S_std
        
        # Network: (S, t) -> drift
        self._network = NeuralNetwork(
            input_dim=2,  # (S, t)
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            output_activation=None,  # Drift can be any real number
            seed=seed,
        )
    
    def __call__(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Compute drift at (S, t).
        
        Parameters
        ----------
        S : float or ndarray
            Spot price(s).
        t : float or ndarray
            Time(s).
        
        Returns
        -------
        ndarray
            Drift value(s).
        """
        S = np.atleast_1d(S)
        t = np.atleast_1d(t)
        
        # Normalize inputs
        if self.normalize_inputs:
            S_norm = (S - self.S_mean) / self.S_std
            t_norm = t  # Time usually in [0, 1]
        else:
            S_norm = S
            t_norm = t
        
        # Build input
        x = np.stack([S_norm, t_norm], axis=-1)
        
        # Forward pass
        drift = self._network(x)
        
        return drift.squeeze()
    
    def get_parameters(self) -> dict:
        """Get network parameters."""
        return {
            "network": self._network.get_parameters(),
            "S_mean": self.S_mean,
            "S_std": self.S_std,
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set network parameters."""
        self._network.set_parameters(params["network"])
        self.S_mean = params.get("S_mean", self.S_mean)
        self.S_std = params.get("S_std", self.S_std)


# =============================================================================
# Neural Diffusion Network
# =============================================================================


class NeuralDiffusionNetwork:
    """
    Neural network for learned diffusion function σ_θ(S, t).
    
    The diffusion (volatility) function maps (spot, time) to
    instantaneous volatility. Uses softplus output to ensure positivity.
    
    Example:
        diffusion_net = NeuralDiffusionNetwork(hidden_dims=[64, 64])
        
        S = np.array([100.0, 101.0, 99.0])
        t = np.array([0.0, 0.1, 0.2])
        
        vol = diffusion_net(S, t)  # Positive volatility values
    """
    
    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "tanh",
        min_vol: float = 0.01,
        max_vol: float = 2.0,
        normalize_inputs: bool = True,
        S_mean: float = 100.0,
        S_std: float = 20.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize diffusion network.
        
        Parameters
        ----------
        hidden_dims : list of int
            Hidden layer sizes.
        activation : str
            Activation function.
        min_vol : float
            Minimum volatility (floor).
        max_vol : float
            Maximum volatility (ceiling).
        normalize_inputs : bool
            Whether to normalize inputs.
        S_mean, S_std : float
            Normalization parameters.
        seed : int, optional
            Random seed.
        """
        hidden_dims = hidden_dims or [64, 64]
        
        self.min_vol = min_vol
        self.max_vol = max_vol
        self.normalize_inputs = normalize_inputs
        self.S_mean = S_mean
        self.S_std = S_std
        
        # Network: (S, t) -> log_vol (use softplus for positivity)
        self._network = NeuralNetwork(
            input_dim=2,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            output_activation="softplus",  # Ensures positive output
            seed=seed,
        )
    
    def __call__(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Compute diffusion (volatility) at (S, t).
        
        Parameters
        ----------
        S : float or ndarray
            Spot price(s).
        t : float or ndarray
            Time(s).
        
        Returns
        -------
        ndarray
            Volatility value(s) (always positive).
        """
        S = np.atleast_1d(S)
        t = np.atleast_1d(t)
        
        # Normalize inputs
        if self.normalize_inputs:
            S_norm = (S - self.S_mean) / self.S_std
            t_norm = t
        else:
            S_norm = S
            t_norm = t
        
        # Build input
        x = np.stack([S_norm, t_norm], axis=-1)
        
        # Forward pass
        vol = self._network(x)
        
        # Clip to valid range
        vol = np.clip(vol.squeeze(), self.min_vol, self.max_vol)
        
        return vol
    
    def get_parameters(self) -> dict:
        """Get network parameters."""
        return {
            "network": self._network.get_parameters(),
            "S_mean": self.S_mean,
            "S_std": self.S_std,
            "min_vol": self.min_vol,
            "max_vol": self.max_vol,
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set network parameters."""
        self._network.set_parameters(params["network"])
        self.S_mean = params.get("S_mean", self.S_mean)
        self.S_std = params.get("S_std", self.S_std)
        self.min_vol = params.get("min_vol", self.min_vol)
        self.max_vol = params.get("max_vol", self.max_vol)


__all__ = [
    "NeuralNetwork",
    "NeuralDriftNetwork",
    "NeuralDiffusionNetwork",
    "NetworkConfig",
]
