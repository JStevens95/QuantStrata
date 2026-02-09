"""
Neural SDE Dynamics.

Combines neural drift/diffusion networks with SDE solvers
for simulation and training.

Example:
    from src.models.neural_sde import (
        NeuralSDEDynamics,
        NeuralDriftNetwork,
        NeuralDiffusionNetwork,
    )
    
    sde = NeuralSDEDynamics(
        drift_network=NeuralDriftNetwork(hidden_dims=[64, 64]),
        diffusion_network=NeuralDiffusionNetwork(hidden_dims=[64, 64]),
    )
    
    # Simulate paths
    paths = sde.simulate(S0=100.0, T=1.0, n_steps=252, n_paths=10000)
    
    # Calibrate to data
    sde.calibrate(historical_paths=real_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from src.models.neural_sde.networks import (
    NeuralDriftNetwork,
    NeuralDiffusionNetwork,
)
from src.models.neural_sde.solvers import (
    SDESolver,
    EulerMaruyamaSolver,
    MilsteinSolver,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NeuralSDEConfig:
    """Configuration for Neural SDE dynamics."""
    
    # Network architecture
    drift_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    diffusion_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"
    
    # Volatility constraints
    min_vol: float = 0.01
    max_vol: float = 2.0
    
    # Solver
    solver_type: str = "euler"  # "euler", "milstein"
    antithetic: bool = True
    
    # Normalization
    S_mean: float = 100.0
    S_std: float = 20.0


# =============================================================================
# Neural SDE Dynamics
# =============================================================================


class NeuralSDEDynamics:
    """
    Neural SDE dynamics model.
    
    Learns the drift and diffusion functions from data:
        dS = μ_θ(S, t)dt + σ_θ(S, t)dW
    
    Features:
    - Flexible neural architectures for drift and diffusion
    - Multiple SDE solvers
    - Calibration to historical data
    - Monte Carlo simulation
    
    Example:
        # Create with default networks
        sde = NeuralSDEDynamics()
        
        # Or with custom networks
        sde = NeuralSDEDynamics(
            drift_network=NeuralDriftNetwork(hidden_dims=[128, 64]),
            diffusion_network=NeuralDiffusionNetwork(hidden_dims=[128, 64]),
        )
        
        # Simulate
        paths = sde.simulate(S0=100.0, T=1.0, n_steps=252, n_paths=10000)
    """
    
    def __init__(
        self,
        config: Optional[NeuralSDEConfig] = None,
        drift_network: Optional[NeuralDriftNetwork] = None,
        diffusion_network: Optional[NeuralDiffusionNetwork] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize Neural SDE.
        
        Parameters
        ----------
        config : NeuralSDEConfig, optional
            Configuration.
        drift_network : NeuralDriftNetwork, optional
            Custom drift network.
        diffusion_network : NeuralDiffusionNetwork, optional
            Custom diffusion network.
        seed : int, optional
            Random seed.
        """
        self.config = config or NeuralSDEConfig()
        self.seed = seed
        
        # Create networks
        self.drift_network = drift_network or NeuralDriftNetwork(
            hidden_dims=self.config.drift_hidden_dims,
            activation=self.config.activation,
            S_mean=self.config.S_mean,
            S_std=self.config.S_std,
            seed=seed,
        )
        
        self.diffusion_network = diffusion_network or NeuralDiffusionNetwork(
            hidden_dims=self.config.diffusion_hidden_dims,
            activation=self.config.activation,
            min_vol=self.config.min_vol,
            max_vol=self.config.max_vol,
            S_mean=self.config.S_mean,
            S_std=self.config.S_std,
            seed=seed,
        )
        
        # Create solver
        self._solver = self._create_solver()
    
    def _create_solver(self) -> SDESolver:
        """Create SDE solver based on config."""
        if self.config.solver_type == "milstein":
            return MilsteinSolver(seed=self.seed)
        else:
            return EulerMaruyamaSolver(seed=self.seed)
    
    def drift(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Evaluate drift function.
        
        Parameters
        ----------
        S : ndarray
            Spot prices.
        t : ndarray
            Times.
        
        Returns
        -------
        ndarray
            Drift values.
        """
        return self.drift_network(S, t)
    
    def diffusion(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Evaluate diffusion function.
        
        Parameters
        ----------
        S : ndarray
            Spot prices.
        t : ndarray
            Times.
        
        Returns
        -------
        ndarray
            Diffusion (volatility) values.
        """
        return self.diffusion_network(S, t)
    
    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
    ) -> np.ndarray:
        """
        Simulate paths under the learned dynamics.
        
        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time horizon.
        n_steps : int
            Number of time steps.
        n_paths : int
            Number of paths.
        
        Returns
        -------
        ndarray
            Paths of shape (n_paths, n_steps + 1).
        """
        # Use networks as drift/diffusion functions
        paths = self._solver.solve(
            drift=self._drift_wrapper,
            diffusion=self._diffusion_wrapper,
            S0=S0,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
        )
        
        return paths
    
    def _drift_wrapper(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Wrapper for drift network."""
        return self.drift_network(S, t)
    
    def _diffusion_wrapper(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Wrapper for diffusion network."""
        return self.diffusion_network(S, t)
    
    def compute_statistics(
        self,
        S0: float,
        T: float,
        n_steps: int = 252,
        n_paths: int = 10000,
    ) -> Dict[str, float]:
        """
        Compute summary statistics from simulated paths.
        
        Parameters
        ----------
        S0 : float
            Initial spot.
        T : float
            Time horizon.
        n_steps : int
            Time steps.
        n_paths : int
            Number of paths.
        
        Returns
        -------
        dict
            Statistics including mean, std, skew, kurtosis.
        """
        paths = self.simulate(S0, T, n_steps, n_paths)
        
        final_prices = paths[:, -1]
        returns = np.log(paths[:, -1] / paths[:, 0])
        
        return {
            "mean_final": float(np.mean(final_prices)),
            "std_final": float(np.std(final_prices)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "skewness": float(self._compute_skewness(returns)),
            "kurtosis": float(self._compute_kurtosis(returns)),
        }
    
    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness."""
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-8:
            return 0.0
        return float(np.mean(((x - m) / s) ** 3))
    
    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis."""
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-8:
            return 0.0
        return float(np.mean(((x - m) / s) ** 4) - 3)
    
    def save(self, path: str) -> None:
        """
        Save model parameters to file.
        
        Parameters
        ----------
        path : str
            File path.
        """
        params = {
            "drift": self.drift_network.get_parameters(),
            "diffusion": self.diffusion_network.get_parameters(),
            "config": {
                "drift_hidden_dims": self.config.drift_hidden_dims,
                "diffusion_hidden_dims": self.config.diffusion_hidden_dims,
                "activation": self.config.activation,
                "min_vol": self.config.min_vol,
                "max_vol": self.config.max_vol,
                "solver_type": self.config.solver_type,
                "S_mean": self.config.S_mean,
                "S_std": self.config.S_std,
            },
        }
        path_str = str(path)
        if not path_str.endswith(".npy"):
            path_str = path_str + ".npy"
        np.save(path_str, params, allow_pickle=True)
    
    @classmethod
    def load(cls, path: str) -> "NeuralSDEDynamics":
        """
        Load model from file.
        
        Parameters
        ----------
        path : str
            File path.
        
        Returns
        -------
        NeuralSDEDynamics
            Loaded model.
        """
        path_str = str(path)
        if not path_str.endswith(".npy"):
            path_str = path_str + ".npy"
        params = np.load(path_str, allow_pickle=True).item()
        
        config = NeuralSDEConfig(**params["config"])
        model = cls(config=config)
        
        model.drift_network.set_parameters(params["drift"])
        model.diffusion_network.set_parameters(params["diffusion"])
        
        return model


__all__ = [
    "NeuralSDEDynamics",
    "NeuralSDEConfig",
]
