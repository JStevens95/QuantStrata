"""
SDE Solvers for Neural SDEs.

Numerical solvers for stochastic differential equations:
- Euler-Maruyama: O(sqrt(dt)) strong convergence
- Milstein: O(dt) strong convergence (for scalar SDEs)

dS = μ(S,t)dt + σ(S,t)dW

Example:
    from src.models.neural_sde.solvers import EulerMaruyamaSolver
    
    solver = EulerMaruyamaSolver(seed=42)
    
    # Define drift and diffusion
    def drift(S, t):
        return 0.05 * S  # GBM drift
    
    def diffusion(S, t):
        return 0.2 * S   # GBM diffusion
    
    # Simulate
    paths = solver.solve(
        drift=drift,
        diffusion=diffusion,
        S0=100.0,
        T=1.0,
        n_steps=252,
        n_paths=10000,
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import numpy as np


# =============================================================================
# Type aliases
# =============================================================================

DriftFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
DiffusionFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


# =============================================================================
# Solver Configuration
# =============================================================================


@dataclass
class SolverConfig:
    """Configuration for SDE solvers."""
    
    seed: Optional[int] = None
    antithetic: bool = True
    positivity: bool = True  # Ensure non-negative prices
    min_value: float = 1e-6


# =============================================================================
# Base Solver
# =============================================================================


class SDESolver(ABC):
    """
    Abstract base class for SDE solvers.
    
    Solves SDEs of the form:
        dS = μ(S, t)dt + σ(S, t)dW
    """
    
    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize solver.
        
        Parameters
        ----------
        config : SolverConfig, optional
            Solver configuration.
        seed : int, optional
            Random seed (overrides config).
        """
        self.config = config or SolverConfig()
        self._rng = np.random.default_rng(seed or self.config.seed)
    
    @abstractmethod
    def solve(
        self,
        drift: DriftFunc,
        diffusion: DiffusionFunc,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
    ) -> np.ndarray:
        """
        Solve SDE and return paths.
        
        Parameters
        ----------
        drift : callable
            Drift function μ(S, t) -> float.
        diffusion : callable
            Diffusion function σ(S, t) -> float.
        S0 : float
            Initial value.
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
        pass
    
    def _generate_brownian(
        self,
        n_paths: int,
        n_steps: int,
    ) -> np.ndarray:
        """
        Generate Brownian increments.
        
        Parameters
        ----------
        n_paths : int
            Number of paths.
        n_steps : int
            Number of steps.
        
        Returns
        -------
        ndarray
            Standard normal increments of shape (n_paths, n_steps).
        """
        if self.config.antithetic:
            half = n_paths // 2
            Z = self._rng.standard_normal((half, n_steps))
            Z = np.vstack([Z, -Z])
            
            # Handle odd n_paths
            if n_paths % 2 == 1:
                extra = self._rng.standard_normal((1, n_steps))
                Z = np.vstack([Z, extra])
        else:
            Z = self._rng.standard_normal((n_paths, n_steps))
        
        return Z


# =============================================================================
# Euler-Maruyama Solver
# =============================================================================


class EulerMaruyamaSolver(SDESolver):
    """
    Euler-Maruyama solver for SDEs.
    
    Discretization:
        S_{t+dt} = S_t + μ(S_t, t)dt + σ(S_t, t)√dt * Z
    
    Where Z ~ N(0, 1).
    
    Convergence:
    - Strong: O(sqrt(dt))
    - Weak: O(dt)
    
    Example:
        solver = EulerMaruyamaSolver(seed=42)
        
        # GBM dynamics
        paths = solver.solve(
            drift=lambda S, t: 0.05 * S,
            diffusion=lambda S, t: 0.2 * S,
            S0=100.0,
            T=1.0,
            n_steps=252,
            n_paths=10000,
        )
        
        print(f"Final mean: {paths[:, -1].mean():.2f}")
    """
    
    def solve(
        self,
        drift: DriftFunc,
        diffusion: DiffusionFunc,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
    ) -> np.ndarray:
        """Solve SDE using Euler-Maruyama scheme."""
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        # Generate all Brownian increments
        Z = self._generate_brownian(n_paths, n_steps)
        
        # Time grid
        times = np.linspace(0, T, n_steps + 1)
        
        # Simulate
        S = np.full(n_paths, S0)
        
        for i in range(n_steps):
            t = times[i]
            
            # Evaluate drift and diffusion
            t_arr = np.full(n_paths, t)
            mu = drift(S, t_arr)
            sigma = diffusion(S, t_arr)
            
            # Euler-Maruyama step
            dW = sqrt_dt * Z[:, i]
            S = S + mu * dt + sigma * dW
            
            # Enforce positivity if required
            if self.config.positivity:
                S = np.maximum(S, self.config.min_value)
            
            paths[:, i + 1] = S
        
        return paths


# =============================================================================
# Milstein Solver
# =============================================================================


class MilsteinSolver(SDESolver):
    """
    Milstein solver for SDEs.
    
    Higher-order scheme that improves strong convergence.
    
    Discretization:
        S_{t+dt} = S_t + μ(S_t, t)dt + σ(S_t, t)√dt * Z
                   + 0.5 * σ(S_t, t) * σ'(S_t, t) * (Z² - 1)dt
    
    Where σ' = ∂σ/∂S (computed numerically).
    
    Convergence:
    - Strong: O(dt)
    - Weak: O(dt)
    
    Example:
        solver = MilsteinSolver(seed=42)
        
        paths = solver.solve(
            drift=lambda S, t: 0.05 * S,
            diffusion=lambda S, t: 0.2 * S,
            S0=100.0,
            T=1.0,
            n_steps=252,
            n_paths=10000,
        )
    """
    
    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        seed: Optional[int] = None,
        bump_size: float = 1e-4,
    ) -> None:
        """
        Initialize Milstein solver.
        
        Parameters
        ----------
        config : SolverConfig, optional
            Solver configuration.
        seed : int, optional
            Random seed.
        bump_size : float
            Relative bump for numerical derivative.
        """
        super().__init__(config, seed)
        self.bump_size = bump_size
    
    def solve(
        self,
        drift: DriftFunc,
        diffusion: DiffusionFunc,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
    ) -> np.ndarray:
        """Solve SDE using Milstein scheme."""
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        # Generate all Brownian increments
        Z = self._generate_brownian(n_paths, n_steps)
        
        # Time grid
        times = np.linspace(0, T, n_steps + 1)
        
        # Simulate
        S = np.full(n_paths, S0)
        
        for i in range(n_steps):
            t = times[i]
            t_arr = np.full(n_paths, t)
            
            # Evaluate drift and diffusion
            mu = drift(S, t_arr)
            sigma = diffusion(S, t_arr)
            
            # Compute σ' numerically (∂σ/∂S)
            dS = np.maximum(np.abs(S) * self.bump_size, 1e-8)
            sigma_up = diffusion(S + dS, t_arr)
            sigma_prime = (sigma_up - sigma) / dS
            
            # Milstein step
            dW = sqrt_dt * Z[:, i]
            milstein_correction = 0.5 * sigma * sigma_prime * (dW**2 - dt)
            
            S = S + mu * dt + sigma * dW + milstein_correction
            
            # Enforce positivity
            if self.config.positivity:
                S = np.maximum(S, self.config.min_value)
            
            paths[:, i + 1] = S
        
        return paths


# =============================================================================
# Log-Euler Solver (for GBM-like SDEs)
# =============================================================================


class LogEulerSolver(SDESolver):
    """
    Log-Euler solver for SDEs.
    
    Applies Euler-Maruyama to log(S), which is exact for GBM
    and more stable for positive processes.
    
    For dS = μ(S,t)Sdt + σ(S,t)SdW:
        d(log S) = (μ - σ²/2)dt + σdW
    
    Example:
        solver = LogEulerSolver(seed=42)
        
        # GBM (exact)
        paths = solver.solve(
            drift_rate=lambda S, t: 0.05,
            vol_rate=lambda S, t: 0.2,
            S0=100.0,
            T=1.0,
            n_steps=252,
            n_paths=10000,
        )
    """
    
    def solve(
        self,
        drift: DriftFunc,
        diffusion: DiffusionFunc,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
    ) -> np.ndarray:
        """
        Solve SDE using log transformation.
        
        Note: drift and diffusion should be rate functions:
            dS/S = μ(S,t)dt + σ(S,t)dW
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        # Generate Brownian increments
        Z = self._generate_brownian(n_paths, n_steps)
        
        # Time grid
        times = np.linspace(0, T, n_steps + 1)
        
        # Work in log space
        log_S = np.full(n_paths, np.log(S0))
        
        for i in range(n_steps):
            t = times[i]
            S = np.exp(log_S)
            t_arr = np.full(n_paths, t)
            
            # Get drift and vol rates
            mu = drift(S, t_arr)  # Drift rate
            sigma = diffusion(S, t_arr)  # Vol rate
            
            # Log dynamics
            dW = sqrt_dt * Z[:, i]
            log_S = log_S + (mu - 0.5 * sigma**2) * dt + sigma * dW
            
            paths[:, i + 1] = np.exp(log_S)
        
        return paths


__all__ = [
    "SDESolver",
    "EulerMaruyamaSolver",
    "MilsteinSolver",
    "LogEulerSolver",
    "SolverConfig",
]
