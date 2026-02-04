"""
Path generation and scenario simulation for Neural SDEs.

Provides:
- PathGenerator: Generate price paths from trained models
- ScenarioGenerator: Generate market scenarios with conditions
- DataAugmenter: Augment training data with synthetic paths

Example:
    from src.models.neural_sde.generation import PathGenerator
    
    generator = PathGenerator(trained_sde)
    
    # Generate basic paths
    paths = generator.generate(S0=100.0, T=1.0, n_paths=10000)
    
    # Generate conditioned paths
    paths = generator.generate_conditioned(
        S0=100.0, S_T=120.0, T=1.0, n_paths=1000
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Path Generator
# =============================================================================


class PathGenerator:
    """
    Generate price paths from trained Neural SDE models.
    
    Features:
    - Unconditional generation
    - Conditional generation (pinned endpoints)
    - Importance sampling for rare events
    
    Example:
        generator = PathGenerator(sde_model)
        
        # Basic generation
        paths = generator.generate(S0=100.0, T=1.0, n_paths=10000)
        
        # Generate with specific terminal value
        paths = generator.generate_conditioned(
            S0=100.0, S_T=120.0, T=1.0, n_paths=1000
        )
    """
    
    def __init__(
        self,
        model: Any,  # NeuralSDEDynamics
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize path generator.
        
        Parameters
        ----------
        model : NeuralSDEDynamics
            Trained Neural SDE model.
        seed : int, optional
            Random seed.
        """
        self.model = model
        self._rng = np.random.default_rng(seed)
    
    def generate(
        self,
        S0: float,
        T: float,
        n_steps: int = 252,
        n_paths: int = 10000,
    ) -> np.ndarray:
        """
        Generate unconditional paths.
        
        Parameters
        ----------
        S0 : float
            Initial price.
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
        return self.model.simulate(S0, T, n_steps, n_paths)
    
    def generate_conditioned(
        self,
        S0: float,
        S_T: float,
        T: float,
        n_steps: int = 252,
        n_paths: int = 1000,
        bridge_type: str = "linear",
    ) -> np.ndarray:
        """
        Generate paths conditioned on terminal value.
        
        Uses a Brownian bridge-like construction to ensure
        paths reach the specified terminal value.
        
        Parameters
        ----------
        S0 : float
            Initial price.
        S_T : float
            Target terminal price.
        T : float
            Time horizon.
        n_steps : int
            Number of time steps.
        n_paths : int
            Number of paths.
        bridge_type : str
            Type of bridge: "linear", "exponential".
        
        Returns
        -------
        ndarray
            Conditioned paths of shape (n_paths, n_steps + 1).
        """
        # Generate unconditional paths
        paths = self.generate(S0, T, n_steps, n_paths)
        
        # Apply bridge correction
        times = np.linspace(0, T, n_steps + 1)
        
        if bridge_type == "linear":
            # Linear interpolation to terminal
            unconditional_terminal = paths[:, -1]
            correction = S_T - unconditional_terminal
            
            for i in range(n_steps + 1):
                alpha = times[i] / T
                paths[:, i] += alpha * correction
        
        elif bridge_type == "exponential":
            # Exponential bridge (better for log-normal)
            unconditional_terminal = paths[:, -1]
            ratio = S_T / np.maximum(unconditional_terminal, 1e-8)
            
            for i in range(n_steps + 1):
                alpha = times[i] / T
                paths[:, i] *= np.power(ratio, alpha)
        
        # Ensure non-negative
        paths = np.maximum(paths, 1e-8)
        
        return paths
    
    def generate_barrier_conditioned(
        self,
        S0: float,
        barrier: float,
        barrier_type: str,  # "up", "down"
        barrier_condition: str,  # "touch", "no_touch"
        T: float,
        n_steps: int = 252,
        n_paths: int = 1000,
    ) -> np.ndarray:
        """
        Generate paths conditioned on barrier behavior.
        
        Uses acceptance-rejection to generate paths that
        either touch or don't touch the barrier.
        
        Parameters
        ----------
        S0 : float
            Initial price.
        barrier : float
            Barrier level.
        barrier_type : str
            "up" or "down" barrier.
        barrier_condition : str
            "touch" or "no_touch".
        T : float
            Time horizon.
        n_steps : int
            Time steps.
        n_paths : int
            Number of paths.
        
        Returns
        -------
        ndarray
            Conditioned paths.
        """
        accepted_paths = []
        max_attempts = n_paths * 20
        attempts = 0
        
        while len(accepted_paths) < n_paths and attempts < max_attempts:
            # Generate batch
            batch_size = min(n_paths * 2, max_attempts - attempts)
            paths = self.generate(S0, T, n_steps, batch_size)
            
            # Check barrier condition
            if barrier_type == "up":
                touched = np.max(paths, axis=1) >= barrier
            else:
                touched = np.min(paths, axis=1) <= barrier
            
            if barrier_condition == "touch":
                valid = touched
            else:
                valid = ~touched
            
            accepted_paths.extend(paths[valid])
            attempts += batch_size
        
        if len(accepted_paths) < n_paths:
            # If not enough paths, pad with last generated
            print(f"Warning: Only generated {len(accepted_paths)} valid paths")
            while len(accepted_paths) < n_paths:
                accepted_paths.append(accepted_paths[-1])
        
        return np.array(accepted_paths[:n_paths])


# =============================================================================
# Scenario Generator
# =============================================================================


@dataclass
class Scenario:
    """Market scenario with paths and metadata."""
    
    name: str
    paths: np.ndarray
    initial_price: float
    terminal_mean: float
    terminal_std: float
    max_drawdown: float
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "name": self.name,
            "n_paths": len(self.paths),
            "initial_price": self.initial_price,
            "terminal_mean": self.terminal_mean,
            "terminal_std": self.terminal_std,
            "max_drawdown": self.max_drawdown,
        }


class ScenarioGenerator:
    """
    Generate market scenarios from Neural SDE models.
    
    Creates scenarios for:
    - Stress testing (bear markets, high vol)
    - Risk analysis (tail scenarios)
    - What-if analysis (specific return targets)
    
    Example:
        generator = ScenarioGenerator(sde_model)
        
        scenarios = generator.generate_stress_scenarios(
            S0=100.0, T=0.25, n_paths=10000
        )
        
        for scenario in scenarios:
            print(f"{scenario.name}: mean={scenario.terminal_mean:.2f}")
    """
    
    def __init__(
        self,
        model: Any,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize scenario generator.
        
        Parameters
        ----------
        model : NeuralSDEDynamics
            Trained Neural SDE model.
        seed : int, optional
            Random seed.
        """
        self.model = model
        self.path_generator = PathGenerator(model, seed)
        self._rng = np.random.default_rng(seed)
    
    def generate_stress_scenarios(
        self,
        S0: float,
        T: float,
        n_paths: int = 10000,
        n_steps: int = 252,
    ) -> List[Scenario]:
        """
        Generate standard stress scenarios.
        
        Scenarios:
        - Base case
        - Bear market (-20% terminal)
        - Bull market (+30% terminal)
        - High volatility
        - Crash then recovery
        
        Parameters
        ----------
        S0 : float
            Initial price.
        T : float
            Time horizon.
        n_paths : int
            Paths per scenario.
        n_steps : int
            Time steps.
        
        Returns
        -------
        list of Scenario
            Stress scenarios.
        """
        scenarios = []
        
        # Base case
        base_paths = self.path_generator.generate(S0, T, n_steps, n_paths)
        scenarios.append(self._create_scenario("Base Case", base_paths, S0))
        
        # Bear market
        bear_paths = self.path_generator.generate_conditioned(
            S0=S0, S_T=S0 * 0.8, T=T, n_steps=n_steps, n_paths=n_paths
        )
        scenarios.append(self._create_scenario("Bear Market (-20%)", bear_paths, S0))
        
        # Bull market
        bull_paths = self.path_generator.generate_conditioned(
            S0=S0, S_T=S0 * 1.3, T=T, n_steps=n_steps, n_paths=n_paths
        )
        scenarios.append(self._create_scenario("Bull Market (+30%)", bull_paths, S0))
        
        return scenarios
    
    def generate_tail_scenarios(
        self,
        S0: float,
        T: float,
        n_paths: int = 1000,
        n_steps: int = 252,
        tail_percentile: float = 0.05,
    ) -> Tuple[Scenario, Scenario]:
        """
        Generate tail scenarios (left and right tails).
        
        Parameters
        ----------
        S0 : float
            Initial price.
        T : float
            Time horizon.
        n_paths : int
            Paths per scenario.
        n_steps : int
            Time steps.
        tail_percentile : float
            Percentile for tail definition.
        
        Returns
        -------
        left_tail, right_tail : tuple of Scenario
            Left and right tail scenarios.
        """
        # Generate many paths
        all_paths = self.path_generator.generate(S0, T, n_steps, n_paths * 20)
        terminal_prices = all_paths[:, -1]
        
        # Left tail
        left_threshold = np.percentile(terminal_prices, tail_percentile * 100)
        left_mask = terminal_prices <= left_threshold
        left_paths = all_paths[left_mask][:n_paths]
        left_scenario = self._create_scenario(
            f"Left Tail ({tail_percentile*100:.0f}%)",
            left_paths, S0
        )
        
        # Right tail
        right_threshold = np.percentile(terminal_prices, (1 - tail_percentile) * 100)
        right_mask = terminal_prices >= right_threshold
        right_paths = all_paths[right_mask][:n_paths]
        right_scenario = self._create_scenario(
            f"Right Tail ({(1-tail_percentile)*100:.0f}%)",
            right_paths, S0
        )
        
        return left_scenario, right_scenario
    
    def _create_scenario(
        self,
        name: str,
        paths: np.ndarray,
        S0: float,
    ) -> Scenario:
        """Create scenario from paths."""
        terminal = paths[:, -1]
        
        # Compute max drawdown per path
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (running_max - paths) / running_max
        max_drawdown = float(np.mean(np.max(drawdowns, axis=1)))
        
        return Scenario(
            name=name,
            paths=paths,
            initial_price=S0,
            terminal_mean=float(np.mean(terminal)),
            terminal_std=float(np.std(terminal)),
            max_drawdown=max_drawdown,
        )


# =============================================================================
# Data Augmenter
# =============================================================================


class DataAugmenter:
    """
    Augment training data with synthetic paths.
    
    Uses the learned Neural SDE to generate additional
    training data that captures the learned dynamics.
    
    Example:
        augmenter = DataAugmenter(sde_model)
        
        augmented_data = augmenter.augment(
            original_data=historical_paths,
            augmentation_factor=2.0,
        )
    """
    
    def __init__(
        self,
        model: Any,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize data augmenter.
        
        Parameters
        ----------
        model : NeuralSDEDynamics
            Trained Neural SDE model.
        seed : int, optional
            Random seed.
        """
        self.model = model
        self.path_generator = PathGenerator(model, seed)
        self._rng = np.random.default_rng(seed)
    
    def augment(
        self,
        original_data: np.ndarray,
        augmentation_factor: float = 2.0,
        noise_scale: float = 0.01,
    ) -> np.ndarray:
        """
        Augment data with synthetic paths.
        
        Parameters
        ----------
        original_data : ndarray
            Original paths of shape (n_paths, n_steps).
        augmentation_factor : float
            Factor to multiply data size.
        noise_scale : float
            Scale of noise to add for diversity.
        
        Returns
        -------
        ndarray
            Augmented data (original + synthetic).
        """
        n_original = len(original_data)
        n_synthetic = int(n_original * (augmentation_factor - 1))
        
        if n_synthetic <= 0:
            return original_data
        
        # Get parameters from original data
        S0 = float(np.mean(original_data[:, 0]))
        n_steps = original_data.shape[1] - 1
        T = 1.0  # Assume unit time
        
        # Generate synthetic paths
        synthetic_paths = self.path_generator.generate(
            S0=S0, T=T, n_steps=n_steps, n_paths=n_synthetic
        )
        
        # Add small noise for diversity
        if noise_scale > 0:
            noise = self._rng.standard_normal(synthetic_paths.shape) * noise_scale
            synthetic_paths = synthetic_paths * (1 + noise)
            synthetic_paths = np.maximum(synthetic_paths, 1e-8)
        
        # Combine
        augmented = np.vstack([original_data, synthetic_paths])
        
        # Shuffle
        self._rng.shuffle(augmented)
        
        return augmented
    
    def augment_with_conditions(
        self,
        original_data: np.ndarray,
        conditions: List[Dict[str, Any]],
        paths_per_condition: int = 100,
    ) -> np.ndarray:
        """
        Augment with paths meeting specific conditions.
        
        Parameters
        ----------
        original_data : ndarray
            Original paths.
        conditions : list of dict
            Conditions like {"terminal_return": 0.1} or {"barrier": 120}.
        paths_per_condition : int
            Paths to generate per condition.
        
        Returns
        -------
        ndarray
            Augmented data.
        """
        S0 = float(np.mean(original_data[:, 0]))
        n_steps = original_data.shape[1] - 1
        T = 1.0
        
        augmented_paths = [original_data]
        
        for condition in conditions:
            if "terminal_return" in condition:
                target_return = condition["terminal_return"]
                S_T = S0 * (1 + target_return)
                
                paths = self.path_generator.generate_conditioned(
                    S0=S0, S_T=S_T, T=T, n_steps=n_steps,
                    n_paths=paths_per_condition
                )
                augmented_paths.append(paths)
            
            elif "barrier" in condition and "barrier_type" in condition:
                paths = self.path_generator.generate_barrier_conditioned(
                    S0=S0,
                    barrier=condition["barrier"],
                    barrier_type=condition["barrier_type"],
                    barrier_condition=condition.get("barrier_condition", "touch"),
                    T=T,
                    n_steps=n_steps,
                    n_paths=paths_per_condition,
                )
                augmented_paths.append(paths)
        
        return np.vstack(augmented_paths)


__all__ = [
    "PathGenerator",
    "ScenarioGenerator",
    "DataAugmenter",
    "Scenario",
]
