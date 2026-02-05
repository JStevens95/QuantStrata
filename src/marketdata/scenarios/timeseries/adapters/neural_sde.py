"""
Neural SDE adapter for time series generation.

This adapter integrates trained Neural SDE models from `src/models/neural_sde/`
into the TimeseriesGenerator framework.

Neural SDE Model
----------------
The learned SDE has the form:

    dS_t = μ_θ(S_t, t) dt + σ_θ(S_t, t) dW_t

where μ_θ and σ_θ are neural networks parameterized by θ.

Advantages over Parametric Models
---------------------------------
1. **Learned Dynamics**: Captures complex market behavior from data
2. **Non-Gaussian**: Learns non-Gaussian return distributions
3. **Time-Varying**: Learns regime-dependent dynamics implicitly
4. **Conditional Generation**: Can generate paths conditioned on endpoints

When to Use
-----------
- When parametric models (GBM, Heston) don't fit historical data well
- When you have sufficient historical data to train on
- For stress testing with learned tail behavior
- For data augmentation in ML pipelines

Prerequisites
-------------
Requires a trained NeuralSDEDynamics model from `src/models/neural_sde/`.
See `examples/pipelines/run_train_neural_sde.py` for training.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

# Avoid circular import - only import for type checking
if TYPE_CHECKING:
    from src.models.neural_sde.dynamics import NeuralSDEDynamics
    from src.models.neural_sde.generation import PathGenerator


@dataclass(frozen=True, slots=True)
class NeuralSDEDynamicsSpec:
    """
    Specification for Neural SDE dynamics.

    Parameters
    ----------
    model : NeuralSDEDynamics or path to model
        Trained Neural SDE model, or path to saved model weights.
    drift_adjustment : float
        Additional drift adjustment (e.g., for risk-neutral pricing).
    conditional_terminal : float, optional
        If specified, generate paths conditioned on this terminal value.
    bridge_type : str
        Type of bridge for conditional generation: "linear" or "exponential".

    Examples
    --------
    >>> # Load trained model
    >>> from src.models.neural_sde.dynamics import NeuralSDEDynamics
    >>> model = NeuralSDEDynamics.load("models/trained_nsde.pt")
    >>>
    >>> spec = NeuralSDEDynamicsSpec(
    ...     model=model,
    ...     drift_adjustment=0.0,
    ... )
    """

    model: Any  # NeuralSDEDynamics or path string
    drift_adjustment: float = 0.0
    conditional_terminal: Optional[float] = None
    bridge_type: str = "exponential"

    def __post_init__(self) -> None:
        if self.bridge_type not in ("linear", "exponential"):
            raise ValueError(
                f"bridge_type must be 'linear' or 'exponential', got {self.bridge_type}"
            )


@dataclass(slots=True)
class NeuralSDEAdapter:
    """
    Adapter for Neural SDE dynamics.

    This adapter wraps a trained NeuralSDEDynamics model and provides
    the standard interface for TimeseriesGenerator.

    Parameters
    ----------
    spec : NeuralSDEDynamicsSpec
        Neural SDE specification with trained model.
    seed : int, optional
        Random seed for reproducibility.

    Notes
    -----
    Unlike other adapters that use the provided shocks directly, the
    NeuralSDEAdapter may ignore the shocks parameter and use its own
    internal sampling, as the trained model has learned its own noise
    distribution.

    Examples
    --------
    >>> from src.models.neural_sde.dynamics import NeuralSDEDynamics
    >>>
    >>> # Load or create trained model
    >>> model = NeuralSDEDynamics.load("models/equity_nsde.pt")
    >>>
    >>> spec = NeuralSDEDynamicsSpec(model=model)
    >>> adapter = NeuralSDEAdapter(spec=spec, seed=42)
    >>>
    >>> # Generate paths (shocks may be ignored)
    >>> paths = adapter.simulate(
    ...     initial_value=100.0,
    ...     n_time=252,
    ...     n_scenarios=10000,
    ...     shocks=np.zeros((252, 10000)),  # May be ignored
    ...     dt=1/252,
    ... )
    """

    spec: NeuralSDEDynamicsSpec
    seed: Optional[int] = None

    # Lazy-loaded generator
    _path_generator: Any = None

    @property
    def requires_variance_paths(self) -> bool:
        """Neural SDE may produce variance paths depending on architecture."""
        return False

    def _get_path_generator(self) -> "PathGenerator":
        """Lazily create PathGenerator from model."""
        if self._path_generator is None:
            from src.models.neural_sde.generation import PathGenerator
            
            model = self.spec.model
            
            # If model is a path string, load it
            if isinstance(model, str):
                from src.models.neural_sde.dynamics import NeuralSDEDynamics
                model = NeuralSDEDynamics.load(model)
            
            self._path_generator = PathGenerator(model, seed=self.seed)
        
        return self._path_generator

    def simulate(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Simulate paths using the trained Neural SDE.

        Parameters
        ----------
        initial_value : float
            Starting value S_0.
        n_time : int
            Number of time steps.
        n_scenarios : int
            Number of scenarios.
        shocks : np.ndarray
            Standard normal shocks (may be ignored by Neural SDE).
        dt : float
            Time step in years.

        Returns
        -------
        np.ndarray
            Simulated paths, shape (n_time + 1, n_scenarios).

        Notes
        -----
        The Neural SDE model uses its own internal noise generation,
        so the provided shocks may be ignored. The seed parameter
        controls reproducibility.
        """
        generator = self._get_path_generator()
        
        # Compute total time horizon
        T = n_time * dt
        
        # Check for conditional generation
        if self.spec.conditional_terminal is not None:
            # Generate conditioned paths
            paths = generator.generate_conditioned(
                S0=initial_value,
                S_T=self.spec.conditional_terminal,
                T=T,
                n_steps=n_time,
                n_paths=n_scenarios,
                bridge_type=self.spec.bridge_type,
            )
        else:
            # Unconditional generation
            paths = generator.generate(
                S0=initial_value,
                T=T,
                n_steps=n_time,
                n_paths=n_scenarios,
            )
        
        # Apply drift adjustment if specified
        if abs(self.spec.drift_adjustment) > 1e-10:
            times = np.linspace(0, T, n_time + 1)
            drift_factor = np.exp(self.spec.drift_adjustment * times)
            paths = paths * drift_factor[np.newaxis, :]
        
        # Transpose from (n_scenarios, n_time+1) to (n_time+1, n_scenarios)
        return paths.T

    def simulate_with_variance(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate Neural SDE paths (variance is N/A, returned as NaN).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (paths, variance_paths filled with NaN).
        """
        paths = self.simulate(initial_value, n_time, n_scenarios, shocks, dt)
        variance = np.full_like(paths, np.nan)
        return paths, variance

    def generate_stress_scenarios(
        self,
        initial_value: float,
        T: float,
        n_steps: int,
        n_scenarios: int,
    ) -> dict[str, np.ndarray]:
        """
        Generate predefined stress scenarios using the trained model.

        Parameters
        ----------
        initial_value : float
            Starting value S_0.
        T : float
            Time horizon in years.
        n_steps : int
            Number of time steps.
        n_scenarios : int
            Scenarios per stress type.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping scenario name to paths.
        """
        from src.models.neural_sde.generation import ScenarioGenerator
        
        model = self.spec.model
        if isinstance(model, str):
            from src.models.neural_sde.dynamics import NeuralSDEDynamics
            model = NeuralSDEDynamics.load(model)
        
        scenario_gen = ScenarioGenerator(model, seed=self.seed)
        scenarios = scenario_gen.generate_stress_scenarios(
            S0=initial_value,
            T=T,
            n_paths=n_scenarios,
            n_steps=n_steps,
        )
        
        return {s.name: s.paths.T for s in scenarios}
