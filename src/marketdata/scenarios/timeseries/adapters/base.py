"""
Base protocol for dynamics adapters.

All adapters transform correlated Gaussian shocks into risk factor paths.
"""

from __future__ import annotations

import numpy as np
from typing import Protocol, runtime_checkable


@runtime_checkable
class DynamicsAdapter(Protocol):
    """
    Protocol for dynamics adapters.

    An adapter transforms correlated Gaussian shocks Z[t,s] into
    simulated paths X[t,s] for a single risk factor.

    The shocks are pre-correlated via Cholesky decomposition in the
    TimeseriesGenerator, so adapters receive factor-specific shocks.

    Methods
    -------
    simulate(initial_value, n_time, n_scenarios, shocks, dt) -> np.ndarray
        Generate simulated paths from Gaussian shocks.

    Properties
    ----------
    requires_variance_paths : bool
        True if this adapter produces variance paths (e.g., Heston).
        Used by TimeseriesGenerator to allocate variance storage.
    """

    def simulate(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Simulate risk factor paths from Gaussian shocks.

        Parameters
        ----------
        initial_value : float
            Starting value X_0.
        n_time : int
            Number of time steps.
        n_scenarios : int
            Number of scenarios.
        shocks : np.ndarray
            Pre-correlated standard normal shocks, shape (n_time, n_scenarios).
            shocks[t, s] is the shock at time t for scenario s.
        dt : float
            Time step in years.

        Returns
        -------
        np.ndarray
            Simulated paths, shape (n_time + 1, n_scenarios).
            paths[0, :] = initial_value
            paths[t, s] is the value at time t for scenario s.
        """
        ...

    @property
    def requires_variance_paths(self) -> bool:
        """Whether this adapter produces variance paths."""
        return False

    def simulate_with_variance(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths with variance (for Heston and similar models).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (spot_paths, variance_paths), each shape (n_time + 1, n_scenarios).
        """
        paths = self.simulate(initial_value, n_time, n_scenarios, shocks, dt)
        return paths, np.full_like(paths, np.nan)
