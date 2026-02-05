"""
Bootstrap Methods for Historical Simulation.

Provides various bootstrap approaches for resampling historical returns:

1. **Simple Bootstrap**: IID resampling (ignores autocorrelation)
2. **Block Bootstrap**: Fixed-length block resampling (preserves short-term structure)
3. **Stationary Bootstrap**: Random-length blocks (better for stationary series)

Mathematical Framework
----------------------
Given historical log-returns r_1, ..., r_T, we generate synthetic paths by
resampling these returns. The block methods preserve autocorrelation structure.

Block Bootstrap (Künsch, 1989):
- Divide returns into overlapping blocks of length ℓ
- Sample blocks with replacement
- Concatenate to form new series

Stationary Bootstrap (Politis & Romano, 1994):
- Block length is geometric random variable with mean ℓ
- Results in stationary resampled series

References
----------
- Künsch, H.R. (1989). "The Jackknife and the Bootstrap for General Stationary
  Observations." Annals of Statistics.
- Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap."
  Journal of the American Statistical Association.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """
    Configuration for bootstrap resampling.

    Parameters
    ----------
    historical_returns : np.ndarray
        Historical log-returns, shape (n_assets, n_observations).
    block_length : int
        Expected block length for block/stationary bootstrap.
        Ignored for simple bootstrap (method="iid").
    method : str
        Bootstrap method: "iid", "block", or "stationary".
    circular : bool
        If True, wrap around at data boundaries (avoids edge effects).

    Examples
    --------
    >>> returns = np.random.randn(5, 1000) * 0.01  # 5 assets, 1000 days
    >>> config = BootstrapConfig(
    ...     historical_returns=returns,
    ...     block_length=20,
    ...     method="stationary",
    ... )
    """

    historical_returns: np.ndarray
    block_length: int = 20
    method: Literal["iid", "block", "stationary"] = "block"
    circular: bool = True

    def __post_init__(self) -> None:
        if self.historical_returns.ndim != 2:
            raise ValueError(
                "historical_returns must be 2D: (n_assets, n_observations)"
            )
        if self.block_length < 1:
            raise ValueError("block_length must be >= 1")
        if self.method not in ("iid", "block", "stationary"):
            raise ValueError(
                f"method must be 'iid', 'block', or 'stationary', got {self.method}"
            )

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.historical_returns.shape[0]

    @property
    def n_observations(self) -> int:
        """Number of historical observations."""
        return self.historical_returns.shape[1]


@dataclass(slots=True)
class BlockBootstrap:
    """
    Block Bootstrap for multivariate time series.

    Preserves cross-sectional dependence by resampling blocks of
    returns across all assets simultaneously.

    Parameters
    ----------
    config : BootstrapConfig
        Bootstrap configuration.
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> returns = np.random.randn(3, 500) * 0.01  # 3 assets, 500 days
    >>> config = BootstrapConfig(returns, block_length=10)
    >>> bootstrap = BlockBootstrap(config)
    >>>
    >>> # Generate 10000 scenarios of 252-day returns
    >>> scenarios = bootstrap.sample(n_scenarios=10000, horizon=252)
    >>> print(scenarios.shape)  # (3, 252, 10000)
    """

    config: BootstrapConfig
    _rng: np.random.Generator = None

    def __post_init__(self) -> None:
        pass

    def _init_rng(self, seed: Optional[int] = None) -> None:
        """Initialize random number generator."""
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate bootstrap scenarios.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios to generate.
        horizon : int
            Length of each scenario (time steps).
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Bootstrapped returns, shape (n_assets, horizon, n_scenarios).
        """
        self._init_rng(seed)

        n_assets = self.config.n_assets
        n_obs = self.config.n_observations
        block_len = self.config.block_length
        returns = self.config.historical_returns

        # Output array
        scenarios = np.empty((n_assets, horizon, n_scenarios), dtype=np.float64)

        for s in range(n_scenarios):
            # Generate block starting indices
            n_blocks = int(np.ceil(horizon / block_len))
            
            if self.config.circular:
                # Circular: any starting point is valid
                starts = self._rng.integers(0, n_obs, size=n_blocks)
            else:
                # Non-circular: must leave room for full block
                max_start = max(1, n_obs - block_len)
                starts = self._rng.integers(0, max_start, size=n_blocks)

            # Extract and concatenate blocks
            t = 0
            for start in starts:
                for b in range(block_len):
                    if t >= horizon:
                        break
                    idx = (start + b) % n_obs if self.config.circular else start + b
                    scenarios[:, t, s] = returns[:, idx]
                    t += 1

        return scenarios

    def sample_paths(
        self,
        initial_values: np.ndarray,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate price paths from bootstrapped returns.

        Parameters
        ----------
        initial_values : np.ndarray
            Initial prices for each asset, shape (n_assets,).
        n_scenarios : int
            Number of scenarios.
        horizon : int
            Path length.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Price paths, shape (n_assets, horizon + 1, n_scenarios).
        """
        returns = self.sample(n_scenarios, horizon, seed)
        
        # Convert returns to prices
        n_assets = self.config.n_assets
        paths = np.empty((n_assets, horizon + 1, n_scenarios), dtype=np.float64)
        
        # Set initial values
        paths[:, 0, :] = initial_values[:, np.newaxis]
        
        # Cumulative product of returns
        for t in range(horizon):
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(returns[:, t, :])
        
        return paths


@dataclass(slots=True)
class StationaryBootstrap:
    """
    Stationary Bootstrap (Politis & Romano, 1994).

    Uses geometrically distributed block lengths, resulting in a
    stationary resampled series (better properties for inference).

    Parameters
    ----------
    config : BootstrapConfig
        Bootstrap configuration (block_length = expected block length).
    seed : int, optional
        Random seed.

    Notes
    -----
    The expected block length is 1/p where p is the probability of
    ending a block at each step. For block_length = ℓ, p = 1/ℓ.

    Examples
    --------
    >>> returns = np.random.randn(3, 500) * 0.01
    >>> config = BootstrapConfig(returns, block_length=20, method="stationary")
    >>> bootstrap = StationaryBootstrap(config)
    >>>
    >>> scenarios = bootstrap.sample(n_scenarios=10000, horizon=252)
    """

    config: BootstrapConfig
    _rng: np.random.Generator = None

    def _init_rng(self, seed: Optional[int] = None) -> None:
        """Initialize random number generator."""
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate stationary bootstrap scenarios.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios.
        horizon : int
            Length of each scenario.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Bootstrapped returns, shape (n_assets, horizon, n_scenarios).
        """
        self._init_rng(seed)

        n_assets = self.config.n_assets
        n_obs = self.config.n_observations
        block_len = self.config.block_length
        returns = self.config.historical_returns

        # Probability of ending block at each step
        p = 1.0 / block_len

        scenarios = np.empty((n_assets, horizon, n_scenarios), dtype=np.float64)

        for s in range(n_scenarios):
            # Start at random position
            idx = self._rng.integers(0, n_obs)
            
            for t in range(horizon):
                # Copy return for all assets
                scenarios[:, t, s] = returns[:, idx % n_obs]
                
                # Decide whether to continue block or start new
                if self._rng.random() < p:
                    # Start new block at random position
                    idx = self._rng.integers(0, n_obs)
                else:
                    # Continue current block
                    idx += 1

        return scenarios

    def sample_paths(
        self,
        initial_values: np.ndarray,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate price paths from bootstrapped returns.

        Parameters
        ----------
        initial_values : np.ndarray
            Initial prices, shape (n_assets,).
        n_scenarios : int
            Number of scenarios.
        horizon : int
            Path length.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Price paths, shape (n_assets, horizon + 1, n_scenarios).
        """
        returns = self.sample(n_scenarios, horizon, seed)
        
        n_assets = self.config.n_assets
        paths = np.empty((n_assets, horizon + 1, n_scenarios), dtype=np.float64)
        paths[:, 0, :] = initial_values[:, np.newaxis]
        
        for t in range(horizon):
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(returns[:, t, :])
        
        return paths
