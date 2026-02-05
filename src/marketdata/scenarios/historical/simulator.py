"""
Unified Historical Simulation Interface.

Provides a single entry point for all historical simulation methods:
- Bootstrap (block, stationary)
- Filtered (EWMA, GARCH)
- Regime-aware (coming soon)

Example
-------
>>> config = HistoricalConfig(
...     historical_returns=returns,
...     method="filtered_block",
...     current_volatility=current_vol,
...     block_length=20,
... )
>>> simulator = HistoricalSimulator(config)
>>> dataset = simulator.generate_dataset(
...     initial_values={"FX.SPOT.EUR": 1.10, "FX.SPOT.GBP": 1.25},
...     n_scenarios=10000,
...     horizon=252,
...     start_date="2024-01-01",
... )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, List, Union, Sequence
from datetime import datetime, timedelta

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.panel import Panel
from src.marketdata.core.ids import MarketId


@dataclass(slots=True)
class HistoricalConfig:
    """
    Unified configuration for historical simulation.

    Parameters
    ----------
    historical_returns : np.ndarray
        Historical log-returns, shape (n_assets, n_observations).
    asset_ids : list[str]
        Identifiers for each asset (e.g., "FX.SPOT.EUR").
    method : str
        Simulation method:
        - "bootstrap": Simple IID bootstrap
        - "block": Block bootstrap
        - "stationary": Stationary bootstrap
        - "filtered": Filtered historical (EWMA/GARCH)
        - "filtered_block": Filtered + block bootstrap
    current_volatility : np.ndarray, optional
        Current annualized volatility. Required for filtered methods.
    block_length : int
        Block length for bootstrap methods.
    volatility_method : str
        How to estimate historical vol: "rolling", "ewma", "garch".
    ewma_lambda : float
        EWMA decay factor.
    annualization_factor : float
        Factor to annualize daily returns.

    Examples
    --------
    >>> # Load historical returns for FX pairs
    >>> returns = np.random.randn(3, 1000) * 0.01
    >>> asset_ids = ["FX.SPOT.EUR", "FX.SPOT.GBP", "FX.SPOT.JPY"]
    >>> current_vol = np.array([0.08, 0.10, 0.12])
    >>>
    >>> config = HistoricalConfig(
    ...     historical_returns=returns,
    ...     asset_ids=asset_ids,
    ...     method="filtered_block",
    ...     current_volatility=current_vol,
    ...     block_length=20,
    ... )
    """

    historical_returns: np.ndarray
    asset_ids: Sequence[str]
    method: Literal[
        "bootstrap", "block", "stationary", "filtered", "filtered_block"
    ] = "block"
    current_volatility: Optional[np.ndarray] = None
    block_length: int = 20
    volatility_method: Literal["rolling", "ewma", "garch"] = "ewma"
    ewma_lambda: float = 0.94
    annualization_factor: float = np.sqrt(252)

    def __post_init__(self) -> None:
        if self.historical_returns.ndim != 2:
            raise ValueError("historical_returns must be 2D")
        if len(self.asset_ids) != self.historical_returns.shape[0]:
            raise ValueError("asset_ids length must match number of assets")
        if self.method in ("filtered", "filtered_block"):
            if self.current_volatility is None:
                raise ValueError(
                    "current_volatility required for filtered methods"
                )
            if len(self.current_volatility) != len(self.asset_ids):
                raise ValueError(
                    "current_volatility length must match number of assets"
                )


@dataclass(slots=True)
class HistoricalSimulator:
    """
    Unified Historical Simulation Engine.

    Orchestrates different historical simulation methods and produces
    MarketDataset output compatible with the pricing engine.

    Parameters
    ----------
    config : HistoricalConfig
        Simulation configuration.

    Examples
    --------
    >>> # Create simulator
    >>> simulator = HistoricalSimulator(config)
    >>>
    >>> # Generate raw paths
    >>> paths = simulator.generate_paths(
    ...     initial_values=np.array([1.10, 1.25, 150.0]),
    ...     n_scenarios=10000,
    ...     horizon=252,
    ...     seed=42,
    ... )
    >>> print(paths.shape)  # (3, 253, 10000)
    >>>
    >>> # Generate MarketDataset
    >>> dataset = simulator.generate_dataset(
    ...     initial_values={"FX.SPOT.EUR": 1.10, "FX.SPOT.GBP": 1.25},
    ...     n_scenarios=10000,
    ...     horizon=252,
    ...     start_date="2024-01-01",
    ... )
    """

    config: HistoricalConfig
    _bootstrap: "BlockBootstrap | StationaryBootstrap" = field(
        default=None, init=False
    )
    _filtered: "FilteredHistorical" = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize underlying simulators based on method."""
        from src.marketdata.scenarios.historical.bootstrap import (
            BlockBootstrap,
            StationaryBootstrap,
            BootstrapConfig,
        )
        from src.marketdata.scenarios.historical.filtered import (
            FilteredHistorical,
            FilteredConfig,
        )

        method = self.config.method

        if method in ("bootstrap", "block"):
            boot_config = BootstrapConfig(
                historical_returns=self.config.historical_returns,
                block_length=self.config.block_length,
                method="block",
            )
            self._bootstrap = BlockBootstrap(boot_config)

        elif method == "stationary":
            boot_config = BootstrapConfig(
                historical_returns=self.config.historical_returns,
                block_length=self.config.block_length,
                method="stationary",
            )
            self._bootstrap = StationaryBootstrap(boot_config)

        elif method in ("filtered", "filtered_block"):
            filt_config = FilteredConfig(
                historical_returns=self.config.historical_returns,
                current_volatility=self.config.current_volatility,
                volatility_method=self.config.volatility_method,
                ewma_lambda=self.config.ewma_lambda,
                annualization_factor=self.config.annualization_factor,
            )
            self._filtered = FilteredHistorical(filt_config)

    def generate_returns(
        self,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate scenario returns.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios.
        horizon : int
            Return horizon (number of time steps).
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Returns, shape (n_assets, horizon, n_scenarios).
        """
        method = self.config.method

        if method in ("bootstrap", "block"):
            return self._bootstrap.sample(n_scenarios, horizon, seed)
        elif method == "stationary":
            return self._bootstrap.sample(n_scenarios, horizon, seed)
        elif method == "filtered":
            return self._filtered.sample(n_scenarios, horizon, seed)
        elif method == "filtered_block":
            return self._filtered.sample_block(
                n_scenarios,
                horizon,
                self.config.block_length,
                seed,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_paths(
        self,
        initial_values: np.ndarray,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate price paths.

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
        returns = self.generate_returns(n_scenarios, horizon, seed)

        n_assets = returns.shape[0]
        paths = np.empty((n_assets, horizon + 1, n_scenarios), dtype=np.float64)
        paths[:, 0, :] = initial_values[:, np.newaxis]

        for t in range(horizon):
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(returns[:, t, :])

        return paths

    def generate_dataset(
        self,
        initial_values: Dict[str, float],
        n_scenarios: int,
        horizon: int,
        start_date: str,
        freq: str = "B",
        seed: Optional[int] = None,
    ) -> MarketDataset:
        """
        Generate MarketDataset from historical simulation.

        Parameters
        ----------
        initial_values : dict
            Initial prices keyed by asset_id.
        n_scenarios : int
            Number of scenarios.
        horizon : int
            Number of time steps.
        start_date : str
            Start date (YYYY-MM-DD format).
        freq : str
            Date frequency ("B" for business days, "D" for daily).
        seed : int, optional
            Random seed.

        Returns
        -------
        MarketDataset
            Dataset with simulated paths for each asset.
        """
        # Convert initial values to array
        init_arr = np.array([
            initial_values[aid] for aid in self.config.asset_ids
        ])

        # Generate paths
        paths = self.generate_paths(init_arr, n_scenarios, horizon, seed)

        # Generate dates
        dates = pd.date_range(start=start_date, periods=horizon + 1, freq=freq)
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        # Build panels
        panels: Dict[MarketId, Panel] = {}
        for i, asset_id in enumerate(self.config.asset_ids):
            mkt_id = MarketId.parse(asset_id)

            # Shape: (n_time, n_scenarios)
            asset_paths = paths[i, :, :]

            panel = Panel(
                data=asset_paths,
                axis_names=("time", "scenario"),
            )
            panels[mkt_id] = panel

        return MarketDataset(
            dates=date_strs,
            n_scenarios=n_scenarios,
            panels=panels,
            curve_params={},
            curve_factories={},
            vol_params={},
            vol_factories={},
            meta={
                "generator": "HistoricalSimulator",
                "method": self.config.method,
                "seed": seed,
            },
        )

    def compute_statistics(
        self,
        initial_values: np.ndarray,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute summary statistics of simulated paths.

        Parameters
        ----------
        initial_values : np.ndarray
            Initial prices.
        n_scenarios : int
            Number of scenarios.
        horizon : int
            Path length.
        seed : int, optional
            Random seed.

        Returns
        -------
        dict
            Dictionary with:
            - mean: Mean terminal values (n_assets,)
            - std: Std of terminal values (n_assets,)
            - var_95: 5th percentile (VaR 95%)
            - var_99: 1st percentile (VaR 99%)
            - es_95: Expected shortfall at 95%
            - es_99: Expected shortfall at 99%
            - realized_correlation: Realized return correlation
        """
        paths = self.generate_paths(initial_values, n_scenarios, horizon, seed)
        returns = self.generate_returns(n_scenarios, horizon, seed)

        # Terminal values
        terminal = paths[:, -1, :]

        # Terminal returns
        terminal_returns = np.log(terminal / paths[:, 0, :])

        # Compute realized correlation from returns
        # Flatten time and scenarios for correlation
        n_assets = returns.shape[0]
        returns_flat = returns.reshape(n_assets, -1)  # (n_assets, horizon * n_scenarios)
        realized_corr = np.corrcoef(returns_flat)

        return {
            "mean": np.mean(terminal, axis=1),
            "std": np.std(terminal, axis=1),
            "var_95": np.percentile(terminal_returns, 5, axis=1),
            "var_99": np.percentile(terminal_returns, 1, axis=1),
            "es_95": np.array([
                np.mean(terminal_returns[i, terminal_returns[i] <= np.percentile(terminal_returns[i], 5)])
                for i in range(n_assets)
            ]),
            "es_99": np.array([
                np.mean(terminal_returns[i, terminal_returns[i] <= np.percentile(terminal_returns[i], 1)])
                for i in range(n_assets)
            ]),
            "realized_correlation": realized_corr,
        }
