"""
Main time series generator for risk factor simulation.

This module provides the core TimeseriesGenerator class that orchestrates
correlated risk factor path simulation using various dynamics models.

Architecture
------------
1. Generate independent standard normal shocks Z[t,s,f]
2. Apply Cholesky decomposition to correlate shocks: Z_corr = Z @ L^T
3. For each factor, use the appropriate DynamicsAdapter to transform
   correlated shocks into paths
4. Build MarketDataset from generated paths

Correlation Handling
--------------------
- Cross-factor correlation is handled via Cholesky decomposition
- Internal correlations (e.g., Heston spot-vol) are handled within adapters
- For Heston factors, only spot Brownian receives cross-factor correlation;
  variance Brownian uses internal correlation with spot Brownian
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel
from src.marketdata.scenarios.timeseries.config import (
    TimeseriesConfig,
    RiskFactorSpec,
    GBMDynamicsSpec,
    HestonDynamicsSpec,
    OUDynamicsSpec,
    FactorDynamicsSpec,
)
from src.marketdata.scenarios.timeseries.adapters import (
    DynamicsAdapter,
    GBMAdapter,
    HestonAdapter,
    OUAdapter,
    FactorAdapter,
)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """
    Result container for time series generation.

    Attributes
    ----------
    paths : Dict[MarketId, np.ndarray]
        Simulated paths for each risk factor.
        Each array has shape (n_time + 1, n_scenarios).
    variance_paths : Dict[MarketId, np.ndarray]
        Variance paths for stochastic vol factors (Heston).
        NaN for factors without variance (GBM, OU).
    dates : List[str]
        List of dates corresponding to time axis.
    n_scenarios : int
        Number of scenarios.
    seed : int
        Random seed used for generation.
    """

    paths: Dict[MarketId, np.ndarray]
    variance_paths: Dict[MarketId, np.ndarray]
    dates: List[str]
    n_scenarios: int
    seed: int


@dataclass(slots=True)
class TimeseriesGenerator:
    """
    Main orchestrator for correlated risk factor time series generation.

    This generator:
    1. Takes a TimeseriesConfig specifying factors, correlation, and dates
    2. Generates correlated Gaussian shocks using Cholesky decomposition
    3. Applies appropriate dynamics adapters to transform shocks into paths
    4. Produces a MarketDataset suitable for downstream pricing/risk workflows

    Parameters
    ----------
    config : TimeseriesConfig
        Configuration specifying factors, correlation, dates, and scenarios.

    Examples
    --------
    >>> import numpy as np
    >>> from src.marketdata.core.ids import MarketId
    >>> from src.marketdata.generation import (
    ...     TimeseriesGenerator,
    ...     TimeseriesConfig,
    ...     RiskFactorSpec,
    ...     GBMDynamicsSpec,
    ... )
    >>>
    >>> # Define correlated FX factors
    >>> factors = [
    ...     RiskFactorSpec(
    ...         market_id=MarketId("FX", "SPOT", "EURUSD"),
    ...         initial_value=1.08,
    ...         dynamics=GBMDynamicsSpec(drift=0.0, vol=0.08),
    ...     ),
    ...     RiskFactorSpec(
    ...         market_id=MarketId("FX", "SPOT", "GBPUSD"),
    ...         initial_value=1.26,
    ...         dynamics=GBMDynamicsSpec(drift=0.0, vol=0.09),
    ...     ),
    ... ]
    >>>
    >>> correlation = np.array([[1.0, 0.6], [0.6, 1.0]])
    >>>
    >>> config = TimeseriesConfig(
    ...     factors=factors,
    ...     correlation=correlation,
    ...     start_date="2024-01-01",
    ...     end_date="2024-12-31",
    ...     freq="D",
    ...     n_scenarios=10000,
    ... )
    >>>
    >>> generator = TimeseriesGenerator(config)
    >>> dataset = generator.generate(seed=42)
    >>>
    >>> # Get market snapshot at time 100, scenario 0
    >>> market = dataset.snapshot(time_idx=100, scenario_idx=0)
    >>> spot = market.quote(MarketId("FX", "SPOT", "EURUSD"))
    """

    config: TimeseriesConfig

    # Internal state
    _adapters: Dict[MarketId, DynamicsAdapter] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Initialize dynamics adapters for each factor."""
        self._adapters = self._create_adapters()

    def _create_adapters(self) -> Dict[MarketId, DynamicsAdapter]:
        """Create appropriate adapter for each risk factor."""
        adapters = {}

        for factor in self.config.factors:
            mkt_id = factor.market_id
            dynamics = factor.dynamics

            if isinstance(dynamics, GBMDynamicsSpec):
                adapters[mkt_id] = GBMAdapter(spec=dynamics)
            elif isinstance(dynamics, HestonDynamicsSpec):
                adapters[mkt_id] = HestonAdapter(spec=dynamics)
            elif isinstance(dynamics, OUDynamicsSpec):
                adapters[mkt_id] = OUAdapter(spec=dynamics)
            elif isinstance(dynamics, FactorDynamicsSpec):
                adapters[mkt_id] = FactorAdapter(spec=dynamics)
            else:
                raise ValueError(f"Unknown dynamics type: {type(dynamics)}")

        return adapters

    def generate(self, seed: Optional[int] = None) -> MarketDataset:
        """
        Generate correlated risk factor time series.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        MarketDataset
            Dataset containing simulated time series for all risk factors.
            Use dataset.snapshot(time_idx, scenario_idx) to get Market objects.
        """
        result = self.generate_paths(seed=seed)
        return self._build_dataset(result)

    def generate_paths(self, seed: Optional[int] = None) -> GenerationResult:
        """
        Generate raw paths without building MarketDataset.

        Useful for debugging or when you need direct access to path arrays.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        GenerationResult
            Container with paths, variance paths, dates, and metadata.
        """
        seed = seed if seed is not None else np.random.default_rng().integers(0, 2**31)
        rng = np.random.default_rng(seed=seed)

        n_time = self.config.n_time_steps
        n_scenarios = self.config.n_scenarios
        n_factors = self.config.n_factors
        dt = self.config.time_step

        # Step 1: Generate independent standard normal shocks
        # Shape: (n_time, n_scenarios, n_factors)
        z_independent = rng.standard_normal((n_time, n_scenarios, n_factors))

        # Step 2: Apply Cholesky correlation
        # L @ z^T for each (t, s) slice
        chol = self.config.cholesky  # Shape: (n_factors, n_factors)
        z_correlated = z_independent @ chol.T  # Shape: (n_time, n_scenarios, n_factors)

        # Step 3: Simulate each factor using its adapter
        paths: Dict[MarketId, np.ndarray] = {}
        variance_paths: Dict[MarketId, np.ndarray] = {}

        for i, factor in enumerate(self.config.factors):
            mkt_id = factor.market_id
            adapter = self._adapters[mkt_id]

            # Extract shocks for this factor: (n_time, n_scenarios)
            factor_shocks = z_correlated[:, :, i]

            # Simulate paths
            if adapter.requires_variance_paths:
                spot_path, var_path = adapter.simulate_with_variance(
                    initial_value=factor.initial_value,
                    n_time=n_time,
                    n_scenarios=n_scenarios,
                    shocks=factor_shocks,
                    dt=dt,
                )
                paths[mkt_id] = spot_path
                variance_paths[mkt_id] = var_path
            else:
                spot_path = adapter.simulate(
                    initial_value=factor.initial_value,
                    n_time=n_time,
                    n_scenarios=n_scenarios,
                    shocks=factor_shocks,
                    dt=dt,
                )
                paths[mkt_id] = spot_path
                variance_paths[mkt_id] = np.full_like(spot_path, np.nan)

        return GenerationResult(
            paths=paths,
            variance_paths=variance_paths,
            dates=self.config.dates,
            n_scenarios=n_scenarios,
            seed=seed,
        )

    def _build_dataset(self, result: GenerationResult) -> MarketDataset:
        """
        Build MarketDataset from generation result.

        Parameters
        ----------
        result : GenerationResult
            Raw paths from generate_paths().

        Returns
        -------
        MarketDataset
            Dataset with panels for each risk factor.
        """
        panels: Dict[MarketId, Panel] = {}

        for mkt_id, path_array in result.paths.items():
            # path_array has shape (n_time + 1, n_scenarios)
            # Panel expects shape (T, S) with axis_names ("time", "scenario")
            panel = Panel(
                data=path_array,
                axis_names=("time", "scenario"),
            )
            panels[mkt_id] = panel

        # For variance paths (Heston), create additional panels
        # Use a modified MarketId with "VARIANCE" mkt_type
        variance_panels: Dict[MarketId, Panel] = {}
        for mkt_id, var_array in result.variance_paths.items():
            if not np.all(np.isnan(var_array)):
                # Only add if we have actual variance data
                var_mkt_id = MarketId(
                    asset_class=mkt_id.asset_class,
                    mkt_type="VARIANCE",
                    name=mkt_id.name,
                )
                variance_panels[var_mkt_id] = Panel(
                    data=var_array,
                    axis_names=("time", "scenario"),
                )

        # Combine all panels
        all_panels = {**panels, **variance_panels}

        return MarketDataset(
            dates=result.dates,
            n_scenarios=result.n_scenarios,
            panels=all_panels,
            curve_params={},
            curve_factories={},
            vol_params={},
            vol_factories={},
            meta={
                "generator": "TimeseriesGenerator",
                "seed": result.seed,
                "n_factors": self.config.n_factors,
                "factor_names": self.config.factor_names(),
            },
        )

    def compute_statistics(self, result: GenerationResult) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for generated paths.

        Parameters
        ----------
        result : GenerationResult
            Raw paths from generate_paths().

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping factor names to statistics dict.
            Statistics include: mean, std, min, max, terminal_mean, terminal_std.
        """
        stats = {}

        for i, factor in enumerate(self.config.factors):
            mkt_id = factor.market_id
            paths = result.paths[mkt_id]

            terminal = paths[-1, :]  # Terminal values across scenarios

            stats[factor.display_name] = {
                "initial": float(factor.initial_value),
                "terminal_mean": float(np.mean(terminal)),
                "terminal_std": float(np.std(terminal)),
                "terminal_min": float(np.min(terminal)),
                "terminal_max": float(np.max(terminal)),
                "path_mean": float(np.mean(paths)),
                "path_std": float(np.std(paths)),
            }

        return stats

    def compute_realized_correlation(
        self,
        result: GenerationResult,
        log_returns: bool = True,
    ) -> np.ndarray:
        """
        Compute realized correlation matrix from generated paths.

        Useful for validating that generated correlation matches config.

        Parameters
        ----------
        result : GenerationResult
            Raw paths from generate_paths().
        log_returns : bool
            If True, compute correlation of log-returns.
            If False, compute correlation of simple returns.

        Returns
        -------
        np.ndarray
            Realized correlation matrix, shape (n_factors, n_factors).
        """
        n_factors = self.config.n_factors
        n_time = self.config.n_time_steps
        n_scenarios = self.config.n_scenarios

        # Collect returns for all factors
        returns = np.empty((n_time * n_scenarios, n_factors))

        for i, factor in enumerate(self.config.factors):
            mkt_id = factor.market_id
            paths = result.paths[mkt_id]

            if log_returns:
                # Log returns: log(S_{t+1}/S_t)
                factor_returns = np.log(paths[1:, :] / paths[:-1, :])
            else:
                # Simple returns: (S_{t+1} - S_t) / S_t
                factor_returns = (paths[1:, :] - paths[:-1, :]) / paths[:-1, :]

            # Flatten (n_time, n_scenarios) -> (n_time * n_scenarios,)
            returns[:, i] = factor_returns.flatten()

        # Compute correlation matrix
        return np.corrcoef(returns.T)
