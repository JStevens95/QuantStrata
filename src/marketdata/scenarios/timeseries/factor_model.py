"""
Factor Model Scenario Generator for Production Market Data.

This module generates correlated scenarios for full term structures
(curves, surfaces) using PCA-based factor models - the standard
approach at hedge funds and banks.

Key Concepts
------------
1. **Factors**: Small number of latent risk drivers (level, slope, etc.)
2. **Loadings**: How each factor affects each point on the curve/surface
3. **Dynamics**: How factors evolve (typically OU for mean-reversion)
4. **Reconstruction**: Map factor values → full term structure

Example
-------
>>> from src.marketdata.scenarios.timeseries.factor_model import (
...     FactorModelGenerator, CurveFactorSpec, VolSurfaceFactorSpec
... )
>>> 
>>> # Define curve with 3 PCA factors
>>> curve_spec = CurveFactorSpec(
...     market_id=MarketId("IR", "CURVE", "USD"),
...     tenors=np.array([0.25, 0.5, 1, 2, 5, 10, 30]),
...     initial_rates=np.array([0.05, 0.048, 0.045, 0.042, 0.040, 0.041, 0.043]),
...     factor_loadings={
...         "level": np.ones(7),
...         "slope": np.array([-0.8, -0.5, -0.2, 0.0, 0.3, 0.6, 0.8]),
...         "curve": np.array([0.3, 0.1, -0.2, -0.4, -0.2, 0.1, 0.3]),
...     },
...     factor_dynamics={
...         "level": OUDynamicsSpec(mean=0, kappa=0.5, vol=0.008),
...         "slope": OUDynamicsSpec(mean=0, kappa=1.0, vol=0.004),
...         "curve": OUDynamicsSpec(mean=0, kappa=2.0, vol=0.002),
...     },
... )
>>> 
>>> generator = FactorModelGenerator(curves=[curve_spec], correlation_matrix=corr)
>>> result = generator.generate(n_time=252, n_scenarios=1000, seed=42)
>>> 
>>> # Result contains full term structures:
>>> result.curve_paths[curve_spec.market_id].shape  # (252, 1000, 7)

Architecture Note
-----------------
This is the PRODUCTION approach to scenario generation, complementing
the simplified single-factor `TimeseriesGenerator`. Use this when you
need realistic term structure dynamics for:
- VaR/ES calculations
- Stress testing
- XVA simulations
- Trading strategy backtests

Author: QuantStrata Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.curves.factory import ZeroRateCurveFactory
from src.marketdata.surfaces.factory import GridVolFactory


# =============================================================================
# FACTOR SPECIFICATIONS
# =============================================================================

@dataclass
class FactorDynamics:
    """
    Dynamics specification for a single factor.
    
    Supports:
    - OU (Ornstein-Uhlenbeck): Mean-reverting, standard for rates/vol
    - GBM: Geometric Brownian Motion, standard for spot prices
    """
    dynamics_type: str  # "ou" or "gbm"
    
    # OU parameters
    mean: float = 0.0
    kappa: float = 1.0  # Mean reversion speed
    vol: float = 0.01
    
    # GBM parameters (only if dynamics_type == "gbm")
    drift: float = 0.0


@dataclass
class CurveFactorSpec:
    """
    Specification for generating a yield curve via factor model.
    
    The curve at (t, scenario) is:
        curve[t, s, :] = initial_rates + Σ_i (factor_paths[t, s, i] × loadings[i])
    
    Parameters
    ----------
    market_id : MarketId
        Identifier for this curve (e.g., MarketId("IR", "CURVE", "USD"))
    tenors : np.ndarray
        Tenor points in years [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    initial_rates : np.ndarray
        Starting zero rates at each tenor
    factor_loadings : Dict[str, np.ndarray]
        Named loadings for each factor. Common: level, slope, curvature
    factor_dynamics : Dict[str, FactorDynamics]
        Dynamics for each factor (must have same keys as loadings)
    """
    market_id: MarketId
    tenors: np.ndarray
    initial_rates: np.ndarray
    factor_loadings: Dict[str, np.ndarray]
    factor_dynamics: Dict[str, FactorDynamics]
    
    def __post_init__(self):
        if set(self.factor_loadings.keys()) != set(self.factor_dynamics.keys()):
            raise ValueError("factor_loadings and factor_dynamics must have same keys")
        for name, loading in self.factor_loadings.items():
            if len(loading) != len(self.tenors):
                raise ValueError(f"Loading '{name}' length must match tenors")
        if len(self.initial_rates) != len(self.tenors):
            raise ValueError("initial_rates length must match tenors")


@dataclass
class VolSurfaceFactorSpec:
    """
    Specification for generating a volatility surface via factor model.
    
    Parameters
    ----------
    market_id : MarketId
        Identifier (e.g., MarketId("FX", "VOL", "EURUSD"))
    expiries : np.ndarray
        Expiry points in years [0.25, 0.5, 1, 2]
    strikes : np.ndarray
        Strike points (absolute or moneyness) [0.8, 0.9, 1.0, 1.1, 1.2]
    initial_vols : np.ndarray
        Starting vol surface, shape (n_exp, n_strike)
    factor_loadings : Dict[str, np.ndarray]
        Named loadings for each factor, each shape (n_exp, n_strike)
        Common: atm (level), skew (risk reversal), smile (butterfly)
    factor_dynamics : Dict[str, FactorDynamics]
        Dynamics for each factor
    """
    market_id: MarketId
    expiries: np.ndarray
    strikes: np.ndarray
    initial_vols: np.ndarray
    factor_loadings: Dict[str, np.ndarray]
    factor_dynamics: Dict[str, FactorDynamics]
    vol_floor: float = 0.001  # Minimum vol (prevent negative)
    
    def __post_init__(self):
        expected_shape = (len(self.expiries), len(self.strikes))
        if self.initial_vols.shape != expected_shape:
            raise ValueError(f"initial_vols shape must be {expected_shape}")
        for name, loading in self.factor_loadings.items():
            if loading.shape != expected_shape:
                raise ValueError(f"Loading '{name}' shape must be {expected_shape}")


@dataclass
class SpotFactorSpec:
    """
    Specification for a spot price (scalar per time/scenario).
    
    Parameters
    ----------
    market_id : MarketId
        Identifier (e.g., MarketId("FX", "SPOT", "EURUSD"))
    initial_value : float
        Starting spot price
    dynamics : FactorDynamics
        Typically GBM for spots
    """
    market_id: MarketId
    initial_value: float
    dynamics: FactorDynamics


# =============================================================================
# FACTOR MODEL GENERATOR
# =============================================================================

@dataclass
class FactorModelResult:
    """
    Result from factor model scenario generation.
    
    Contains raw paths for each market data type, plus metadata.
    Can be converted to MarketDataset via `to_dataset()`.
    """
    dates: List[str]
    n_scenarios: int
    seed: int
    
    # Paths by market ID
    spot_paths: Dict[MarketId, np.ndarray]       # Each: [T+1, S]
    curve_paths: Dict[MarketId, np.ndarray]      # Each: [T+1, S, n_tenors]
    vol_paths: Dict[MarketId, np.ndarray]        # Each: [T+1, S, n_exp, n_K]
    
    # Factor paths for diagnostics
    factor_paths: Dict[str, np.ndarray]          # Each: [T+1, S]
    
    # Specs for factory creation
    curve_specs: Dict[MarketId, CurveFactorSpec] = field(default_factory=dict)
    vol_specs: Dict[MarketId, VolSurfaceFactorSpec] = field(default_factory=dict)
    
    def to_dataset(self) -> MarketDataset:
        """Convert to MarketDataset with factories."""
        panels = {}
        curve_params = {}
        curve_factories = {}
        vol_params = {}
        vol_factories = {}
        
        # Spots → panels
        for mid, paths in self.spot_paths.items():
            panels[mid] = Panel(
                data=paths,
                axis_names=("time", "scenario"),
            )
        
        # Curves → curve_params + factories
        for mid, paths in self.curve_paths.items():
            spec = self.curve_specs[mid]
            n_tenors = len(spec.tenors)
            n_time = paths.shape[0]
            n_scen = paths.shape[1]
            
            # Build param array: [T, S, n_tenors, 2] for (tenor, rate) pairs
            param_data = np.zeros((n_time, n_scen, n_tenors, 2))
            param_data[:, :, :, 0] = spec.tenors  # Tenors constant
            param_data[:, :, :, 1] = paths  # Rates vary
            
            curve_params[mid] = Panel(
                data=param_data,
                axis_names=("time", "scenario", "tenor", "value"),
            )
            curve_factories[mid] = ZeroRateCurveFactory()
        
        # Vols → vol_params + factories
        for mid, paths in self.vol_paths.items():
            spec = self.vol_specs[mid]
            
            vol_params[mid] = Panel(
                data=paths,
                axis_names=("time", "scenario", "expiry", "strike"),
            )
            vol_factories[mid] = GridVolFactory(
                expiries=spec.expiries,
                strikes=spec.strikes,
            )
        
        return MarketDataset(
            dates=self.dates,
            n_scenarios=self.n_scenarios,
            panels=panels,
            curve_params=curve_params,
            curve_factories=curve_factories,
            vol_params=vol_params,
            vol_factories=vol_factories,
            meta={"generator": "FactorModelGenerator", "seed": self.seed},
        )


class FactorModelGenerator:
    """
    Generate correlated scenarios for spots, curves, and vol surfaces.
    
    This is the production approach used at hedge funds for:
    - VaR/ES scenarios
    - XVA simulations
    - Stress testing
    - Strategy backtesting
    
    Example
    -------
    >>> generator = FactorModelGenerator(
    ...     spots=[spot_spec],
    ...     curves=[usd_curve_spec, eur_curve_spec],
    ...     vol_surfaces=[eurusd_vol_spec],
    ...     correlation_matrix=corr,  # Correlations between all factors
    ... )
    >>> result = generator.generate(n_time=252, n_scenarios=1000, seed=42)
    >>> dataset = result.to_dataset()
    >>> market = dataset.snapshot(time_idx=100, scenario_idx=0)
    """
    
    def __init__(
        self,
        spots: Optional[List[SpotFactorSpec]] = None,
        curves: Optional[List[CurveFactorSpec]] = None,
        vol_surfaces: Optional[List[VolSurfaceFactorSpec]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        dt: float = 1/252,
    ):
        """
        Parameters
        ----------
        spots : List[SpotFactorSpec]
            Spot specifications (each produces scalar per scenario)
        curves : List[CurveFactorSpec]
            Curve specifications (each produces full term structure)
        vol_surfaces : List[VolSurfaceFactorSpec]
            Vol surface specifications (each produces full grid)
        correlation_matrix : np.ndarray
            Correlation between ALL factors (spots + curve factors + vol factors)
            Order: [spot_factors..., curve_factors..., vol_factors...]
        dt : float
            Time step (default: 1/252 = daily)
        """
        self.spots = spots or []
        self.curves = curves or []
        self.vol_surfaces = vol_surfaces or []
        self.dt = dt
        
        # Build factor registry
        self._build_factor_registry()
        
        # Validate or default correlation matrix
        if correlation_matrix is not None:
            if correlation_matrix.shape != (self.n_factors, self.n_factors):
                raise ValueError(
                    f"correlation_matrix shape {correlation_matrix.shape} doesn't match "
                    f"n_factors={self.n_factors}"
                )
            self.correlation = correlation_matrix
        else:
            self.correlation = np.eye(self.n_factors)
    
    def _build_factor_registry(self):
        """Build registry of all factors and their properties."""
        self.factor_names: List[str] = []
        self.factor_dynamics: List[FactorDynamics] = []
        self.factor_initial: List[float] = []
        
        # Spot factors (1 per spot)
        for spec in self.spots:
            name = f"spot.{spec.market_id.key()}"
            self.factor_names.append(name)
            self.factor_dynamics.append(spec.dynamics)
            self.factor_initial.append(0.0)  # Log-space for GBM
        
        # Curve factors (multiple per curve)
        for spec in self.curves:
            for factor_name in spec.factor_loadings.keys():
                name = f"curve.{spec.market_id.key()}.{factor_name}"
                self.factor_names.append(name)
                self.factor_dynamics.append(spec.factor_dynamics[factor_name])
                self.factor_initial.append(0.0)
        
        # Vol factors (multiple per surface)
        for spec in self.vol_surfaces:
            for factor_name in spec.factor_loadings.keys():
                name = f"vol.{spec.market_id.key()}.{factor_name}"
                self.factor_names.append(name)
                self.factor_dynamics.append(spec.factor_dynamics[factor_name])
                self.factor_initial.append(0.0)
        
        self.n_factors = len(self.factor_names)
    
    def generate(
        self,
        n_time: int,
        n_scenarios: int,
        seed: int = 42,
        start_date: str = "2024-01-01",
    ) -> FactorModelResult:
        """
        Generate correlated scenarios.
        
        Parameters
        ----------
        n_time : int
            Number of time steps
        n_scenarios : int
            Number of Monte Carlo paths
        seed : int
            Random seed for reproducibility
        start_date : str
            Starting date for date labels
        
        Returns
        -------
        FactorModelResult
            Contains paths for all market data types
        """
        rng = np.random.default_rng(seed)
        
        # Cholesky decomposition for correlation
        chol = np.linalg.cholesky(self.correlation)
        
        # Simulate all factors
        factor_paths = np.zeros((n_time + 1, n_scenarios, self.n_factors))
        
        for t in range(n_time):
            # Correlated Gaussian shocks
            z = rng.standard_normal((n_scenarios, self.n_factors))
            z_corr = z @ chol.T
            
            for f, (name, dyn) in enumerate(zip(self.factor_names, self.factor_dynamics)):
                if dyn.dynamics_type == "gbm":
                    # GBM in log-space
                    factor_paths[t+1, :, f] = (
                        factor_paths[t, :, f] +
                        (dyn.drift - 0.5 * dyn.vol**2) * self.dt +
                        dyn.vol * np.sqrt(self.dt) * z_corr[:, f]
                    )
                else:  # OU
                    kappa = dyn.kappa
                    if kappa > 0:
                        factor_paths[t+1, :, f] = (
                            dyn.mean +
                            (factor_paths[t, :, f] - dyn.mean) * np.exp(-kappa * self.dt) +
                            dyn.vol * np.sqrt((1 - np.exp(-2*kappa*self.dt)) / (2*kappa)) * z_corr[:, f]
                        )
                    else:
                        # Brownian motion if kappa == 0
                        factor_paths[t+1, :, f] = (
                            factor_paths[t, :, f] +
                            dyn.vol * np.sqrt(self.dt) * z_corr[:, f]
                        )
        
        # Convert factor paths to market data
        spot_paths = self._factors_to_spots(factor_paths)
        curve_paths = self._factors_to_curves(factor_paths)
        vol_paths = self._factors_to_vol_surfaces(factor_paths)
        
        # Generate dates
        dates = [f"{start_date}+{i}d" for i in range(n_time + 1)]  # Simplified
        
        # Factor paths dict for diagnostics
        factor_paths_dict = {
            name: factor_paths[:, :, f]
            for f, name in enumerate(self.factor_names)
        }
        
        return FactorModelResult(
            dates=dates,
            n_scenarios=n_scenarios,
            seed=seed,
            spot_paths=spot_paths,
            curve_paths=curve_paths,
            vol_paths=vol_paths,
            factor_paths=factor_paths_dict,
            curve_specs={spec.market_id: spec for spec in self.curves},
            vol_specs={spec.market_id: spec for spec in self.vol_surfaces},
        )
    
    def _factors_to_spots(self, factor_paths: np.ndarray) -> Dict[MarketId, np.ndarray]:
        """Convert factor paths to spot prices."""
        result = {}
        f_idx = 0
        
        for spec in self.spots:
            # Spot in log-space → exponentiate
            log_spot = factor_paths[:, :, f_idx]
            result[spec.market_id] = spec.initial_value * np.exp(log_spot)
            f_idx += 1
        
        return result
    
    def _factors_to_curves(self, factor_paths: np.ndarray) -> Dict[MarketId, np.ndarray]:
        """Convert factor paths to full yield curves."""
        result = {}
        
        # Skip spot factors
        f_idx = len(self.spots)
        
        for spec in self.curves:
            n_tenors = len(spec.tenors)
            n_time = factor_paths.shape[0]
            n_scen = factor_paths.shape[1]
            
            # Initialize with base curve
            curves = np.zeros((n_time, n_scen, n_tenors))
            curves[:, :, :] = spec.initial_rates
            
            # Add factor contributions
            for factor_name, loading in spec.factor_loadings.items():
                factor = factor_paths[:, :, f_idx]  # [T, S]
                # Broadcasting: [T, S, 1] * [1, 1, n_tenors]
                curves += factor[:, :, np.newaxis] * loading[np.newaxis, np.newaxis, :]
                f_idx += 1
            
            result[spec.market_id] = curves
        
        return result
    
    def _factors_to_vol_surfaces(self, factor_paths: np.ndarray) -> Dict[MarketId, np.ndarray]:
        """Convert factor paths to full vol surfaces."""
        result = {}
        
        # Skip spot + curve factors
        f_idx = len(self.spots)
        for spec in self.curves:
            f_idx += len(spec.factor_loadings)
        
        for spec in self.vol_surfaces:
            n_exp = len(spec.expiries)
            n_K = len(spec.strikes)
            n_time = factor_paths.shape[0]
            n_scen = factor_paths.shape[1]
            
            # Initialize with base surface
            surfaces = np.zeros((n_time, n_scen, n_exp, n_K))
            surfaces[:, :, :, :] = spec.initial_vols
            
            # Add factor contributions
            for factor_name, loading in spec.factor_loadings.items():
                factor = factor_paths[:, :, f_idx]  # [T, S]
                # Broadcasting: [T, S, 1, 1] * [1, 1, n_exp, n_K]
                surfaces += factor[:, :, np.newaxis, np.newaxis] * loading[np.newaxis, np.newaxis, :, :]
                f_idx += 1
            
            # Apply vol floor
            surfaces = np.maximum(surfaces, spec.vol_floor)
            
            result[spec.market_id] = surfaces
        
        return result
