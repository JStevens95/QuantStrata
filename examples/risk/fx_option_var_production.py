#!/usr/bin/env python3
"""
===============================================================================
FX Option VaR - Production Grade
===============================================================================

This example demonstrates a FULLY PRODUCTION-GRADE VaR workflow as would be
implemented at a front-office quant hedge fund.

Key Production Features
-----------------------
1. **Full Term Structures**: Uses `ZeroRateCurve` and `GridVolSurface` with
   actual tenor/strike grids - NOT flat curves/vols.

2. **Factor Model Scenarios**: Uses `FactorModelGenerator` with PCA-based
   dynamics for curves and vol surfaces.

3. **Proper Error Handling**: Validates inputs, catches pricing failures,
   provides meaningful error messages.

4. **Structured Logging**: Uses Python logging, not print statements.

5. **Configuration Support**: Can load config from YAML file or use defaults.

6. **Type Hints**: Full type annotations throughout.

7. **Modular Design**: Separated concerns (config, market building, pricing,
   risk metrics).

Market Data Structures (Production Standard)
--------------------------------------------
- IR Curves: 11 tenors from 3M to 30Y
- Vol Surface: 4 expiries × 5 strikes (delta-space converted to absolute)
- Correlation: 10×10 factor correlation matrix

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/risk/fx_option_var_production.py
    
    # With custom config
    PYTHONPATH=. python examples/risk/fx_option_var_production.py --config my_config.yaml

Author: QuantStrata Team
===============================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Proper import path handling
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Market data infrastructure - PRODUCTION CLASSES
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.curves.term_structure import ZeroRateCurve  # NOT FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import GridVolSurface  # NOT FlatVolSurface

# Factor model generator for full term structure scenarios
from src.marketdata.scenarios.timeseries import (
    FactorModelGenerator,
    FactorModelResult,
    CurveFactorSpec,
    VolSurfaceFactorSpec,
    SpotFactorSpec,
    FactorDynamics,
)

# Instrument and pricer
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("fx_option_var")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CurveConfig:
    """Production curve configuration."""
    tenors: List[float] = field(default_factory=lambda: [
        0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0
    ])
    usd_rates: List[float] = field(default_factory=lambda: [
        0.054, 0.053, 0.052, 0.048, 0.045, 0.042, 0.041, 0.040, 0.041, 0.042, 0.043
    ])
    eur_rates: List[float] = field(default_factory=lambda: [
        0.040, 0.039, 0.038, 0.035, 0.033, 0.031, 0.030, 0.029, 0.029, 0.030, 0.031
    ])
    
    def __post_init__(self):
        if len(self.tenors) != len(self.usd_rates):
            raise ValueError(f"tenors ({len(self.tenors)}) and usd_rates ({len(self.usd_rates)}) must have same length")
        if len(self.tenors) != len(self.eur_rates):
            raise ValueError(f"tenors ({len(self.tenors)}) and eur_rates ({len(self.eur_rates)}) must have same length")


@dataclass
class VolSurfaceConfig:
    """Production vol surface configuration."""
    expiries: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])
    strikes_moneyness: List[float] = field(default_factory=lambda: [0.75, 0.90, 1.00, 1.10, 1.25])
    # Initial vol grid (expiry × strike)
    vols: List[List[float]] = field(default_factory=lambda: [
        [0.120, 0.095, 0.085, 0.090, 0.105],  # 3M
        [0.115, 0.092, 0.083, 0.087, 0.100],  # 6M
        [0.110, 0.090, 0.082, 0.085, 0.095],  # 1Y
        [0.105, 0.088, 0.082, 0.084, 0.092],  # 2Y
    ])
    
    def __post_init__(self):
        if len(self.vols) != len(self.expiries):
            raise ValueError("vols rows must match expiries")
        for i, row in enumerate(self.vols):
            if len(row) != len(self.strikes_moneyness):
                raise ValueError(f"vols row {i} length must match strikes")


@dataclass
class ScenarioConfig:
    """Scenario generation configuration."""
    n_scenarios: int = 10_000
    n_time: int = 10  # Business days horizon
    seed: int = 42
    spot_eurusd: float = 1.0850
    spot_vol: float = 0.085


@dataclass
class OptionConfig:
    """Option configuration."""
    option_type: str = "call"
    strike: float = 1.10
    expiry_years: float = 0.25
    notional: float = 10_000_000
    
    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{self.option_type}'")
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.expiry_years <= 0:
            raise ValueError(f"expiry_years must be positive, got {self.expiry_years}")


@dataclass
class Config:
    """Master configuration."""
    curves: CurveConfig = field(default_factory=CurveConfig)
    vol_surface: VolSurfaceConfig = field(default_factory=VolSurfaceConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    option: OptionConfig = field(default_factory=OptionConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
            return cls(
                curves=CurveConfig(**data.get("curves", {})),
                vol_surface=VolSurfaceConfig(**data.get("vol_surface", {})),
                scenarios=ScenarioConfig(**data.get("scenarios", {})),
                option=OptionConfig(**data.get("option", {})),
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {path}")
        except Exception as e:
            raise ValueError(f"Failed to parse config: {e}")


# =============================================================================
# MARKET IDS (PRODUCTION NAMING CONVENTION)
# =============================================================================

class MarketIds:
    """Centralized market ID definitions."""
    EURUSD_SPOT = MarketId("FX", "SPOT", "EURUSD")
    USD_CURVE = MarketId("IR", "CURVE", "USD")
    EUR_CURVE = MarketId("IR", "CURVE", "EUR")
    EURUSD_VOL = MarketId("FX", "VOL", "EURUSD")


# =============================================================================
# PRODUCTION MARKET BUILDER
# =============================================================================

class ProductionMarketBuilder:
    """
    Builds Market objects with FULL term structures.
    
    This is the production approach - NOT using FlatCurve/FlatVolSurface.
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.curve_tenors = np.array(cfg.curves.tenors)
        self.vol_expiries = np.array(cfg.vol_surface.expiries)
        self.vol_strikes_moneyness = np.array(cfg.vol_surface.strikes_moneyness)
    
    def build_base_market(self, asof: str = "2024-01-15") -> Market:
        """Build base market with full term structures."""
        return self._build_market(
            asof=asof,
            spot=self.cfg.scenarios.spot_eurusd,
            usd_rates=np.array(self.cfg.curves.usd_rates),
            eur_rates=np.array(self.cfg.curves.eur_rates),
            vol_grid=np.array(self.cfg.vol_surface.vols),
        )
    
    def build_scenario_market(
        self,
        asof: str,
        spot: float,
        usd_rates: np.ndarray,
        eur_rates: np.ndarray,
        vol_grid: np.ndarray,
    ) -> Market:
        """Build market from scenario values."""
        return self._build_market(asof, spot, usd_rates, eur_rates, vol_grid)
    
    def _build_market(
        self,
        asof: str,
        spot: float,
        usd_rates: np.ndarray,
        eur_rates: np.ndarray,
        vol_grid: np.ndarray,
    ) -> Market:
        """
        Build Market with PRODUCTION term structures.
        
        This uses:
        - ZeroRateCurve (NOT FlatZeroRateCurve)
        - GridVolSurface (NOT FlatVolSurface)
        """
        # Validate inputs
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if np.any(~np.isfinite(usd_rates)):
            raise ValueError("usd_rates contains non-finite values")
        if np.any(~np.isfinite(eur_rates)):
            raise ValueError("eur_rates contains non-finite values")
        if np.any(vol_grid <= 0):
            raise ValueError("vol_grid contains non-positive values")
        
        # Quotes
        quotes = {
            MarketIds.EURUSD_SPOT: Quote(value=float(spot)),
        }
        
        # PRODUCTION CURVES - Full term structure
        curves = {
            MarketIds.USD_CURVE: ZeroRateCurve(
                tenors=self.curve_tenors,
                zero_rates=usd_rates,
                extrapolation="flat",
            ),
            MarketIds.EUR_CURVE: ZeroRateCurve(
                tenors=self.curve_tenors,
                zero_rates=eur_rates,
                extrapolation="flat",
            ),
        }
        
        # PRODUCTION VOL SURFACE - Full grid
        # Convert moneyness to absolute strikes
        absolute_strikes = self.vol_strikes_moneyness * spot
        
        vols = {
            MarketIds.EURUSD_VOL: GridVolSurface(
                expiries=self.vol_expiries,
                strikes=absolute_strikes,
                implied_vols=vol_grid,
                extrapolation="flat",
            ),
        }
        
        return Market(
            asof=asof,
            quotes=quotes,
            curves=curves,
            vols=vols,
            meta={"builder": "ProductionMarketBuilder"},
        )


# =============================================================================
# FACTOR MODEL SPECS
# =============================================================================

def build_factor_specs(cfg: Config) -> Tuple[
    SpotFactorSpec,
    CurveFactorSpec,
    CurveFactorSpec,
    VolSurfaceFactorSpec,
]:
    """Build factor specifications for scenario generation."""
    
    tenors = np.array(cfg.curves.tenors)
    n_tenors = len(tenors)
    
    # FX Spot (GBM)
    spot_spec = SpotFactorSpec(
        market_id=MarketIds.EURUSD_SPOT,
        initial_value=cfg.scenarios.spot_eurusd,
        dynamics=FactorDynamics(dynamics_type="gbm", vol=cfg.scenarios.spot_vol),
    )
    
    # USD Curve - 3 PCA factors
    usd_curve_spec = CurveFactorSpec(
        market_id=MarketIds.USD_CURVE,
        tenors=tenors,
        initial_rates=np.array(cfg.curves.usd_rates),
        factor_loadings={
            "level": np.ones(n_tenors) * 0.01,
            "slope": np.linspace(-0.015, 0.011, n_tenors),
            "curve": np.array([0.005, 0.003, 0.0, -0.004, -0.006, -0.006, -0.004, 0.0, 0.003, 0.005, 0.006]),
        },
        factor_dynamics={
            "level": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=0.3, vol=0.5),
            "slope": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=0.5, vol=0.3),
            "curve": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=1.0, vol=0.2),
        },
    )
    
    # EUR Curve - 3 PCA factors
    eur_curve_spec = CurveFactorSpec(
        market_id=MarketIds.EUR_CURVE,
        tenors=tenors,
        initial_rates=np.array(cfg.curves.eur_rates),
        factor_loadings={
            "level": np.ones(n_tenors) * 0.01,
            "slope": np.linspace(-0.012, 0.010, n_tenors),
            "curve": np.array([0.004, 0.002, 0.0, -0.003, -0.005, -0.005, -0.003, 0.0, 0.002, 0.004, 0.005]),
        },
        factor_dynamics={
            "level": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=0.3, vol=0.4),
            "slope": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=0.5, vol=0.25),
            "curve": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=1.0, vol=0.15),
        },
    )
    
    # Vol Surface - 3 factors
    expiries = np.array(cfg.vol_surface.expiries)
    strikes = np.array(cfg.vol_surface.strikes_moneyness)
    n_exp, n_K = len(expiries), len(strikes)
    
    vol_spec = VolSurfaceFactorSpec(
        market_id=MarketIds.EURUSD_VOL,
        expiries=expiries,
        strikes=strikes,
        initial_vols=np.array(cfg.vol_surface.vols),
        factor_loadings={
            "atm": np.ones((n_exp, n_K)) * 0.1,
            "skew": np.tile(np.linspace(-0.02, 0.02, n_K), (n_exp, 1)),
            "smile": np.tile(np.array([0.015, 0.005, 0.0, 0.005, 0.015]), (n_exp, 1)),
        },
        factor_dynamics={
            "atm": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=2.0, vol=0.8),
            "skew": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=3.0, vol=0.5),
            "smile": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=4.0, vol=0.3),
        },
        vol_floor=0.01,
    )
    
    return spot_spec, usd_curve_spec, eur_curve_spec, vol_spec


def build_correlation_matrix() -> np.ndarray:
    """
    Build 10×10 factor correlation matrix.
    
    Factors: spot, usd_level, usd_slope, usd_curve, eur_level, eur_slope, eur_curve, vol_atm, vol_skew, vol_smile
    """
    n = 10
    corr = np.eye(n)
    
    # Spot correlations
    corr[0, 1] = corr[1, 0] = -0.15  # Spot vs USD level
    corr[0, 4] = corr[4, 0] = 0.10   # Spot vs EUR level
    corr[0, 7] = corr[7, 0] = -0.35  # Spot vs Vol ATM
    
    # USD curve internal
    corr[1, 2] = corr[2, 1] = 0.3
    corr[1, 3] = corr[3, 1] = 0.2
    corr[2, 3] = corr[3, 2] = 0.4
    
    # EUR curve internal
    corr[4, 5] = corr[5, 4] = 0.3
    corr[4, 6] = corr[6, 4] = 0.2
    corr[5, 6] = corr[6, 5] = 0.4
    
    # Cross-currency
    corr[1, 4] = corr[4, 1] = 0.6  # USD-EUR level
    corr[2, 5] = corr[5, 2] = 0.4  # USD-EUR slope
    
    # Vol internal
    corr[7, 8] = corr[8, 7] = 0.2
    corr[7, 9] = corr[9, 7] = 0.3
    corr[8, 9] = corr[9, 8] = 0.15
    
    # Ensure PSD
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 0:
        eigvals_fixed, eigvecs = np.linalg.eigh(corr)
        eigvals_fixed = np.maximum(eigvals_fixed, 1e-8)
        corr = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T
    
    return corr


# =============================================================================
# VAR CALCULATOR
# =============================================================================

class VaRCalculator:
    """Production VaR calculator with proper error handling."""
    
    def __init__(
        self,
        option: FxVanillaEuropeanOption,
        market_builder: ProductionMarketBuilder,
    ):
        self.option = option
        self.market_builder = market_builder
        self.pricer = FxVanillaEuropeanOptionBsmPricer()
        self._pricing_errors = 0
    
    def compute_scenario_pnl(
        self,
        scenarios: FactorModelResult,
        base_price: float,
        time_idx: int = -1,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute PnL for all scenarios with proper error handling.
        
        Returns
        -------
        Tuple[np.ndarray, Dict]
            PnL array and risk metrics dictionary.
        """
        n_scenarios = scenarios.n_scenarios
        asof = scenarios.dates[time_idx]
        
        # Extract paths
        spot_paths = scenarios.spot_paths[MarketIds.EURUSD_SPOT]
        usd_paths = scenarios.curve_paths[MarketIds.USD_CURVE]
        eur_paths = scenarios.curve_paths[MarketIds.EUR_CURVE]
        vol_paths = scenarios.vol_paths[MarketIds.EURUSD_VOL]
        
        # Price each scenario
        prices = np.zeros(n_scenarios)
        self._pricing_errors = 0
        
        for s in range(n_scenarios):
            try:
                market = self.market_builder.build_scenario_market(
                    asof=asof,
                    spot=spot_paths[time_idx, s],
                    usd_rates=usd_paths[time_idx, s, :],
                    eur_rates=eur_paths[time_idx, s, :],
                    vol_grid=vol_paths[time_idx, s, :, :],
                )
                prices[s] = self.pricer.price(self.option, market)
            except Exception as e:
                logger.warning(f"Pricing failed for scenario {s}: {e}")
                prices[s] = np.nan
                self._pricing_errors += 1
        
        if self._pricing_errors > 0:
            logger.warning(f"Total pricing errors: {self._pricing_errors}/{n_scenarios}")
        
        # Remove failed scenarios
        valid_mask = ~np.isnan(prices)
        valid_prices = prices[valid_mask]
        
        if len(valid_prices) < n_scenarios * 0.95:
            raise RuntimeError(f"Too many pricing failures: {n_scenarios - len(valid_prices)}")
        
        # Compute PnL
        pnl = valid_prices - base_price
        
        # Compute risk metrics
        metrics = self._compute_risk_metrics(pnl, base_price)
        
        return pnl, metrics
    
    def _compute_risk_metrics(
        self,
        pnl: np.ndarray,
        base_price: float,
    ) -> Dict[str, float]:
        """Compute standard risk metrics."""
        return {
            "base_price": base_price,
            "mean_pnl": float(np.mean(pnl)),
            "std_pnl": float(np.std(pnl)),
            "var_95": float(np.percentile(pnl, 5)),
            "var_99": float(np.percentile(pnl, 1)),
            "es_95": float(np.mean(pnl[pnl <= np.percentile(pnl, 5)])),
            "es_99": float(np.mean(pnl[pnl <= np.percentile(pnl, 1)])),
            "min_pnl": float(np.min(pnl)),
            "max_pnl": float(np.max(pnl)),
            "n_scenarios": len(pnl),
        }


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_var_analysis(cfg: Config) -> Dict:
    """
    Run production VaR analysis.
    
    Returns
    -------
    Dict
        Complete results including scenarios, PnL, and metrics.
    """
    logger.info("=" * 70)
    logger.info("FX Option VaR - Production Grade")
    logger.info("=" * 70)
    
    # Step 1: Build factor specs
    logger.info("Building factor specifications...")
    spot_spec, usd_spec, eur_spec, vol_spec = build_factor_specs(cfg)
    correlation = build_correlation_matrix()
    
    logger.info(f"  Spot: GBM, vol={spot_spec.dynamics.vol:.1%}")
    logger.info(f"  USD Curve: {len(usd_spec.factor_loadings)} factors, {len(usd_spec.tenors)} tenors")
    logger.info(f"  EUR Curve: {len(eur_spec.factor_loadings)} factors, {len(eur_spec.tenors)} tenors")
    logger.info(f"  Vol Surface: {len(vol_spec.factor_loadings)} factors, {vol_spec.initial_vols.shape} grid")
    
    # Step 2: Generate scenarios
    logger.info("Generating scenarios...")
    t0 = time.time()
    
    generator = FactorModelGenerator(
        spots=[spot_spec],
        curves=[usd_spec, eur_spec],
        vol_surfaces=[vol_spec],
        correlation_matrix=correlation,
    )
    
    scenarios = generator.generate(
        n_time=cfg.scenarios.n_time,
        n_scenarios=cfg.scenarios.n_scenarios,
        seed=cfg.scenarios.seed,
    )
    
    gen_time = time.time() - t0
    logger.info(f"  Generated {cfg.scenarios.n_scenarios:,} scenarios in {gen_time:.2f}s")
    
    # Step 3: Build market and option
    logger.info("Building market and option...")
    market_builder = ProductionMarketBuilder(cfg)
    base_market = market_builder.build_base_market()
    
    option = FxVanillaEuropeanOption(
        option_type=cfg.option.option_type,
        notional=cfg.option.notional,
        strike=cfg.option.strike,
        expiry=cfg.option.expiry_years,
        spot_id=MarketIds.EURUSD_SPOT,
        vol_id=MarketIds.EURUSD_VOL,
        domestic_curve_id=MarketIds.USD_CURVE,
        foreign_curve_id=MarketIds.EUR_CURVE,
    )
    
    # Step 4: Compute VaR
    logger.info("Computing VaR...")
    t0 = time.time()
    
    var_calc = VaRCalculator(option, market_builder)
    base_price = var_calc.pricer.price(option, base_market)
    pnl, metrics = var_calc.compute_scenario_pnl(scenarios, base_price)
    
    calc_time = time.time() - t0
    logger.info(f"  Computed PnL for {metrics['n_scenarios']:,} scenarios in {calc_time:.2f}s")
    
    # Step 5: Report
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Base Price:     ${metrics['base_price']:>12,.2f}")
    logger.info(f"Mean PnL:       ${metrics['mean_pnl']:>12,.2f}")
    logger.info(f"Std PnL:        ${metrics['std_pnl']:>12,.2f}")
    logger.info(f"VaR (95%):      ${metrics['var_95']:>12,.2f}")
    logger.info(f"VaR (99%):      ${metrics['var_99']:>12,.2f}")
    logger.info(f"ES (95%):       ${metrics['es_95']:>12,.2f}")
    logger.info(f"ES (99%):       ${metrics['es_99']:>12,.2f}")
    logger.info(f"Max Loss:       ${metrics['min_pnl']:>12,.2f}")
    logger.info(f"Max Gain:       ${metrics['max_pnl']:>12,.2f}")
    
    return {
        "scenarios": scenarios,
        "pnl": pnl,
        "metrics": metrics,
        "option": option,
        "config": cfg,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="FX Option VaR - Production Grade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=10_000,
        help="Number of scenarios (default: 10000)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Horizon in days (default: 10)",
    )
    args = parser.parse_args()
    
    try:
        # Load config
        if args.config:
            logger.info(f"Loading config from {args.config}")
            cfg = Config.from_yaml(args.config)
        else:
            cfg = Config()
        
        # Override with CLI args
        cfg.scenarios.n_scenarios = args.scenarios
        cfg.scenarios.n_time = args.horizon
        
        # Run analysis
        results = run_var_analysis(cfg)
        
        logger.info("=" * 70)
        logger.info("Analysis complete")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
