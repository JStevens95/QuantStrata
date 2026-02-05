#!/usr/bin/env python3
"""
===============================================================================
Time Series Generation for Risk Factor Simulation
===============================================================================

This example provides a comprehensive demonstration of the TimeseriesGenerator
framework for generating realistic, correlated risk factor time series suitable
for production hedge fund risk management workflows.

Learning Objectives
-------------------
1. **Dynamics Models**: Understand GBM, Heston, and OU processes
2. **Correlation**: Build correlated multi-factor scenarios
3. **Model Selection**: Choose appropriate dynamics for each asset class
4. **Stress Testing**: Generate crisis scenarios with modified parameters
5. **Integration**: Connect generators to MarketDataset for pricing

================================================================================
WHEN TO USE EACH DYNAMICS MODEL
================================================================================

1. GBM (Geometric Brownian Motion)
   - Use for: FX spot, commodity prices, equity indices (simple case)
   - Characteristics: Log-normal paths, constant volatility
   - Pros: Simple, fast, well-understood
   - Cons: No vol smile/skew, can't capture fat tails
   
2. Heston (Stochastic Volatility)
   - Use for: Equity with realistic vol dynamics, options risk
   - Characteristics: Time-varying volatility, vol clustering, leverage effect
   - Pros: Generates vol smile/skew, captures volatility clustering
   - Cons: More complex, requires 5 parameters, computationally heavier
   
3. Ornstein-Uhlenbeck (Mean-Reverting)
   - Use for: Interest rates, credit spreads, volatility factors
   - Characteristics: Mean-reversion to long-term level
   - Pros: Captures rate dynamics, stationary distribution
   - Cons: Can go negative (may need bounds for rates)

4. Factor Model
   - Use for: Yield curve (Level/Slope/Curvature), vol surface factors
   - Characteristics: OU dynamics with tenor loadings
   - Pros: PCA-style decomposition, interpretable
   - Cons: Requires calibrated loadings from historical data

================================================================================
CORRELATION STRUCTURE DESIGN
================================================================================

When designing correlation matrices, consider:

1. Economic Relationships:
   - EUR/USD vs GBP/USD: 0.5-0.7 (both dollar pairs)
   - Equity vs USD rates: -0.1 to -0.3 (higher rates → lower equity)
   - Gold vs USD: -0.3 to -0.5 (safe haven)
   - Oil vs inflation: 0.3-0.5

2. Risk-On/Risk-Off:
   - During stress: correlations spike toward ±1
   - Consider regime-switching or stress scenarios separately

3. Positive Semi-Definiteness:
   - Correlation matrix must be PSD for Cholesky
   - Test with np.linalg.eigvalsh(corr) >= 0

================================================================================
MATHEMATICAL BACKGROUND
================================================================================

**GBM** (Geometric Brownian Motion):
    dS_t = μ S_t dt + σ S_t dW_t
    
    Exact solution: S_t = S_0 exp((μ - σ²/2)t + σW_t)

**Heston** (Stochastic Volatility):
    dS_t = μ S_t dt + √V_t S_t dW_t^S
    dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V
    Corr(dW^S, dW^V) = ρ
    
    Feller condition: 2κθ > ξ² ensures V_t > 0

**Ornstein-Uhlenbeck** (Mean-Reverting):
    dX_t = κ(θ - X_t) dt + σ dW_t
    
    Expected value: E[X_t] = θ + (X_0 - θ)e^(-κt)
    Half-life: t_1/2 = ln(2)/κ

**Correlation via Cholesky**:
    Given correlation matrix Σ = LL^T
    Independent shocks Z ~ N(0, I)
    Correlated shocks: Z̃ = Z·L^T

Production Context
------------------
At a hedge fund, time series generation is used for:
- Monte Carlo VaR: Simulated P&L distributions
- Expected Shortfall: Tail risk measures
- Stress testing: Modified dynamics for crisis scenarios
- Model validation: Backtesting against historical paths

Prerequisites
-------------
- Examples 01-06: Market fundamentals

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/fundamentals/07_timeseries_generation.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations  # Enable modern type hints (PEP 604)

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.timeseries import (
    TimeseriesGenerator,
    TimeseriesConfig,
    RiskFactorSpec,
    GBMDynamicsSpec,
    HestonDynamicsSpec,
    OUDynamicsSpec,
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

ENABLE_PLOTTING = True

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - plotting disabled")


# =============================================================================
# HELPER: Plot Paths
# =============================================================================

def plot_paths(
    paths: np.ndarray,
    dates: List[str],
    title: str,
    ylabel: str,
    save: bool = False,
) -> None:
    """
    Helper to plot sample paths.
    
    Parameters
    ----------
    paths : np.ndarray
        Array of shape [T, n_scenarios] with path data.
    dates : List[str]
        Date strings for x-axis.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save : bool
        Whether to save the figure.
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_time, n_scenarios = paths.shape
    
    for i in range(n_scenarios):
        ax.plot(paths[:, i], alpha=0.7)
    
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(filename, dpi=150)
        logger.info(f"Saved: {filename}")
    
    plt.close()


# =============================================================================
# EXAMPLE 1: Single Factor GBM (FX Spot)
# =============================================================================

def example_1_single_factor_gbm() -> None:
    """
    Example 1: Single GBM factor (FX spot).
    
    Generate EUR/USD spot paths using Geometric Brownian Motion.
    This is the simplest case - a single log-normal process.
    
    GBM Mathematics
    ---------------
    dS_t = μ S_t dt + σ S_t dW_t
    
    Solution: S_t = S_0 exp((μ - σ²/2)t + σW_t)
    
    Parameters:
    - drift (μ): Expected return (0 for risk-neutral FX)
    - vol (σ): Annualized volatility (8% typical for EUR/USD)
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Example 1: Single GBM Factor (EUR/USD Spot)")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Define single FX spot factor
    # -------------------------------------------------------------------------
    eurusd = RiskFactorSpec(
        market_id=MarketId("FX", "SPOT", "EURUSD"),
        initial_value=1.08,               # Current spot
        dynamics=GBMDynamicsSpec(
            drift=0.0,                    # Zero drift (martingale under risk-neutral)
            vol=0.08,                     # 8% annual volatility
        ),
        name="EUR/USD Spot",              # Display name
    )
    
    # -------------------------------------------------------------------------
    # Create configuration
    # Single factor → 1x1 correlation matrix
    # -------------------------------------------------------------------------
    config = TimeseriesConfig(
        factors=[eurusd],
        correlation=np.array([[1.0]]),    # Trivial for single factor
        start_date="2024-01-01",
        end_date="2024-12-31",
        freq="D",                         # Daily frequency
        n_scenarios=1000,                 # 1000 Monte Carlo paths
    )
    
    # -------------------------------------------------------------------------
    # Generate paths
    # -------------------------------------------------------------------------
    generator = TimeseriesGenerator(config)
    result = generator.generate_paths(seed=42)  # Seed for reproducibility
    
    # -------------------------------------------------------------------------
    # Print statistics
    # -------------------------------------------------------------------------
    stats = generator.compute_statistics(result)
    logger.info("")
    logger.info(f"Generation Statistics for {eurusd.display_name}:")
    for key, value in stats[eurusd.display_name].items():
        logger.info(f"  {key}: {value:.6f}")
    
    # Plot sample paths
    paths = result.paths[eurusd.market_id]
    plot_paths(
        paths=paths[:, :10],              # First 10 scenarios
        dates=result.dates,
        title="EUR/USD Spot Paths (GBM)",
        ylabel="EUR/USD",
    )


# =============================================================================
# EXAMPLE 2: Correlated FX Pairs
# =============================================================================

def example_2_correlated_fx_pairs() -> None:
    """
    Example 2: Correlated FX pairs.
    
    Generate EUR/USD and GBP/USD with 60% correlation.
    Both are dollar pairs and historically move together.
    
    Correlation Implementation
    --------------------------
    Uses Cholesky decomposition of correlation matrix:
        Σ = LL^T
        Z_correlated = Z_independent × L^T
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Example 2: Correlated FX Pairs")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Define two correlated FX factors
    # -------------------------------------------------------------------------
    factors = [
        RiskFactorSpec(
            market_id=MarketId("FX", "SPOT", "EURUSD"),
            initial_value=1.08,
            dynamics=GBMDynamicsSpec(drift=0.0, vol=0.08),
            name="EUR/USD",
        ),
        RiskFactorSpec(
            market_id=MarketId("FX", "SPOT", "GBPUSD"),
            initial_value=1.26,
            dynamics=GBMDynamicsSpec(drift=0.0, vol=0.09),
            name="GBP/USD",
        ),
    ]
    
    # -------------------------------------------------------------------------
    # 60% correlation between EUR/USD and GBP/USD
    # This is typical for dollar pairs
    # -------------------------------------------------------------------------
    correlation = np.array([
        [1.0, 0.6],
        [0.6, 1.0],
    ])
    
    config = TimeseriesConfig(
        factors=factors,
        correlation=correlation,
        start_date="2024-01-01",
        end_date="2024-12-31",
        freq="D",
        n_scenarios=10000,                # More scenarios for better statistics
    )
    
    generator = TimeseriesGenerator(config)
    result = generator.generate_paths(seed=42)
    
    # -------------------------------------------------------------------------
    # Validate realized correlation
    # Should be close to input (60%)
    # -------------------------------------------------------------------------
    realized_corr = generator.compute_realized_correlation(result)
    
    logger.info("")
    logger.info("Input Correlation Matrix:")
    logger.info(f"  {correlation[0]}")
    logger.info(f"  {correlation[1]}")
    
    logger.info("")
    logger.info("Realized Correlation Matrix (from generated paths):")
    logger.info(f"  {realized_corr[0].round(4)}")
    logger.info(f"  {realized_corr[1].round(4)}")
    
    logger.info(f"")
    logger.info(f"Correlation error: {np.abs(correlation - realized_corr).max():.6f}")
    
    # -------------------------------------------------------------------------
    # Print statistics for each factor
    # -------------------------------------------------------------------------
    stats = generator.compute_statistics(result)
    for factor in factors:
        logger.info("")
        logger.info(f"{factor.display_name}:")
        for key, value in stats[factor.display_name].items():
            logger.info(f"  {key}: {value:.6f}")


# =============================================================================
# EXAMPLE 3: Heston Stochastic Volatility (Equity)
# =============================================================================

def example_3_heston_equity() -> None:
    """
    Example 3: Heston stochastic volatility for equity.
    
    Generate S&P 500 paths with stochastic variance.
    Captures the leverage effect (negative spot-vol correlation).
    
    Heston Model
    ------------
    dS_t = μ S_t dt + √V_t S_t dW_t^S
    dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V
    Corr(dW^S, dW^V) = ρ
    
    Parameters:
    - kappa (κ): Variance mean-reversion speed
    - theta (θ): Long-term variance (square of long-term vol)
    - xi (ξ): Vol-of-vol
    - v0: Initial variance
    - rho_internal (ρ): Spot-vol correlation (negative = leverage effect)
    
    Feller condition: 2κθ > ξ² ensures V_t > 0
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Example 3: Heston Stochastic Volatility (S&P 500)")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Define equity with Heston dynamics
    # -------------------------------------------------------------------------
    spx = RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "SPX"),
        initial_value=4500.0,
        dynamics=HestonDynamicsSpec(
            drift=0.05,                   # 5% risk-neutral drift (r - q)
            kappa=2.0,                    # Mean reversion speed
            theta=0.04,                   # Long-term variance (20% vol squared)
            xi=0.3,                       # Vol of vol
            v0=0.04,                      # Initial variance (20% vol)
            rho_internal=-0.7,            # Negative leverage effect
        ),
        name="S&P 500",
    )
    
    config = TimeseriesConfig(
        factors=[spx],
        correlation=np.array([[1.0]]),
        start_date="2024-01-01",
        end_date="2024-12-31",
        freq="D",
        n_scenarios=5000,
    )
    
    generator = TimeseriesGenerator(config)
    result = generator.generate_paths(seed=42)
    
    # -------------------------------------------------------------------------
    # Print statistics
    # -------------------------------------------------------------------------
    stats = generator.compute_statistics(result)
    logger.info("")
    logger.info(f"Generation Statistics for {spx.display_name}:")
    for key, value in stats[spx.display_name].items():
        logger.info(f"  {key}: {value:.4f}")
    
    # -------------------------------------------------------------------------
    # Variance path statistics (Heston-specific)
    # -------------------------------------------------------------------------
    if spx.market_id in result.variance_paths:
        variance_paths = result.variance_paths[spx.market_id]
        if not np.all(np.isnan(variance_paths)):
            vol_paths = np.sqrt(variance_paths)
            logger.info("")
            logger.info("Variance Path Statistics:")
            logger.info(f"  Initial Vol: {np.sqrt(0.04) * 100:.2f}%")
            logger.info(f"  Terminal Mean Vol: {np.mean(vol_paths[-1, :]) * 100:.2f}%")
            logger.info(f"  Terminal Min Vol: {np.min(vol_paths[-1, :]) * 100:.2f}%")
            logger.info(f"  Terminal Max Vol: {np.max(vol_paths[-1, :]) * 100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Plot spot and vol paths
    # -------------------------------------------------------------------------
    if MATPLOTLIB_AVAILABLE and ENABLE_PLOTTING:
        spot_paths = result.paths[spx.market_id]
        var_paths = result.variance_paths[spx.market_id]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Spot paths
        for i in range(10):
            ax1.plot(spot_paths[:, i], alpha=0.7)
        ax1.set_ylabel("S&P 500 Index")
        ax1.set_title("S&P 500 Spot Paths (Heston Model)")
        ax1.grid(True, alpha=0.3)
        
        # Vol paths
        for i in range(10):
            ax2.plot(np.sqrt(var_paths[:, i]) * 100, alpha=0.7)
        ax2.axhline(y=20.0, color='red', linestyle='--', label='Long-term Vol (20%)')
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Instantaneous Vol (%)")
        ax2.set_title("Instantaneous Volatility Paths (Heston Model)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=True)


# =============================================================================
# EXAMPLE 4: Multi-Asset Portfolio
# =============================================================================

def example_4_multi_asset_portfolio() -> None:
    """
    Example 4: Multi-asset portfolio with various dynamics.
    
    Generate correlated paths for FX, equity, and rate factors.
    Demonstrates integration with MarketDataset for pricing.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Example 4: Multi-Asset Portfolio")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Define diverse risk factors
    # -------------------------------------------------------------------------
    factors = [
        # FX factors (GBM)
        RiskFactorSpec(
            market_id=MarketId("FX", "SPOT", "EURUSD"),
            initial_value=1.08,
            dynamics=GBMDynamicsSpec(drift=0.0, vol=0.08),
            name="EUR/USD",
        ),
        # Equity with Heston
        RiskFactorSpec(
            market_id=MarketId("EQ", "SPOT", "SPX"),
            initial_value=4500.0,
            dynamics=HestonDynamicsSpec(
                drift=0.05,
                kappa=2.0,
                theta=0.04,
                xi=0.3,
                v0=0.04,
                rho_internal=-0.7,
            ),
            name="S&P 500",
        ),
        # Rate level (Ornstein-Uhlenbeck)
        RiskFactorSpec(
            market_id=MarketId("IR", "LEVEL", "USD"),
            initial_value=0.05,           # 5% starting rate
            dynamics=OUDynamicsSpec(
                mean=0.04,                # Long-term mean 4%
                kappa=0.5,                # Mean reversion speed
                vol=0.01,                 # 100bp annual vol
            ),
            name="USD Rate Level",
        ),
    ]
    
    # -------------------------------------------------------------------------
    # Correlation structure
    # EUR/USD and SPX: 0.3 (positive during risk-on)
    # EUR/USD and USD Rate: -0.2 (higher US rates → stronger USD)
    # SPX and USD Rate: -0.1 (higher rates → lower equity)
    # -------------------------------------------------------------------------
    correlation = np.array([
        [1.00,  0.30, -0.20],   # EUR/USD
        [0.30,  1.00, -0.10],   # S&P 500
        [-0.20, -0.10, 1.00],   # USD Rate
    ])
    
    config = TimeseriesConfig(
        factors=factors,
        correlation=correlation,
        start_date="2024-01-01",
        end_date="2024-12-31",
        freq="D",
        n_scenarios=10000,
    )
    
    generator = TimeseriesGenerator(config)
    
    # -------------------------------------------------------------------------
    # Generate full MarketDataset
    # This integrates with the pricing framework
    # -------------------------------------------------------------------------
    dataset = generator.generate(seed=42)
    
    logger.info("")
    logger.info("Generated Dataset:")
    logger.info(f"  Dates: {len(dataset.dates)} ({dataset.dates[0]} to {dataset.dates[-1]})")
    logger.info(f"  Scenarios: {dataset.n_scenarios}")
    logger.info(f"  Panels: {list(dataset.panels.keys())}")
    
    # -------------------------------------------------------------------------
    # Access market snapshots
    # -------------------------------------------------------------------------
    market_t100_s0 = dataset.snapshot(time_idx=100, scenario_idx=0)
    
    logger.info("")
    logger.info("Market Snapshot at t=100, scenario=0:")
    logger.info(f"  Date: {market_t100_s0.asof}")
    logger.info(f"  EUR/USD: {market_t100_s0.quote(factors[0].market_id):.4f}")
    logger.info(f"  S&P 500: {market_t100_s0.quote(factors[1].market_id):.2f}")
    logger.info(f"  USD Rate: {market_t100_s0.quote(factors[2].market_id) * 100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Verify realized correlation
    # -------------------------------------------------------------------------
    result = generator.generate_paths(seed=42)
    realized_corr = generator.compute_realized_correlation(result)
    
    logger.info("")
    logger.info("Realized vs Input Correlation:")
    logger.info(f"  Max absolute error: {np.abs(correlation - realized_corr).max():.4f}")


# =============================================================================
# EXAMPLE 5: Rate Curve Factors (PCA-Style)
# =============================================================================

def example_5_rate_curve_factors() -> None:
    """
    Example 5: Rate curve factors (Level, Slope, Curvature).
    
    Use OU dynamics to simulate PCA-style rate curve factors.
    This is common for interest rate risk management.
    
    PCA Decomposition
    -----------------
    Typical factor loadings:
    - Level: All tenors move together (+1 on all)
    - Slope: Short vs long rates (negative short, positive long)
    - Curvature: Butterfly (negative 2Y, positive wings)
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Example 5: Rate Curve Factors (Level/Slope/Curvature)")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Define three rate curve factors
    # -------------------------------------------------------------------------
    factors = [
        # Level factor (parallel shift)
        RiskFactorSpec(
            market_id=MarketId("IR", "LEVEL", "USD"),
            initial_value=0.0,            # Start at zero (shock factor)
            dynamics=OUDynamicsSpec(
                mean=0.0,                 # Zero mean
                kappa=0.1,                # Slow reversion
                vol=0.005,                # 50bp annual vol
            ),
            name="USD Level",
        ),
        # Slope factor (twist)
        RiskFactorSpec(
            market_id=MarketId("IR", "SLOPE", "USD"),
            initial_value=0.0,
            dynamics=OUDynamicsSpec(
                mean=0.0,
                kappa=0.2,                # Faster reversion
                vol=0.003,                # 30bp annual vol
            ),
            name="USD Slope",
        ),
        # Curvature factor (butterfly)
        RiskFactorSpec(
            market_id=MarketId("IR", "CURVE", "USD"),
            initial_value=0.0,
            dynamics=OUDynamicsSpec(
                mean=0.0,
                kappa=0.3,                # Fastest reversion
                vol=0.002,                # 20bp annual vol
            ),
            name="USD Curvature",
        ),
    ]
    
    # -------------------------------------------------------------------------
    # Typical PCA-based factor correlations
    # Level tends to be uncorrelated with slope/curve
    # -------------------------------------------------------------------------
    correlation = np.array([
        [1.00,  0.15, -0.10],  # Level
        [0.15,  1.00, -0.20],  # Slope
        [-0.10, -0.20, 1.00],  # Curvature
    ])
    
    config = TimeseriesConfig(
        factors=factors,
        correlation=correlation,
        start_date="2024-01-01",
        end_date="2024-12-31",
        freq="D",
        n_scenarios=5000,
    )
    
    generator = TimeseriesGenerator(config)
    result = generator.generate_paths(seed=42)
    
    # -------------------------------------------------------------------------
    # Print statistics in basis points
    # -------------------------------------------------------------------------
    stats = generator.compute_statistics(result)
    
    logger.info("")
    logger.info("Factor Statistics (in basis points):")
    for factor in factors:
        s = stats[factor.display_name]
        logger.info("")
        logger.info(f"{factor.display_name}:")
        logger.info(f"  Terminal Mean: {s['terminal_mean'] * 10000:.2f} bp")
        logger.info(f"  Terminal Std: {s['terminal_std'] * 10000:.2f} bp")
        logger.info(f"  Terminal Range: [{s['terminal_min'] * 10000:.2f}, {s['terminal_max'] * 10000:.2f}] bp")


# =============================================================================
# EXAMPLE 6: Choosing the Right Dynamics Model
# =============================================================================

def example_6_choosing_the_right_model() -> None:
    """
    Example 6: Guide to choosing the right dynamics model.
    
    Demonstrates the practical differences between GBM and Heston
    to help select the appropriate model for each use case.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Example 6: Choosing the Right Dynamics Model")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Compare GBM vs Heston for equity
    # -------------------------------------------------------------------------
    n_scenarios = 5000
    initial_spot = 100.0
    
    # GBM: Simple, constant volatility
    gbm_factor = RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "TEST_GBM"),
        initial_value=initial_spot,
        dynamics=GBMDynamicsSpec(drift=0.05, vol=0.20),
        name="GBM (σ=20%)",
    )
    
    # Heston: Stochastic volatility with leverage effect
    heston_factor = RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "TEST_HESTON"),
        initial_value=initial_spot,
        dynamics=HestonDynamicsSpec(
            drift=0.05,
            kappa=2.0,
            theta=0.04,               # Long-term 20% vol
            xi=0.3,
            v0=0.04,                  # Start at 20% vol
            rho_internal=-0.7,        # Negative leverage
        ),
        name="Heston (stoch vol)",
    )
    
    # -------------------------------------------------------------------------
    # Simulate both
    # -------------------------------------------------------------------------
    results = {}
    for factor in [gbm_factor, heston_factor]:
        config = TimeseriesConfig(
            factors=[factor],
            correlation=np.array([[1.0]]),
            start_date="2024-01-01",
            end_date="2024-12-31",
            freq="D",
            n_scenarios=n_scenarios,
        )
        gen = TimeseriesGenerator(config)
        result = gen.generate_paths(seed=42)
        results[factor.display_name] = result.paths[factor.market_id][-1, :]
    
    # -------------------------------------------------------------------------
    # Compare distributions
    # -------------------------------------------------------------------------
    try:
        from scipy import stats as scipy_stats
        
        logger.info("")
        logger.info("Terminal Distribution Comparison (after 1 year):")
        logger.info("-" * 50)
        logger.info(f"{'Model':<25} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurtosis':>10}")
        logger.info("-" * 50)
        
        for name, terminals in results.items():
            mean = np.mean(terminals)
            std = np.std(terminals)
            skew = scipy_stats.skew(terminals)
            kurt = scipy_stats.kurtosis(terminals)
            logger.info(f"{name:<25} {mean:>10.2f} {std:>10.2f} {skew:>10.3f} {kurt:>10.3f}")
        
        logger.info("")
        logger.info("Key Observations:")
        logger.info("  - GBM: Symmetric distribution (skew ≈ 0)")
        logger.info("  - Heston: Negative skew due to leverage effect (ρ < 0)")
        logger.info("  - Heston: Higher kurtosis (fatter tails) due to vol clustering")
        logger.info("")
        logger.info("When to use:")
        logger.info("  - GBM: Quick estimates, FX (which has less pronounced skew)")
        logger.info("  - Heston: Options pricing, equity VaR (captures tail risk better)")
        
    except ImportError:
        logger.warning("scipy not available - skipping distribution comparison")


# =============================================================================
# EXAMPLE 7: Stress Scenario Generation
# =============================================================================

def example_7_stress_scenario_generation() -> None:
    """
    Example 7: Generating stress scenarios.
    
    Demonstrates how to create stressed market scenarios with
    modified dynamics parameters (e.g., market crash, vol spike).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Example 7: Stress Scenario Generation")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Normal market conditions
    # -------------------------------------------------------------------------
    normal_equity = RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "SPX"),
        initial_value=4500.0,
        dynamics=HestonDynamicsSpec(
            drift=0.05,                   # 5% expected return
            kappa=2.0,
            theta=0.04,                   # 20% long-term vol
            xi=0.3,
            v0=0.04,                      # 20% current vol
            rho_internal=-0.7,
        ),
        name="S&P 500 (Normal)",
    )
    
    # -------------------------------------------------------------------------
    # Stress scenario: 2008-style crash
    # Modified parameters for crisis conditions
    # -------------------------------------------------------------------------
    stress_equity = RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "SPX"),
        initial_value=4500.0,
        dynamics=HestonDynamicsSpec(
            drift=-0.15,                  # Negative drift (recession)
            kappa=5.0,                    # Faster mean reversion
            theta=0.09,                   # 30% long-term vol
            xi=0.8,                       # High vol-of-vol
            v0=0.16,                      # 40% starting vol (crisis level)
            rho_internal=-0.9,            # Extreme leverage
        ),
        name="S&P 500 (Stress)",
    )
    
    # -------------------------------------------------------------------------
    # Simulate both over 3 months
    # -------------------------------------------------------------------------
    for label, factor in [("Normal", normal_equity), ("Stress", stress_equity)]:
        config = TimeseriesConfig(
            factors=[factor],
            correlation=np.array([[1.0]]),
            start_date="2024-01-01",
            end_date="2024-03-31",         # 90 days
            freq="D",
            n_scenarios=10000,
        )
        gen = TimeseriesGenerator(config)
        result = gen.generate_paths(seed=42)
        terminals = result.paths[factor.market_id][-1, :]
        
        returns = (terminals / 4500.0 - 1) * 100  # Percentage returns
        
        logger.info("")
        logger.info(f"{label} Scenario (90 days):")
        logger.info(f"  Mean return: {np.mean(returns):.1f}%")
        logger.info(f"  Std dev: {np.std(returns):.1f}%")
        logger.info(f"  5th percentile (VaR 95): {np.percentile(returns, 5):.1f}%")
        logger.info(f"  1st percentile (VaR 99): {np.percentile(returns, 1):.1f}%")
        logger.info(f"  Worst scenario: {np.min(returns):.1f}%")


# =============================================================================
# EXAMPLE 8: Full Portfolio Simulation Workflow
# =============================================================================

def example_8_full_portfolio_simulation() -> None:
    """
    Example 8: Complete portfolio simulation workflow.
    
    Demonstrates end-to-end generation for a multi-asset portfolio
    with realistic correlation structure and integration with
    MarketDataset for pricing.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Example 8: Full Portfolio Simulation Workflow")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Define a realistic hedge fund portfolio of risk factors
    # -------------------------------------------------------------------------
    factors = [
        # FX positions
        RiskFactorSpec(
            market_id=MarketId("FX", "SPOT", "EURUSD"),
            initial_value=1.08,
            dynamics=GBMDynamicsSpec(drift=0.01, vol=0.08),
            name="EUR/USD",
        ),
        RiskFactorSpec(
            market_id=MarketId("FX", "SPOT", "USDJPY"),
            initial_value=150.0,
            dynamics=GBMDynamicsSpec(drift=-0.01, vol=0.10),
            name="USD/JPY",
        ),
        # Equity
        RiskFactorSpec(
            market_id=MarketId("EQ", "SPOT", "SPX"),
            initial_value=4500.0,
            dynamics=HestonDynamicsSpec(
                drift=0.07,
                kappa=2.0,
                theta=0.04,
                xi=0.3,
                v0=0.04,
                rho_internal=-0.7,
            ),
            name="S&P 500",
        ),
        # Rates
        RiskFactorSpec(
            market_id=MarketId("IR", "LEVEL", "USD"),
            initial_value=0.05,
            dynamics=OUDynamicsSpec(mean=0.04, kappa=0.5, vol=0.01),
            name="USD Rate",
        ),
        RiskFactorSpec(
            market_id=MarketId("IR", "LEVEL", "EUR"),
            initial_value=0.03,
            dynamics=OUDynamicsSpec(mean=0.025, kappa=0.5, vol=0.008),
            name="EUR Rate",
        ),
    ]
    
    # -------------------------------------------------------------------------
    # Realistic correlation matrix
    # Rows/cols: EUR/USD, USD/JPY, SPX, USD Rate, EUR Rate
    # -------------------------------------------------------------------------
    correlation = np.array([
        # EUR/USD  USD/JPY    SPX    USD_R   EUR_R
        [  1.00,   -0.30,    0.35,  -0.15,   0.10],   # EUR/USD
        [ -0.30,    1.00,   -0.20,   0.10,  -0.05],   # USD/JPY
        [  0.35,   -0.20,    1.00,  -0.10,   0.05],   # SPX
        [ -0.15,    0.10,   -0.10,   1.00,   0.60],   # USD Rate
        [  0.10,   -0.05,    0.05,   0.60,   1.00],   # EUR Rate
    ])
    
    config = TimeseriesConfig(
        factors=factors,
        correlation=correlation,
        start_date="2024-01-01",
        end_date="2024-12-31",
        freq="D",
        n_scenarios=10000,
    )
    
    logger.info("Generating 10,000 scenarios for 5 risk factors over 1 year...")
    generator = TimeseriesGenerator(config)
    dataset = generator.generate(seed=42)
    
    logger.info("")
    logger.info("Dataset created:")
    logger.info(f"  Time points: {len(dataset.dates)}")
    logger.info(f"  Scenarios: {dataset.n_scenarios}")
    logger.info(f"  Risk factors: {len(dataset.panels)}")
    
    # -------------------------------------------------------------------------
    # Demonstrate snapshot access
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Sample market snapshots:")
    end_idx = len(dataset.dates) - 1
    
    for t_idx, label in [(0, "Start"), (125, "Mid-year"), (end_idx, "End")]:
        market = dataset.snapshot(time_idx=t_idx, scenario_idx=0)
        logger.info("")
        logger.info(f"  {label} ({market.asof}):")
        logger.info(f"    EUR/USD: {market.quote(factors[0].market_id):.4f}")
        logger.info(f"    USD/JPY: {market.quote(factors[1].market_id):.2f}")
        logger.info(f"    S&P 500: {market.quote(factors[2].market_id):.0f}")
        logger.info(f"    USD Rate: {market.quote(factors[3].market_id)*100:.2f}%")
        logger.info(f"    EUR Rate: {market.quote(factors[4].market_id)*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Validate correlation
    # -------------------------------------------------------------------------
    result = generator.generate_paths(seed=42)
    realized = generator.compute_realized_correlation(result)
    
    logger.info("")
    logger.info("Correlation validation:")
    logger.info(f"  Max error vs input: {np.abs(correlation - realized).max():.4f}")
    
    # -------------------------------------------------------------------------
    # Compute return statistics
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Annualized return statistics:")
    stats = generator.compute_statistics(result)
    
    for factor in factors:
        s = stats[factor.display_name]
        annual_ret = (s['terminal_mean'] / s['initial'] - 1) * 100
        annual_vol = (s['terminal_std'] / s['initial']) * 100
        logger.info(f"  {factor.display_name}: {annual_ret:+.1f}% return, {annual_vol:.1f}% vol")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary() -> None:
    """Print summary of key concepts."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    summary = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         KEY TAKEAWAYS                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. Dynamics Selection:                                             │
    │     - GBM: FX, commodities, simple equity                           │
    │     - Heston: Equity with vol dynamics, options risk                │
    │     - OU: Interest rates, credit spreads, mean-reverting            │
    │                                                                      │
    │  2. Correlation:                                                    │
    │     - Essential for portfolio risk                                  │
    │     - Uses Cholesky decomposition                                   │
    │     - Must be positive semi-definite                                │
    │                                                                      │
    │  3. Integration:                                                    │
    │     - generator.generate() → MarketDataset                          │
    │     - dataset.snapshot(t, s) → Market for pricing                   │
    │                                                                      │
    │  4. Stress Testing:                                                 │
    │     - Modify dynamics parameters for crisis scenarios               │
    │     - Higher vol, negative drift, extreme leverage                  │
    │                                                                      │
    │  See docs/reference/marketdata/timeseries_generation.md             │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the example.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    global ENABLE_PLOTTING
    ENABLE_PLOTTING = args.plot
    
    try:
        logger.info("=" * 70)
        logger.info("QuantStrata Time Series Generation - Comprehensive Examples")
        logger.info("=" * 70)
        logger.info("""
This script demonstrates:
  1. Single factor GBM generation (FX spot)
  2. Correlated multi-asset generation (FX pairs)
  3. Heston stochastic volatility (equity)
  4. Multi-asset portfolio (FX + Equity + Rates)
  5. Rate curve factors (Level/Slope/Curvature)
  6. Choosing the right dynamics model
  7. Stress scenario generation
  8. Full portfolio simulation workflow
""")
        
        # Run all examples
        example_1_single_factor_gbm()
        example_2_correlated_fx_pairs()
        example_3_heston_equity()
        example_4_multi_asset_portfolio()
        example_5_rate_curve_factors()
        example_6_choosing_the_right_model()
        example_7_stress_scenario_generation()
        example_8_full_portfolio_simulation()
        
        # Summary
        print_summary()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time Series Generation Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Enable plotting (default: True)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable plotting",
    )
    
    args = parser.parse_args()
    main(args)
