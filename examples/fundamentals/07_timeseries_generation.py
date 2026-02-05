"""
Time Series Generation for Risk Factors.

This example provides a comprehensive demonstration of the TimeseriesGenerator 
framework for generating realistic, correlated risk factor time series suitable 
for production hedge fund risk management workflows.

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

Usage
-----
Run this script directly:
    python examples/fundamentals/07_timeseries_generation.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.timeseries import (
    TimeseriesGenerator,
    TimeseriesConfig,
    RiskFactorSpec,
    GBMDynamicsSpec,
    HestonDynamicsSpec,
    OUDynamicsSpec,
)


def example_1_single_factor_gbm():
    """
    Example 1: Single GBM factor (FX spot).

    Generate EUR/USD spot paths using Geometric Brownian Motion.
    """
    print("\n" + "=" * 60)
    print("Example 1: Single GBM Factor (EUR/USD Spot)")
    print("=" * 60)

    # Define single FX spot factor
    eurusd = RiskFactorSpec(
        market_id=MarketId("FX", "SPOT", "EURUSD"),
        initial_value=1.08,
        dynamics=GBMDynamicsSpec(
            drift=0.0,   # Zero drift (martingale under domestic measure)
            vol=0.08,    # 8% annual volatility
        ),
        name="EUR/USD Spot",
    )

    # Single factor -> 1x1 correlation matrix
    config = TimeseriesConfig(
        factors=[eurusd],
        correlation=np.array([[1.0]]),
        start_date="2024-01-01",
        end_date="2024-12-31",
        freq="D",
        n_scenarios=1000,
    )

    # Generate paths
    generator = TimeseriesGenerator(config)
    result = generator.generate_paths(seed=42)

    # Print statistics
    stats = generator.compute_statistics(result)
    print(f"\nGeneration Statistics for {eurusd.display_name}:")
    for key, value in stats[eurusd.display_name].items():
        print(f"  {key}: {value:.6f}")

    # Plot sample paths
    paths = result.paths[eurusd.market_id]
    _plot_paths(
        paths=paths[:, :10],  # First 10 scenarios
        dates=result.dates,
        title="EUR/USD Spot Paths (GBM)",
        ylabel="EUR/USD",
    )


def example_2_correlated_fx_pairs():
    """
    Example 2: Correlated FX pairs.

    Generate EUR/USD and GBP/USD with 60% correlation.
    """
    print("\n" + "=" * 60)
    print("Example 2: Correlated FX Pairs")
    print("=" * 60)

    # Define two correlated FX factors
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

    # 60% correlation between EUR/USD and GBP/USD
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
        n_scenarios=10000,
    )

    generator = TimeseriesGenerator(config)
    result = generator.generate_paths(seed=42)

    # Validate realized correlation
    realized_corr = generator.compute_realized_correlation(result)
    print(f"\nInput Correlation Matrix:")
    print(correlation)
    print(f"\nRealized Correlation Matrix (from generated paths):")
    print(realized_corr.round(4))
    print(f"\nCorrelation error: {np.abs(correlation - realized_corr).max():.6f}")

    # Print statistics
    stats = generator.compute_statistics(result)
    for factor in factors:
        print(f"\n{factor.display_name}:")
        for key, value in stats[factor.display_name].items():
            print(f"  {key}: {value:.6f}")


def example_3_heston_equity():
    """
    Example 3: Heston stochastic volatility for equity.

    Generate S&P 500 paths with stochastic variance.
    """
    print("\n" + "=" * 60)
    print("Example 3: Heston Stochastic Volatility (S&P 500)")
    print("=" * 60)

    # Define equity with Heston dynamics
    spx = RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "SPX"),
        initial_value=4500.0,
        dynamics=HestonDynamicsSpec(
            drift=0.05,         # 5% risk-neutral drift (r - q)
            kappa=2.0,          # Mean reversion speed
            theta=0.04,         # Long-term variance (20% vol)
            xi=0.3,             # Vol of vol
            v0=0.04,            # Initial variance
            rho_internal=-0.7,  # Negative leverage effect
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

    # Print statistics
    stats = generator.compute_statistics(result)
    print(f"\nGeneration Statistics for {spx.display_name}:")
    for key, value in stats[spx.display_name].items():
        print(f"  {key}: {value:.4f}")

    # Also have variance paths
    var_mkt_id = MarketId("EQ", "VARIANCE", "SPX")
    if var_mkt_id in result.variance_paths and not np.all(np.isnan(result.variance_paths[spx.market_id])):
        variance_paths = result.variance_paths[spx.market_id]
        vol_paths = np.sqrt(variance_paths)
        print(f"\nVariance Path Statistics:")
        print(f"  Initial Vol: {np.sqrt(0.04) * 100:.2f}%")
        print(f"  Terminal Mean Vol: {np.mean(vol_paths[-1, :]) * 100:.2f}%")
        print(f"  Terminal Min Vol: {np.min(vol_paths[-1, :]) * 100:.2f}%")
        print(f"  Terminal Max Vol: {np.max(vol_paths[-1, :]) * 100:.2f}%")

    # Plot spot and vol paths
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
    plt.savefig("heston_paths.png", dpi=150)
    print("\nSaved: heston_paths.png")


def example_4_multi_asset_portfolio():
    """
    Example 4: Multi-asset portfolio with various dynamics.

    Generate correlated paths for FX, equity, and rate factors.
    """
    print("\n" + "=" * 60)
    print("Example 4: Multi-Asset Portfolio")
    print("=" * 60)

    # Define diverse risk factors
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
                drift=0.05, kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho_internal=-0.7
            ),
            name="S&P 500",
        ),
        # Rate level (Ornstein-Uhlenbeck)
        RiskFactorSpec(
            market_id=MarketId("IR", "LEVEL", "USD"),
            initial_value=0.05,  # 5% starting rate
            dynamics=OUDynamicsSpec(
                mean=0.04,     # Long-term mean 4%
                kappa=0.5,     # Mean reversion speed
                vol=0.01,      # 100bp annual vol
            ),
            name="USD Rate Level",
        ),
    ]

    # Correlation structure:
    # EUR/USD and SPX: 0.3 (positive during risk-on)
    # EUR/USD and USD Rate: -0.2 (higher US rates -> stronger USD)
    # SPX and USD Rate: -0.1 (higher rates -> lower equity)
    correlation = np.array([
        [1.00,  0.30, -0.20],  # EUR/USD
        [0.30,  1.00, -0.10],  # S&P 500
        [-0.20, -0.10, 1.00],  # USD Rate
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

    # Generate full MarketDataset
    dataset = generator.generate(seed=42)

    print(f"\nGenerated Dataset:")
    print(f"  Dates: {len(dataset.dates)} ({dataset.dates[0]} to {dataset.dates[-1]})")
    print(f"  Scenarios: {dataset.n_scenarios}")
    print(f"  Panels: {list(dataset.panels.keys())}")

    # Access market snapshots
    market_t100_s0 = dataset.snapshot(time_idx=100, scenario_idx=0)
    print(f"\nMarket Snapshot at t=100, scenario=0:")
    print(f"  Date: {market_t100_s0.asof}")
    print(f"  EUR/USD: {market_t100_s0.quote(factors[0].market_id):.4f}")
    print(f"  S&P 500: {market_t100_s0.quote(factors[1].market_id):.2f}")
    print(f"  USD Rate: {market_t100_s0.quote(factors[2].market_id) * 100:.2f}%")

    # Verify realized correlation
    result = generator.generate_paths(seed=42)
    realized_corr = generator.compute_realized_correlation(result)
    print(f"\nRealized vs Input Correlation:")
    print(f"  Max absolute error: {np.abs(correlation - realized_corr).max():.4f}")


def example_5_rate_curve_factors():
    """
    Example 5: Rate curve factors (Level, Slope, Curvature).

    Use OU dynamics to simulate PCA-style rate curve factors.
    """
    print("\n" + "=" * 60)
    print("Example 5: Rate Curve Factors (Level/Slope/Curvature)")
    print("=" * 60)

    # Define three rate curve factors
    factors = [
        # Level factor (parallel shift)
        RiskFactorSpec(
            market_id=MarketId("IR", "LEVEL", "USD"),
            initial_value=0.0,  # Start at zero (shock factor)
            dynamics=OUDynamicsSpec(mean=0.0, kappa=0.1, vol=0.005),
            name="USD Level",
        ),
        # Slope factor (twist)
        RiskFactorSpec(
            market_id=MarketId("IR", "SLOPE", "USD"),
            initial_value=0.0,
            dynamics=OUDynamicsSpec(mean=0.0, kappa=0.2, vol=0.003),
            name="USD Slope",
        ),
        # Curvature factor (butterfly)
        RiskFactorSpec(
            market_id=MarketId("IR", "CURVE", "USD"),
            initial_value=0.0,
            dynamics=OUDynamicsSpec(mean=0.0, kappa=0.3, vol=0.002),
            name="USD Curvature",
        ),
    ]

    # Typical PCA-based factor correlations
    # Level tends to be uncorrelated with slope/curve
    correlation = np.array([
        [1.00, 0.15, -0.10],  # Level
        [0.15, 1.00, -0.20],  # Slope
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

    # Print statistics
    stats = generator.compute_statistics(result)
    print("\nFactor Statistics (in basis points):")
    for factor in factors:
        s = stats[factor.display_name]
        print(f"\n{factor.display_name}:")
        print(f"  Terminal Mean: {s['terminal_mean'] * 10000:.2f} bp")
        print(f"  Terminal Std: {s['terminal_std'] * 10000:.2f} bp")
        print(f"  Terminal Range: [{s['terminal_min'] * 10000:.2f}, {s['terminal_max'] * 10000:.2f}] bp")


def _plot_paths(paths: np.ndarray, dates: list, title: str, ylabel: str, save: bool = False):
    """Helper to plot sample paths."""
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
        # Create safe filename (remove special chars)
        filename = title.lower()
        for char in " /\\()[]{}":
            filename = filename.replace(char, "_")
        filename = filename + ".png"
        plt.savefig(filename, dpi=150)
        print(f"  Plot saved: {filename}")
    
    plt.close()


def example_6_choosing_the_right_model():
    """
    Example 6: Guide to choosing the right dynamics model.

    This example demonstrates the practical differences between models
    to help you select the appropriate one for your use case.
    """
    print("\n" + "=" * 60)
    print("Example 6: Choosing the Right Dynamics Model")
    print("=" * 60)

    # We'll simulate the same asset with different models and compare

    # Common setup
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
            theta=0.04,      # Long-term 20% vol
            xi=0.3,
            v0=0.04,         # Start at 20% vol
            rho_internal=-0.7,  # Negative leverage
        ),
        name="Heston (stoch vol)",
    )

    # Simulate both
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

    # Compare distributions
    print("\nTerminal Distribution Comparison (after 1 year):")
    print("-" * 50)
    print(f"{'Model':<25} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurtosis':>10}")
    print("-" * 50)

    from scipy import stats as scipy_stats

    for name, terminals in results.items():
        mean = np.mean(terminals)
        std = np.std(terminals)
        skew = scipy_stats.skew(terminals)
        kurt = scipy_stats.kurtosis(terminals)
        print(f"{name:<25} {mean:>10.2f} {std:>10.2f} {skew:>10.3f} {kurt:>10.3f}")

    print("\nKey Observations:")
    print("  - GBM: Symmetric distribution (skew ≈ 0)")
    print("  - Heston: Negative skew due to leverage effect (ρ < 0)")
    print("  - Heston: Higher kurtosis (fatter tails) due to vol clustering")
    print("\nWhen to use:")
    print("  - GBM: Quick estimates, FX (which has less pronounced skew)")
    print("  - Heston: Options pricing, equity VaR (captures tail risk better)")


def example_7_stress_scenario_generation():
    """
    Example 7: Generating stress scenarios.

    Demonstrates how to create stressed market scenarios with
    modified dynamics parameters (e.g., market crash, vol spike).
    """
    print("\n" + "=" * 60)
    print("Example 7: Stress Scenario Generation")
    print("=" * 60)

    # Normal market conditions
    normal_equity = RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "SPX"),
        initial_value=4500.0,
        dynamics=HestonDynamicsSpec(
            drift=0.05,        # 5% expected return
            kappa=2.0,
            theta=0.04,        # 20% long-term vol
            xi=0.3,
            v0=0.04,           # 20% current vol
            rho_internal=-0.7,
        ),
        name="S&P 500 (Normal)",
    )

    # Stress scenario: 2008-style crash
    stress_equity = RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "SPX"),
        initial_value=4500.0,
        dynamics=HestonDynamicsSpec(
            drift=-0.15,       # Negative drift (recession)
            kappa=5.0,         # Faster mean reversion
            theta=0.09,        # 30% long-term vol
            xi=0.8,            # High vol-of-vol
            v0=0.16,           # 40% starting vol (crisis level)
            rho_internal=-0.9, # Extreme leverage
        ),
        name="S&P 500 (Stress)",
    )

    # Simulate both over 3 months
    for label, factor in [("Normal", normal_equity), ("Stress", stress_equity)]:
        config = TimeseriesConfig(
            factors=[factor],
            correlation=np.array([[1.0]]),
            start_date="2024-01-01",
            end_date="2024-03-31",  # 90 days
            freq="D",
            n_scenarios=10000,
        )
        gen = TimeseriesGenerator(config)
        result = gen.generate_paths(seed=42)
        terminals = result.paths[factor.market_id][-1, :]

        returns = (terminals / 4500.0 - 1) * 100  # Percentage returns

        print(f"\n{label} Scenario (90 days):")
        print(f"  Mean return: {np.mean(returns):.1f}%")
        print(f"  Std dev: {np.std(returns):.1f}%")
        print(f"  5th percentile (VaR 95): {np.percentile(returns, 5):.1f}%")
        print(f"  1st percentile (VaR 99): {np.percentile(returns, 1):.1f}%")
        print(f"  Worst scenario: {np.min(returns):.1f}%")


def example_8_full_portfolio_simulation():
    """
    Example 8: Complete portfolio simulation workflow.

    Demonstrates end-to-end generation for a multi-asset portfolio
    with realistic correlation structure and integration with
    MarketDataset for pricing.
    """
    print("\n" + "=" * 60)
    print("Example 8: Full Portfolio Simulation Workflow")
    print("=" * 60)

    # Define a realistic hedge fund portfolio of risk factors
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
                drift=0.07, kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho_internal=-0.7
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

    # Realistic correlation matrix
    # Rows/cols: EUR/USD, USD/JPY, SPX, USD Rate, EUR Rate
    correlation = np.array([
        # EUR/USD  USD/JPY    SPX    USD_R   EUR_R
        [  1.00,   -0.30,    0.35,  -0.15,   0.10],  # EUR/USD
        [ -0.30,    1.00,   -0.20,   0.10,  -0.05],  # USD/JPY
        [  0.35,   -0.20,    1.00,  -0.10,   0.05],  # SPX
        [ -0.15,    0.10,   -0.10,   1.00,   0.60],  # USD Rate
        [  0.10,   -0.05,    0.05,   0.60,   1.00],  # EUR Rate
    ])

    config = TimeseriesConfig(
        factors=factors,
        correlation=correlation,
        start_date="2024-01-01",
        end_date="2024-12-31",
        freq="D",
        n_scenarios=10000,
    )

    print("\nGenerating 10,000 scenarios for 5 risk factors over 1 year...")
    generator = TimeseriesGenerator(config)
    dataset = generator.generate(seed=42)

    print(f"\nDataset created:")
    print(f"  Time points: {len(dataset.dates)}")
    print(f"  Scenarios: {dataset.n_scenarios}")
    print(f"  Risk factors: {len(dataset.panels)}")

    # Demonstrate snapshot access
    print("\nSample market snapshots:")
    end_idx = len(dataset.dates) - 1
    for t_idx, label in [(0, "Start"), (125, "Mid-year"), (end_idx, "End")]:
        market = dataset.snapshot(time_idx=t_idx, scenario_idx=0)
        print(f"\n  {label} ({market.asof}):")
        print(f"    EUR/USD: {market.quote(factors[0].market_id):.4f}")
        print(f"    USD/JPY: {market.quote(factors[1].market_id):.2f}")
        print(f"    S&P 500: {market.quote(factors[2].market_id):.0f}")
        print(f"    USD Rate: {market.quote(factors[3].market_id)*100:.2f}%")
        print(f"    EUR Rate: {market.quote(factors[4].market_id)*100:.2f}%")

    # Validate correlation
    result = generator.generate_paths(seed=42)
    realized = generator.compute_realized_correlation(result)

    print("\nCorrelation validation:")
    print(f"  Max error vs input: {np.abs(correlation - realized).max():.4f}")

    # Compute return statistics
    print("\nAnnualized return statistics:")
    stats = generator.compute_statistics(result)
    for factor in factors:
        s = stats[factor.display_name]
        # Annualized return from terminal mean
        annual_ret = (s['terminal_mean'] / s['initial'] - 1) * 100
        annual_vol = (s['terminal_std'] / s['initial']) * 100
        print(f"  {factor.display_name}: {annual_ret:+.1f}% return, {annual_vol:.1f}% vol")


if __name__ == "__main__":
    print("=" * 70)
    print("QuantStrata Time Series Generation - Comprehensive Examples")
    print("=" * 70)
    print("""
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

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("""
Key Takeaways:
  - GBM: Use for FX, commodities, simple equity simulation
  - Heston: Use for equity options, when vol dynamics matter
  - OU: Use for rates, spreads, any mean-reverting process
  - Correlation: Essential for portfolio risk - use Cholesky decomposition
  - Stress testing: Modify dynamics parameters for crisis scenarios
""")
    print("See docs/reference/marketdata/timeseries_generation.md for full documentation.")
