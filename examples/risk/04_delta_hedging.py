#!/usr/bin/env python3
"""
===============================================================================
Delta Hedging: Dynamic Hedging with QuantStrata Pricers
===============================================================================

This example demonstrates dynamic delta hedging using QuantStrata's production
FX pricers and market data infrastructure.

Learning Objectives
-------------------
1. **Delta Hedging**: Maintain delta-neutral position via continuous rebalancing
2. **Library Integration**: Use FxEuropeanVanillaBsmPricer for Greeks
3. **GBM Simulation**: Use library dynamics for spot simulation
4. **Hedge Effectiveness**: Measure P&L volatility reduction from hedging

Mathematical Framework
----------------------
Delta hedge position for a short option:
    Hedge Position = +Δ (buy delta shares to offset option delta)

For a call with Δ = 0.5:
    - Short 1 call (Δ = -0.5)
    - Buy 0.5 shares (Δ = +0.5)
    - Net delta = 0 (delta neutral)

As spot moves, delta changes (gamma effect):
    ΔΔ ≈ Γ × ΔS

Requiring rebalancing to maintain neutrality.

Hedging P&L decomposition:
    Hedge P&L = Gamma P&L - Transaction Costs - Theta Decay

Production Context
------------------
At a hedge fund:
- Options desks hedge to isolate vol/gamma exposure
- Rebalancing frequency is a key decision (cost vs accuracy)
- Perfect hedging is impossible (discrete rebalancing, transaction costs)
- Gamma P&L should offset theta over time (hedging earns vol carry)

Prerequisites
-------------
- Greeks computation (examples/risk/02_sensitivities_computation.py)
- P&L attribution (examples/risk/03_pnl_attribution.py)

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/risk/04_delta_hedging.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports - using actual library modules
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer

# BSM model for direct Greeks computation
from src.models.analytic.black_scholes_merton.base import (
    vanilla_price,
    vanilla_delta,
    vanilla_gamma,
    vanilla_theta,
    vanilla_vega,
)

# GBM dynamics for path simulation
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator


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
# CONSTANTS
# =============================================================================

EURUSD_SPOT = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HedgingConfig:
    """Configuration for hedging simulation."""
    # Option parameters
    spot: float = 1.0850
    strike: float = 1.0850  # ATM
    expiry: float = 0.25  # 3 months
    vol: float = 0.10
    r_dom: float = 0.05
    r_for: float = 0.04
    notional: float = 10_000_000
    
    # Simulation parameters
    n_steps: int = 63  # Daily steps (~3 months)
    n_paths: int = 1000  # MC paths
    
    # Transaction costs
    proportional_cost: float = 0.0001  # 1 bp
    
    # Rebalancing
    rebalance_frequency: int = 1  # Every n steps


@dataclass
class HedgeState:
    """State of the hedge at a point in time."""
    time: float
    spot: float
    option_value: float
    delta: float
    gamma: float
    hedge_position: float
    cash: float
    portfolio_value: float
    cumulative_costs: float


@dataclass
class HedgingResult:
    """Result of a hedging simulation."""
    states: List[HedgeState] = field(default_factory=list)
    final_pnl: float = 0.0
    pnl_std: float = 0.0
    total_costs: float = 0.0
    n_rebalances: int = 0


# =============================================================================
# MARKET AND INSTRUMENT SETUP
# =============================================================================

def create_market(
    spot: float,
    r_dom: float,
    r_for: float,
    vol: float,
    val_date: date,
) -> Market:
    """Create market snapshot with given parameters."""
    return Market(
        val_date=val_date,
        quotes={EURUSD_SPOT: Quote(value=spot)},
        curves={
            USD_CURVE: FlatZeroRateCurve(USD_CURVE, r_dom),
            EUR_CURVE: FlatZeroRateCurve(EUR_CURVE, r_for),
        },
        vol_surfaces={EURUSD_VOL: FlatVolSurface(EURUSD_VOL, vol)},
    )


def create_option(config: HedgingConfig) -> EuropeanFxVanillaOption:
    """Create FX vanilla option from config."""
    return EuropeanFxVanillaOption(
        option_type="call",
        spot_id=EURUSD_SPOT,
        domestic_curve_id=USD_CURVE,
        foreign_curve_id=EUR_CURVE,
        vol_id=EURUSD_VOL,
        strike=config.strike,
        expiry=config.expiry,
        notional=config.notional,
    )


# =============================================================================
# HEDGING FUNCTIONS USING LIBRARY BSM
# =============================================================================

def compute_greeks_from_model(
    spot: float,
    strike: float,
    expiry: float,
    r_dom: float,
    r_for: float,
    vol: float,
) -> Tuple[float, float, float, float, float]:
    """
    Compute Greeks using library BSM functions.
    
    For FX options:
    - discount_rate = r_dom (domestic rate)
    - carry = r_dom - r_for (interest rate differential)
    
    Returns
    -------
    Tuple
        (price, delta, gamma, theta, vega)
    """
    carry = r_dom - r_for
    
    price = vanilla_price(
        option_type="call",
        spot=spot,
        strike=strike,
        expiry=expiry,
        discount_rate=r_dom,
        carry=carry,
        vol=vol,
    )
    
    delta = vanilla_delta(
        option_type="call",
        spot=spot,
        strike=strike,
        expiry=expiry,
        discount_rate=r_dom,
        carry=carry,
        vol=vol,
    )
    
    gamma = vanilla_gamma(
        option_type="call",
        spot=spot,
        strike=strike,
        expiry=expiry,
        discount_rate=r_dom,
        carry=carry,
        vol=vol,
    )
    
    theta = vanilla_theta(
        option_type="call",
        spot=spot,
        strike=strike,
        expiry=expiry,
        discount_rate=r_dom,
        carry=carry,
        vol=vol,
    )
    
    vega = vanilla_vega(
        spot=spot,
        strike=strike,
        expiry=expiry,
        discount_rate=r_dom,
        carry=carry,
        vol=vol,
    )
    
    return price, delta, gamma, theta, vega


def simulate_gbm_paths(
    S0: float,
    drift: float,
    vol: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate GBM paths using library dynamics.
    
    Parameters
    ----------
    S0 : float
        Initial spot.
    drift : float
        Risk-neutral drift (r - q for FX: r_dom - r_for).
    vol : float
        Volatility.
    T : float
        Time horizon.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    ndarray
        Paths of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    n_half = n_paths // 2
    Z = rng.standard_normal((n_half, n_steps))
    Z = np.concatenate([Z, -Z], axis=0)
    simulator = GbmDynamicsSimulator(drift=drift, vol=vol)
    paths = simulator.simulate_paths(
        spot0=S0,
        maturity=T,
        n_steps=n_steps,
        n_paths=Z.shape[0],
        normals=Z,
        scheme="exact",
    )
    return paths


def run_single_hedge_path(
    path: np.ndarray,
    config: HedgingConfig,
    pricer: FxEuropeanVanillaBsmPricer,
    option: EuropeanFxVanillaOption,
) -> HedgingResult:
    """
    Run hedging simulation on a single path.
    
    Parameters
    ----------
    path : ndarray
        Spot price path of shape (n_steps + 1,).
    config : HedgingConfig
        Hedging configuration.
    pricer : FxEuropeanVanillaBsmPricer
        Library pricer.
    option : EuropeanFxVanillaOption
        Option being hedged.
    
    Returns
    -------
    HedgingResult
        Hedging simulation result.
    """
    dt = config.expiry / config.n_steps
    n_steps = len(path) - 1
    
    states: List[HedgeState] = []
    hedge_position = 0.0
    cash = 0.0
    cumulative_costs = 0.0
    n_rebalances = 0
    
    for i in range(n_steps + 1):
        t = i * dt
        time_remaining = config.expiry - t
        spot = path[i]
        
        # Get Greeks from library model
        if time_remaining > 1e-6:
            price, delta, gamma, theta, vega = compute_greeks_from_model(
                spot=spot,
                strike=config.strike,
                expiry=time_remaining,
                r_dom=config.r_dom,
                r_for=config.r_for,
                vol=config.vol,
            )
        else:
            # At expiry
            price = max(spot - config.strike, 0)
            delta = 1.0 if spot > config.strike else 0.0
            gamma = 0.0
        
        option_value = price * config.notional
        
        # Rebalance hedge
        should_rebalance = (i % config.rebalance_frequency == 0) and (i < n_steps)
        
        if should_rebalance:
            target_hedge = delta * config.notional
            trade_size = target_hedge - hedge_position
            
            # Transaction cost
            trade_cost = abs(trade_size) * spot * config.proportional_cost
            cumulative_costs += trade_cost
            
            # Execute trade
            cash -= trade_size * spot + trade_cost
            hedge_position = target_hedge
            n_rebalances += 1
        
        # Portfolio value: short option + hedge position + cash
        portfolio_value = -option_value + hedge_position * spot + cash
        
        states.append(HedgeState(
            time=t,
            spot=spot,
            option_value=option_value,
            delta=delta,
            gamma=gamma,
            hedge_position=hedge_position,
            cash=cash,
            portfolio_value=portfolio_value,
            cumulative_costs=cumulative_costs,
        ))
    
    # Final P&L
    final_pnl = states[-1].portfolio_value - states[0].portfolio_value
    
    return HedgingResult(
        states=states,
        final_pnl=final_pnl,
        total_costs=cumulative_costs,
        n_rebalances=n_rebalances,
    )


def run_hedging_simulation(config: HedgingConfig) -> Tuple[List[HedgingResult], np.ndarray]:
    """
    Run full hedging simulation across multiple paths.
    
    Returns
    -------
    Tuple
        (list of HedgingResult, array of final P&Ls)
    """
    # Create pricer and option
    pricer = FxEuropeanVanillaBsmPricer()
    option = create_option(config)
    
    # Simulate paths using library GBM
    drift = config.r_dom - config.r_for  # FX risk-neutral drift
    paths = simulate_gbm_paths(
        S0=config.spot,
        drift=drift,
        vol=config.vol,
        T=config.expiry,
        n_steps=config.n_steps,
        n_paths=config.n_paths,
        seed=42,
    )
    
    # Run hedging on each path
    results = []
    final_pnls = []
    
    for i in range(config.n_paths):
        result = run_single_hedge_path(paths[i], config, pricer, option)
        results.append(result)
        final_pnls.append(result.final_pnl)
    
    return results, np.array(final_pnls)


def compare_rebalancing_frequencies(
    base_config: HedgingConfig,
    frequencies: List[int],
) -> dict:
    """
    Compare hedging effectiveness across different rebalancing frequencies.
    
    Returns
    -------
    dict
        Results by frequency.
    """
    comparison = {}
    
    for freq in frequencies:
        config = HedgingConfig(
            spot=base_config.spot,
            strike=base_config.strike,
            expiry=base_config.expiry,
            vol=base_config.vol,
            r_dom=base_config.r_dom,
            r_for=base_config.r_for,
            notional=base_config.notional,
            n_steps=base_config.n_steps,
            n_paths=base_config.n_paths,
            proportional_cost=base_config.proportional_cost,
            rebalance_frequency=freq,
        )
        
        results, pnls = run_hedging_simulation(config)
        
        comparison[freq] = {
            'pnl_mean': float(np.mean(pnls)),
            'pnl_std': float(np.std(pnls)),
            'total_cost': float(np.mean([r.total_costs for r in results])),
            'n_rebalances': results[0].n_rebalances,
        }
    
    return comparison


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_delta_hedging() -> Tuple[List[HedgingResult], np.ndarray, dict]:
    """
    Run the complete delta hedging workflow.
    
    Returns
    -------
    Tuple
        (results, pnls, frequency comparison)
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Configuration")
    logger.info("=" * 70)
    
    config = HedgingConfig(
        spot=1.0850,
        strike=1.0850,
        expiry=0.25,  # 3 months
        vol=0.10,
        r_dom=0.05,
        r_for=0.04,
        notional=10_000_000,
        n_steps=63,  # Daily
        n_paths=1000,
        proportional_cost=0.0001,  # 1 bp
        rebalance_frequency=1,  # Daily
    )
    
    logger.info("")
    logger.info(f"  Option: EUR/USD Call")
    logger.info(f"  Spot:     {config.spot}")
    logger.info(f"  Strike:   {config.strike}")
    logger.info(f"  Expiry:   {config.expiry}y ({int(config.expiry * 252)} days)")
    logger.info(f"  Vol:      {config.vol:.1%}")
    logger.info(f"  r_dom:    {config.r_dom:.2%}")
    logger.info(f"  r_for:    {config.r_for:.2%}")
    logger.info(f"  Notional: ${config.notional:,.0f}")
    logger.info(f"  Paths:    {config.n_paths:,}")
    logger.info(f"  Txn cost: {config.proportional_cost:.2%}")
    
    # Initial Greeks
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Initial Greeks (using library BSM)")
    logger.info("=" * 70)
    
    price, delta, gamma, theta, vega = compute_greeks_from_model(
        spot=config.spot,
        strike=config.strike,
        expiry=config.expiry,
        r_dom=config.r_dom,
        r_for=config.r_for,
        vol=config.vol,
    )
    
    logger.info("")
    logger.info(f"  Price (per unit):  {price:.6f}")
    logger.info(f"  Delta:             {delta:.4f}")
    logger.info(f"  Gamma:             {gamma:.4f}")
    logger.info(f"  Theta (per year):  {theta:.6f}")
    logger.info(f"  Vega (per 1 vol):  {vega:.6f}")
    logger.info(f"  Option PV:         ${price * config.notional:,.0f}")
    
    # Run simulation
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Hedging Simulation")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("Running hedging simulation...")
    results, pnls = run_hedging_simulation(config)
    
    logger.info("")
    logger.info(f"  Daily Rebalancing Results:")
    logger.info(f"    Mean P&L:     ${np.mean(pnls):>12,.0f}")
    logger.info(f"    Std P&L:      ${np.std(pnls):>12,.0f}")
    logger.info(f"    Min P&L:      ${np.min(pnls):>12,.0f}")
    logger.info(f"    Max P&L:      ${np.max(pnls):>12,.0f}")
    logger.info(f"    Avg Cost:     ${np.mean([r.total_costs for r in results]):>12,.0f}")
    logger.info(f"    Rebalances:   {results[0].n_rebalances:>12}")
    
    # Compare frequencies
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Rebalancing Frequency Comparison")
    logger.info("=" * 70)
    
    frequencies = [1, 2, 5, 10, 21]  # Daily, every 2d, weekly, bi-weekly, monthly
    comparison = compare_rebalancing_frequencies(config, frequencies)
    
    logger.info("")
    logger.info(f"{'Frequency':<12} {'Mean P&L':>12} {'Std P&L':>12} {'Avg Cost':>12} {'Rebalances':>12}")
    logger.info("-" * 64)
    
    for freq, data in comparison.items():
        freq_label = "Daily" if freq == 1 else f"Every {freq}d"
        logger.info(
            f"{freq_label:<12} ${data['pnl_mean']:>11,.0f} ${data['pnl_std']:>11,.0f} "
            f"${data['total_cost']:>11,.0f} {data['n_rebalances']:>12}"
        )
    
    logger.info("-" * 64)
    
    return results, pnls, comparison


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_hedging(
    results: List[HedgingResult],
    pnls: np.ndarray,
    comparison: dict,
) -> None:
    """Visualize hedging results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Sample hedge path
    ax = axes[0, 0]
    sample_result = results[0]
    times = [s.time for s in sample_result.states]
    spots = [s.spot for s in sample_result.states]
    deltas = [s.delta for s in sample_result.states]
    
    ax2 = ax.twinx()
    ax.plot(times, spots, color='#2E86AB', linewidth=2, label='Spot')
    ax2.plot(times, deltas, color='#E94F37', linewidth=2, linestyle='--', label='Delta')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Spot', color='#2E86AB')
    ax2.set_ylabel('Delta', color='#E94F37')
    ax.set_title('Sample Hedging Path')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: P&L distribution
    ax = axes[0, 1]
    ax.hist(pnls, bins=50, color='#2E86AB', alpha=0.7, density=True)
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.axvline(np.mean(pnls), color='#E94F37', linestyle='-', linewidth=2, label=f'Mean: ${np.mean(pnls):,.0f}')
    ax.set_xlabel('Final P&L ($)')
    ax.set_ylabel('Density')
    ax.set_title('Hedging P&L Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Portfolio value evolution
    ax = axes[1, 0]
    for i in range(min(50, len(results))):
        times = [s.time for s in results[i].states]
        pv = [s.portfolio_value for s in results[i].states]
        ax.plot(times, pv, alpha=0.3, color='#2E86AB', linewidth=0.5)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Portfolio Value Paths')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cost vs Risk Trade-off
    ax = axes[1, 1]
    freqs = list(comparison.keys())
    costs = [comparison[f]['total_cost'] for f in freqs]
    stds = [comparison[f]['pnl_std'] for f in freqs]
    
    ax.scatter(costs, stds, s=100, c='#2E86AB', zorder=5)
    for f, c, s in zip(freqs, costs, stds):
        label = "Daily" if f == 1 else f"{f}d"
        ax.annotate(label, (c, s), textcoords="offset points", xytext=(5, 5))
    
    ax.set_xlabel('Average Transaction Cost ($)')
    ax.set_ylabel('P&L Standard Deviation ($)')
    ax.set_title('Cost-Risk Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    logger.info("Visualization complete")


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
    │  1. Library Integration:                                            │
    │     - FxEuropeanVanillaBsmPricer for market-aware pricing           │
    │     - BSM functions for direct Greeks computation                   │
    │     - GbmDynamicsSimulator for path simulation                      │
    │                                                                      │
    │  2. Delta Hedging Mechanics:                                        │
    │     - Hedge position = Δ × notional                                 │
    │     - Rebalance when delta changes                                  │
    │     - Transaction costs erode profits                               │
    │                                                                      │
    │  3. Cost-Risk Trade-off:                                            │
    │     - More frequent rebalancing → lower P&L variance                │
    │     - More frequent rebalancing → higher transaction costs          │
    │     - Optimal frequency depends on gamma and cost structure         │
    │                                                                      │
    │  4. Production Considerations:                                      │
    │     - Use library pricers for consistency                           │
    │     - Monitor hedge effectiveness daily                             │
    │     - Consider gamma hedging for large positions                    │
    │     - Account for bid-ask spreads in costs                          │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    global ENABLE_PLOTTING
    ENABLE_PLOTTING = args.plot
    
    try:
        results, pnls, comparison = run_delta_hedging()
        visualize_hedging(results, pnls, comparison)
        print_summary()
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta Hedging Example")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    
    args = parser.parse_args()
    main(args)
