#!/usr/bin/env python3
"""
===============================================================================
Risk: Model Validation Using QuantStrata Pricers
===============================================================================

This example demonstrates comprehensive model validation techniques using
QuantStrata's production pricers - comparing BSM analytic, Monte Carlo, and
Finite Difference methods.

Learning Objectives
-------------------
1. **Cross-Model Validation**: Compare BSM vs Monte Carlo vs FDE pricers
2. **Convergence Testing**: Verify numerical methods converge properly
3. **Extreme Scenario Stress**: Test models under extreme conditions
4. **Library Integration**: Use production pricers for validation

Mathematical Framework
----------------------
Model validation involves:

1. Consistency: Different methods converge to same value
   |V_BSM - V_MC| < ε for large N simulations
   |V_BSM - V_FDE| < ε for fine grids

2. Convergence: Numerical methods converge at expected rates
   Error_MC ∝ 1/√N    (Monte Carlo)
   Error_FDE ∝ Δx²    (Finite difference, 2nd order)

3. Boundary Behavior: Correct limits
   V(S→∞, Call) → S - K·e^(-rT)
   V(S→0, Call) → 0

Production Context
------------------
At a hedge fund:
- Model validation is required by risk management
- Regulators demand documented model testing
- P&L attribution reveals model weaknesses
- Stress tests expose hidden risks

Prerequisites
-------------
- Basic option pricing
- Numerical methods understanding
- Previous risk examples

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/risk/05_model_validation.py

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
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports - Library pricers and models
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

from src.instruments.equity.options.vanilla import EquityVanillaEuropeanOption

# Library BSM pricer
from src.pricers.equity.european_bsm import EquityVanillaEuropeanOptionBsmPricer

# Library MC pricer
from src.pricers.equity.european_bsm_mc import (
    EquityVanillaEuropeanOptionMcPricer,
    EquityVanillOptionMcSimulation,
)

# Library FDE pricer
from src.pricers.equity.european_bsm_fde import EquityVanillaEuropeanOptionFdPricer

# Direct BSM functions for reference
from src.models.analytic.black_scholes_merton.base import (
    vanilla_price,
    vanilla_delta,
    vanilla_gamma,
    vanilla_vega,
    vanilla_theta,
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
# CONSTANTS
# =============================================================================

SPOT_ID = MarketId(asset_class="EQ", mkt_type="SPOT", name="SPX")
CURVE_ID = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
VOL_ID = MarketId(asset_class="EQ", mkt_type="VOL", name="SPX")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationResult:
    """Result of model validation."""
    model_name: str
    price: float
    delta: float
    gamma: float
    vega: float
    time_ms: float
    error_vs_bsm: Optional[float] = None
    std_error: Optional[float] = None


@dataclass
class ConvergenceResult:
    """Result of convergence testing."""
    parameter_values: List
    prices: List[float]
    errors: List[float]
    times_ms: List[float]


# =============================================================================
# MARKET AND INSTRUMENT SETUP
# =============================================================================

def create_market(
    spot: float,
    rate: float,
    vol: float,
    val_date: date = date.today(),
) -> Market:
    """Create market snapshot with given parameters."""
    return Market(
        val_date=val_date,
        quotes={SPOT_ID: Quote(SPOT_ID, spot)},
        curves={CURVE_ID: FlatZeroRateCurve(CURVE_ID, rate)},
        vol_surfaces={VOL_ID: FlatVolSurface(VOL_ID, vol)},
    )


def create_option(
    strike: float,
    expiry: float,
    option_type: str = "call",
    notional: float = 1.0,
) -> EquityVanillaEuropeanOption:
    """Create equity vanilla option."""
    return EquityVanillaEuropeanOption(
        option_type=option_type,
        spot_id=SPOT_ID,
        curve_id=CURVE_ID,
        vol_id=VOL_ID,
        strike=strike,
        expiry=expiry,
        notional=notional,
        dividend_yield=0.0,
    )


# =============================================================================
# MODEL VALIDATION FUNCTIONS
# =============================================================================

def validate_model_consistency(
    spot: float = 100.0,
    strike: float = 100.0,
    expiry: float = 0.5,
    rate: float = 0.05,
    vol: float = 0.20,
) -> List[ValidationResult]:
    """
    Compare prices across BSM, MC, and FDE pricers.
    
    Returns
    -------
    List[ValidationResult]
        Results from each pricer.
    """
    market = create_market(spot=spot, rate=rate, vol=vol)
    option = create_option(strike=strike, expiry=expiry)
    
    results = []
    
    # 1. BSM Analytic (reference)
    bsm_pricer = EquityVanillaEuropeanOptionBsmPricer()
    
    start = time.perf_counter()
    bsm_price = bsm_pricer.price(option, market)
    bsm_greeks = bsm_pricer.greeks(option, market)
    bsm_time = (time.perf_counter() - start) * 1000
    
    results.append(ValidationResult(
        model_name="BSM (Analytic)",
        price=bsm_price,
        delta=bsm_greeks.get("delta", 0.0),
        gamma=bsm_greeks.get("gamma", 0.0),
        vega=bsm_greeks.get("vega", 0.0),
        time_ms=bsm_time,
        error_vs_bsm=0.0,
    ))
    
    # 2. Monte Carlo
    mc_pricer = EquityVanillaEuropeanOptionMcPricer(
        n_paths=100_000,
        seed=42,
        antithetic=True,
    )
    
    start = time.perf_counter()
    mc_result = mc_pricer.run(option, market)
    mc_time = (time.perf_counter() - start) * 1000
    
    mc_price = mc_result.discounted_payoffs.mean()
    mc_stderr = mc_result.discounted_payoffs.std() / np.sqrt(len(mc_result.discounted_payoffs))
    
    # MC Greeks via bump-and-reprice
    mc_greeks = mc_pricer.greeks(option, market)
    
    results.append(ValidationResult(
        model_name="Monte Carlo",
        price=mc_price,
        delta=mc_greeks.get("delta", 0.0),
        gamma=mc_greeks.get("gamma", 0.0),
        vega=mc_greeks.get("vega", 0.0),
        time_ms=mc_time,
        error_vs_bsm=abs(mc_price - bsm_price),
        std_error=mc_stderr,
    ))
    
    # 3. Finite Difference
    fde_pricer = EquityVanillaEuropeanOptionFdPricer(
        n_space=401,
        n_time_steps=200,
        theta=0.5,  # Crank-Nicolson
    )
    
    start = time.perf_counter()
    fde_price = fde_pricer.price(option, market)
    fde_greeks = fde_pricer.greeks(option, market)
    fde_time = (time.perf_counter() - start) * 1000
    
    results.append(ValidationResult(
        model_name="Finite Difference",
        price=fde_price,
        delta=fde_greeks.get("delta", 0.0),
        gamma=fde_greeks.get("gamma", 0.0),
        vega=fde_greeks.get("vega", 0.0),
        time_ms=fde_time,
        error_vs_bsm=abs(fde_price - bsm_price),
    ))
    
    return results


def test_mc_convergence(
    spot: float = 100.0,
    strike: float = 100.0,
    expiry: float = 0.5,
    rate: float = 0.05,
    vol: float = 0.20,
) -> ConvergenceResult:
    """
    Test Monte Carlo convergence rate.
    
    Expected: Error ∝ 1/√N
    """
    market = create_market(spot=spot, rate=rate, vol=vol)
    option = create_option(strike=strike, expiry=expiry)
    
    # Reference price from BSM
    bsm_pricer = EquityVanillaEuropeanOptionBsmPricer()
    bsm_price = bsm_pricer.price(option, market)
    
    # Test different path counts
    path_counts = [1000, 5000, 10000, 50000, 100000, 500000]
    prices = []
    errors = []
    times_ms = []
    
    for n_paths in path_counts:
        mc_pricer = EquityVanillaEuropeanOptionMcPricer(
            n_paths=n_paths,
            seed=42,
            antithetic=True,
        )
        
        start = time.perf_counter()
        mc_result = mc_pricer.run(option, market)
        elapsed = (time.perf_counter() - start) * 1000
        
        mc_price = mc_result.discounted_payoffs.mean()
        prices.append(mc_price)
        errors.append(abs(mc_price - bsm_price))
        times_ms.append(elapsed)
    
    return ConvergenceResult(
        parameter_values=path_counts,
        prices=prices,
        errors=errors,
        times_ms=times_ms,
    )


def test_fde_convergence(
    spot: float = 100.0,
    strike: float = 100.0,
    expiry: float = 0.5,
    rate: float = 0.05,
    vol: float = 0.20,
) -> ConvergenceResult:
    """
    Test Finite Difference convergence rate.
    
    Expected: Error ∝ Δx² for second-order schemes
    """
    market = create_market(spot=spot, rate=rate, vol=vol)
    option = create_option(strike=strike, expiry=expiry)
    
    # Reference price from BSM
    bsm_pricer = EquityVanillaEuropeanOptionBsmPricer()
    bsm_price = bsm_pricer.price(option, market)
    
    # Test different grid sizes
    grid_sizes = [51, 101, 201, 401, 801]
    prices = []
    errors = []
    times_ms = []
    
    for n_space in grid_sizes:
        fde_pricer = EquityVanillaEuropeanOptionFdPricer(
            n_space=n_space,
            n_time_steps=n_space // 2,
            theta=0.5,
        )
        
        start = time.perf_counter()
        fde_price = fde_pricer.price(option, market)
        elapsed = (time.perf_counter() - start) * 1000
        
        prices.append(fde_price)
        errors.append(abs(fde_price - bsm_price))
        times_ms.append(elapsed)
    
    return ConvergenceResult(
        parameter_values=grid_sizes,
        prices=prices,
        errors=errors,
        times_ms=times_ms,
    )


def test_extreme_scenarios() -> Dict[str, List[ValidationResult]]:
    """
    Test model behavior under extreme market conditions.
    
    Tests:
    - Deep ITM / OTM
    - Near expiry
    - High / low volatility
    - Zero rates
    """
    scenarios = {}
    
    # Deep ITM (spot = 150, strike = 100)
    scenarios["Deep ITM"] = validate_model_consistency(
        spot=150.0, strike=100.0, expiry=0.5, rate=0.05, vol=0.20
    )
    
    # Deep OTM (spot = 50, strike = 100)
    scenarios["Deep OTM"] = validate_model_consistency(
        spot=50.0, strike=100.0, expiry=0.5, rate=0.05, vol=0.20
    )
    
    # Near expiry (1 week)
    scenarios["Near Expiry"] = validate_model_consistency(
        spot=100.0, strike=100.0, expiry=0.02, rate=0.05, vol=0.20
    )
    
    # High volatility
    scenarios["High Vol"] = validate_model_consistency(
        spot=100.0, strike=100.0, expiry=0.5, rate=0.05, vol=0.80
    )
    
    # Low volatility
    scenarios["Low Vol"] = validate_model_consistency(
        spot=100.0, strike=100.0, expiry=0.5, rate=0.05, vol=0.05
    )
    
    # Zero rate
    scenarios["Zero Rate"] = validate_model_consistency(
        spot=100.0, strike=100.0, expiry=0.5, rate=0.0, vol=0.20
    )
    
    return scenarios


def test_put_call_parity() -> Dict[str, float]:
    """
    Validate put-call parity: C - P = S - K·e^(-rT)
    """
    spot = 100.0
    strike = 100.0
    expiry = 0.5
    rate = 0.05
    vol = 0.20
    
    market = create_market(spot=spot, rate=rate, vol=vol)
    
    call_option = create_option(strike=strike, expiry=expiry, option_type="call")
    put_option = create_option(strike=strike, expiry=expiry, option_type="put")
    
    bsm_pricer = EquityVanillaEuropeanOptionBsmPricer()
    
    call_price = bsm_pricer.price(call_option, market)
    put_price = bsm_pricer.price(put_option, market)
    
    # Put-call parity
    forward_diff = spot - strike * np.exp(-rate * expiry)
    parity_diff = call_price - put_price
    parity_error = abs(parity_diff - forward_diff)
    
    return {
        "call_price": call_price,
        "put_price": put_price,
        "C - P": parity_diff,
        "S - K·e^(-rT)": forward_diff,
        "parity_error": parity_error,
    }


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_model_validation() -> Tuple[List[ValidationResult], Dict]:
    """
    Run the complete model validation workflow.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Cross-Model Consistency")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("Comparing BSM, Monte Carlo, and Finite Difference pricers...")
    
    base_results = validate_model_consistency()
    
    logger.info("")
    logger.info(f"{'Model':<20} {'Price':>12} {'Delta':>10} {'Gamma':>10} {'Error':>12} {'Time':>10}")
    logger.info("-" * 80)
    
    for r in base_results:
        error_str = f"{r.error_vs_bsm:.6f}" if r.error_vs_bsm is not None else "N/A"
        logger.info(
            f"{r.model_name:<20} {r.price:>12.6f} {r.delta:>10.4f} {r.gamma:>10.6f} "
            f"{error_str:>12} {r.time_ms:>9.2f}ms"
        )
    
    # Monte Carlo convergence
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Monte Carlo Convergence")
    logger.info("=" * 70)
    
    mc_conv = test_mc_convergence()
    
    logger.info("")
    logger.info(f"{'Paths':>10} {'Price':>12} {'Error':>12} {'Time':>10}")
    logger.info("-" * 50)
    
    for paths, price, error, t in zip(
        mc_conv.parameter_values, mc_conv.prices, mc_conv.errors, mc_conv.times_ms
    ):
        logger.info(f"{paths:>10,} {price:>12.6f} {error:>12.6f} {t:>9.2f}ms")
    
    # FDE convergence
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Finite Difference Convergence")
    logger.info("=" * 70)
    
    fde_conv = test_fde_convergence()
    
    logger.info("")
    logger.info(f"{'Grid Size':>10} {'Price':>12} {'Error':>12} {'Time':>10}")
    logger.info("-" * 50)
    
    for size, price, error, t in zip(
        fde_conv.parameter_values, fde_conv.prices, fde_conv.errors, fde_conv.times_ms
    ):
        logger.info(f"{size:>10} {price:>12.6f} {error:>12.6f} {t:>9.2f}ms")
    
    # Put-Call Parity
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Put-Call Parity Validation")
    logger.info("=" * 70)
    
    parity = test_put_call_parity()
    
    logger.info("")
    logger.info(f"  Call Price:       {parity['call_price']:.6f}")
    logger.info(f"  Put Price:        {parity['put_price']:.6f}")
    logger.info(f"  C - P:            {parity['C - P']:.6f}")
    logger.info(f"  S - K·e^(-rT):    {parity['S - K·e^(-rT)']:.6f}")
    logger.info(f"  Parity Error:     {parity['parity_error']:.2e}")
    
    # Extreme scenarios
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Extreme Scenario Testing")
    logger.info("=" * 70)
    
    extreme_results = test_extreme_scenarios()
    
    for scenario_name, results in extreme_results.items():
        logger.info("")
        logger.info(f"  {scenario_name}:")
        bsm_result = results[0]
        max_error = max(r.error_vs_bsm for r in results[1:] if r.error_vs_bsm is not None)
        logger.info(f"    BSM Price: {bsm_result.price:.6f}")
        logger.info(f"    Max Error: {max_error:.6f}")
    
    return base_results, {
        "mc_convergence": mc_conv,
        "fde_convergence": fde_conv,
        "parity": parity,
        "extreme_scenarios": extreme_results,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_validation(base_results: List[ValidationResult], analysis: Dict) -> None:
    """Visualize validation results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Model comparison
    ax = axes[0, 0]
    models = [r.model_name for r in base_results]
    prices = [r.price for r in base_results]
    colors = ['#2E86AB', '#E94F37', '#4CAF50']
    
    bars = ax.bar(models, prices, color=colors)
    ax.axhline(prices[0], color='black', linestyle='--', linewidth=1, label='BSM Reference')
    ax.set_ylabel('Price')
    ax.set_title('Model Price Comparison')
    ax.legend()
    
    # Add error labels
    for bar, r in zip(bars[1:], base_results[1:]):
        if r.error_vs_bsm:
            ax.annotate(f'Err: {r.error_vs_bsm:.4f}',
                       (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
    
    # Plot 2: MC Convergence
    ax = axes[0, 1]
    mc_conv = analysis["mc_convergence"]
    ax.loglog(mc_conv.parameter_values, mc_conv.errors, 'o-', color='#2E86AB', 
              linewidth=2, markersize=8, label='MC Error')
    
    # Reference 1/sqrt(N) line
    n = np.array(mc_conv.parameter_values)
    ref_line = mc_conv.errors[0] * np.sqrt(mc_conv.parameter_values[0]) / np.sqrt(n)
    ax.loglog(n, ref_line, '--', color='gray', label=r'$1/\sqrt{N}$ reference')
    
    ax.set_xlabel('Number of Paths')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Monte Carlo Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: FDE Convergence
    ax = axes[1, 0]
    fde_conv = analysis["fde_convergence"]
    ax.loglog(fde_conv.parameter_values, fde_conv.errors, 's-', color='#4CAF50',
              linewidth=2, markersize=8, label='FDE Error')
    
    # Reference 1/N² line
    n = np.array(fde_conv.parameter_values)
    ref_line = fde_conv.errors[0] * (fde_conv.parameter_values[0]**2) / (n**2)
    ax.loglog(n, ref_line, '--', color='gray', label=r'$1/N^2$ reference')
    
    ax.set_xlabel('Grid Points')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Finite Difference Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Execution Time Comparison
    ax = axes[1, 1]
    times = [r.time_ms for r in base_results]
    bars = ax.bar(models, times, color=colors)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Execution Time Comparison')
    ax.set_yscale('log')
    
    for bar, t in zip(bars, times):
        ax.annotate(f'{t:.1f}ms',
                   (bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
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
    │  1. Library Pricers Used:                                           │
    │     - EquityVanillaEuropeanOptionBsmPricer (analytic)               │
    │     - EquityVanillaEuropeanOptionMcPricer (Monte Carlo)             │
    │     - EquityVanillaEuropeanOptionFdPricer (Finite Difference)       │
    │                                                                      │
    │  2. Cross-Model Consistency:                                        │
    │     - All methods should converge to same price                     │
    │     - Differences indicate numerical error or bugs                  │
    │     - Greeks should also be consistent                              │
    │                                                                      │
    │  3. Convergence Properties:                                         │
    │     - MC: Error ∝ 1/√N (slow, but dimension-independent)            │
    │     - FDE: Error ∝ Δx² (fast for low dimensions)                    │
    │     - BSM: Exact (closed-form, fastest)                             │
    │                                                                      │
    │  4. Production Validation:                                          │
    │     - Test extreme scenarios (deep ITM/OTM, near expiry)            │
    │     - Verify put-call parity                                        │
    │     - Check boundary conditions                                     │
    │     - Document convergence rates                                    │
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
        base_results, analysis = run_model_validation()
        visualize_validation(base_results, analysis)
        print_summary()
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Validation Example")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    
    args = parser.parse_args()
    main(args)
