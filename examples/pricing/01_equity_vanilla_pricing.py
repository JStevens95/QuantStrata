#!/usr/bin/env python3
"""
===============================================================================
Equity European & American Vanilla Options Pricing
===============================================================================

This example prices equity vanilla options using multiple methods:
- Black-Scholes-Merton (closed-form analytic)
- Monte Carlo simulation
- Finite Difference (PDE) method

Learning Objectives
-------------------
1. **Equity Option Modeling**: Understand cost-of-carry with dividend yield
2. **European vs American**: Early exercise premium and optimal stopping
3. **Put-Call Parity**: Verify theoretical relationships
4. **Greeks Analysis**: Compute and interpret risk sensitivities

Mathematical Framework
----------------------
For equity options with continuous dividend yield q:

    C = S·e^(-q·T)·N(d1) - K·e^(-r·T)·N(d2)
    P = K·e^(-r·T)·N(-d2) - S·e^(-q·T)·N(-d1)

Where:
    d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    
Cost-of-carry: b = r - q (vs FX where b = r_d - r_f)

Put-Call Parity:
    C - P = S·e^(-q·T) - K·e^(-r·T)

Production Context
------------------
At a hedge fund:
- Equity options are highly liquid and well-arbitraged
- American options dominate single-stock trading (early exercise)
- Index options are typically European (SPX, DAX)
- Dividend modeling is critical for long-dated options

Prerequisites
-------------
- Examples in fundamentals/
- Understanding of BSM model

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/pricing/01_equity_vanilla_pricing.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import List, Tuple

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
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

# Try to import equity instruments and pricers
try:
    from src.instruments.equity.options.vanilla import (
        EuropeanEquityVanillaOption,
        AmericanEquityVanillaOption,
    )
    from src.pricers.equity.european_bsm import EquityEuropeanVanillaBsmPricer
    EQUITY_AVAILABLE = True
except ImportError:
    EQUITY_AVAILABLE = False

try:
    from src.pricers.equity.european_bsm_mc import EquityEuropeanVanillaMcPricer
except ImportError:
    EquityEuropeanVanillaMcPricer = None  # type: ignore

try:
    from src.pricers.equity.european_bsm_fde import EquityEuropeanVanillaFdPricer
except ImportError:
    EquityEuropeanVanillaFdPricer = None  # type: ignore

try:
    from src.pricers.equity.american_bsm_fde import EquityAmericanVanillaFdPricer
except ImportError:
    EquityAmericanVanillaFdPricer = None  # type: ignore


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

# Market IDs
AAPL_SPOT = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
AAPL_VOL = MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")


# =============================================================================
# SETUP: Market and Instruments
# =============================================================================

def create_market_and_instruments() -> Tuple[Market, dict]:
    """
    Create market snapshot and option parameters.
    
    Returns
    -------
    Tuple[Market, dict]
        Market and parameters dictionary.
    
    Market Parameters
    -----------------
    - Spot: $150 (AAPL)
    - Risk-free rate: 5%
    - Dividend yield: 1%
    - Volatility: 25%
    
    Option Terms
    ------------
    - Strike: $150 (ATM)
    - Expiry: 1 year
    """
    logger.info("=" * 70)
    logger.info("Equity Vanilla Options Pricing Example")
    logger.info("=" * 70)
    
    # Market parameters
    S0 = 150.0          # Spot price
    r = 0.05            # Risk-free rate (5%)
    q = 0.01            # Dividend yield (1%)
    sigma = 0.25        # Volatility (25%)
    K = 150.0           # Strike (ATM)
    T = 1.0             # Time to expiry (1 year)
    
    # Create market snapshot with correct API
    market = Market(
        asof="2026-01-28",
        quotes={AAPL_SPOT: Quote(value=S0)},
        curves={USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r)},
        vols={AAPL_VOL: FlatVolSurface(sigma=sigma)},
    )
    
    params = {
        "S0": S0, "r": r, "q": q, "sigma": sigma, "K": K, "T": T,
    }
    
    logger.info("")
    logger.info("Market Setup:")
    logger.info(f"  Spot (S):           ${S0:.2f}")
    logger.info(f"  Strike (K):         ${K:.2f}")
    logger.info(f"  Expiry (T):         {T:.2f} years")
    logger.info(f"  Risk-free rate (r): {r*100:.1f}%")
    logger.info(f"  Dividend yield (q): {q*100:.1f}%")
    logger.info(f"  Volatility (σ):     {sigma*100:.1f}%")
    logger.info(f"  Cost-of-carry (b):  {(r-q)*100:.1f}% (r - q)")
    
    return market, params


# =============================================================================
# SECTION 1: European Vanilla - Method Comparison
# =============================================================================

def run_european_pricing(market: Market, params: dict) -> dict:
    """
    Price European options with BSM, MC, and FD methods.
    
    Returns
    -------
    dict
        Pricing results including PVs and Greeks.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 1: European Vanilla Pricing - Method Comparison")
    logger.info("=" * 70)
    
    if not EQUITY_AVAILABLE:
        logger.warning("Equity module not available - skipping")
        return {}
    
    K, T, q = params["K"], params["T"], params["q"]
    
    # -------------------------------------------------------------------------
    # Create European options
    # -------------------------------------------------------------------------
    eu_call = EuropeanEquityVanillaOption(
        ticker="AAPL",
        option_type="call",
        strike=K,
        expiry=T,
        notional=1,
        dividend_yield=q,
        spot_id=AAPL_SPOT,
        vol_id=AAPL_VOL,
        curve_id=USD_CURVE,
    )
    
    eu_put = EuropeanEquityVanillaOption(
        ticker="AAPL",
        option_type="put",
        strike=K,
        expiry=T,
        notional=1,
        dividend_yield=q,
        spot_id=AAPL_SPOT,
        vol_id=AAPL_VOL,
        curve_id=USD_CURVE,
    )
    
    # -------------------------------------------------------------------------
    # Create pricers
    # -------------------------------------------------------------------------
    bsm_pricer = EquityEuropeanVanillaBsmPricer()
    
    mc_pricer = None
    if EquityEuropeanVanillaMcPricer is not None:
        mc_pricer = EquityEuropeanVanillaMcPricer(n_paths=500_000, seed=42, antithetic=True)
    
    fd_pricer = None
    if EquityEuropeanVanillaFdPricer is not None:
        fd_pricer = EquityEuropeanVanillaFdPricer(n_space=401, n_time_steps=200)
    
    # -------------------------------------------------------------------------
    # Price with all methods
    # -------------------------------------------------------------------------
    bsm_call = bsm_pricer.price(eu_call, market)
    bsm_put = bsm_pricer.price(eu_put, market)
    
    mc_call = mc_pricer.price(eu_call, market) if mc_pricer else None
    mc_put = mc_pricer.price(eu_put, market) if mc_pricer else None
    
    fd_call = fd_pricer.price(eu_call, market) if fd_pricer else None
    fd_put = fd_pricer.price(eu_put, market) if fd_pricer else None
    
    # -------------------------------------------------------------------------
    # Display results
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("European Option Prices:")
    logger.info(f"{'Method':<15} {'Call':<12} {'Put':<12}")
    logger.info("-" * 39)
    logger.info(f"{'BSM (Analytic)':<15} ${bsm_call:<11.4f} ${bsm_put:<11.4f}")
    
    if mc_call is not None:
        logger.info(f"{'Monte Carlo':<15} ${mc_call:<11.4f} ${mc_put:<11.4f}")
        logger.info("")
        logger.info("Convergence to BSM:")
        logger.info(f"  MC Call Error:  {abs(mc_call - bsm_call):.6f} ({100*abs(mc_call - bsm_call)/bsm_call:.3f}%)")
        logger.info(f"  MC Put Error:   {abs(mc_put - bsm_put):.6f} ({100*abs(mc_put - bsm_put)/bsm_put:.3f}%)")
    
    if fd_call is not None:
        logger.info(f"{'Finite Diff':<15} ${fd_call:<11.4f} ${fd_put:<11.4f}")
        logger.info(f"  FD Call Error:  {abs(fd_call - bsm_call):.6f} ({100*abs(fd_call - bsm_call)/bsm_call:.3f}%)")
        logger.info(f"  FD Put Error:   {abs(fd_put - bsm_put):.6f} ({100*abs(fd_put - bsm_put)/bsm_put:.3f}%)")
    
    return {
        "eu_call": eu_call, "eu_put": eu_put,
        "bsm_call": bsm_call, "bsm_put": bsm_put,
        "bsm_pricer": bsm_pricer,
    }


# =============================================================================
# SECTION 2: Put-Call Parity Verification
# =============================================================================

def verify_put_call_parity(params: dict, bsm_call: float, bsm_put: float) -> None:
    """
    Verify put-call parity relationship.
    
    Put-Call Parity for equity with dividends:
        C - P = S·e^(-q·T) - K·e^(-r·T)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Put-Call Parity Verification")
    logger.info("=" * 70)
    
    S0, K, r, q, T = params["S0"], params["K"], params["r"], params["q"], params["T"]
    
    # Calculate both sides
    parity_lhs = bsm_call - bsm_put
    parity_rhs = S0 * math.exp(-q * T) - K * math.exp(-r * T)
    
    logger.info("")
    logger.info("Put-Call Parity: C - P = S·e^(-q·T) - K·e^(-r·T)")
    logger.info(f"  LHS (C - P):                   ${parity_lhs:.6f}")
    logger.info(f"  RHS (S·e^(-qT) - K·e^(-rT)):   ${parity_rhs:.6f}")
    logger.info(f"  Difference:                     ${abs(parity_lhs - parity_rhs):.10f}")
    
    if abs(parity_lhs - parity_rhs) < 1e-8:
        logger.info("  ✓ Parity holds!")
    else:
        logger.warning("  ✗ Parity violated!")


# =============================================================================
# SECTION 3: Greeks Analysis
# =============================================================================

def run_greeks_analysis(
    market: Market,
    eu_call: "EuropeanEquityVanillaOption",
    eu_put: "EuropeanEquityVanillaOption",
    bsm_pricer: "EquityEuropeanVanillaBsmPricer",
    params: dict,
) -> dict:
    """
    Compute and analyze Greeks.
    
    Returns
    -------
    dict
        Greeks for call and put.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Greeks Analysis")
    logger.info("=" * 70)
    
    greeks_call = bsm_pricer.greeks(eu_call, market)
    greeks_put = bsm_pricer.greeks(eu_put, market)
    
    logger.info("")
    logger.info("BSM Greeks (per share):")
    logger.info(f"{'Greek':<10} {'Call':<12} {'Put':<12} {'Relationship'}")
    logger.info("-" * 50)
    logger.info(f"{'Delta':<10} {greeks_call['delta']:<12.4f} {greeks_put['delta']:<12.4f} Δ_put = Δ_call - exp(-qT)")
    logger.info(f"{'Gamma':<10} {greeks_call['gamma']:<12.4f} {greeks_put['gamma']:<12.4f} Same for call/put")
    logger.info(f"{'Vega':<10} {greeks_call['vega']:<12.4f} {greeks_put['vega']:<12.4f} Same for call/put")
    logger.info(f"{'Rho':<10} {greeks_call['rho']:<12.4f} {greeks_put['rho']:<12.4f} Opposite signs")
    
    # Verify delta relationship
    q, T = params["q"], params["T"]
    delta_diff = greeks_put['delta'] - (greeks_call['delta'] - math.exp(-q * T))
    logger.info("")
    logger.info(f"Delta Parity Check: Δ_put - (Δ_call - exp(-qT)) = {delta_diff:.8f}")
    
    return {"greeks_call": greeks_call, "greeks_put": greeks_put}


# =============================================================================
# SECTION 4: American vs European Comparison
# =============================================================================

def run_american_comparison(
    market: Market,
    params: dict,
    bsm_call: float,
    bsm_put: float,
) -> Tuple[float, float]:
    """
    Compare American vs European option values.
    
    Key insight: American options can be exercised early, giving them
    additional value (early exercise premium).
    
    Returns
    -------
    Tuple[float, float]
        American call and put prices.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: American vs European - Early Exercise Premium")
    logger.info("=" * 70)
    
    if not EQUITY_AVAILABLE or EquityAmericanVanillaFdPricer is None:
        logger.warning("American pricer not available - skipping")
        return (0.0, 0.0)
    
    K, T, q = params["K"], params["T"], params["q"]
    
    # American pricers
    am_fd_pricer = EquityAmericanVanillaFdPricer(n_space=401, n_time_steps=200)
    
    # Create American options
    am_call = AmericanEquityVanillaOption(
        ticker="AAPL",
        option_type="call",
        strike=K,
        expiry=T,
        notional=1,
        dividend_yield=q,
        spot_id=AAPL_SPOT,
        vol_id=AAPL_VOL,
        curve_id=USD_CURVE,
    )
    
    am_put = AmericanEquityVanillaOption(
        ticker="AAPL",
        option_type="put",
        strike=K,
        expiry=T,
        notional=1,
        dividend_yield=q,
        spot_id=AAPL_SPOT,
        vol_id=AAPL_VOL,
        curve_id=USD_CURVE,
    )
    
    # Price American options
    am_call_pv = am_fd_pricer.price(am_call, market)
    am_put_pv = am_fd_pricer.price(am_put, market)
    
    logger.info("")
    logger.info("American vs European Prices:")
    logger.info(f"{'Option':<15} {'European':<12} {'American':<12} {'Premium':<12}")
    logger.info("-" * 51)
    logger.info(f"{'Call':<15} ${bsm_call:<11.4f} ${am_call_pv:<11.4f} ${am_call_pv - bsm_call:<11.4f}")
    logger.info(f"{'Put':<15} ${bsm_put:<11.4f} ${am_put_pv:<11.4f} ${am_put_pv - bsm_put:<11.4f}")
    
    logger.info("")
    logger.info("Key Insights:")
    logger.info(f"  - American Call Premium: ${am_call_pv - bsm_call:.4f}")
    logger.info(f"    (Small because dividend yield q={q*100:.1f}% is low)")
    logger.info(f"  - American Put Premium:  ${am_put_pv - bsm_put:.4f}")
    logger.info(f"    (Puts always have early exercise value when ITM)")
    
    return (am_call_pv, am_put_pv)


# =============================================================================
# SECTION 5: Dividend Impact Analysis
# =============================================================================

def run_dividend_analysis(
    market: Market,
    params: dict,
    bsm_pricer: "EquityEuropeanVanillaBsmPricer",
) -> List[Tuple[float, float, float, float, float]]:
    """
    Analyze impact of dividend yield on option prices.
    
    Returns
    -------
    List[Tuple]
        List of (q, eu_call, am_call, eu_put, am_put).
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Dividend Impact Analysis")
    logger.info("=" * 70)
    
    if not EQUITY_AVAILABLE:
        return []
    
    K, T = params["K"], params["T"]
    dividend_yields = np.linspace(0, 0.10, 11)
    results: List[Tuple[float, float, float, float, float]] = []
    
    am_fd_pricer = None
    if EquityAmericanVanillaFdPricer is not None:
        am_fd_pricer = EquityAmericanVanillaFdPricer(n_space=401, n_time_steps=200)
    
    for q_val in dividend_yields:
        # Create options with this dividend yield
        eu_c = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=K, expiry=T,
            notional=1, dividend_yield=q_val, spot_id=AAPL_SPOT,
            vol_id=AAPL_VOL, curve_id=USD_CURVE,
        )
        eu_p = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=K, expiry=T,
            notional=1, dividend_yield=q_val, spot_id=AAPL_SPOT,
            vol_id=AAPL_VOL, curve_id=USD_CURVE,
        )
        
        eu_call_pv = bsm_pricer.price(eu_c, market)
        eu_put_pv = bsm_pricer.price(eu_p, market)
        
        am_call_pv, am_put_pv = 0.0, 0.0
        if am_fd_pricer is not None:
            am_c = AmericanEquityVanillaOption(
                ticker="AAPL", option_type="call", strike=K, expiry=T,
                notional=1, dividend_yield=q_val, spot_id=AAPL_SPOT,
                vol_id=AAPL_VOL, curve_id=USD_CURVE,
            )
            am_p = AmericanEquityVanillaOption(
                ticker="AAPL", option_type="put", strike=K, expiry=T,
                notional=1, dividend_yield=q_val, spot_id=AAPL_SPOT,
                vol_id=AAPL_VOL, curve_id=USD_CURVE,
            )
            am_call_pv = am_fd_pricer.price(am_c, market)
            am_put_pv = am_fd_pricer.price(am_p, market)
        
        results.append((q_val, eu_call_pv, am_call_pv, eu_put_pv, am_put_pv))
    
    logger.info("")
    logger.info("Dividend Yield Impact on Option Prices:")
    logger.info(f"{'q':<8} {'EU Call':<10} {'AM Call':<10} {'EU Put':<10} {'AM Put':<10}")
    logger.info("-" * 48)
    
    for i, (q_val, eu_c, am_c, eu_p, am_p) in enumerate(results):
        if i % 2 == 0:  # Show every other
            logger.info(f"{q_val*100:.1f}%    ${eu_c:<9.4f} ${am_c:<9.4f} ${eu_p:<9.4f} ${am_p:<9.4f}")
    
    return results


# =============================================================================
# SECTION 6: Visualization
# =============================================================================

def visualize_results(
    market: Market,
    params: dict,
    pricing_results: dict,
    greeks_results: dict,
    dividend_results: List[Tuple],
) -> None:
    """
    Create comprehensive visualizations.
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    if not EQUITY_AVAILABLE or not pricing_results:
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    S0, K, r, sigma = params["S0"], params["K"], params["r"], params["sigma"]
    eu_call = pricing_results["eu_call"]
    eu_put = pricing_results["eu_put"]
    bsm_pricer = pricing_results["bsm_pricer"]
    
    # -------------------------------------------------------------------------
    # Plot 1: Option prices vs spot
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    spots = np.linspace(100, 200, 50)
    call_prices = []
    put_prices = []
    
    for s in spots:
        mkt = Market(
            asof="2026-01-28",
            quotes={AAPL_SPOT: Quote(value=s)},
            curves={USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r)},
            vols={AAPL_VOL: FlatVolSurface(sigma=sigma)},
        )
        call_prices.append(bsm_pricer.price(eu_call, mkt))
        put_prices.append(bsm_pricer.price(eu_put, mkt))
    
    ax.plot(spots, call_prices, 'b-', label='Call')
    ax.plot(spots, put_prices, 'r-', label='Put')
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5, label=f'Strike = ${K}')
    ax.set_xlabel('Spot Price ($)')
    ax.set_ylabel('Option Price ($)')
    ax.set_title('Option Price vs Spot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: American vs European premium
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    if dividend_results:
        div_yields = [r[0] * 100 for r in dividend_results]
        am_call_premium = [r[2] - r[1] for r in dividend_results]
        am_put_premium = [r[4] - r[3] for r in dividend_results]
        
        ax.plot(div_yields, am_call_premium, 'b-', label='Call Premium')
        ax.plot(div_yields, am_put_premium, 'r-', label='Put Premium')
        ax.set_xlabel('Dividend Yield (%)')
        ax.set_ylabel('Early Exercise Premium ($)')
        ax.set_title('American - European Premium vs Dividend Yield')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Data unavailable", ha='center', va='center', transform=ax.transAxes)
    
    # -------------------------------------------------------------------------
    # Plot 3: Delta vs spot
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    call_deltas = []
    put_deltas = []
    
    for s in spots:
        mkt = Market(
            asof="2026-01-28",
            quotes={AAPL_SPOT: Quote(value=s)},
            curves={USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r)},
            vols={AAPL_VOL: FlatVolSurface(sigma=sigma)},
        )
        call_deltas.append(bsm_pricer.greeks(eu_call, mkt)['delta'])
        put_deltas.append(bsm_pricer.greeks(eu_put, mkt)['delta'])
    
    ax.plot(spots, call_deltas, 'b-', label='Call Delta')
    ax.plot(spots, put_deltas, 'r-', label='Put Delta')
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Spot Price ($)')
    ax.set_ylabel('Delta')
    ax.set_title('Delta vs Spot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Gamma vs spot
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    call_gammas = []
    
    for s in spots:
        mkt = Market(
            asof="2026-01-28",
            quotes={AAPL_SPOT: Quote(value=s)},
            curves={USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r)},
            vols={AAPL_VOL: FlatVolSurface(sigma=sigma)},
        )
        call_gammas.append(bsm_pricer.greeks(eu_call, mkt)['gamma'])
    
    ax.plot(spots, call_gammas, 'g-', label='Gamma')
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spot Price ($)')
    ax.set_ylabel('Gamma')
    ax.set_title('Gamma vs Spot (Peak at ATM)')
    ax.legend()
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
    │  1. Equity Option Pricing:                                          │
    │     - Cost-of-carry: b = r - q (dividend reduces forward)           │
    │     - Put-Call Parity: C - P = S·e^(-qT) - K·e^(-rT)                │
    │                                                                      │
    │  2. American vs European:                                           │
    │     - American puts always have early exercise premium              │
    │     - American calls may have premium if dividends are high         │
    │     - FD (Bermudan/American) required for early exercise            │
    │                                                                      │
    │  3. Greeks:                                                         │
    │     - Delta: Call positive, Put negative                            │
    │     - Gamma: Same for call/put, peaks at ATM                        │
    │     - Vega: Same for call/put, higher for longer expiries           │
    │                                                                      │
    │  4. Dividend Impact:                                                │
    │     - Higher dividend → lower call value, higher put value          │
    │     - Higher dividend → larger American call premium                 │
    │                                                                      │
    │  NEXT: See 02_exotic_options.py for path-dependent options          │
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
        # Setup
        market, params = create_market_and_instruments()
        
        # Section 1: European pricing
        pricing_results = run_european_pricing(market, params)
        
        if pricing_results:
            # Section 2: Put-call parity
            verify_put_call_parity(params, pricing_results["bsm_call"], pricing_results["bsm_put"])
            
            # Section 3: Greeks
            greeks_results = run_greeks_analysis(
                market,
                pricing_results["eu_call"],
                pricing_results["eu_put"],
                pricing_results["bsm_pricer"],
                params,
            )
            
            # Section 4: American comparison
            am_call, am_put = run_american_comparison(
                market, params, pricing_results["bsm_call"], pricing_results["bsm_put"]
            )
            
            # Section 5: Dividend analysis
            dividend_results = run_dividend_analysis(market, params, pricing_results["bsm_pricer"])
            
            # Section 6: Visualization
            visualize_results(market, params, pricing_results, greeks_results, dividend_results)
        else:
            greeks_results = {}
            dividend_results = []
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Equity Vanilla Options Pricing Example",
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
