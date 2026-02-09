#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: risk.compute_sensitivities
===============================================================================

This script demonstrates how to use the `risk.compute_sensitivities` pipeline
to compute portfolio Greeks (delta, gamma, vega, theta, rho) with aggregation.

Learning Objectives
-------------------
1. **Pipeline Execution**: Use the orchestrator to run risk workflows
2. **Greeks Computation**: Understand bump-and-reprice methodology
3. **Aggregation**: Aggregate Greeks by underlying, currency, desk
4. **State Management**: Provide and extract data via pipeline state

What This Pipeline Does
-----------------------
1. Loads portfolio and market snapshot from state
2. Configures which Greeks to compute
3. Computes Greeks per position via bump-and-reprice
4. Aggregates Greeks by underlying, currency, desk
5. Optionally computes cross-gamma matrix
6. Writes a comprehensive sensitivity report

Mathematical Framework
----------------------
The Greeks measure option sensitivity to market inputs:

    Delta (Δ) = ∂V/∂S        First-order sensitivity to spot
    Gamma (Γ) = ∂²V/∂S²      Second-order sensitivity (convexity)
    Vega (ν)  = ∂V/∂σ        Sensitivity to implied volatility
    Theta (Θ) = ∂V/∂t        Time decay (value lost per day)
    Rho (ρ)   = ∂V/∂r        Sensitivity to interest rates

Computed via finite difference (bump-and-reprice):
    Δ ≈ [V(S + ε) - V(S - ε)] / (2ε)
    Γ ≈ [V(S + ε) - 2V(S) + V(S - ε)] / ε²

Production Context
------------------
At a hedge fund:
- Greeks are computed intraday for risk monitoring
- Aggregation by desk/trader/strategy for limit monitoring
- P&L attribution uses Greeks to explain daily P&L
- Hedging decisions are driven by delta and vega exposure

Prerequisites
-------------
- Understanding of examples/fundamentals/ and examples/risk/
- Familiarity with orchestrator framework

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/pipelines/run_compute_greeks.py

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
from datetime import date
from pathlib import Path
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# Orchestrator framework
# -----------------------------------------------------------------------------
from src.orchestrator.config.schemas import RunConfig, IOConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.core.state_keys import StateKeys as Keys

# -----------------------------------------------------------------------------
# Market data and instruments
# -----------------------------------------------------------------------------
from src.marketdata.core.market import Market
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.portfolio.core import Portfolio, Position
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption


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
# MARKET AND PORTFOLIO BUILDERS
# =============================================================================

def build_market() -> Market:
    """
    Build a market snapshot with all data needed for pricing and Greeks.
    
    A complete market needs:
    - Spot quotes (underlying prices)
    - Yield curves (discount factors, forward rates)
    - Vol surfaces (implied volatility)
    
    Returns
    -------
    Market
        Market snapshot for pricing.
    """
    return Market(
        asof=date.today(),
        
        # Spot quotes for FX pairs
        quotes={
            MarketId.parse("FX.SPOT.EURUSD"): Quote(value=1.0850),
            MarketId.parse("FX.SPOT.GBPUSD"): Quote(value=1.2650),
        },
        
        # Yield curves (zero rate curves) - use FlatZeroRateCurve
        curves={
            MarketId.parse("IR.ZERO.USD"): FlatZeroRateCurve(continuously_compounded_rate=0.050),
            MarketId.parse("IR.ZERO.EUR"): FlatZeroRateCurve(continuously_compounded_rate=0.040),
            MarketId.parse("IR.ZERO.GBP"): FlatZeroRateCurve(continuously_compounded_rate=0.045),
        },
        
        # Vol surfaces - use FlatVolSurface with sigma parameter
        vols={
            MarketId.parse("FX.VOL.EURUSD"): FlatVolSurface(sigma=0.10),
            MarketId.parse("FX.VOL.GBPUSD"): FlatVolSurface(sigma=0.12),
        },
    )


def build_portfolio() -> Portfolio:
    """
    Build a sample portfolio for Greeks computation.
    
    We create a diverse portfolio with:
    - Different underlyings (EURUSD, GBPUSD)
    - Different option types (calls, puts)
    - Different expiries and strikes
    
    Returns
    -------
    Portfolio
        Sample portfolio with multiple positions.
    """
    positions = []
    
    # -------------------------------------------------------------------------
    # EURUSD Options
    # -------------------------------------------------------------------------
    
    # Long EURUSD call - positive delta, positive gamma, positive vega
    positions.append(Position(
        position_id="EURUSD_CALL_3M",
        instrument=FxVanillaEuropeanOption(
            option_type="call",
            notional=10_000_000,
            strike=1.10,
            expiry=0.25,  # 3 months
            spot_id=MarketId.parse("FX.SPOT.EURUSD"),
            vol_id=MarketId.parse("FX.VOL.EURUSD"),
            domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
            foreign_curve_id=MarketId.parse("IR.ZERO.EUR"),
        ),
        quantity=1,
    ))
    
    # Short EURUSD put - positive delta (short put), negative vega
    positions.append(Position(
        position_id="EURUSD_PUT_3M",
        instrument=FxVanillaEuropeanOption(
            option_type="put",
            notional=10_000_000,
            strike=1.05,
            expiry=0.25,
            spot_id=MarketId.parse("FX.SPOT.EURUSD"),
            vol_id=MarketId.parse("FX.VOL.EURUSD"),
            domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
            foreign_curve_id=MarketId.parse("IR.ZERO.EUR"),
        ),
        quantity=-1,  # Short position
    ))
    
    # Long EURUSD straddle (1Y) - gamma/vega position
    positions.append(Position(
        position_id="EURUSD_STRADDLE_1Y_CALL",
        instrument=FxVanillaEuropeanOption(
            option_type="call",
            notional=5_000_000,
            strike=1.085,  # ATM
            expiry=1.0,
            spot_id=MarketId.parse("FX.SPOT.EURUSD"),
            vol_id=MarketId.parse("FX.VOL.EURUSD"),
            domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
            foreign_curve_id=MarketId.parse("IR.ZERO.EUR"),
        ),
        quantity=1,
    ))
    
    positions.append(Position(
        position_id="EURUSD_STRADDLE_1Y_PUT",
        instrument=FxVanillaEuropeanOption(
            option_type="put",
            notional=5_000_000,
            strike=1.085,
            expiry=1.0,
            spot_id=MarketId.parse("FX.SPOT.EURUSD"),
            vol_id=MarketId.parse("FX.VOL.EURUSD"),
            domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
            foreign_curve_id=MarketId.parse("IR.ZERO.EUR"),
        ),
        quantity=1,
    ))
    
    # -------------------------------------------------------------------------
    # GBPUSD Options (diversification)
    # -------------------------------------------------------------------------
    
    positions.append(Position(
        position_id="GBPUSD_CALL_6M",
        instrument=FxVanillaEuropeanOption(
            option_type="call",
            notional=8_000_000,
            strike=1.30,
            expiry=0.5,
            spot_id=MarketId.parse("FX.SPOT.GBPUSD"),
            vol_id=MarketId.parse("FX.VOL.GBPUSD"),
            domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
            foreign_curve_id=MarketId.parse("IR.ZERO.GBP"),
        ),
        quantity=1,
    ))
    
    return Portfolio(positions=positions)


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config() -> RunConfig:
    """
    Build configuration for Greeks computation.
    
    Returns
    -------
    RunConfig
        Validated configuration for the pipeline.
    """
    config = RunConfig(
        pipeline="risk.compute_sensitivities",
        
        io=IOConfig(workdir="./artifacts/greeks_example"),
        
        params={
            "sensitivities": {
                # Which Greeks to compute
                "greeks": ["delta", "gamma", "vega", "theta", "rho"],
                
                # Bump sizes for finite difference computation
                "bump_sizes": {
                    "spot": 0.01,     # 1% bump for delta/gamma
                    "vol": 0.01,      # 1 vol point for vega
                    "rate": 0.0001,   # 1bp for rho
                },
                
                # Aggregation dimensions
                "aggregation": ["underlying", "currency"],
                
                # Cross-gamma (optional, computationally expensive)
                "cross_gamma": False,
            }
        },
    )
    
    return validate_run_config(config)


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def display_position_greeks(position_greeks: Dict[str, Dict[str, float]]) -> None:
    """Display position-level Greeks."""
    logger.info("")
    logger.info("Position-Level Greeks:")
    logger.info("-" * 80)
    logger.info(f"{'Position ID':<25} {'Delta':>12} {'Gamma':>12} {'Vega':>12} {'Theta':>12}")
    logger.info("-" * 80)
    
    for pos_id, greeks in position_greeks.items():
        delta = greeks.get("delta", 0)
        gamma = greeks.get("gamma", 0)
        vega = greeks.get("vega", 0)
        theta = greeks.get("theta", 0)
        
        logger.info(f"{pos_id:<25} {delta:>12,.0f} {gamma:>12,.0f} {vega:>12,.0f} {theta:>12,.0f}")
    
    logger.info("-" * 80)


def display_totals(totals: Dict[str, float]) -> None:
    """Display portfolio totals."""
    logger.info("")
    logger.info("Portfolio Totals:")
    logger.info("-" * 50)
    logger.info(f"  Delta (Δ):   {totals.get('delta', 0):>15,.0f}")
    logger.info(f"  Gamma (Γ):   {totals.get('gamma', 0):>15,.0f}")
    logger.info(f"  Vega (ν):    {totals.get('vega', 0):>15,.0f}")
    logger.info(f"  Theta (Θ):   {totals.get('theta', 0):>15,.0f}")
    logger.info(f"  Rho (ρ):     {totals.get('rho', 0):>15,.0f}")


def display_interpretation(totals: Dict[str, float]) -> None:
    """Display Greeks interpretation."""
    logger.info("")
    logger.info("Greeks Interpretation:")
    logger.info("-" * 70)
    
    total_delta = totals.get('delta', 0)
    total_gamma = totals.get('gamma', 0)
    total_vega = totals.get('vega', 0)
    total_theta = totals.get('theta', 0)
    
    # Delta interpretation
    if total_delta > 0:
        logger.info(f"  Delta: Portfolio is LONG the underlying")
        logger.info(f"         A 1% spot move generates ~${total_delta * 0.01:,.0f} P&L")
    else:
        logger.info(f"  Delta: Portfolio is SHORT the underlying")
        logger.info(f"         A 1% spot move generates ~${total_delta * 0.01:,.0f} P&L")
    
    # Gamma interpretation
    if total_gamma > 0:
        logger.info(f"  Gamma: Portfolio is LONG gamma (convexity)")
        logger.info(f"         Benefits from large moves in either direction")
    else:
        logger.info(f"  Gamma: Portfolio is SHORT gamma")
        logger.info(f"         Exposed to large moves (may need rebalancing)")
    
    # Vega interpretation
    if total_vega > 0:
        logger.info(f"  Vega:  Portfolio is LONG volatility")
        logger.info(f"         A 1 vol point increase adds ~${total_vega * 0.01:,.0f}")
    else:
        logger.info(f"  Vega:  Portfolio is SHORT volatility")
        logger.info(f"         A 1 vol point increase costs ~${abs(total_vega) * 0.01:,.0f}")
    
    # Theta interpretation
    logger.info(f"  Theta: Portfolio decays ~${abs(total_theta):,.0f}/day due to time")


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
    │  1. Pipeline Execution:                                             │
    │     - RunConfig specifies pipeline + params                         │
    │     - Initial state provides market and portfolio                   │
    │     - Results extracted from context state                          │
    │                                                                      │
    │  2. Greeks Computation:                                             │
    │     - Bump-and-reprice for finite difference                        │
    │     - Configurable bump sizes per risk factor                       │
    │                                                                      │
    │  3. Aggregation:                                                    │
    │     - Position-level and portfolio-level Greeks                     │
    │     - Aggregation by underlying, currency, desk                     │
    │                                                                      │
    │  4. Production Use:                                                 │
    │     - Intraday risk monitoring                                      │
    │     - Limit checking and breach alerts                              │
    │     - P&L attribution and explain                                   │
    │                                                                      │
    │  NEXT: See run_var.py for VaR computation                           │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Execute the Greeks pipeline and display results.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    logger.info("=" * 70)
    logger.info("Pipeline Example: risk.compute_sensitivities")
    logger.info("=" * 70)
    
    try:
        # ---------------------------------------------------------------------
        # Step 1: Build prerequisites
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[1/5] Building market and portfolio...")
        market = build_market()
        portfolio = build_portfolio()
        logger.info(f"      Market as of: {market.asof}")
        logger.info(f"      Portfolio size: {len(portfolio.positions)} positions")
        
        # ---------------------------------------------------------------------
        # Step 2: Build configuration
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[2/5] Building configuration...")
        cfg = build_config()
        logger.info(f"      Pipeline: {cfg.pipeline}")
        logger.info(f"      Greeks: delta, gamma, vega, theta, rho")
        
        # ---------------------------------------------------------------------
        # Step 3: Execute the pipeline with initial state
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[3/5] Executing pipeline...")
        
        initial_state = {
            Keys.MARKET: market,
            Keys.PORTFOLIO: portfolio,
        }
        
        ctx = run_pipeline_from_config(cfg, initial_state=initial_state)
        logger.info("      Pipeline completed successfully!")
        
        # ---------------------------------------------------------------------
        # Step 4: Extract results
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[4/5] Extracting results...")
        
        position_greeks = ctx.state.get(Keys.POSITION_GREEKS, {})
        aggregated_greeks = ctx.state.get(Keys.AGGREGATED_GREEKS, {})
        
        logger.info(f"      Positions with Greeks: {len(position_greeks)}")
        
        # ---------------------------------------------------------------------
        # Step 5: Display Greeks Report
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[5/5] Portfolio Greeks Report")
        logger.info("=" * 80)
        
        display_position_greeks(position_greeks)
        
        totals = aggregated_greeks.get("TOTAL", {})
        display_totals(totals)
        display_interpretation(totals)
        
        logger.info("")
        logger.info(f"Artifacts saved to: {cfg.io.workdir}")
        
        # Summary
        print_summary()
        
        logger.info("Pipeline example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Greeks Computation Pipeline Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    args = parser.parse_args()
    main(args)
