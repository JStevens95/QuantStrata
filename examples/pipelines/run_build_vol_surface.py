#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: marketdata.build_vol_surface
===============================================================================

This script demonstrates how to use the `marketdata.build_vol_surface` pipeline
to construct an implied volatility surface from option quotes.

Learning Objectives
-------------------
1. **Vol Surface Construction**: Build surfaces from delta-quoted vols
2. **Delta Convention**: Understand FX market quoting conventions
3. **Arbitrage Checking**: Validate calendar and butterfly constraints
4. **Pipeline Integration**: Use orchestrator for market data workflows

What This Pipeline Does
-----------------------
1. Loads vol quotes (delta or strike convention)
2. Converts quotes to standard strike/expiry format
3. Builds a raw volatility surface grid
4. Validates arbitrage constraints (calendar spread, butterfly)
5. Applies interpolation to fill gaps
6. Stores the resulting VolSurface for pricing

Mathematical Framework
----------------------
Delta-to-Strike Conversion:
    For FX options with Garman-Kohlhagen:
    Δ_call = e^(-r_f·T) · N(d1)
    Δ_put = -e^(-r_f·T) · N(-d1)
    
    where d1 = [ln(S/K) + (r_d - r_f + σ²/2)T] / (σ√T)

Arbitrage Constraints:
    - Calendar: σ(T₁)√T₁ ≤ σ(T₂)√T₂ for T₁ < T₂ (no calendar arb)
    - Butterfly: C(K-ΔK) + C(K+ΔK) ≥ 2C(K) (convexity in strikes)

Smile Parameterizations:
    - Raw quotes: Direct interpolation on grid
    - SABR: α, β, ρ, ν parameterization
    - SVI: a, b, ρ, m, σ (Stochastic Volatility Inspired)

Production Context
------------------
At a hedge fund:
- Vol surfaces are built from broker quotes or exchange data
- FX uses delta convention (25Δ put, ATM, 25Δ call)
- Equity uses strike/moneyness convention
- Surfaces are validated for arbitrage before use

Prerequisites
-------------
- Understanding of examples/fundamentals/03_volatility_surface.py
- Basic knowledge of options and vol conventions

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/pipelines/run_build_vol_surface.py

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
from pathlib import Path
from typing import List, Dict, Any

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

def build_config() -> RunConfig:
    """
    Build configuration for vol surface construction.
    
    We'll create an FX vol surface for EURUSD using delta-quoted vols,
    which is the standard convention in the FX market.
    
    Returns
    -------
    RunConfig
        Validated configuration for the pipeline.
    """
    
    # -------------------------------------------------------------------------
    # Define volatility quotes (FX delta convention)
    # -------------------------------------------------------------------------
    # FX vols are typically quoted by delta: 25D put, ATM (50D), 25D call
    # Delta = N(d1) for calls, so 0.25 delta call is OTM, 0.75 delta call is ITM
    
    vol_quotes = [
        # 1-week expiry - short-dated, usually higher vol
        {"expiry": "1W", "delta": 0.25, "vol": 0.088},   # 25-delta put
        {"expiry": "1W", "delta": 0.50, "vol": 0.082},   # ATM (at-the-money)
        {"expiry": "1W", "delta": 0.75, "vol": 0.086},   # 25-delta call
        
        # 1-month expiry
        {"expiry": "1M", "delta": 0.25, "vol": 0.092},
        {"expiry": "1M", "delta": 0.50, "vol": 0.085},
        {"expiry": "1M", "delta": 0.75, "vol": 0.090},
        
        # 3-month expiry
        {"expiry": "3M", "delta": 0.25, "vol": 0.098},
        {"expiry": "3M", "delta": 0.50, "vol": 0.090},
        {"expiry": "3M", "delta": 0.75, "vol": 0.095},
        
        # 6-month expiry
        {"expiry": "6M", "delta": 0.25, "vol": 0.102},
        {"expiry": "6M", "delta": 0.50, "vol": 0.094},
        {"expiry": "6M", "delta": 0.75, "vol": 0.099},
        
        # 1-year expiry - longer-dated, typically higher vol
        {"expiry": "1Y", "delta": 0.25, "vol": 0.108},
        {"expiry": "1Y", "delta": 0.50, "vol": 0.098},
        {"expiry": "1Y", "delta": 0.75, "vol": 0.104},
    ]
    
    # -------------------------------------------------------------------------
    # Build the RunConfig
    # -------------------------------------------------------------------------
    config = RunConfig(
        pipeline="marketdata.build_vol_surface",
        
        io=IOConfig(workdir="./artifacts/vol_surface_example"),
        
        params={
            "vol_surface": {
                # Underlying asset
                "underlying": "EURUSD",
                
                # Current spot rate (needed for delta-to-strike conversion)
                "spot": 1.0850,
                
                # Surface type
                "surface_type": "implied",  # "implied" or "local"
                
                # Quote convention
                "quote_convention": "delta",  # "delta", "strike", or "moneyness"
                
                # The vol quotes
                "quotes": vol_quotes,
                
                # Interest rates for delta conversion (domestic USD, foreign EUR)
                "r_domestic": 0.05,   # USD rate
                "r_foreign": 0.04,    # EUR rate
                
                # Interpolation settings
                "interpolation": {
                    "strike": "linear",          # Interpolation in strike dimension
                    "time": "linear_variance",   # Interpolate total variance, then convert
                },
                
                # Arbitrage validation
                "arbitrage_check": True,
                "arbitrage_tolerance": 0.001,  # Allow 0.1% tolerance
            }
        },
    )
    
    return validate_run_config(config)


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def display_vol_smile(vol_quotes: List[Any]) -> None:
    """
    Display the volatility smile at each expiry.
    
    Parameters
    ----------
    vol_quotes : List[Any]
        List of vol quote objects.
    """
    logger.info("")
    logger.info("Volatility Smile by Expiry:")
    logger.info("-" * 60)
    logger.info(f"{'Expiry':<10} {'25D Put':>12} {'ATM':>12} {'25D Call':>12}")
    logger.info("-" * 60)
    
    # Group quotes by expiry
    expiries: Dict[str, Dict[str, float]] = {}
    for q in vol_quotes:
        exp = q.expiry
        if exp not in expiries:
            expiries[exp] = {}
        delta_key = f"{int(q.delta * 100)}D"
        expiries[exp][delta_key] = q.vol
    
    # Display sorted by expiry
    for exp in sorted(expiries.keys(), key=lambda x: x):
        exp_data = expiries[exp]
        put_vol = exp_data.get("25D", "-")
        atm_vol = exp_data.get("50D", "-")
        call_vol = exp_data.get("75D", "-")
        
        def fmt_vol(v: Any) -> str:
            return f"{v:.2%}" if isinstance(v, (int, float)) else str(v)
        
        logger.info(f"{exp:<10} {fmt_vol(put_vol):>12} {fmt_vol(atm_vol):>12} {fmt_vol(call_vol):>12}")
    
    logger.info("-" * 60)


def display_surface_characteristics(vol_quotes: List[Any]) -> None:
    """
    Display surface characteristics (smile type, term structure).
    
    Parameters
    ----------
    vol_quotes : List[Any]
        List of vol quote objects.
    """
    logger.info("")
    logger.info("Surface Characteristics:")
    logger.info("-" * 60)
    
    if not vol_quotes:
        return
    
    # Check for smile (vol at wings > ATM)
    atm_vols = [q.vol for q in vol_quotes if abs(q.delta - 0.5) < 0.01]
    wing_vols = [q.vol for q in vol_quotes if abs(q.delta - 0.25) < 0.01 or abs(q.delta - 0.75) < 0.01]
    
    if atm_vols and wing_vols:
        avg_atm = sum(atm_vols) / len(atm_vols)
        avg_wing = sum(wing_vols) / len(wing_vols)
        
        smile_type = "Smile" if avg_wing > avg_atm else "Flat/Frown"
        logger.info(f"  Shape:           {smile_type}")
        logger.info(f"  Avg ATM vol:     {avg_atm:.2%}")
        logger.info(f"  Avg Wing vol:    {avg_wing:.2%}")
    
    # Term structure (short vs long expiry)
    short_expiry_vols = [q.vol for q in vol_quotes if hasattr(q, 'expiry_years') and q.expiry_years < 0.25]
    long_expiry_vols = [q.vol for q in vol_quotes if hasattr(q, 'expiry_years') and q.expiry_years >= 0.5]
    
    if short_expiry_vols and long_expiry_vols:
        avg_short = sum(short_expiry_vols) / len(short_expiry_vols)
        avg_long = sum(long_expiry_vols) / len(long_expiry_vols)
        term_structure = "Upward sloping" if avg_long > avg_short else "Downward sloping"
        logger.info(f"  Term structure:  {term_structure}")


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
    │  1. Vol Surface Construction:                                       │
    │     - Build from market quotes (delta or strike convention)         │
    │     - FX uses delta: 25Δ put, ATM, 25Δ call                         │
    │                                                                      │
    │  2. Delta Convention:                                               │
    │     - Δ_call = e^(-r_f·T) · N(d1)                                   │
    │     - 25Δ = OTM wings, 50Δ = ATM                                    │
    │     - Requires spot and rates for strike conversion                 │
    │                                                                      │
    │  3. Arbitrage Checking:                                             │
    │     - Calendar: total variance must increase with time              │
    │     - Butterfly: convexity in strike space                          │
    │                                                                      │
    │  4. Smile Shapes:                                                   │
    │     - FX: symmetric smile (wings higher than ATM)                   │
    │     - Equity: skew (put wings higher than calls)                    │
    │                                                                      │
    │  NEXT: See run_calibrate_sabr.py for model calibration              │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Execute the vol surface pipeline and display results.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    logger.info("=" * 70)
    logger.info("Pipeline Example: marketdata.build_vol_surface")
    logger.info("=" * 70)
    
    try:
        # ---------------------------------------------------------------------
        # Step 1: Build configuration
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[1/4] Building configuration...")
        cfg = build_config()
        logger.info(f"      Pipeline: {cfg.pipeline}")
        logger.info(f"      Underlying: EURUSD")
        
        # ---------------------------------------------------------------------
        # Step 2: Execute the pipeline
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[2/4] Executing pipeline...")
        ctx = run_pipeline_from_config(cfg)
        logger.info("      Pipeline completed successfully!")
        
        # ---------------------------------------------------------------------
        # Step 3: Extract results
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[3/4] Extracting results...")
        
        vol_surface = ctx.state.get(Keys.VOL_SURFACE)
        vol_quotes = ctx.state.get(Keys.VOL_QUOTES, [])
        arbitrage_report = ctx.state.get(Keys.ARBITRAGE_REPORT, {})
        
        if vol_surface is None:
            logger.warning("No vol surface produced. Check configuration.")
            return
        
        logger.info(f"      Surface type: {type(vol_surface).__name__}")
        logger.info(f"      Input quotes: {len(vol_quotes)}")
        
        arb_passed = arbitrage_report.get('passed', True)
        logger.info(f"      Arbitrage check: {'PASSED' if arb_passed else 'FAILED'}")
        
        if not arb_passed:
            logger.info(f"      Calendar violations: {arbitrage_report.get('calendar_violations', 0)}")
            logger.info(f"      Butterfly violations: {arbitrage_report.get('butterfly_violations', 0)}")
        
        # ---------------------------------------------------------------------
        # Step 4: Display the surface
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[4/4] Implied Volatility Surface (EURUSD)")
        logger.info("-" * 60)
        
        display_vol_smile(vol_quotes)
        display_surface_characteristics(vol_quotes)
        
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
        description="Vol Surface Building Pipeline Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    args = parser.parse_args()
    main(args)
