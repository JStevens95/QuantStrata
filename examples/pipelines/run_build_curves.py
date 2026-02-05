#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: marketdata.build_curves
===============================================================================

This script demonstrates how to use the `marketdata.build_curves` pipeline to
bootstrap a yield curve from market rate quotes (deposits and swaps).

Learning Objectives
-------------------
1. **Curve Bootstrapping**: Build zero curves from market instruments
2. **Market Instruments**: Understand deposits, FRAs, swaps as inputs
3. **Interpolation**: Apply log-linear or cubic spline methods
4. **Pipeline Framework**: Use orchestrator for market data workflows

What This Pipeline Does
-----------------------
1. Loads rate quotes from configuration (deposits, FRAs, swaps)
2. Validates quote consistency (no duplicates, reasonable rate bounds)
3. Bootstraps discount factors using iterative solving
4. Applies interpolation (log-linear by default)
5. Stores the resulting ZeroRateCurve for downstream use

Mathematical Framework
----------------------
Deposit rates give us short-end discount factors:
    DF(T) = 1 / (1 + r_deposit × T)

Swap rates imply discount factors via par swap pricing:
    Σ DF(t_i) × c × Δt = DF(t_0) - DF(t_n)
    
where c is the fixed coupon rate and Δt is the payment period.

The bootstrap solves for DF(T) iteratively, then converts to zero rates:
    z(T) = -ln(DF(T)) / T

Production Context
------------------
At a hedge fund:
- Curves are built daily from live market quotes
- Multiple curves per currency (OIS, LIBOR, repo)
- Curve construction is the foundation for all pricing
- Bootstrapping quality affects pricing accuracy

Prerequisites
-------------
- Understanding of examples/fundamentals/02_curves_and_term_structures.py
- Basic knowledge of interest rate products

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/pipelines/run_build_curves.py

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
from datetime import date
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
    Build the pipeline configuration for curve bootstrapping.
    
    This configuration specifies:
    - Which pipeline to run
    - Where to store artifacts (logs, outputs)
    - The curve parameters (currency, quotes, interpolation method)
    
    Returns
    -------
    RunConfig
        Validated configuration object ready for pipeline execution.
    """
    
    # -------------------------------------------------------------------------
    # Define market rate quotes
    # -------------------------------------------------------------------------
    # These represent observable market rates that we'll use to build the curve.
    # In production, these would come from a market data feed (Bloomberg, Reuters).
    
    # Deposit rates: Short-end of the curve (overnight to 6 months)
    # These are simple interest rates for unsecured interbank lending
    deposit_quotes = [
        {"tenor": "1M",  "rate": 0.0525},   # 1-month deposit at 5.25%
        {"tenor": "3M",  "rate": 0.0535},   # 3-month deposit at 5.35%
        {"tenor": "6M",  "rate": 0.0540},   # 6-month deposit at 5.40%
    ]
    
    # Swap rates: Mid to long-end of the curve (1 year to 30 years)
    # These are par swap rates where fixed leg PV = floating leg PV
    swap_quotes = [
        {"tenor": "1Y",  "rate": 0.0520},   # 1-year swap at 5.20%
        {"tenor": "2Y",  "rate": 0.0490},   # 2-year swap at 4.90%
        {"tenor": "3Y",  "rate": 0.0470},   # 3-year swap at 4.70%
        {"tenor": "5Y",  "rate": 0.0450},   # 5-year swap at 4.50%
        {"tenor": "7Y",  "rate": 0.0440},   # 7-year swap at 4.40%
        {"tenor": "10Y", "rate": 0.0435},   # 10-year swap at 4.35%
        {"tenor": "15Y", "rate": 0.0430},   # 15-year swap at 4.30%
        {"tenor": "20Y", "rate": 0.0428},   # 20-year swap at 4.28%
        {"tenor": "30Y", "rate": 0.0425},   # 30-year swap at 4.25%
    ]
    
    # -------------------------------------------------------------------------
    # Build the RunConfig
    # -------------------------------------------------------------------------
    config = RunConfig(
        # Which pipeline to execute (registered in src/orchestrator/runtime/discovery.py)
        pipeline="marketdata.build_curves",
        
        # I/O settings: where to write artifacts (logs, CSVs, JSON reports)
        io=IOConfig(
            workdir="./artifacts/curves_example",
        ),
        
        # Pipeline-specific parameters
        params={
            "curves": {
                # Currency for the curve (used in naming/logging)
                "currency": "USD",
                
                # Type of curve to build
                # Options: "zero" (zero rates), "discount" (discount factors), "forward"
                "curve_type": "zero",
                
                # The market quotes to bootstrap from
                "quotes": {
                    "deposits": deposit_quotes,
                    "swaps": swap_quotes,
                },
                
                # Interpolation method between quote tenors
                # Options: "log_linear", "cubic_spline", "monotone"
                "interpolation": "log_linear",
                
                # Extrapolation beyond the last quote
                # Options: "flat", "linear"
                "extrapolation": "flat",
                
                # Day count convention (affects year fraction calculations)
                "day_count": "ACT/360",
            }
        },
    )
    
    # Validate the configuration (catches errors early)
    return validate_run_config(config)


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def display_curve(term_structure: Any, sample_tenors: List[float]) -> None:
    """
    Display the bootstrapped curve at sample tenors.
    
    Parameters
    ----------
    term_structure : Any
        The bootstrapped curve object.
    sample_tenors : List[float]
        Tenors at which to display rates.
    """
    logger.info("")
    logger.info("Bootstrapped Zero Curve (USD):")
    logger.info("-" * 50)
    logger.info(f"{'Tenor':<10} {'Zero Rate':>12} {'DF':>12}")
    logger.info("-" * 50)
    
    for t in sample_tenors:
        try:
            # Get zero rate at this tenor
            zero_rate = term_structure.zero_rate(t)
            # Calculate discount factor: DF(T) = exp(-r * T)
            df = math.exp(-zero_rate * t)
            
            # Format tenor for display
            if t < 1:
                tenor_str = f"{int(t * 12)}M"
            else:
                tenor_str = f"{int(t)}Y"
            
            logger.info(f"{tenor_str:<10} {zero_rate:>11.4%} {df:>12.6f}")
        except Exception:
            pass
    
    logger.info("-" * 50)


def display_curve_summary(term_structure: Any) -> None:
    """Display curve summary statistics."""
    logger.info("")
    logger.info("Curve Summary:")
    logger.info("-" * 50)
    
    short_rate = term_structure.zero_rate(0.25)
    long_rate = term_structure.zero_rate(30.0)
    
    logger.info(f"  Curve currency:     USD")
    logger.info(f"  Short rate (3M):    {short_rate:.4%}")
    logger.info(f"  Long rate (30Y):    {long_rate:.4%}")
    logger.info(f"  Curve shape:        {'Inverted' if short_rate > long_rate else 'Normal'}")


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
    │  1. Curve Bootstrapping:                                            │
    │     - Build zero curves from market instruments                     │
    │     - Deposits for short-end, swaps for long-end                    │
    │                                                                      │
    │  2. Interpolation:                                                  │
    │     - Log-linear: standard, no arbitrage in DF space                │
    │     - Cubic spline: smooth forwards                                 │
    │     - Monotone: preserves monotonicity                              │
    │                                                                      │
    │  3. Pipeline Structure:                                             │
    │     - RunConfig specifies pipeline + params                         │
    │     - Results stored in context state                               │
    │     - Artifacts saved to workdir                                    │
    │                                                                      │
    │  4. Production Use:                                                 │
    │     - Daily curve builds from live quotes                           │
    │     - Multiple curves per currency (OIS, LIBOR)                     │
    │     - Quality checks and validation                                 │
    │                                                                      │
    │  NEXT: See run_build_vol_surface.py for vol surface calibration     │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Execute the curve building pipeline and display results.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    logger.info("=" * 70)
    logger.info("Pipeline Example: marketdata.build_curves")
    logger.info("=" * 70)
    
    try:
        # ---------------------------------------------------------------------
        # Step 1: Build configuration
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[1/4] Building configuration...")
        cfg = build_config()
        logger.info(f"      Pipeline: {cfg.pipeline}")
        logger.info(f"      Artifacts: {cfg.io.workdir}")
        
        # ---------------------------------------------------------------------
        # Step 2: Execute the pipeline
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[2/4] Executing pipeline...")
        ctx = run_pipeline_from_config(cfg)
        logger.info("      Pipeline completed successfully!")
        
        # ---------------------------------------------------------------------
        # Step 3: Extract results from context
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[3/4] Extracting results...")
        
        term_structure = ctx.state.get(Keys.TERM_STRUCTURE)
        rate_quotes = ctx.state.get(Keys.RATE_QUOTES)
        discount_factors = ctx.state.get(Keys.DISCOUNT_FACTORS)
        
        if term_structure is None:
            logger.warning("No term structure produced. Check configuration.")
            return
        
        logger.info(f"      Curve type: {type(term_structure).__name__}")
        
        n_deposits = len([q for q in rate_quotes if q.instrument_type == 'deposit'])
        n_swaps = len([q for q in rate_quotes if q.instrument_type == 'swap'])
        logger.info(f"      Input quotes: {len(rate_quotes)} ({n_deposits} deposits, {n_swaps} swaps)")
        logger.info(f"      Discount factors computed: {len(discount_factors)} tenors")
        
        # ---------------------------------------------------------------------
        # Step 4: Display the curve
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[4/4] Displaying Results")
        
        sample_tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        display_curve(term_structure, sample_tenors)
        display_curve_summary(term_structure)
        
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
        description="Curve Building Pipeline Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    args = parser.parse_args()
    main(args)
