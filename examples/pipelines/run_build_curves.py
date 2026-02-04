#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: marketdata.build_curves
===============================================================================

This script demonstrates how to use the `marketdata.build_curves` pipeline to
bootstrap a yield curve from market rate quotes (deposits and swaps).

What This Pipeline Does
-----------------------
1. Loads rate quotes from configuration (deposits, FRAs, swaps)
2. Validates quote consistency (no duplicates, reasonable rate bounds)
3. Bootstraps discount factors using iterative solving
4. Applies interpolation (log-linear by default)
5. Stores the resulting ZeroRateCurve for downstream use

When to Use This Pipeline
-------------------------
- Building yield curves for pricing interest rate products
- Constructing discount curves for present value calculations
- Creating forward rate curves for FRA/swap pricing
- Any scenario requiring a term structure from market quotes

Prerequisites
-------------
- QuantStrata library installed (pip install -e .)
- Python 3.12+

Run This Example
----------------
    python examples/pipelines/run_build_curves.py

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import sys
from pathlib import Path
from datetime import date

# Ensure the src package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Orchestrator framework - the core pipeline execution engine
from src.orchestrator.config.schemas import RunConfig, IOConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.core.state_keys import StateKeys as Keys


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
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """
    Execute the curve building pipeline and display results.
    
    This function:
    1. Builds the configuration
    2. Runs the pipeline
    3. Extracts and displays the bootstrapped curve
    4. Shows sample zero rates at key tenors
    """
    
    print("=" * 70)
    print("Pipeline Example: marketdata.build_curves")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Build configuration
    # -------------------------------------------------------------------------
    print("[1/4] Building configuration...")
    cfg = build_config()
    print(f"      Pipeline: {cfg.pipeline}")
    print(f"      Artifacts: {cfg.io.workdir}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Execute the pipeline
    # -------------------------------------------------------------------------
    print("[2/4] Executing pipeline...")
    ctx = run_pipeline_from_config(cfg)
    print("      Pipeline completed successfully!")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Extract results from context
    # -------------------------------------------------------------------------
    print("[3/4] Extracting results...")
    
    # The pipeline stores the bootstrapped curve in ctx.state
    term_structure = ctx.state.get(Keys.TERM_STRUCTURE)
    rate_quotes = ctx.state.get(Keys.RATE_QUOTES)
    discount_factors = ctx.state.get(Keys.DISCOUNT_FACTORS)
    
    if term_structure is None:
        print("      WARNING: No term structure produced. Check configuration.")
        return
    
    print(f"      Curve type: {type(term_structure).__name__}")
    print(f"      Input quotes: {len(rate_quotes)} ({len([q for q in rate_quotes if q.instrument_type == 'deposit'])} deposits, {len([q for q in rate_quotes if q.instrument_type == 'swap'])} swaps)")
    print(f"      Discount factors computed: {len(discount_factors)} tenors")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Display the curve
    # -------------------------------------------------------------------------
    print("[4/4] Bootstrapped Zero Curve (USD)")
    print("-" * 50)
    print(f"{'Tenor':<10} {'Zero Rate':>12} {'DF':>12}")
    print("-" * 50)
    
    # Sample tenors to display
    sample_tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    
    for t in sample_tenors:
        try:
            # Get zero rate at this tenor
            zero_rate = term_structure.zero_rate(t)
            # Calculate discount factor: DF(T) = exp(-r * T)
            import math
            df = math.exp(-zero_rate * t)
            
            # Format tenor for display
            if t < 1:
                tenor_str = f"{int(t * 12)}M"
            else:
                tenor_str = f"{int(t)}Y"
            
            print(f"{tenor_str:<10} {zero_rate:>11.4%} {df:>12.6f}")
        except Exception:
            pass
    
    print("-" * 50)
    print()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("Summary")
    print("-" * 50)
    print(f"  Curve currency:     USD")
    print(f"  Short rate (3M):    {term_structure.zero_rate(0.25):.4%}")
    print(f"  Long rate (30Y):    {term_structure.zero_rate(30.0):.4%}")
    print(f"  Curve shape:        {'Inverted' if term_structure.zero_rate(0.25) > term_structure.zero_rate(30.0) else 'Normal'}")
    print()
    print("Artifacts saved to:", cfg.io.workdir)
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
