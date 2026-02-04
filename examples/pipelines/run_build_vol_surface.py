#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: marketdata.build_vol_surface
===============================================================================

This script demonstrates how to use the `marketdata.build_vol_surface` pipeline
to construct an implied volatility surface from option quotes.

What This Pipeline Does
-----------------------
1. Loads vol quotes (delta or strike convention)
2. Converts quotes to standard strike/expiry format
3. Builds a raw volatility surface grid
4. Validates arbitrage constraints (calendar spread, butterfly)
5. Applies interpolation to fill gaps
6. Stores the resulting VolSurface for pricing

When to Use This Pipeline
-------------------------
- Building vol surfaces for FX, equity, or commodity options
- Validating market quotes for arbitrage
- Preparing volatility data for exotic option pricing
- Calibrating local or stochastic volatility models

Key Concepts
------------
- **Delta convention**: Quotes given as delta (0.25, 0.50, 0.75) vs strike
- **Smile**: Vol varies with strike at fixed expiry (typically U-shaped)
- **Term structure**: Vol varies with expiry (typically upward sloping)
- **Arbitrage-free**: No calendar or butterfly arbitrage in the surface

Run This Example
----------------
    python examples/pipelines/run_build_vol_surface.py

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.core.state_keys import StateKeys as Keys


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config() -> RunConfig:
    """
    Build configuration for vol surface construction.
    
    We'll create an FX vol surface for EURUSD using delta-quoted vols,
    which is the standard convention in the FX market.
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
        
        io={
            "artifacts_dir": "./artifacts/vol_surface_example",
            "enable_save": True,
        },
        
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
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Execute the vol surface pipeline and display results."""
    
    print("=" * 70)
    print("Pipeline Example: marketdata.build_vol_surface")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Build configuration
    # -------------------------------------------------------------------------
    print("[1/4] Building configuration...")
    cfg = build_config()
    print(f"      Pipeline: {cfg.pipeline}")
    print(f"      Underlying: EURUSD")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Execute the pipeline
    # -------------------------------------------------------------------------
    print("[2/4] Executing pipeline...")
    ctx = run_pipeline_from_config(cfg)
    print("      Pipeline completed successfully!")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Extract results
    # -------------------------------------------------------------------------
    print("[3/4] Extracting results...")
    
    vol_surface = ctx.state.get(Keys.VOL_SURFACE)
    vol_quotes = ctx.state.get(Keys.VOL_QUOTES)
    arbitrage_report = ctx.state.get(Keys.ARBITRAGE_REPORT, {})
    
    if vol_surface is None:
        print("      WARNING: No vol surface produced.")
        return
    
    print(f"      Surface type: {type(vol_surface).__name__}")
    print(f"      Input quotes: {len(vol_quotes)}")
    print(f"      Arbitrage check: {'PASSED' if arbitrage_report.get('passed', True) else 'FAILED'}")
    if not arbitrage_report.get('passed', True):
        print(f"      Calendar violations: {arbitrage_report.get('calendar_violations', 0)}")
        print(f"      Butterfly violations: {arbitrage_report.get('butterfly_violations', 0)}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Display the surface
    # -------------------------------------------------------------------------
    print("[4/4] Implied Volatility Surface (EURUSD)")
    print("-" * 60)
    
    # Display the vol smile at each expiry
    print("\nVolatility Smile by Expiry:")
    print("-" * 60)
    print(f"{'Expiry':<10} {'25D Put':>12} {'ATM':>12} {'25D Call':>12}")
    print("-" * 60)
    
    # Group quotes by expiry
    expiries = {}
    for q in vol_quotes:
        exp = q.expiry
        if exp not in expiries:
            expiries[exp] = {}
        delta_key = f"{int(q.delta * 100)}D"
        expiries[exp][delta_key] = q.vol
    
    # Display sorted by expiry
    for exp in sorted(expiries.keys(), key=lambda x: x):
        exp_data = expiries[exp]
        put_vol = exp_data.get("25D", exp_data.get("25D", "-"))
        atm_vol = exp_data.get("50D", "-")
        call_vol = exp_data.get("75D", "-")
        
        def fmt_vol(v):
            return f"{v:.2%}" if isinstance(v, (int, float)) else v
        
        print(f"{exp:<10} {fmt_vol(put_vol):>12} {fmt_vol(atm_vol):>12} {fmt_vol(call_vol):>12}")
    
    print("-" * 60)
    
    # -------------------------------------------------------------------------
    # Surface characteristics
    # -------------------------------------------------------------------------
    print("\nSurface Characteristics:")
    print("-" * 60)
    
    # Check for smile (vol at wings > ATM)
    if vol_quotes:
        atm_vols = [q.vol for q in vol_quotes if abs(q.delta - 0.5) < 0.01]
        wing_vols = [q.vol for q in vol_quotes if abs(q.delta - 0.25) < 0.01 or abs(q.delta - 0.75) < 0.01]
        
        if atm_vols and wing_vols:
            avg_atm = sum(atm_vols) / len(atm_vols)
            avg_wing = sum(wing_vols) / len(wing_vols)
            
            smile_type = "Smile" if avg_wing > avg_atm else "Flat/Frown"
            print(f"  Shape:           {smile_type}")
            print(f"  Avg ATM vol:     {avg_atm:.2%}")
            print(f"  Avg Wing vol:    {avg_wing:.2%}")
    
    # Term structure (short vs long expiry)
    short_expiry_vols = [q.vol for q in vol_quotes if q.expiry_years < 0.25]
    long_expiry_vols = [q.vol for q in vol_quotes if q.expiry_years >= 0.5]
    
    if short_expiry_vols and long_expiry_vols:
        avg_short = sum(short_expiry_vols) / len(short_expiry_vols)
        avg_long = sum(long_expiry_vols) / len(long_expiry_vols)
        term_structure = "Upward sloping" if avg_long > avg_short else "Downward sloping"
        print(f"  Term structure:  {term_structure}")
    
    print()
    print("Artifacts saved to:", cfg.io.get("artifacts_dir", "N/A"))
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
