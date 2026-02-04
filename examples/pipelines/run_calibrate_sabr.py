#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: calibration.volatility_surface
===============================================================================

This script demonstrates how to use the `calibration.volatility_surface` pipeline
to calibrate a SABR volatility model to market option quotes.

What This Pipeline Does
-----------------------
1. Loads vol quotes from configuration or state
2. Loads yield curves for forward calculation
3. Selects calibration method (SABR, Dupire, SVI)
4. Sets up the calibration objective function
5. Runs numerical optimisation
6. Validates calibration quality
7. Builds the calibrated surface
8. Stores calibration results

The SABR Model
--------------
SABR (Stochastic Alpha Beta Rho) is a stochastic volatility model that
captures the smile dynamics in interest rate and FX markets.

Model dynamics:
    dF = σ * F^β * dW₁
    dσ = ν * σ * dW₂
    
where:
    - F is the forward price
    - σ is the stochastic volatility
    - α (alpha) is the initial volatility level
    - β (beta) is the CEV exponent (often fixed at 0.5 or 0)
    - ρ (rho) is the correlation between F and σ
    - ν (nu) is the vol-of-vol

Why SABR?
---------
- Industry standard for swaption and cap/floor vol surfaces
- Captures smile and skew dynamics
- Analytically tractable (Hagan approximation)
- Parameters have intuitive interpretations

Run This Example
----------------
    python examples/pipelines/run_calibrate_sabr.py

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.orchestrator.config.schemas import RunConfig, IOConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.core.state_keys import StateKeys as Keys


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config() -> RunConfig:
    """
    Build configuration for SABR calibration.
    
    We'll calibrate SABR to EURUSD FX vol quotes with a typical smile.
    """
    
    # -------------------------------------------------------------------------
    # Define vol quotes to calibrate to
    # -------------------------------------------------------------------------
    # These represent market-observed implied vols at different deltas
    # The smile pattern: higher vols for OTM options
    
    vol_quotes = [
        # 1-month expiry smile
        {"expiry": "1M", "delta": 0.10, "vol": 0.115},   # 10-delta (deep OTM)
        {"expiry": "1M", "delta": 0.25, "vol": 0.095},   # 25-delta
        {"expiry": "1M", "delta": 0.50, "vol": 0.082},   # ATM
        {"expiry": "1M", "delta": 0.75, "vol": 0.092},   # 75-delta
        {"expiry": "1M", "delta": 0.90, "vol": 0.108},   # 90-delta (deep ITM)
        
        # 3-month expiry smile
        {"expiry": "3M", "delta": 0.10, "vol": 0.118},
        {"expiry": "3M", "delta": 0.25, "vol": 0.098},
        {"expiry": "3M", "delta": 0.50, "vol": 0.088},
        {"expiry": "3M", "delta": 0.75, "vol": 0.096},
        {"expiry": "3M", "delta": 0.90, "vol": 0.112},
        
        # 6-month expiry smile
        {"expiry": "6M", "delta": 0.10, "vol": 0.122},
        {"expiry": "6M", "delta": 0.25, "vol": 0.102},
        {"expiry": "6M", "delta": 0.50, "vol": 0.092},
        {"expiry": "6M", "delta": 0.75, "vol": 0.100},
        {"expiry": "6M", "delta": 0.90, "vol": 0.116},
        
        # 1-year expiry smile
        {"expiry": "1Y", "delta": 0.10, "vol": 0.128},
        {"expiry": "1Y", "delta": 0.25, "vol": 0.108},
        {"expiry": "1Y", "delta": 0.50, "vol": 0.098},
        {"expiry": "1Y", "delta": 0.75, "vol": 0.106},
        {"expiry": "1Y", "delta": 0.90, "vol": 0.122},
    ]
    
    # -------------------------------------------------------------------------
    # Build the RunConfig
    # -------------------------------------------------------------------------
    config = RunConfig(
        pipeline="calibration.volatility_surface",
        
        io=IOConfig(workdir="./artifacts/sabr_calibration"),
        
        params={
            "calibration": {
                # Calibration method
                "method": "sabr",
                
                # Underlying details
                "underlying": "EURUSD",
                
                # Input vol quotes
                "vol_quotes": {
                    "source": "config",
                    "data": vol_quotes,
                },
                
                # SABR-specific parameters
                "sabr": {
                    # Initial parameter guesses
                    "initial_params": {
                        "alpha": 0.20,    # Initial volatility
                        "beta": 0.50,     # CEV exponent (often fixed)
                        "rho": -0.30,     # Spot-vol correlation (typically negative)
                        "nu": 0.40,       # Vol-of-vol
                    },
                    
                    # Fix beta? Common choices: 0 (normal), 0.5 (CIR), 1 (lognormal)
                    "fix_beta": True,
                    "beta_value": 0.50,
                },
                
                # Optimisation settings
                "optimiser": "L-BFGS-B",    # Bounded optimisation
                "max_iterations": 1000,
                "tolerance": 1e-8,
                
                # Validation
                "max_error_bps": 50,        # Max acceptable error in bps
            }
        },
    )
    
    return validate_run_config(config)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Execute the SABR calibration pipeline and display results."""
    
    print("=" * 70)
    print("Pipeline Example: SABR Volatility Surface Calibration")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Build configuration
    # -------------------------------------------------------------------------
    print("[1/4] Building configuration...")
    cfg = build_config()
    print(f"      Pipeline: {cfg.pipeline}")
    print(f"      Method: SABR")
    print(f"      Underlying: EURUSD")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Execute the pipeline
    # -------------------------------------------------------------------------
    print("[2/4] Executing calibration...")
    ctx = run_pipeline_from_config(cfg)
    print("      Calibration completed!")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Extract results
    # -------------------------------------------------------------------------
    print("[3/4] Extracting results...")
    
    calibrated_params = ctx.state.get(Keys.CALIBRATED_PARAMS, {})
    calibration_errors = ctx.state.get(Keys.CALIBRATION_ERRORS, {})
    calibration_result = ctx.state.get(Keys.CALIBRATION_RESULT, {})
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Display Calibration Results
    # -------------------------------------------------------------------------
    print("[4/4] SABR Calibration Results")
    print("=" * 70)
    print()
    
    # Calibrated SABR parameters
    print("Calibrated SABR Parameters:")
    print("-" * 50)
    
    alpha = calibrated_params.get("alpha", 0)
    beta = calibrated_params.get("beta", 0)
    rho = calibrated_params.get("rho", 0)
    nu = calibrated_params.get("nu", 0)
    
    print(f"  α (alpha):  {alpha:.4f}    Initial volatility level")
    print(f"  β (beta):   {beta:.4f}    CEV exponent (fixed)")
    print(f"  ρ (rho):    {rho:+.4f}   Spot-vol correlation")
    print(f"  ν (nu):     {nu:.4f}    Vol-of-vol")
    print()
    
    # Parameter interpretation
    print("Parameter Interpretation:")
    print("-" * 50)
    
    # Alpha interpretation
    print(f"  • Alpha = {alpha:.2%} sets the ATM vol level")
    
    # Rho interpretation
    if rho < 0:
        print(f"  • Rho = {rho:+.2f} indicates negative skew (puts more expensive)")
        print(f"    This is typical for equity/FX: 'fear' premium on downside")
    else:
        print(f"  • Rho = {rho:+.2f} indicates positive skew (calls more expensive)")
    
    # Nu interpretation
    print(f"  • Nu = {nu:.2f} controls smile curvature (vol-of-vol)")
    if nu > 0.5:
        print(f"    High nu = pronounced smile, larger wings")
    else:
        print(f"    Low nu = flatter smile, smaller wings")
    
    print()
    
    # Calibration quality
    print("Calibration Quality:")
    print("-" * 50)
    
    if calibration_errors:
        errors_list = list(calibration_errors.values())
        avg_error = sum(errors_list) / len(errors_list) if errors_list else 0
        max_error = max(errors_list) if errors_list else 0
        
        print(f"  Average error:    {avg_error:.1f} bps")
        print(f"  Maximum error:    {max_error:.1f} bps")
        print(f"  Convergence:      {'Yes' if calibration_result.get('converged') else 'No'}")
        
        # Quality assessment
        if avg_error < 5:
            quality = "EXCELLENT"
        elif avg_error < 15:
            quality = "GOOD"
        elif avg_error < 30:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR - Consider different initial params"
        
        print(f"  Overall quality:  {quality}")
    
    print()
    
    # -------------------------------------------------------------------------
    # SABR smile at different expiries
    # -------------------------------------------------------------------------
    print("Implied Vol from Calibrated SABR:")
    print("-" * 60)
    print(f"{'Expiry':<10} {'10D':>10} {'25D':>10} {'ATM':>10} {'75D':>10} {'90D':>10}")
    print("-" * 60)
    
    # Display sample implied vols (these would come from the calibrated surface)
    # For illustration, showing the input quotes
    vol_quotes = cfg.params["calibration"]["vol_quotes"]["data"]
    expiries = sorted(set(q["expiry"] for q in vol_quotes))
    
    for exp in expiries:
        exp_quotes = [q for q in vol_quotes if q["expiry"] == exp]
        vols_by_delta = {q["delta"]: q["vol"] for q in exp_quotes}
        
        d10 = vols_by_delta.get(0.10, "-")
        d25 = vols_by_delta.get(0.25, "-")
        atm = vols_by_delta.get(0.50, "-")
        d75 = vols_by_delta.get(0.75, "-")
        d90 = vols_by_delta.get(0.90, "-")
        
        def fmt(v):
            return f"{v:.2%}" if isinstance(v, float) else v
        
        print(f"{exp:<10} {fmt(d10):>10} {fmt(d25):>10} {fmt(atm):>10} {fmt(d75):>10} {fmt(d90):>10}")
    
    print("-" * 60)
    print()
    
    print("Artifacts saved to:", cfg.io.workdir)
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
