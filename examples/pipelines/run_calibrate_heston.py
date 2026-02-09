#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: calibration.stochastic_vol (Heston Model)
===============================================================================

This script demonstrates how to use the `calibration.stochastic_vol` pipeline
to calibrate the Heston stochastic volatility model to option prices.

What This Pipeline Does
-----------------------
1. Loads option prices/vol quotes from configuration
2. Sets up the Heston model
3. Configures calibration objective (price or vol matching)
4. Runs global + local optimisation
5. Validates Feller condition
6. Stores calibrated parameters

The Heston Model
----------------
The Heston model (1993) is a continuous-time stochastic volatility model:

    dS = μS dt + √V S dW₁
    dV = κ(θ - V) dt + σᵥ √V dW₂
    
where:
    - S is the asset price
    - V is the variance (NOT volatility!)
    - κ (kappa) is mean reversion speed
    - θ (theta) is long-term variance
    - σᵥ (sigma) is volatility of variance (vol-of-vol)
    - ρ (rho) is correlation between S and V
    - V₀ is initial variance

Key constraints:
    - Feller condition: 2κθ > σᵥ² ensures variance stays positive
    - κ > 0 (mean-reversion)
    - θ > 0 (positive long-term variance)

Why Heston?
-----------
- Captures volatility smile/skew
- Correlation parameter drives skew
- Semi-analytic pricing via characteristic function
- Widely used for equity and FX options

Run This Example
----------------
    python examples/pipelines/run_calibrate_heston.py

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
    Build configuration for Heston calibration.
    
    We provide option prices at various strikes and expiries,
    and the calibration will find Heston parameters that best fit.
    """
    
    # -------------------------------------------------------------------------
    # Define option prices to calibrate to
    # -------------------------------------------------------------------------
    # Market prices of options at different strikes and expiries
    # In practice, these come from option chain data
    
    option_data = [
        # 3-month expiry options
        {"expiry": 0.25, "strike": 95,  "type": "call", "price": 10.50},
        {"expiry": 0.25, "strike": 100, "type": "call", "price": 6.85},
        {"expiry": 0.25, "strike": 105, "type": "call", "price": 3.90},
        {"expiry": 0.25, "strike": 95,  "type": "put",  "price": 4.50},
        {"expiry": 0.25, "strike": 100, "type": "put",  "price": 5.85},
        {"expiry": 0.25, "strike": 105, "type": "put",  "price": 7.90},
        
        # 6-month expiry options
        {"expiry": 0.50, "strike": 90,  "type": "call", "price": 14.20},
        {"expiry": 0.50, "strike": 95,  "type": "call", "price": 11.30},
        {"expiry": 0.50, "strike": 100, "type": "call", "price": 8.80},
        {"expiry": 0.50, "strike": 105, "type": "call", "price": 6.60},
        {"expiry": 0.50, "strike": 110, "type": "call", "price": 4.80},
        
        # 1-year expiry options (longer term, more sensitive to vol dynamics)
        {"expiry": 1.00, "strike": 85,  "type": "call", "price": 20.80},
        {"expiry": 1.00, "strike": 90,  "type": "call", "price": 17.50},
        {"expiry": 1.00, "strike": 95,  "type": "call", "price": 14.40},
        {"expiry": 1.00, "strike": 100, "type": "call", "price": 11.70},
        {"expiry": 1.00, "strike": 105, "type": "call", "price": 9.30},
        {"expiry": 1.00, "strike": 110, "type": "call", "price": 7.30},
        {"expiry": 1.00, "strike": 115, "type": "call", "price": 5.60},
    ]
    
    # -------------------------------------------------------------------------
    # Build the RunConfig
    # -------------------------------------------------------------------------
    config = RunConfig(
        pipeline="calibration.stochastic_vol",
        
        io=IOConfig(workdir="./artifacts/heston_calibration"),
        
        params={
            "calibration": {
                # Model to calibrate
                "model": "heston",
                
                # Spot price (needed for pricing during calibration)
                "spot": 100.0,
                
                # Risk-free rate and dividend yield
                "rate": 0.05,
                "dividend": 0.02,
                
                # Option data to calibrate to
                "option_data": {
                    "source": "config",
                    "data": option_data,
                },
                
                # Heston-specific settings
                "heston": {
                    # Initial parameter guesses
                    "initial_params": {
                        "v0": 0.04,       # Initial variance (20% vol)
                        "kappa": 2.0,     # Mean reversion speed
                        "theta": 0.04,    # Long-term variance (20% vol)
                        "sigma": 0.4,     # Vol of vol
                        "rho": -0.7,      # Spot-vol correlation
                    },
                    
                    # Parameter bounds for optimisation
                    "bounds": {
                        "v0":    [0.001, 0.50],    # 3% to 70% vol
                        "kappa": [0.01, 10.0],     # Mean reversion
                        "theta": [0.001, 0.50],    # Long-term variance
                        "sigma": [0.01, 2.0],      # Vol of vol
                        "rho":   [-0.99, 0.99],    # Correlation
                    },
                },
                
                # Pricing method for calibration
                "pricing_method": "characteristic_function",  # Or "monte_carlo"
                
                # Optimisation settings
                "optimiser": {
                    "global": "differential_evolution",
                    "local": "L-BFGS-B",
                },
                "max_iterations": 500,
                "tolerance": 1e-6,
                
                # Enforce Feller condition
                "feller_constraint": True,
            }
        },
    )
    
    return validate_run_config(config)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Execute the Heston calibration pipeline and display results."""
    
    print("=" * 70)
    print("Pipeline Example: Heston Stochastic Volatility Calibration")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Build configuration
    # -------------------------------------------------------------------------
    print("[1/4] Building configuration...")
    cfg = build_config()
    print(f"      Pipeline: {cfg.pipeline}")
    print(f"      Model: Heston")
    print(f"      Options: {len(cfg.params['calibration']['option_data']['data'])}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Execute the pipeline
    # -------------------------------------------------------------------------
    print("[2/4] Executing calibration...")
    print("      (This may take a few seconds for global optimisation)")
    ctx = run_pipeline_from_config(cfg)
    print("      Calibration completed!")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Extract results
    # -------------------------------------------------------------------------
    print("[3/4] Extracting results...")
    
    heston_params = ctx.state.get(Keys.HESTON_PARAMS, {})
    calibration_errors = ctx.state.get(Keys.CALIBRATION_ERRORS, {})
    feller_check = ctx.state.get(Keys.FELLER_CONDITION, {})
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Display Calibration Results
    # -------------------------------------------------------------------------
    print("[4/4] Heston Calibration Results")
    print("=" * 70)
    print()
    
    # Calibrated Heston parameters
    print("Calibrated Heston Parameters:")
    print("-" * 60)
    
    v0 = heston_params.get("v0", 0)
    kappa = heston_params.get("kappa", 0)
    theta = heston_params.get("theta", 0)
    sigma = heston_params.get("sigma", 0)
    rho = heston_params.get("rho", 0)
    
    import math
    
    print(f"  V₀ (v0):      {v0:.4f}    Initial variance ({math.sqrt(v0)*100:.1f}% vol)")
    print(f"  κ (kappa):    {kappa:.4f}    Mean reversion speed")
    print(f"  θ (theta):    {theta:.4f}    Long-term variance ({math.sqrt(theta)*100:.1f}% vol)")
    print(f"  σ (sigma):    {sigma:.4f}    Vol of variance")
    print(f"  ρ (rho):      {rho:+.4f}   Spot-variance correlation")
    print()
    
    # Feller condition check
    print("Model Validity Checks:")
    print("-" * 60)
    
    feller_ratio = 2 * kappa * theta / (sigma ** 2) if sigma > 0 else float('inf')
    feller_satisfied = feller_ratio > 1
    
    print(f"  Feller condition: 2κθ/σ² = {feller_ratio:.2f} {'> 1 ✓' if feller_satisfied else '< 1 ✗'}")
    if feller_satisfied:
        print(f"  → Variance process stays strictly positive")
    else:
        print(f"  → WARNING: Variance may touch zero (requires reflection)")
    
    # Half-life of mean reversion
    half_life = 0.693 / kappa if kappa > 0 else float('inf')
    print(f"  Mean reversion half-life: {half_life:.2f} years")
    print()
    
    # Parameter interpretation
    print("Parameter Interpretation:")
    print("-" * 60)
    
    # V0 vs Theta interpretation
    if v0 > theta:
        print(f"  • V₀ > θ: Current vol ({math.sqrt(v0)*100:.1f}%) above long-term ({math.sqrt(theta)*100:.1f}%)")
        print(f"    → Volatility expected to DECREASE over time")
    else:
        print(f"  • V₀ < θ: Current vol ({math.sqrt(v0)*100:.1f}%) below long-term ({math.sqrt(theta)*100:.1f}%)")
        print(f"    → Volatility expected to INCREASE over time")
    
    # Kappa interpretation
    print(f"  • κ = {kappa:.2f}: {'Fast' if kappa > 2 else 'Moderate' if kappa > 1 else 'Slow'} mean reversion")
    print(f"    → Variance shock decays to half in {half_life:.1f} years")
    
    # Rho interpretation
    if rho < -0.5:
        print(f"  • ρ = {rho:+.2f}: Strong NEGATIVE correlation (leverage effect)")
        print(f"    → Stock drops ↓ volatility jumps ↑ (fear premium)")
    elif rho < 0:
        print(f"  • ρ = {rho:+.2f}: Moderate negative correlation")
    else:
        print(f"  • ρ = {rho:+.2f}: Positive/neutral correlation (unusual)")
    
    # Sigma (vol of vol) interpretation
    print(f"  • σ = {sigma:.2f}: {'High' if sigma > 0.5 else 'Moderate' if sigma > 0.3 else 'Low'} vol-of-vol")
    print(f"    → Controls smile curvature and term structure dynamics")
    print()
    
    # Calibration quality
    print("Calibration Quality:")
    print("-" * 60)
    
    if calibration_errors:
        errors = list(calibration_errors.values())
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else 0
        max_error = max(abs(e) for e in errors) if errors else 0
        
        print(f"  RMSE price error:   ${rmse:.4f}")
        print(f"  Max price error:    ${max_error:.4f}")
        
        # Quality assessment
        if rmse < 0.05:
            quality = "EXCELLENT"
        elif rmse < 0.10:
            quality = "GOOD"
        elif rmse < 0.20:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR - Consider different initial params"
        
        print(f"  Overall quality:    {quality}")
    
    print()
    print("Artifacts saved to:", cfg.io.workdir)
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
