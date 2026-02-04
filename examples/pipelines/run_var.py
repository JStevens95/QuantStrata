#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: risk.compute_var
===============================================================================

This script demonstrates how to use the `risk.compute_var` pipeline to compute
Value-at-Risk (VaR) using multiple methodologies.

What This Pipeline Does
-----------------------
1. Loads portfolio and market from state
2. Loads/generates historical returns data
3. Computes Historical Simulation VaR
4. Computes Parametric (Delta-Normal) VaR
5. Computes Monte Carlo VaR
6. Computes Expected Shortfall (CVaR)
7. Compares methods and writes report

Key Concepts: Value-at-Risk
---------------------------
**VaR** answers: "What is the maximum loss at a given confidence level?"

- **VaR(95%)**: The loss that will NOT be exceeded 95% of the time
- **VaR(99%)**: The loss that will NOT be exceeded 99% of the time

VaR Methods
-----------
1. **Historical Simulation**: Use actual historical returns
   - Pros: No distribution assumptions, captures fat tails
   - Cons: Limited by historical data, backward-looking

2. **Parametric (Delta-Normal)**: Assume normal distribution
   - Pros: Fast, closed-form, easy to decompose
   - Cons: Underestimates tail risk, ignores gamma/convexity

3. **Monte Carlo**: Simulate future scenarios
   - Pros: Flexible, can model any distribution
   - Cons: Computationally intensive, model-dependent

Expected Shortfall (CVaR)
-------------------------
"What is the AVERAGE loss when VaR is exceeded?"
CVaR is more conservative and coherent than VaR.

Run This Example
----------------
    python examples/pipelines/run_var.py

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.orchestrator.config.schemas import RunConfig, IOConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.core.state_keys import StateKeys as Keys

# Prerequisites
from src.marketdata.core.market import Market
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.portfolio.core import Portfolio, Position
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption


# =============================================================================
# HELPER: BUILD MARKET AND PORTFOLIO
# =============================================================================

def build_market() -> Market:
    """Build market snapshot for VaR computation."""
    return Market(
        asof=date.today(),
        quotes={
            MarketId.parse("FX.SPOT.EURUSD"): Quote(value=1.0850),
        },
        curves={
            MarketId.parse("IR.ZERO.USD"): FlatZeroRateCurve(continuously_compounded_rate=0.05),
            MarketId.parse("IR.ZERO.EUR"): FlatZeroRateCurve(continuously_compounded_rate=0.04),
        },
        vols={
            MarketId.parse("FX.VOL.EURUSD"): FlatVolSurface(sigma=0.10),
        },
    )


def build_portfolio() -> Portfolio:
    """Build a sample portfolio for VaR computation."""
    
    positions = [
        # Large directional position
        Position(
            position_id="EURUSD_LONG_CALL",
            instrument=FxVanillaEuropeanOption(
                option_type="call",
                notional=50_000_000,  # $50M notional
                strike=1.10,
                expiry=0.5,
                spot_id=MarketId.parse("FX.SPOT.EURUSD"),
                vol_id=MarketId.parse("FX.VOL.EURUSD"),
                domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
                foreign_curve_id=MarketId.parse("IR.ZERO.EUR"),
            ),
            quantity=1,
        ),
        
        # Partially hedging put
        Position(
            position_id="EURUSD_LONG_PUT",
            instrument=FxVanillaEuropeanOption(
                option_type="put",
                notional=30_000_000,
                strike=1.05,
                expiry=0.5,
                spot_id=MarketId.parse("FX.SPOT.EURUSD"),
                vol_id=MarketId.parse("FX.VOL.EURUSD"),
                domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
                foreign_curve_id=MarketId.parse("IR.ZERO.EUR"),
            ),
            quantity=1,
        ),
    ]
    
    return Portfolio(positions=positions)


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config() -> RunConfig:
    """Build configuration for VaR computation."""
    
    config = RunConfig(
        pipeline="risk.compute_var",
        
        io=IOConfig(workdir="./artifacts/var_example"),
        
        params={
            "var": {
                # Confidence levels to compute
                "confidence_levels": [0.95, 0.99],
                
                # VaR horizon (in days)
                "horizon_days": 1,
                
                # Method-specific settings
                "methods": {
                    "historical": {
                        "enabled": True,
                        "lookback_days": 252,    # 1 year of history
                        "decay": 0.94,           # Exponential decay for weighting
                    },
                    "parametric": {
                        "enabled": True,
                        "covariance": "exponential",  # How to estimate covariance
                        "decay": 0.94,
                    },
                    "monte_carlo": {
                        "enabled": True,
                        "n_simulations": 10000,
                        "model": "gbm",          # GBM, GARCH, or historical_bootstrap
                    },
                },
                
                # Also compute Expected Shortfall (CVaR)
                "compute_es": True,
                
                # Decompose VaR by risk factor
                "decomposition": True,
            }
        },
    )
    
    return validate_run_config(config)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Execute the VaR pipeline and display results."""
    
    print("=" * 70)
    print("Pipeline Example: risk.compute_var")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Build prerequisites
    # -------------------------------------------------------------------------
    print("[1/5] Building market and portfolio...")
    market = build_market()
    portfolio = build_portfolio()
    
    # Calculate approximate portfolio value
    portfolio_value = sum(
        p.instrument.notional * 0.02  # Rough option premium estimate
        for p in portfolio
    )
    
    print(f"      Portfolio positions: {len(portfolio)}")
    print(f"      Approximate PV: ${portfolio_value:,.0f}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Build configuration
    # -------------------------------------------------------------------------
    print("[2/5] Building configuration...")
    cfg = build_config()
    print(f"      Pipeline: {cfg.pipeline}")
    print(f"      Confidence levels: 95%, 99%")
    print(f"      Methods: Historical, Parametric, Monte Carlo")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Execute the pipeline
    # -------------------------------------------------------------------------
    print("[3/5] Executing pipeline...")
    
    initial_state = {
        Keys.MARKET: market,
        Keys.PORTFOLIO: portfolio,
    }
    
    ctx = run_pipeline_from_config(cfg, initial_state=initial_state)
    print("      Pipeline completed successfully!")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Extract results
    # -------------------------------------------------------------------------
    print("[4/5] Extracting results...")
    
    historical_var = ctx.state.get(Keys.HISTORICAL_VAR, {})
    parametric_var = ctx.state.get(Keys.PARAMETRIC_VAR, {})
    monte_carlo_var = ctx.state.get(Keys.MONTE_CARLO_VAR, {})
    expected_shortfall = ctx.state.get(Keys.EXPECTED_SHORTFALL, {})
    var_report = ctx.state.get(Keys.VAR_REPORT, {})
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Display VaR Report
    # -------------------------------------------------------------------------
    print("[5/5] Value-at-Risk Report")
    print("=" * 70)
    print()
    
    # VaR by method
    print("VaR by Method (1-Day Horizon):")
    print("-" * 70)
    print(f"{'Method':<20} {'VaR(95%)':>15} {'VaR(99%)':>15}")
    print("-" * 70)
    
    # Historical VaR
    h95 = historical_var.get(0.95, 0)
    h99 = historical_var.get(0.99, 0)
    print(f"{'Historical':<20} ${h95:>14,.0f} ${h99:>14,.0f}")
    
    # Parametric VaR
    p95 = parametric_var.get(0.95, 0)
    p99 = parametric_var.get(0.99, 0)
    print(f"{'Parametric':<20} ${p95:>14,.0f} ${p99:>14,.0f}")
    
    # Monte Carlo VaR
    m95 = monte_carlo_var.get(0.95, 0)
    m99 = monte_carlo_var.get(0.99, 0)
    print(f"{'Monte Carlo':<20} ${m95:>14,.0f} ${m99:>14,.0f}")
    
    print("-" * 70)
    print()
    
    # Expected Shortfall
    print("Expected Shortfall (CVaR):")
    print("-" * 50)
    es_historical = expected_shortfall.get("historical", {})
    es95 = es_historical.get(0.95, 0)
    es99 = es_historical.get(0.99, 0)
    print(f"  ES(95%): ${es95:>12,.0f}")
    print(f"  ES(99%): ${es99:>12,.0f}")
    print()
    
    # -------------------------------------------------------------------------
    # VaR Comparison
    # -------------------------------------------------------------------------
    print("Method Comparison:")
    print("-" * 70)
    
    var_95_values = [h95, p95, m95]
    var_99_values = [h99, p99, m99]
    
    # Most/least conservative
    if any(var_95_values):
        most_conservative = max(var_95_values)
        least_conservative = min(var_95_values)
        spread = most_conservative - least_conservative
        
        print(f"  VaR(95%) spread:    ${spread:,.0f} between methods")
        print(f"  Most conservative:  ${most_conservative:,.0f}")
        print(f"  Least conservative: ${least_conservative:,.0f}")
    print()
    
    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------
    print("Interpretation:")
    print("-" * 70)
    print(f"  At 95% confidence, daily losses should not exceed ~${h95:,.0f}")
    print(f"  At 99% confidence, daily losses should not exceed ~${h99:,.0f}")
    print()
    print(f"  However, when VaR is exceeded (5% of days at 95% level),")
    print(f"  the AVERAGE loss (ES) is ~${es95:,.0f}")
    print()
    
    # Risk metrics as % of portfolio
    if portfolio_value > 0:
        print(f"  VaR(95%) as % of portfolio: {h95/portfolio_value*100:.2f}%")
        print(f"  VaR(99%) as % of portfolio: {h99/portfolio_value*100:.2f}%")
    print()
    
    print("Artifacts saved to:", cfg.io.workdir)
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
