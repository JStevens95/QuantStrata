#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: risk.compute_sensitivities
===============================================================================

This script demonstrates how to use the `risk.compute_sensitivities` pipeline
to compute portfolio Greeks (delta, gamma, vega, theta, rho) with aggregation.

What This Pipeline Does
-----------------------
1. Loads portfolio and market snapshot from state
2. Configures which Greeks to compute
3. Computes Greeks per position via bump-and-reprice
4. Aggregates Greeks by underlying, currency, desk
5. Optionally computes cross-gamma matrix
6. Writes a comprehensive sensitivity report

Key Concepts: The Greeks
------------------------
- **Delta (Δ)**: Sensitivity to underlying price (first-order)
- **Gamma (Γ)**: Rate of change of delta (second-order, convexity)
- **Vega (ν)**: Sensitivity to implied volatility
- **Theta (Θ)**: Time decay (value lost per day)
- **Rho (ρ)**: Sensitivity to interest rates

Why Greeks Matter
-----------------
- **Risk management**: Understand portfolio exposures
- **Hedging**: Determine hedge quantities
- **P&L explain**: Attribute daily P&L to risk factors
- **Limits monitoring**: Check against risk limits

Run This Example
----------------
    python examples/pipelines/run_compute_greeks.py

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.core.state_keys import StateKeys as Keys

# We need to provide market and portfolio in initial_state
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
    """
    Build a market snapshot with all data needed for pricing and Greeks.
    
    A complete market needs:
    - Spot quotes (underlying prices)
    - Yield curves (discount factors, forward rates)
    - Vol surfaces (implied volatility)
    """
    return Market(
        asof=date.today(),
        
        # Spot quotes
        quotes={
            MarketId.parse("FX.SPOT.EURUSD"): Quote(value=1.0850),
            MarketId.parse("FX.SPOT.GBPUSD"): Quote(value=1.2650),
        },
        
        # Yield curves (zero rate curves)
        curves={
            MarketId.parse("IR.ZERO.USD"): FlatZeroRateCurve(continuously_compounded_rate=0.050),
            MarketId.parse("IR.ZERO.EUR"): FlatZeroRateCurve(continuously_compounded_rate=0.040),
            MarketId.parse("IR.ZERO.GBP"): FlatZeroRateCurve(continuously_compounded_rate=0.045),
        },
        
        # Vol surfaces (flat for simplicity)
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
    """Build configuration for Greeks computation."""
    
    config = RunConfig(
        pipeline="risk.compute_sensitivities",
        
        io={
            "artifacts_dir": "./artifacts/greeks_example",
            "enable_save": True,
        },
        
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
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Execute the Greeks pipeline and display results."""
    
    print("=" * 70)
    print("Pipeline Example: risk.compute_sensitivities")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Build prerequisites
    # -------------------------------------------------------------------------
    print("[1/5] Building market and portfolio...")
    market = build_market()
    portfolio = build_portfolio()
    print(f"      Market as of: {market.asof}")
    print(f"      Portfolio size: {len(portfolio)} positions")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Build configuration
    # -------------------------------------------------------------------------
    print("[2/5] Building configuration...")
    cfg = build_config()
    print(f"      Pipeline: {cfg.pipeline}")
    print(f"      Greeks: delta, gamma, vega, theta, rho")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Execute the pipeline with initial state
    # -------------------------------------------------------------------------
    print("[3/5] Executing pipeline...")
    
    # Provide market and portfolio in initial state
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
    
    position_greeks = ctx.state.get(Keys.POSITION_GREEKS, {})
    aggregated_greeks = ctx.state.get(Keys.AGGREGATED_GREEKS, {})
    
    print(f"      Positions with Greeks: {len(position_greeks)}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Display Greeks
    # -------------------------------------------------------------------------
    print("[5/5] Portfolio Greeks Report")
    print("=" * 80)
    print()
    
    # Position-level Greeks
    print("Position-Level Greeks:")
    print("-" * 80)
    print(f"{'Position ID':<25} {'Delta':>12} {'Gamma':>12} {'Vega':>12} {'Theta':>12}")
    print("-" * 80)
    
    for pos_id, greeks in position_greeks.items():
        delta = greeks.get("delta", 0)
        gamma = greeks.get("gamma", 0)
        vega = greeks.get("vega", 0)
        theta = greeks.get("theta", 0)
        
        print(f"{pos_id:<25} {delta:>12,.0f} {gamma:>12,.0f} {vega:>12,.0f} {theta:>12,.0f}")
    
    print("-" * 80)
    print()
    
    # Aggregated Greeks (totals)
    totals = aggregated_greeks.get("TOTAL", {})
    
    print("Portfolio Totals:")
    print("-" * 50)
    print(f"  Delta (Δ):   {totals.get('delta', 0):>15,.0f}")
    print(f"  Gamma (Γ):   {totals.get('gamma', 0):>15,.0f}")
    print(f"  Vega (ν):    {totals.get('vega', 0):>15,.0f}")
    print(f"  Theta (Θ):   {totals.get('theta', 0):>15,.0f}")
    print(f"  Rho (ρ):     {totals.get('rho', 0):>15,.0f}")
    print()
    
    # -------------------------------------------------------------------------
    # Greeks interpretation
    # -------------------------------------------------------------------------
    print("Greeks Interpretation:")
    print("-" * 70)
    
    total_delta = totals.get('delta', 0)
    total_gamma = totals.get('gamma', 0)
    total_vega = totals.get('vega', 0)
    total_theta = totals.get('theta', 0)
    
    # Delta interpretation
    if total_delta > 0:
        print(f"  Delta: Portfolio is LONG the underlying")
        print(f"         A 1% spot move generates ~${total_delta * 0.01:,.0f} P&L")
    else:
        print(f"  Delta: Portfolio is SHORT the underlying")
        print(f"         A 1% spot move generates ~${total_delta * 0.01:,.0f} P&L")
    
    # Gamma interpretation
    if total_gamma > 0:
        print(f"  Gamma: Portfolio is LONG gamma (convexity)")
        print(f"         Benefits from large moves in either direction")
    else:
        print(f"  Gamma: Portfolio is SHORT gamma")
        print(f"         Exposed to large moves (may need rebalancing)")
    
    # Vega interpretation
    if total_vega > 0:
        print(f"  Vega:  Portfolio is LONG volatility")
        print(f"         A 1 vol point increase adds ~${total_vega * 0.01:,.0f}")
    else:
        print(f"  Vega:  Portfolio is SHORT volatility")
        print(f"         A 1 vol point increase costs ~${abs(total_vega) * 0.01:,.0f}")
    
    # Theta interpretation
    print(f"  Theta: Portfolio decays ~${abs(total_theta):,.0f}/day due to time")
    
    print()
    print("Artifacts saved to:", cfg.io.get("artifacts_dir", "N/A"))
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
