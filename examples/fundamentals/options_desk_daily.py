#!/usr/bin/env python3
"""
===============================================================================
Workflow: Options Desk Daily Process
===============================================================================

This script implements a **complete daily workflow** for an options trading desk:

1. Load market data (curves, vol surfaces)
2. Build portfolio from trade blotter
3. Price all positions
4. Compute Greeks and exposures
5. Run daily scenario analysis
6. Generate risk report

This mirrors what a front-office quant desk would run every morning to:
- Mark-to-market the book
- Understand risk exposures
- Prepare for trading decisions
- Meet regulatory requirements

Pipeline Flow
-------------
    [Market Data]
         │
         ▼
    [Portfolio Construction]
         │
         ▼
    [Portfolio Pricing]
         │
         ▼
    [Greeks Computation]
         │
         ▼
    [Scenario Analysis]
         │
         ▼
    [Risk Report Generation]

Run This Workflow
-----------------
    python examples/workflows/options_desk_daily.py

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path
from datetime import date, datetime
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Orchestrator
from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.core.state_keys import StateKeys as Keys

# Market data
import numpy as np
from src.marketdata.core.market import Market
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.curves.term_structure import ZeroRateCurve
from src.marketdata.surfaces.vol_surface import GridVolSurface

# Portfolio
from src.portfolio.core import Portfolio, Position
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption

# Pricing
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer


# =============================================================================
# CONFIGURATION
# =============================================================================

# Market data source (in production, this would come from a data vendor)
MARKET_CONFIG = {
    "source": "synthetic",
    "as_of": date.today(),
    
    # FX spots
    "spots": {
        "EURUSD": 1.0850,
        "GBPUSD": 1.2650,
        "USDJPY": 149.50,
    },
    
    # Interest rate term structures (realistic curve shapes)
    # Tenors in years, rates are continuously compounded
    "curves": {
        "USD": {
            # Inverted curve (short rates > long rates) - realistic 2024+ environment
            "tenors": np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]),
            "rates": np.array([0.0540, 0.0535, 0.0520, 0.0480, 0.0460, 0.0440, 0.0435, 0.0430, 0.0425, 0.0420]),
        },
        "EUR": {
            # Normal upward sloping curve
            "tenors": np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]),
            "rates": np.array([0.0380, 0.0385, 0.0390, 0.0395, 0.0400, 0.0405, 0.0408, 0.0410, 0.0412, 0.0415]),
        },
        "GBP": {
            # Humped curve (peak at 2Y)
            "tenors": np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]),
            "rates": np.array([0.0460, 0.0470, 0.0480, 0.0485, 0.0480, 0.0470, 0.0465, 0.0460, 0.0455, 0.0450]),
        },
        "JPY": {
            # Very flat, low rate environment
            "tenors": np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]),
            "rates": np.array([0.0005, 0.0008, 0.0010, 0.0012, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050]),
        },
    },
    
    # Implied volatility surfaces (realistic smile and term structure)
    # Structure: expiries x strikes grid with implied vols
    "vol_surfaces": {
        "EURUSD": {
            "spot": 1.0850,
            "expiries": np.array([0.083, 0.167, 0.25, 0.5, 1.0]),  # 1M, 2M, 3M, 6M, 1Y
            # Strikes around spot (moneyness grid)
            "strikes": np.array([0.95, 0.98, 1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15]) * 1.0850,
            # Vol grid (expiry x strike) - realistic smile with negative skew
            "vols": np.array([
                # 1M: Higher short-dated vol, steep smile
                [0.115, 0.100, 0.088, 0.085, 0.090, 0.095, 0.100, 0.108, 0.118],
                # 2M
                [0.112, 0.098, 0.087, 0.084, 0.088, 0.093, 0.098, 0.105, 0.115],
                # 3M
                [0.110, 0.096, 0.086, 0.084, 0.087, 0.092, 0.096, 0.103, 0.112],
                # 6M: Lower vol, flatter smile
                [0.108, 0.095, 0.086, 0.085, 0.088, 0.092, 0.095, 0.100, 0.108],
                # 1Y: Lowest vol, flattest smile
                [0.105, 0.094, 0.088, 0.087, 0.090, 0.093, 0.096, 0.100, 0.106],
            ]),
        },
        "GBPUSD": {
            "spot": 1.2650,
            "expiries": np.array([0.083, 0.167, 0.25, 0.5, 1.0]),
            "strikes": np.array([0.94, 0.97, 1.00, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18]) * 1.2650,
            # Higher vol for GBP, more pronounced smile
            "vols": np.array([
                [0.135, 0.118, 0.105, 0.100, 0.105, 0.112, 0.120, 0.130, 0.142],
                [0.130, 0.115, 0.103, 0.099, 0.103, 0.110, 0.117, 0.126, 0.138],
                [0.125, 0.112, 0.101, 0.098, 0.102, 0.108, 0.115, 0.123, 0.134],
                [0.120, 0.108, 0.099, 0.097, 0.100, 0.106, 0.112, 0.119, 0.128],
                [0.115, 0.105, 0.098, 0.096, 0.099, 0.104, 0.110, 0.116, 0.124],
            ]),
        },
        "USDJPY": {
            "spot": 149.50,
            "expiries": np.array([0.083, 0.167, 0.25, 0.5, 1.0]),
            "strikes": np.array([135, 140, 145, 148, 150, 152, 155, 160, 165]),
            # Lower vol for JPY, more symmetric smile
            "vols": np.array([
                [0.110, 0.095, 0.085, 0.082, 0.080, 0.082, 0.085, 0.092, 0.102],
                [0.108, 0.094, 0.084, 0.081, 0.079, 0.081, 0.084, 0.090, 0.099],
                [0.105, 0.092, 0.083, 0.080, 0.078, 0.080, 0.083, 0.089, 0.097],
                [0.100, 0.089, 0.082, 0.079, 0.078, 0.079, 0.082, 0.087, 0.094],
                [0.095, 0.086, 0.080, 0.078, 0.077, 0.078, 0.080, 0.085, 0.091],
            ]),
        },
    },
}

# Trade blotter (in production, this would come from a trading system)
TRADE_BLOTTER = [
    # EURUSD book
    {"trade_id": "EU001", "underlying": "EURUSD", "type": "call", "strike": 1.10, 
     "expiry": 0.25, "notional": 10_000_000, "direction": "long"},
    {"trade_id": "EU002", "underlying": "EURUSD", "type": "put", "strike": 1.05, 
     "expiry": 0.25, "notional": 10_000_000, "direction": "short"},
    {"trade_id": "EU003", "underlying": "EURUSD", "type": "call", "strike": 1.085, 
     "expiry": 0.50, "notional": 5_000_000, "direction": "long"},
    {"trade_id": "EU004", "underlying": "EURUSD", "type": "put", "strike": 1.085, 
     "expiry": 0.50, "notional": 5_000_000, "direction": "long"},
    
    # GBPUSD book
    {"trade_id": "GU001", "underlying": "GBPUSD", "type": "call", "strike": 1.30, 
     "expiry": 0.25, "notional": 8_000_000, "direction": "long"},
    {"trade_id": "GU002", "underlying": "GBPUSD", "type": "put", "strike": 1.22, 
     "expiry": 0.25, "notional": 8_000_000, "direction": "long"},
]

# Scenario definitions
SCENARIOS = [
    {"name": "Base Case", "spot_shock": {}, "vol_shock": {}},
    {"name": "USD Strength (+2%)", "spot_shock": {"EURUSD": -0.02, "GBPUSD": -0.02}, "vol_shock": {}},
    {"name": "USD Weakness (-2%)", "spot_shock": {"EURUSD": +0.02, "GBPUSD": +0.02}, "vol_shock": {}},
    {"name": "Vol Spike (+5pts)", "spot_shock": {}, "vol_shock": {"EURUSD": +0.05, "GBPUSD": +0.05}},
    {"name": "Vol Crush (-3pts)", "spot_shock": {}, "vol_shock": {"EURUSD": -0.03, "GBPUSD": -0.03}},
    {"name": "Risk-Off", "spot_shock": {"EURUSD": -0.05, "GBPUSD": -0.05}, "vol_shock": {"EURUSD": +0.08, "GBPUSD": +0.10}},
    {"name": "Risk-On", "spot_shock": {"EURUSD": +0.03, "GBPUSD": +0.03}, "vol_shock": {"EURUSD": -0.02, "GBPUSD": -0.02}},
]


# =============================================================================
# WORKFLOW STEPS
# =============================================================================

def step_1_load_market_data():
    """
    Step 1: Load and build market data.
    
    In production, this would:
    - Connect to market data vendor (Bloomberg, Reuters)
    - Load live quotes
    - Bootstrap yield curves from swap/deposit rates
    - Build vol surfaces from option market quotes
    
    For this example, we use realistic term structures and vol surfaces.
    """
    print("\n" + "="*70)
    print("STEP 1: Loading Market Data")
    print("="*70)
    
    # Build market IDs
    market_ids = {}
    for pair in MARKET_CONFIG["spots"]:
        market_ids[pair] = {
            "spot": MarketId.parse(f"FX.SPOT.{pair}"),
            "vol": MarketId.parse(f"FX.VOL.{pair}"),
        }
    
    for ccy in MARKET_CONFIG["curves"]:
        market_ids[f"IR.{ccy}"] = MarketId.parse(f"IR.ZERO.{ccy}")
    
    # Build market object with realistic structures
    quotes = {}
    curves = {}
    vols = {}
    
    # FX spot quotes
    for pair, spot in MARKET_CONFIG["spots"].items():
        quotes[market_ids[pair]["spot"]] = Quote(value=spot)
    
    # Build realistic zero rate curves
    for ccy, curve_data in MARKET_CONFIG["curves"].items():
        curves[market_ids[f"IR.{ccy}"]] = ZeroRateCurve(
            tenors=curve_data["tenors"],
            zero_rates=curve_data["rates"],
            extrapolation="flat",
        )
    
    # Build realistic volatility surfaces
    for pair, vol_data in MARKET_CONFIG["vol_surfaces"].items():
        vols[market_ids[pair]["vol"]] = GridVolSurface(
            expiries=vol_data["expiries"],
            strikes=vol_data["strikes"],
            implied_vols=vol_data["vols"],
            extrapolation="flat",
            strike_space="absolute",
            surface_id=f"FX.{pair}.VOL",
        )
    
    market = Market(
        asof=MARKET_CONFIG["as_of"],
        quotes=quotes,
        curves=curves,
        vols=vols,
    )
    
    print(f"\n  Market as of: {market.asof}")
    
    print(f"\n  Spot Quotes:")
    for pair, spot in MARKET_CONFIG["spots"].items():
        print(f"    {pair}: {spot:.4f}")
    
    print(f"\n  Zero Rate Curves (term structure):")
    for ccy, curve_data in MARKET_CONFIG["curves"].items():
        short_rate = curve_data["rates"][0]
        long_rate = curve_data["rates"][-1]
        shape = "Inverted" if short_rate > long_rate else "Normal" if short_rate < long_rate else "Flat"
        print(f"    {ccy}: {shape} curve, 3M={short_rate:.2%}, 30Y={long_rate:.2%}")
    
    print(f"\n  Volatility Surfaces:")
    for pair, vol_data in MARKET_CONFIG["vol_surfaces"].items():
        atm_idx = len(vol_data["strikes"]) // 2  # Middle strike (approx ATM)
        atm_vol_3m = vol_data["vols"][2, atm_idx]  # 3M ATM
        atm_vol_1y = vol_data["vols"][4, atm_idx]  # 1Y ATM
        print(f"    {pair}: ATM 3M={atm_vol_3m:.2%}, ATM 1Y={atm_vol_1y:.2%}, {vol_data['vols'].shape[0]} expiries x {vol_data['vols'].shape[1]} strikes")
    
    print("\n  [✓] Market data loaded successfully")
    
    return market, market_ids


def step_2_build_portfolio(market_ids):
    """
    Step 2: Build portfolio from trade blotter.
    
    In production, this would:
    - Connect to trade capture system
    - Load live positions
    - Reconcile with settlements
    - Apply any pending amendments
    """
    print("\n" + "="*70)
    print("STEP 2: Building Portfolio")
    print("="*70)
    
    positions = []
    
    for trade in TRADE_BLOTTER:
        underlying = trade["underlying"]
        base_ccy = underlying[:3]  # EUR, GBP
        quote_ccy = underlying[3:]  # USD
        
        # Map to market IDs
        spot_id = market_ids[underlying]["spot"]
        vol_id = market_ids[underlying]["vol"]
        domestic_curve_id = market_ids[f"IR.{quote_ccy}"]
        foreign_curve_id = market_ids[f"IR.{base_ccy}"]
        
        # Build instrument
        instrument = FxVanillaEuropeanOption(
            option_type=trade["type"],
            notional=trade["notional"],
            strike=trade["strike"],
            expiry=trade["expiry"],
            spot_id=spot_id,
            vol_id=vol_id,
            domestic_curve_id=domestic_curve_id,
            foreign_curve_id=foreign_curve_id,
        )
        
        # Build position
        quantity = 1 if trade["direction"] == "long" else -1
        positions.append(Position(
            position_id=trade["trade_id"],
            instrument=instrument,
            quantity=quantity,
        ))
    
    portfolio = Portfolio(positions=positions)
    
    print(f"\n  Loaded {len(portfolio)} positions from trade blotter")
    print(f"\n  Trade Summary:")
    print(f"    {'ID':<10} {'Underlying':<10} {'Type':<6} {'K':<8} {'T':<6} {'Dir':<6} {'Notional':>12}")
    print(f"    {'-'*70}")
    
    for trade in TRADE_BLOTTER:
        dir_str = trade["direction"].upper()
        print(f"    {trade['trade_id']:<10} {trade['underlying']:<10} {trade['type']:<6} "
              f"{trade['strike']:<8.4f} {trade['expiry']:<6.2f} {dir_str:<6} "
              f"{trade['notional']:>12,}")
    
    print("\n  [✓] Portfolio built successfully")
    
    return portfolio


def step_3_price_portfolio(market, portfolio):
    """
    Step 3: Price all positions and compute Greeks.
    
    This gives us the current mark-to-market and risk exposures.
    """
    print("\n" + "="*70)
    print("STEP 3: Pricing Portfolio")
    print("="*70)
    
    pricer = FxVanillaEuropeanOptionBsmPricer()
    
    results = {}
    totals = {"pv": 0, "delta": 0, "gamma": 0, "vega": 0, "theta": 0}
    
    for pos in portfolio:
        # Price the position
        pv = pricer.price(pos.instrument, market) * pos.quantity
        # Compute Greeks
        greeks = pricer.greeks(pos.instrument, market)
        
        results[pos.position_id] = {
            "pv": pv,
            "delta": greeks.get("delta", 0) * pos.quantity,
            "gamma": greeks.get("gamma", 0) * pos.quantity,
            "vega": greeks.get("vega", 0) * pos.quantity,
            "theta": greeks.get("theta", 0) * pos.quantity,
        }
        
        for key in totals:
            totals[key] += results[pos.position_id][key]
    
    print(f"\n  Position-Level Pricing:")
    print(f"    {'ID':<10} {'PV':>15} {'Delta':>12} {'Gamma':>12} {'Vega':>12} {'Theta':>12}")
    print(f"    {'-'*75}")
    
    for pos_id, vals in results.items():
        print(f"    {pos_id:<10} ${vals['pv']:>13,.0f} {vals['delta']:>12,.0f} "
              f"{vals['gamma']:>12,.0f} {vals['vega']:>12,.0f} {vals['theta']:>12,.0f}")
    
    print(f"    {'-'*75}")
    print(f"    {'TOTAL':<10} ${totals['pv']:>13,.0f} {totals['delta']:>12,.0f} "
          f"{totals['gamma']:>12,.0f} {totals['vega']:>12,.0f} {totals['theta']:>12,.0f}")
    
    print("\n  [✓] Pricing completed")
    
    return results, totals


def step_4_run_scenarios(market, portfolio, market_ids, totals):
    """
    Step 4: Run scenario analysis.
    
    Stress test the portfolio under various market moves.
    Applies parallel shifts to spots and vol surfaces.
    """
    print("\n" + "="*70)
    print("STEP 4: Running Scenario Analysis")
    print("="*70)
    
    pricer = FxVanillaEuropeanOptionBsmPricer()
    base_pv = totals["pv"]
    
    scenario_results = []
    
    for scenario in SCENARIOS:
        # Apply shocks to spot quotes
        shocked_quotes = {}
        for pair, spot in MARKET_CONFIG["spots"].items():
            shock = scenario["spot_shock"].get(pair, 0)
            shocked_quotes[market_ids[pair]["spot"]] = Quote(value=spot * (1 + shock))
        
        # Apply parallel vol shocks to vol surfaces
        shocked_vols = {}
        for pair, vol_data in MARKET_CONFIG["vol_surfaces"].items():
            shock = scenario["vol_shock"].get(pair, 0)
            # Shift the entire vol surface by the shock amount
            shocked_vol_grid = np.maximum(0.01, vol_data["vols"] + shock)
            shocked_vols[market_ids[pair]["vol"]] = GridVolSurface(
                expiries=vol_data["expiries"],
                strikes=vol_data["strikes"],
                implied_vols=shocked_vol_grid,
                extrapolation="flat",
                strike_space="absolute",
            )
        
        # Build shocked market (curves unchanged in these scenarios)
        curves = {}
        for ccy, curve_data in MARKET_CONFIG["curves"].items():
            curves[market_ids[f"IR.{ccy}"]] = ZeroRateCurve(
                tenors=curve_data["tenors"],
                zero_rates=curve_data["rates"],
                extrapolation="flat",
            )
        
        shocked_market = Market(
            asof=market.asof,
            quotes=shocked_quotes,
            curves=curves,
            vols=shocked_vols,
        )
        
        # Price portfolio
        scenario_pv = 0
        for pos in portfolio:
            pv = pricer.price(pos.instrument, shocked_market)
            scenario_pv += pv * pos.quantity
        
        pnl = scenario_pv - base_pv
        
        scenario_results.append({
            "name": scenario["name"],
            "pv": scenario_pv,
            "pnl": pnl,
            "pnl_pct": pnl / abs(base_pv) * 100 if base_pv != 0 else 0,
        })
    
    print(f"\n  Base PV: ${base_pv:,.0f}")
    print(f"\n  Scenario Results:")
    print(f"    {'Scenario':<30} {'Scenario PV':>15} {'P&L':>15} {'P&L %':>10}")
    print(f"    {'-'*70}")
    
    for result in scenario_results:
        print(f"    {result['name']:<30} ${result['pv']:>13,.0f} "
              f"${result['pnl']:>+13,.0f} {result['pnl_pct']:>+9.1f}%")
    
    print("\n  [✓] Scenario analysis completed")
    
    return scenario_results


def step_5_generate_report(totals, scenario_results):
    """
    Step 5: Generate daily risk report.
    
    Creates executive summary for the trading desk.
    """
    print("\n" + "="*70)
    print("STEP 5: Daily Risk Report")
    print("="*70)
    
    # Find worst/best scenarios
    sorted_scenarios = sorted(scenario_results[1:], key=lambda x: x["pnl"])
    worst = sorted_scenarios[0]
    best = sorted_scenarios[-1]
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      OPTIONS DESK DAILY REPORT                       ║
║                      {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^38}                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  PORTFOLIO VALUATION                                                 ║
║  ───────────────────────────────────────────────────────────────────║
║  Total PV (MTM):              ${totals['pv']:>15,.0f}                          ║
║                                                                      ║
║  RISK EXPOSURES                                                      ║
║  ───────────────────────────────────────────────────────────────────║
║  Delta:                       ${totals['delta']:>15,.0f}                          ║
║  Gamma:                       {totals['gamma']:>16,.0f}                          ║
║  Vega:                        ${totals['vega']:>15,.0f}                          ║
║  Theta (daily):               ${totals['theta']:>15,.0f}                          ║
║                                                                      ║
║  SCENARIO ANALYSIS                                                   ║
║  ───────────────────────────────────────────────────────────────────║
║  Worst case: {worst['name']:<25}                              ║
║              P&L: ${worst['pnl']:>+13,.0f} ({worst['pnl_pct']:>+5.1f}%)                       ║
║                                                                      ║
║  Best case:  {best['name']:<25}                              ║
║              P&L: ${best['pnl']:>+13,.0f} ({best['pnl_pct']:>+5.1f}%)                       ║
║                                                                      ║
║  RISK INTERPRETATION                                                 ║
║  ───────────────────────────────────────────────────────────────────║"""
    
    if totals['delta'] > 0:
        report += f"""
║  → Book is LONG underlying (benefits from spot increase)             ║"""
    else:
        report += f"""
║  → Book is SHORT underlying (benefits from spot decrease)            ║"""
    
    if totals['vega'] > 0:
        report += f"""
║  → Book is LONG volatility (benefits from vol increase)              ║"""
    else:
        report += f"""
║  → Book is SHORT volatility (benefits from vol decrease)             ║"""
    
    report += f"""
║  → Daily theta decay: ${abs(totals['theta']):,.0f}                                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    
    print(report)
    
    return report


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Execute the complete daily workflow."""
    
    start_time = datetime.now()
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           OPTIONS DESK DAILY WORKFLOW - STARTING                     ║")
    print(f"║           {start_time.strftime('%Y-%m-%d %H:%M:%S'):^50}           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    # Execute workflow steps
    market, market_ids = step_1_load_market_data()
    portfolio = step_2_build_portfolio(market_ids)
    results, totals = step_3_price_portfolio(market, portfolio)
    scenario_results = step_4_run_scenarios(market, portfolio, market_ids, totals)
    report = step_5_generate_report(totals, scenario_results)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"  Start time:  {start_time.strftime('%H:%M:%S')}")
    print(f"  End time:    {end_time.strftime('%H:%M:%S')}")
    print(f"  Duration:    {duration:.2f} seconds")
    print(f"  Positions:   {len(portfolio)}")
    print(f"  Scenarios:   {len(SCENARIOS)}")
    print("="*70 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
