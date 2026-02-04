#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: portfolio.build_from_config
===============================================================================

This script demonstrates how to use the `portfolio.build_from_config` pipeline
to construct a portfolio from a YAML/JSON-style position specification.

What This Pipeline Does
-----------------------
1. Parses position specifications from configuration
2. Instantiates instrument objects (options, forwards, swaps)
3. Validates instrument parameters (strikes, expiries, notionals)
4. Creates Position objects with quantities and directions
5. Assembles positions into a Portfolio object

When to Use This Pipeline
-------------------------
- Loading portfolios from configuration files
- Building test portfolios for pricing/risk analysis
- Reconstructing portfolios from trade blotters
- Any scenario requiring programmatic portfolio construction

Portfolio Structure
-------------------
A portfolio consists of:
- Positions: Individual holdings with quantity and direction
- Instruments: The financial contracts (options, forwards, etc.)
- Metadata: Portfolio name, base currency, trade IDs

Run This Example
----------------
    python examples/pipelines/run_portfolio_from_config.py

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
    Build configuration for portfolio construction.
    
    We'll create a sample FX options portfolio that a typical
    options trading desk might hold.
    """
    
    # -------------------------------------------------------------------------
    # Define portfolio positions
    # -------------------------------------------------------------------------
    # Each position specifies:
    # - id: Unique identifier for the position
    # - instrument: The financial contract details
    # - quantity: Number of contracts (positive = long, negative = short)
    # - direction: "long" or "short" (alternative to signed quantity)
    
    positions = [
        # =====================================================================
        # Position 1: Long EURUSD Call (directional bullish bet)
        # =====================================================================
        {
            "id": "pos_001",
            "instrument": {
                "type": "FxVanillaOption",
                "underlying": "EURUSD",
                "strike": 1.10,              # Strike price
                "expiry": 0.25,              # 3 months to expiry
                "option_type": "call",       # Call option (right to buy EUR)
                "notional": 10_000_000,      # EUR 10 million notional
            },
            "quantity": 1,
            "direction": "long",
        },
        
        # =====================================================================
        # Position 2: Short EURUSD Put (selling downside protection)
        # =====================================================================
        {
            "id": "pos_002",
            "instrument": {
                "type": "FxVanillaOption",
                "underlying": "EURUSD",
                "strike": 1.05,              # Lower strike (OTM put)
                "expiry": 0.25,              # Same expiry as pos_001
                "option_type": "put",        # Put option (right to sell EUR)
                "notional": 10_000_000,
            },
            "quantity": 1,
            "direction": "short",            # We're selling this put
        },
        
        # =====================================================================
        # Position 3: Long EURUSD Call Spread (debit spread)
        # =====================================================================
        # This is the long leg of a call spread
        {
            "id": "pos_003",
            "instrument": {
                "type": "FxVanillaOption",
                "underlying": "EURUSD",
                "strike": 1.08,              # Lower strike (ITM)
                "expiry": 0.5,               # 6 months
                "option_type": "call",
                "notional": 5_000_000,
            },
            "quantity": 1,
            "direction": "long",
        },
        
        # Short leg of the call spread
        {
            "id": "pos_004",
            "instrument": {
                "type": "FxVanillaOption",
                "underlying": "EURUSD",
                "strike": 1.12,              # Higher strike (OTM)
                "expiry": 0.5,               # Same expiry
                "option_type": "call",
                "notional": 5_000_000,
            },
            "quantity": 1,
            "direction": "short",
        },
        
        # =====================================================================
        # Position 5: Long straddle (volatility play)
        # =====================================================================
        # Long ATM call
        {
            "id": "pos_005",
            "instrument": {
                "type": "FxVanillaOption",
                "underlying": "EURUSD",
                "strike": 1.085,             # ATM strike
                "expiry": 1.0,               # 1 year
                "option_type": "call",
                "notional": 2_000_000,
            },
            "quantity": 1,
            "direction": "long",
        },
        
        # Long ATM put (same strike, same expiry)
        {
            "id": "pos_006",
            "instrument": {
                "type": "FxVanillaOption",
                "underlying": "EURUSD",
                "strike": 1.085,             # Same ATM strike
                "expiry": 1.0,               # Same expiry
                "option_type": "put",
                "notional": 2_000_000,
            },
            "quantity": 1,
            "direction": "long",
        },
    ]
    
    # -------------------------------------------------------------------------
    # Build the RunConfig
    # -------------------------------------------------------------------------
    config = RunConfig(
        pipeline="portfolio.build_from_config",
        
        io={
            "artifacts_dir": "./artifacts/portfolio_example",
            "enable_save": True,
        },
        
        params={
            "portfolio": {
                # Portfolio metadata
                "name": "FX_Options_Demo_Book",
                "base_currency": "USD",
                
                # The position specifications
                "positions": positions,
            }
        },
    )
    
    return validate_run_config(config)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Execute the portfolio building pipeline and display results."""
    
    print("=" * 70)
    print("Pipeline Example: portfolio.build_from_config")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Build configuration
    # -------------------------------------------------------------------------
    print("[1/4] Building configuration...")
    cfg = build_config()
    print(f"      Pipeline: {cfg.pipeline}")
    print(f"      Portfolio name: FX_Options_Demo_Book")
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
    
    portfolio = ctx.state.get(Keys.PORTFOLIO)
    positions = ctx.state.get(Keys.POSITIONS, [])
    instruments = ctx.state.get(Keys.INSTRUMENTS, [])
    
    if portfolio is None:
        print("      WARNING: No portfolio produced.")
        return
    
    print(f"      Portfolio size: {len(portfolio)} positions")
    print(f"      Instruments: {len(instruments)}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Display the portfolio
    # -------------------------------------------------------------------------
    print("[4/4] Portfolio Summary")
    print("=" * 80)
    print()
    
    # Position details table
    print("Position Details:")
    print("-" * 80)
    print(f"{'ID':<12} {'Type':<12} {'Strike':>10} {'Expiry':>10} {'Notional':>15} {'Dir':>8}")
    print("-" * 80)
    
    total_notional_long = 0
    total_notional_short = 0
    
    for pos in portfolio:
        inst = pos.instrument
        
        # Extract instrument details
        opt_type = getattr(inst, 'option_type', 'N/A')
        strike = getattr(inst, 'strike', 0)
        expiry = getattr(inst, 'expiry', 0)
        notional = getattr(inst, 'notional', 0)
        
        direction = "LONG" if pos.quantity > 0 else "SHORT"
        
        if pos.quantity > 0:
            total_notional_long += notional
        else:
            total_notional_short += notional
        
        print(f"{pos.position_id:<12} {opt_type.upper():<12} {strike:>10.4f} {expiry:>10.2f}Y {notional:>15,.0f} {direction:>8}")
    
    print("-" * 80)
    print()
    
    # -------------------------------------------------------------------------
    # Portfolio analytics
    # -------------------------------------------------------------------------
    print("Portfolio Analytics:")
    print("-" * 50)
    print(f"  Total positions:         {len(portfolio)}")
    print(f"  Long positions:          {sum(1 for p in portfolio if p.quantity > 0)}")
    print(f"  Short positions:         {sum(1 for p in portfolio if p.quantity < 0)}")
    print(f"  Total long notional:     ${total_notional_long:,.0f}")
    print(f"  Total short notional:    ${total_notional_short:,.0f}")
    print(f"  Net notional:            ${total_notional_long - total_notional_short:,.0f}")
    print()
    
    # -------------------------------------------------------------------------
    # Strategy breakdown
    # -------------------------------------------------------------------------
    print("Strategy Breakdown:")
    print("-" * 50)
    print("  pos_001 + pos_002:       Risk Reversal (long call, short put)")
    print("  pos_003 + pos_004:       Call Spread (bullish, limited risk)")
    print("  pos_005 + pos_006:       Long Straddle (volatility play)")
    print()
    
    print("Artifacts saved to:", cfg.io.get("artifacts_dir", "N/A"))
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
