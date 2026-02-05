#!/usr/bin/env python3
"""
===============================================================================
Market IDs and Quotes: The Foundation of Market Data
===============================================================================

This example introduces the fundamental building blocks of QuantStrata's
market data system - the primitives that every other module builds upon.

Learning Objectives
-------------------
1. **MarketId**: Understand how market data is uniquely identified
2. **Quote**: Learn the container for scalar market values
3. **Market**: Build complete market snapshots for pricing

Production Context
------------------
At a hedge fund, market data flows from multiple sources (Bloomberg, Reuters,
internal systems). The MarketId system provides a canonical naming convention
that allows:
- Consistent identification across systems
- Type-safe dictionary lookups
- Clear audit trails for pricing inputs

Prerequisites
-------------
None - this is the first example in the fundamentals series.

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/fundamentals/01_market_ids_and_quotes.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations  # Enable modern type hints (PEP 604)

import logging
import sys
from pathlib import Path
from typing import Dict

# -----------------------------------------------------------------------------
# Path setup: Ensure imports work when running as script
# This is the CORRECT way to handle imports in example scripts
# (NOT sys.path.insert(0, '../..') which is fragile)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports
# Core market data primitives that form the foundation of the library
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId          # Unique market data identifiers
from src.marketdata.core.interfaces import Quote      # Scalar value container
from src.marketdata.core.market import Market         # Point-in-time market snapshot

# =============================================================================
# LOGGING SETUP
# =============================================================================

# Configure structured logging (production standard - not print statements)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: MarketId - Uniquely Identifying Market Data
# =============================================================================

def demonstrate_market_ids() -> Dict[str, MarketId]:
    """
    Demonstrate the MarketId system for uniquely identifying market data.
    
    MarketId is a frozen (immutable) dataclass with three components:
    - asset_class: Broad category (FX, IR, EQ, CR, CO)
    - mkt_type: Data type (SPOT, VOL, CURVE, FWD)
    - name: Specific identifier (EURUSD, USD_OIS, SPX)
    
    Returns
    -------
    Dict[str, MarketId]
        Dictionary of created MarketIds for use in later examples.
    
    Production Notes
    ----------------
    - MarketIds are the canonical identifiers used throughout the system
    - They are hashable and can be used as dictionary keys
    - The key() method returns a string for logging/debugging
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: MarketId - Uniquely Identifying Market Data")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create FX Spot MarketIds
    # Convention: asset_class="FX", mkt_type="SPOT", name="{CCY1}{CCY2}"
    # -------------------------------------------------------------------------
    eurusd_spot_id = MarketId(
        asset_class="FX",      # Foreign Exchange asset class
        mkt_type="SPOT",       # Spot price (not forward, not vol)
        name="EURUSD",         # Euro vs US Dollar (standard FX pair naming)
    )
    
    gbpusd_spot_id = MarketId(
        asset_class="FX",
        mkt_type="SPOT", 
        name="GBPUSD",         # British Pound vs US Dollar
    )
    
    usdjpy_spot_id = MarketId(
        asset_class="FX",
        mkt_type="SPOT",
        name="USDJPY",         # US Dollar vs Japanese Yen
    )
    
    logger.info("Created FX Spot MarketIds:")
    logger.info(f"  EUR/USD: {eurusd_spot_id}")
    logger.info(f"  GBP/USD: {gbpusd_spot_id}")
    logger.info(f"  USD/JPY: {usdjpy_spot_id}")
    
    # -------------------------------------------------------------------------
    # The key() method returns a unique string representation
    # This is useful for logging, debugging, and database storage
    # -------------------------------------------------------------------------
    logger.info("MarketId keys (string representation):")
    logger.info(f"  EUR/USD key: '{eurusd_spot_id.key()}'")
    logger.info(f"  GBP/USD key: '{gbpusd_spot_id.key()}'")
    
    # -------------------------------------------------------------------------
    # MarketIds are hashable - can be used as dictionary keys
    # This is essential for the Market class which stores data by MarketId
    # -------------------------------------------------------------------------
    market_id_dict = {
        eurusd_spot_id: "Euro vs US Dollar",
        gbpusd_spot_id: "British Pound vs US Dollar",
    }
    logger.info(f"Using MarketId as dict key: {market_id_dict[eurusd_spot_id]}")
    
    # -------------------------------------------------------------------------
    # Create Volatility Surface MarketIds
    # Convention: asset_class="FX", mkt_type="VOL", name="{CCY1}{CCY2}"
    # -------------------------------------------------------------------------
    eurusd_vol_id = MarketId(
        asset_class="FX",
        mkt_type="VOL",        # Volatility surface
        name="EURUSD",
    )
    logger.info(f"Volatility surface MarketId: {eurusd_vol_id}")
    
    # -------------------------------------------------------------------------
    # Create Interest Rate Curve MarketIds
    # Convention: asset_class="IR", mkt_type="CURVE", name="{CCY}_{TYPE}"
    # Common types: OIS (overnight), LIBOR_3M, SOFR, etc.
    # -------------------------------------------------------------------------
    usd_curve_id = MarketId(
        asset_class="IR",      # Interest Rate asset class
        mkt_type="CURVE",      # Discount/projection curve
        name="USD_OIS",        # USD Overnight Index Swap curve
    )
    
    eur_curve_id = MarketId(
        asset_class="IR",
        mkt_type="CURVE",
        name="EUR_OIS",        # EUR Overnight Index Swap curve
    )
    
    logger.info("Created IR Curve MarketIds:")
    logger.info(f"  USD OIS: {usd_curve_id}")
    logger.info(f"  EUR OIS: {eur_curve_id}")
    
    # Return all created IDs for use in later sections
    return {
        "eurusd_spot": eurusd_spot_id,
        "gbpusd_spot": gbpusd_spot_id,
        "usdjpy_spot": usdjpy_spot_id,
        "eurusd_vol": eurusd_vol_id,
        "usd_curve": usd_curve_id,
        "eur_curve": eur_curve_id,
    }


# =============================================================================
# SECTION 2: Quote - Simple Value Containers
# =============================================================================

def demonstrate_quotes(market_ids: Dict[str, MarketId]) -> Dict[MarketId, Quote]:
    """
    Demonstrate the Quote class for holding scalar market values.
    
    Quote is a frozen (immutable) dataclass containing a single float value.
    It's used for:
    - Spot prices (FX rates, equity prices)
    - Single rates (overnight rates, repo rates)
    - Any scalar market observable
    
    Parameters
    ----------
    market_ids : Dict[str, MarketId]
        Dictionary of MarketIds from Section 1.
    
    Returns
    -------
    Dict[MarketId, Quote]
        Dictionary mapping MarketIds to their Quote values.
    
    Production Notes
    ----------------
    - Quotes are immutable to prevent accidental modification during pricing
    - The value is always a float (not Decimal) for performance
    - For time series data, see Panel and MarketDataset in later examples
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Quote - Simple Value Containers")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create Quote objects for FX spot rates
    # These represent the current market prices
    # -------------------------------------------------------------------------
    eurusd_quote = Quote(value=1.0850)  # 1 EUR = 1.0850 USD
    gbpusd_quote = Quote(value=1.2650)  # 1 GBP = 1.2650 USD
    usdjpy_quote = Quote(value=149.50)  # 1 USD = 149.50 JPY
    
    logger.info("Created FX Spot Quotes:")
    logger.info(f"  EUR/USD: {eurusd_quote.value:.4f}")
    logger.info(f"  GBP/USD: {gbpusd_quote.value:.4f}")
    logger.info(f"  USD/JPY: {usdjpy_quote.value:.2f}")
    
    # -------------------------------------------------------------------------
    # Quotes are IMMUTABLE (frozen dataclass)
    # This is a safety feature - once market data is loaded, it cannot be
    # accidentally modified during pricing calculations
    # -------------------------------------------------------------------------
    try:
        eurusd_quote.value = 1.0900  # type: ignore  # Attempting to modify
    except AttributeError as e:
        # This is EXPECTED behavior - quotes are frozen
        logger.info(f"Quotes are immutable: {type(e).__name__} raised as expected")
    
    # -------------------------------------------------------------------------
    # Build a quotes dictionary mapping MarketId -> Quote
    # This is the format expected by the Market class
    # -------------------------------------------------------------------------
    quotes = {
        market_ids["eurusd_spot"]: eurusd_quote,
        market_ids["gbpusd_spot"]: gbpusd_quote,
        market_ids["usdjpy_spot"]: usdjpy_quote,
    }
    
    return quotes


# =============================================================================
# SECTION 3: Building a Market Snapshot
# =============================================================================

def demonstrate_market_snapshot(
    market_ids: Dict[str, MarketId],
    quotes: Dict[MarketId, Quote],
) -> Market:
    """
    Demonstrate building a complete Market snapshot.
    
    A Market is a point-in-time snapshot of all market data needed for pricing.
    It contains:
    - asof: Valuation date (ISO format string)
    - quotes: Scalar values (spots, rates)
    - curves: Term structures (discount curves, projection curves)
    - vols: Volatility surfaces (FX, equity, IR)
    - meta: Optional metadata dictionary
    
    Parameters
    ----------
    market_ids : Dict[str, MarketId]
        Dictionary of MarketIds.
    quotes : Dict[MarketId, Quote]
        Dictionary of Quote values.
    
    Returns
    -------
    Market
        Complete market snapshot.
    
    Production Notes
    ----------------
    - In production, Market objects are built by market data providers
    - The asof date is critical for time-dependent calculations
    - Empty curves/vols dicts are valid for simple spot-only pricing
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Building a Market Snapshot")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create a Market snapshot
    # This is the primary object passed to pricers
    # -------------------------------------------------------------------------
    market = Market(
        asof="2026-01-28",     # Valuation date (ISO format: YYYY-MM-DD)
        quotes=quotes,         # Dictionary of scalar values
        curves={},             # Empty for now - covered in example 02
        vols={},               # Empty for now - covered in example 03
        meta={                 # Optional metadata for audit/debugging
            "source": "Example data",
            "environment": "development",
        },
    )
    
    logger.info(f"Market snapshot created:")
    logger.info(f"  As-of date: {market.asof}")
    logger.info(f"  Number of quotes: {len(market.quotes)}")
    logger.info(f"  Number of curves: {len(market.curves)}")
    logger.info(f"  Number of vols: {len(market.vols)}")
    logger.info(f"  Metadata: {market.meta}")
    
    # -------------------------------------------------------------------------
    # Access quotes via the Market.quote() method
    # This is the standard way to retrieve market data during pricing
    # -------------------------------------------------------------------------
    logger.info("Accessing quotes from Market:")
    logger.info(f"  EUR/USD spot: {market.quote(market_ids['eurusd_spot']):.4f}")
    logger.info(f"  GBP/USD spot: {market.quote(market_ids['gbpusd_spot']):.4f}")
    logger.info(f"  USD/JPY spot: {market.quote(market_ids['usdjpy_spot']):.2f}")
    
    return market


# =============================================================================
# SECTION 4: MarketId Naming Conventions
# =============================================================================

def demonstrate_naming_conventions() -> None:
    """
    Document the QuantStrata MarketId naming conventions.
    
    Consistent naming is critical in a production environment where:
    - Multiple systems need to reference the same data
    - Audit trails must be traceable
    - Mappings between external and internal IDs must be maintained
    
    Production Notes
    ----------------
    - These conventions are enforced throughout the library
    - External data (Bloomberg, Reuters) is mapped to these IDs
    - The MarketId.parse() method can create IDs from strings
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: MarketId Naming Conventions")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Document the standard naming conventions used throughout QuantStrata
    # -------------------------------------------------------------------------
    conventions = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    MARKETID NAMING CONVENTIONS                       │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  ASSET CLASSES (asset_class parameter):                             │
    │  ─────────────────────────────────────                               │
    │    FX  - Foreign Exchange                                           │
    │    IR  - Interest Rates                                             │
    │    EQ  - Equities                                                   │
    │    CR  - Credit                                                     │
    │    CO  - Commodities                                                │
    │                                                                      │
    │  DATA TYPES (mkt_type parameter):                                   │
    │  ────────────────────────────────                                    │
    │    SPOT   - Spot prices/rates                                       │
    │    VOL    - Volatility surfaces                                     │
    │    CURVE  - Discount/projection curves                              │
    │    FWD    - Forward prices/rates                                    │
    │                                                                      │
    │  FX NAMING (name parameter):                                        │
    │  ──────────────────────────                                          │
    │    EURUSD - EUR/USD (Euro per US Dollar)                            │
    │    GBPUSD - GBP/USD (Pounds per US Dollar)                          │
    │    USDJPY - USD/JPY (Dollar per Yen)                                │
    │                                                                      │
    │  IR CURVE NAMING (name parameter):                                  │
    │  ─────────────────────────────────                                   │
    │    USD_OIS   - USD Overnight Index Swap curve                       │
    │    EUR_OIS   - EUR Overnight Index Swap curve                       │
    │    USD_SOFR  - USD SOFR curve                                       │
    │    GBP_SONIA - GBP SONIA curve                                      │
    │                                                                      │
    │  EXAMPLES:                                                           │
    │  ─────────                                                           │
    │    MarketId("FX", "SPOT", "EURUSD")   → FX.SPOT.EURUSD              │
    │    MarketId("FX", "VOL", "EURUSD")    → FX.VOL.EURUSD               │
    │    MarketId("IR", "CURVE", "USD_OIS") → IR.CURVE.USD_OIS            │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(conventions)
    
    # -------------------------------------------------------------------------
    # Demonstrate MarketId.parse() for creating IDs from strings
    # This is useful when loading configuration from files
    # -------------------------------------------------------------------------
    parsed_id = MarketId.parse("FX.SPOT.EURUSD")
    logger.info(f"Parsed from string 'FX.SPOT.EURUSD': {parsed_id}")
    logger.info(f"  asset_class: {parsed_id.asset_class}")
    logger.info(f"  mkt_type: {parsed_id.mkt_type}")
    logger.info(f"  name: {parsed_id.name}")


# =============================================================================
# SECTION 5: Practical Example - Multi-Currency Market
# =============================================================================

def demonstrate_multicurrency_market() -> Market:
    """
    Build a realistic multi-currency FX market.
    
    This demonstrates how a production system would set up market data
    for a multi-currency FX trading desk.
    
    Returns
    -------
    Market
        Multi-currency FX market snapshot.
    
    Production Notes
    ----------------
    - In production, spot rates would come from a real-time feed
    - The meta dictionary would contain feed timestamps, source info
    - Validation would check for stale or missing data
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Practical Example - Multi-Currency Market")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Define all G10 FX pairs (vs USD)
    # In production, this would be configured externally
    # -------------------------------------------------------------------------
    fx_pairs = {
        "EURUSD": MarketId("FX", "SPOT", "EURUSD"),
        "GBPUSD": MarketId("FX", "SPOT", "GBPUSD"),
        "USDJPY": MarketId("FX", "SPOT", "USDJPY"),
        "AUDUSD": MarketId("FX", "SPOT", "AUDUSD"),
        "USDCAD": MarketId("FX", "SPOT", "USDCAD"),
        "USDCHF": MarketId("FX", "SPOT", "USDCHF"),
        "NZDUSD": MarketId("FX", "SPOT", "NZDUSD"),
        "USDSEK": MarketId("FX", "SPOT", "USDSEK"),
        "USDNOK": MarketId("FX", "SPOT", "USDNOK"),
    }
    
    # -------------------------------------------------------------------------
    # Current spot rates (illustrative values)
    # In production, these would come from a market data provider
    # -------------------------------------------------------------------------
    spot_rates = {
        "EURUSD": 1.0850,   # 1 EUR = 1.0850 USD
        "GBPUSD": 1.2650,   # 1 GBP = 1.2650 USD
        "USDJPY": 149.50,   # 1 USD = 149.50 JPY
        "AUDUSD": 0.6550,   # 1 AUD = 0.6550 USD
        "USDCAD": 1.3450,   # 1 USD = 1.3450 CAD
        "USDCHF": 0.8850,   # 1 USD = 0.8850 CHF
        "NZDUSD": 0.6150,   # 1 NZD = 0.6150 USD
        "USDSEK": 10.450,   # 1 USD = 10.450 SEK
        "USDNOK": 10.850,   # 1 USD = 10.850 NOK
    }
    
    # -------------------------------------------------------------------------
    # Build quotes dictionary using dictionary comprehension
    # This is the idiomatic Python pattern for this transformation
    # -------------------------------------------------------------------------
    quotes = {
        fx_pairs[pair]: Quote(value=rate)
        for pair, rate in spot_rates.items()
    }
    
    # -------------------------------------------------------------------------
    # Create the multi-currency market
    # -------------------------------------------------------------------------
    fx_market = Market(
        asof="2026-01-28",
        quotes=quotes,
        curves={},  # Curves covered in example 02
        vols={},    # Vols covered in example 03
        meta={
            "source": "Example data",
            "ccy_base": "USD",
            "n_pairs": len(fx_pairs),
        },
    )
    
    logger.info(f"Multi-currency FX market created:")
    logger.info(f"  As-of date: {fx_market.asof}")
    logger.info(f"  Number of pairs: {fx_market.meta['n_pairs']}")
    logger.info("")
    logger.info("Spot rates:")
    for pair, mid in fx_pairs.items():
        rate = fx_market.quote(mid)
        logger.info(f"  {pair}: {rate:>10.4f}")
    
    return fx_market


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
    │  1. MarketId uniquely identifies market data points:                │
    │     - asset_class: FX, IR, EQ, CR, CO                               │
    │     - mkt_type: SPOT, VOL, CURVE, FWD                               │
    │     - name: EURUSD, USD_OIS, SPX, etc.                              │
    │                                                                      │
    │  2. Quote is an immutable container for scalar values               │
    │     - Used for spot prices, single rates, etc.                      │
    │     - Immutability prevents accidental modification                 │
    │                                                                      │
    │  3. Market is a point-in-time snapshot combining:                   │
    │     - quotes (scalar values)                                        │
    │     - curves (term structures) - see example 02                     │
    │     - vols (volatility surfaces) - see example 03                   │
    │                                                                      │
    │  4. All objects are immutable (frozen dataclasses)                  │
    │     - Thread-safe by design                                         │
    │     - Safe for parallel pricing                                     │
    │                                                                      │
    │  NEXT: See 02_curves_and_term_structures.py for discount curves     │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Main entry point for the example.
    
    Executes all sections in order, demonstrating the complete
    market data foundation.
    """
    try:
        # Section 1: MarketId
        market_ids = demonstrate_market_ids()
        
        # Section 2: Quote
        quotes = demonstrate_quotes(market_ids)
        
        # Section 3: Market Snapshot
        _ = demonstrate_market_snapshot(market_ids, quotes)
        
        # Section 4: Naming Conventions
        demonstrate_naming_conventions()
        
        # Section 5: Multi-Currency Market
        _ = demonstrate_multicurrency_market()
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
