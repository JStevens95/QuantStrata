#!/usr/bin/env python3
"""
Market IDs and Quotes: The Foundation of Market Data

This example introduces the fundamental building blocks of QuantStrata's
market data system:
- MarketId: Unique identifiers for market data points
- Quote: Simple value containers for spot prices, rates, etc.
- Market: A snapshot combining quotes, curves, and vol surfaces

Understanding these primitives is essential before working with
more complex pricing and risk functionality.

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market

# =============================================================================
# 1. MarketId: Uniquely Identifying Market Data
# =============================================================================

print("=" * 70)
print("1. MarketId: Uniquely Identifying Market Data")
print("=" * 70)

# MarketId is a frozen dataclass that uniquely identifies a piece of market data.
# It consists of three components:
#   - asset_class: The broad category (e.g., "FX", "IR", "EQ")
#   - data_type: What kind of data (e.g., "SPOT", "VOL", "CURVE")
#   - name: A specific identifier (e.g., "EURUSD", "USD_LIBOR_3M")

# Create MarketIds for FX spot rates
eurusd_spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
gbpusd_spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="GBPUSD")
usdjpy_spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="USDJPY")

print(f"\nFX Spot MarketIds:")
print(f"  EUR/USD: {eurusd_spot_id}")
print(f"  GBP/USD: {gbpusd_spot_id}")
print(f"  USD/JPY: {usdjpy_spot_id}")

# The key() method returns a unique string representation
print(f"\nMarketId keys (for dictionary lookups):")
print(f"  EUR/USD key: '{eurusd_spot_id.key()}'")
print(f"  GBP/USD key: '{gbpusd_spot_id.key()}'")

# MarketIds are hashable and can be used as dictionary keys
market_id_dict = {
    eurusd_spot_id: "Euro vs US Dollar",
    gbpusd_spot_id: "British Pound vs US Dollar",
}
print(f"\nUsing MarketId as dict key: {market_id_dict[eurusd_spot_id]}")

# Create MarketIds for volatility surfaces
eurusd_vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
print(f"\nVolatility surface MarketId: {eurusd_vol_id}")

# Create MarketIds for discount curves
usd_curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
eur_curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
print(f"\nCurve MarketIds:")
print(f"  USD OIS: {usd_curve_id}")
print(f"  EUR OIS: {eur_curve_id}")

# =============================================================================
# 2. Quote: Simple Value Containers
# =============================================================================

print("\n" + "=" * 70)
print("2. Quote: Simple Value Containers")
print("=" * 70)

# Quote is a simple frozen dataclass holding a single float value.
# It's used for spot prices, single rates, and other scalar market data.

eurusd_quote = Quote(value=1.0850)
gbpusd_quote = Quote(value=1.2650)
usdjpy_quote = Quote(value=149.50)

print(f"\nFX Spot Quotes:")
print(f"  EUR/USD: {eurusd_quote.value}")
print(f"  GBP/USD: {gbpusd_quote.value}")
print(f"  USD/JPY: {usdjpy_quote.value}")

# Quotes are immutable (frozen dataclass)
try:
    eurusd_quote.value = 1.0900  # This will raise an error
except Exception as e:
    print(f"\nQuotes are immutable: {type(e).__name__}")

# =============================================================================
# 3. Building a Market Snapshot
# =============================================================================

print("\n" + "=" * 70)
print("3. Building a Market Snapshot")
print("=" * 70)

# A Market is a point-in-time snapshot of all market data needed for pricing.
# It contains:
#   - asof: The valuation date (ISO format string)
#   - quotes: Dictionary mapping MarketId -> Quote
#   - curves: Dictionary mapping MarketId -> Curve (discount curves)
#   - vols: Dictionary mapping MarketId -> VolSurface (volatility surfaces)

# For this basic example, we'll create a market with just quotes
# (curves and vols will be covered in later examples)

quotes_dict = {
    eurusd_spot_id: eurusd_quote,
    gbpusd_spot_id: gbpusd_quote,
    usdjpy_spot_id: usdjpy_quote,
}

# Create a Market snapshot
market = Market(
    asof="2026-01-28",
    quotes=quotes_dict,
    curves={},  # Empty for now - covered in example 02
    vols={},    # Empty for now - covered in example 03
)

print(f"\nMarket snapshot created:")
print(f"  As-of date: {market.asof}")
print(f"  Number of quotes: {len(market.quotes)}")

# Access quotes via the market
print(f"\nAccessing quotes from Market:")
print(f"  EUR/USD spot: {market.quote(eurusd_spot_id)}")
print(f"  GBP/USD spot: {market.quote(gbpusd_spot_id)}")
print(f"  USD/JPY spot: {market.quote(usdjpy_spot_id)}")

# =============================================================================
# 4. MarketId Conventions
# =============================================================================

print("\n" + "=" * 70)
print("4. MarketId Conventions in QuantStrata")
print("=" * 70)

# QuantStrata follows these naming conventions:

print("""
Asset Class Conventions:
  FX  - Foreign Exchange
  IR  - Interest Rates
  EQ  - Equities
  CR  - Credit
  CO  - Commodities

Data Type Conventions:
  SPOT   - Spot prices/rates
  VOL    - Volatility surfaces
  CURVE  - Discount/projection curves
  FWD    - Forward prices/rates

Name Conventions (FX):
  EURUSD - EUR/USD (Euro per US Dollar)
  GBPUSD - GBP/USD (Pounds per US Dollar)
  USDJPY - USD/JPY (Dollar per Yen)

Name Conventions (Curves):
  USD_OIS  - USD Overnight Index Swap curve
  EUR_OIS  - EUR Overnight Index Swap curve
  USD_3M   - USD 3-month LIBOR curve
""")

# =============================================================================
# 5. Practical Example: Multi-Currency Market Setup
# =============================================================================

print("=" * 70)
print("5. Practical Example: Multi-Currency Market Setup")
print("=" * 70)

# Set up a realistic multi-currency FX market

# Define all market IDs
fx_pairs = {
    "EURUSD": MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
    "GBPUSD": MarketId(asset_class="FX", mkt_type="SPOT", name="GBPUSD"),
    "USDJPY": MarketId(asset_class="FX", mkt_type="SPOT", name="USDJPY"),
    "AUDUSD": MarketId(asset_class="FX", mkt_type="SPOT", name="AUDUSD"),
    "USDCAD": MarketId(asset_class="FX", mkt_type="SPOT", name="USDCAD"),
    "USDCHF": MarketId(asset_class="FX", mkt_type="SPOT", name="USDCHF"),
}

# Current spot rates (illustrative)
spot_rates = {
    "EURUSD": 1.0850,
    "GBPUSD": 1.2650,
    "USDJPY": 149.50,
    "AUDUSD": 0.6550,
    "USDCAD": 1.3450,
    "USDCHF": 0.8850,
}

# Build quotes dictionary
quotes = {
    fx_pairs[pair]: Quote(value=rate)
    for pair, rate in spot_rates.items()
}

# Create comprehensive market
fx_market = Market(
    asof="2026-01-28",
    quotes=quotes,
    curves={},
    vols={},
    meta={"source": "Example data", "ccy_base": "USD"},
)

print(f"\nMulti-currency FX market created:")
print(f"  As-of date: {fx_market.asof}")
print(f"  Metadata: {fx_market.meta}")
print(f"\nSpot rates:")
for pair, mid in fx_pairs.items():
    print(f"  {pair}: {fx_market.quote(mid):.4f}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. MarketId uniquely identifies market data points using:
   - asset_class (FX, IR, EQ, etc.)
   - data_type (SPOT, VOL, CURVE, etc.)
   - name (EURUSD, USD_OIS, etc.)

2. Quote is a simple container for scalar values (spot prices, rates).

3. Market is a point-in-time snapshot combining:
   - quotes (scalar values)
   - curves (term structures)
   - vols (volatility surfaces)

4. All objects are immutable (frozen dataclasses) for safety.

Next: See 02_curves_and_term_structures.py for discount curves.
""")
