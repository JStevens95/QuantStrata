# Futures Options

**Options on Futures Contracts Across Asset Classes**

This document covers futures options - European options where the underlying is a futures contract. These instruments are priced using the Black76 model.

---

## Overview

A **futures option** is an option to buy or sell a futures contract at a predetermined price (strike). Upon exercise, the holder receives a position in the underlying futures contract.

### When to Use Futures Options

| Scenario | Use Futures Option |
|----------|-------------------|
| Listed exchange products | E-mini S&P 500, crude oil options |
| Liquid volatility markets | Options on liquid futures |
| No dividend/carry modeling | Futures embed all carry costs |
| Leverage and margin efficiency | Futures-based hedging |

### Key Characteristics

1. **Settlement:** Delivery into a futures position (not the underlying asset)
2. **Margin:** Subject to futures margin requirements
3. **Pricing:** Black76 model (forward-based)
4. **Volatility:** Typically higher liquidity than OTC options

---

## Mathematical Framework

### Futures Price

The futures price $F$ for delivery at time $T$ is determined by cost-of-carry:

**Equity Index Futures:**
$$F = S \cdot e^{(r - q) \cdot T}$$

Where:
- $S$ = spot index level
- $r$ = risk-free rate
- $q$ = continuous dividend yield

**Commodity Futures:**
$$F = S \cdot e^{(r + u - y) \cdot T}$$

Where:
- $u$ = storage cost rate
- $y$ = convenience yield

### Black76 Pricing

Futures options are priced using the Black76 model:

$$C = DF \cdot [F \cdot N(d_1) - K \cdot N(d_2)]$$
$$P = DF \cdot [K \cdot N(-d_2) - F \cdot N(-d_1)]$$

Where:
- $DF = e^{-r \cdot T_{opt}}$ = discount factor to option expiry
- $F$ = futures price
- $K$ = strike price
- $d_1 = \frac{\ln(F/K) + \sigma^2 T / 2}{\sigma \sqrt{T}}$
- $d_2 = d_1 - \sigma \sqrt{T}$

### Put-Call Parity

For Black76:
$$C - P = DF \cdot (F - K)$$

---

## Implemented Asset Classes

### Equity Index Futures Options

**Instrument:** `EuropeanEquityFuturesOption`

**Location:** `src/instruments/equity/options/futures.py`

**Pricer:** `EquityFuturesOptionBlack76Pricer`

**Location:** `src/pricers/equity/european_b76.py`

#### Usage Example

```python
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.instruments.equity.options.futures import EuropeanEquityFuturesOption
from src.pricers.equity.european_b76 import EquityFuturesOptionBlack76Pricer

# Define MarketIds
spot_id = MarketId("EQUITY", "SPOT", "SPX", (("ccy", "USD"),))
vol_id = MarketId("EQUITY", "VOL", "SPX", (("ccy", "USD"),))
curve_id = MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),))

# Create market
market = Market(
    asof="2026-01-15",
    quotes={spot_id: Quote(value=5000.0)},
    curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05)},
    vols={vol_id: FlatVolSurface(sigma=0.18)},
)

# Create option (E-mini S&P 500 style)
option = EuropeanEquityFuturesOption(
    ticker="SPX",
    option_type="call",
    strike=5200.0,           # Strike in index points
    expiry=0.25,             # 3-month option
    futures_expiry=0.25,     # Same as option expiry
    notional=50.0,           # E-mini multiplier ($50 per point)
    spot_id=spot_id,
    vol_id=vol_id,
    curve_id=curve_id,
    dividend_yield=0.015,    # 1.5% dividend yield
)

# Price
pricer = EquityFuturesOptionBlack76Pricer()
pv = pricer.price(option, market)
greeks = pricer.greeks(option, market)

print(f"PV: ${pv:,.2f}")
print(f"Futures Delta: {greeks['delta_futures']:,.2f}")
print(f"Spot Delta: {greeks['delta_spot']:,.2f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Vega: ${greeks['vega']:,.2f}")
```

#### Greeks

| Greek | Description |
|-------|-------------|
| `delta_futures` | Sensitivity to futures price (dPV/dF) |
| `delta_spot` | Sensitivity to spot index (dPV/dS) |
| `gamma` | Second derivative w.r.t. futures price |
| `vega` | Sensitivity to volatility |
| `theta` | Time decay |
| `rho` | Sensitivity to risk-free rate |

#### Spot Delta vs Futures Delta

The relationship between spot and futures delta:

$$\Delta_{spot} = \Delta_{futures} \cdot e^{(r - q) \cdot T}$$

For ATM options, this ratio is typically close to 1 for short-dated options.

---

## Common Products

### Equity Index Futures Options

| Product | Underlying | Multiplier | Exchange |
|---------|-----------|------------|----------|
| ES Options | E-mini S&P 500 | $50 | CME |
| NQ Options | E-mini NASDAQ-100 | $20 | CME |
| RTY Options | E-mini Russell 2000 | $50 | CME |
| YM Options | E-mini Dow | $5 | CBOT |

### Interest Rate Futures Options (Planned)

| Product | Underlying | Exchange |
|---------|-----------|----------|
| ED Options | Eurodollar Futures | CME |
| SR3 Options | SOFR Futures | CME |
| Treasury Options | Treasury Futures | CBOT |

### Commodity Futures Options (Future Phase)

| Product | Underlying | Exchange |
|---------|-----------|----------|
| CL Options | Crude Oil Futures | NYMEX |
| GC Options | Gold Futures | COMEX |
| NG Options | Natural Gas Futures | NYMEX |

---

## Planned Asset Classes

### Interest Rate Futures Options

**Status:** Phase 3 (Planned)

Options on interest rate futures (SOFR, Treasury) will use Black76 with the futures price as the underlying.

### Commodity Futures Options

**Status:** Future Phase

Options on commodity futures will follow the same Black76 framework with appropriate convenience yield modeling.

---

## Key Considerations

### 1. Futures Expiry vs Option Expiry

- **Monthly options:** Option and futures expire together
- **Weekly options:** Option expires before underlying futures
- **Serial options:** Option on a futures with different expiry month

### 2. Settlement

| Settlement Type | Description |
|-----------------|-------------|
| **Physical** | Delivery into futures position |
| **Cash** | Cash settlement at expiry (common for index options) |

### 3. Margin

Futures options are subject to:
- **Premium margin:** Option premium paid upfront (buyer)
- **Span margin:** Portfolio-based margin for sellers
- **Variation margin:** Daily mark-to-market

### 4. Volatility Smile

Equity index futures options typically exhibit:
- **Negative skew:** OTM puts more expensive than OTM calls
- **Term structure:** Vol varies by expiry
- **Smile dynamics:** Skew flattens/steepens with market conditions

---

## Simple Pricer

For cases where the futures price is directly observable:

```python
from src.instruments.equity.options.futures import EuropeanEquityFuturesOptionSimple
from src.pricers.equity.european_b76 import EquityFuturesOptionBlack76PricerSimple

# Direct pricing with known futures price
option = EuropeanEquityFuturesOptionSimple(
    ticker="ES",
    option_type="call",
    strike=5200.0,
    expiry=0.25,
    futures_price=5178.5,    # Observable futures price
    vol=0.18,                # Implied vol
    discount_factor=0.9875,  # exp(-r*T)
    notional=50.0,
)

pricer = EquityFuturesOptionBlack76PricerSimple()
pv = pricer.price(option)
greeks = pricer.greeks(option)
```

---

## Related Documentation

- [Black76 Model](../../reference/models/black76.md) - Mathematical details
- [Forward Options](forward_options.md) - OTC forward options
- [Volatility Surfaces](../market-data/volatility_surfaces.md) - Vol surface conventions

---

*Document Version: 1.0 | QuantStrata Phase 3.1 | January 2026*
