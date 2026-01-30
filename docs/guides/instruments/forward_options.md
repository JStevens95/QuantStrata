# Forward Options

**Options on Forward Contracts Across Asset Classes**

This document covers forward options - European options where the underlying is a forward contract rather than the spot price. These instruments are priced using the Black76 model.

---

## Overview

A **forward option** is an option to buy or sell an asset at a predetermined forward rate. The key difference from a spot option is that the underlying is the forward price, not the current spot price.

### When to Use Forward Options

| Scenario | Use Forward Option |
|----------|-------------------|
| FX deferred settlement | Option expiry differs from settlement date |
| Forward vol quoted directly | Market quotes forward-starting volatility |
| Hedging forward exposures | Underlying exposure is a forward contract |
| Structured products | Forward-starting features |

### Relationship to Spot Options

For standard European options where option expiry equals forward delivery:
- Forward option and spot option have the **same price** (no-arbitrage)
- **Different deltas**: forward delta vs spot delta

The key distinction is the **Greek interpretation**:
- Spot option delta: sensitivity to spot price
- Forward option delta: sensitivity to forward price

---

## Mathematical Framework

### Forward Price

The forward price $F$ for delivery at time $T$ is determined by no-arbitrage:

**FX Forward:**
$$F = S \cdot e^{(r_d - r_f) \cdot T}$$

Where:
- $S$ = spot FX rate (domestic per foreign)
- $r_d$ = domestic risk-free rate
- $r_f$ = foreign risk-free rate

**Equity Forward:**
$$F = S \cdot e^{(r - q) \cdot T}$$

Where:
- $S$ = spot index/stock price
- $r$ = risk-free rate
- $q$ = continuous dividend yield

### Black76 Pricing

Forward options are priced using the Black76 model:

$$C = DF \cdot [F \cdot N(d_1) - K \cdot N(d_2)]$$
$$P = DF \cdot [K \cdot N(-d_2) - F \cdot N(-d_1)]$$

Where:
- $DF = e^{-r \cdot T_{opt}}$ = discount factor to option expiry
- $d_1 = \frac{\ln(F/K) + \sigma^2 T / 2}{\sigma \sqrt{T}}$
- $d_2 = d_1 - \sigma \sqrt{T}$

### Put-Call Parity

For Black76:
$$C - P = DF \cdot (F - K)$$

---

## Implemented Asset Classes

### FX Forward Options

**Instrument:** `EuropeanFxForwardOption`

**Location:** `src/instruments/fx/options/forward.py`

**Pricer:** `FxForwardOptionBlack76Pricer`

**Location:** `src/pricers/fx/european_b76.py`

#### Usage Example

```python
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.instruments.fx.options.forward import EuropeanFxForwardOption
from src.pricers.fx.european_b76 import FxForwardOptionBlack76Pricer

# Define MarketIds
spot_id = MarketId("FX", "SPOT", "EURUSD", (("dom", "USD"), ("for", "EUR")))
vol_id = MarketId("FX", "VOL", "EURUSD", (("dom", "USD"), ("for", "EUR")))
usd_curve_id = MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),))
eur_curve_id = MarketId("IR", "CURVE", "EUR.OIS", (("ccy", "EUR"),))

# Create market
market = Market(
    asof="2026-01-15",
    quotes={spot_id: Quote(value=1.10)},
    curves={
        usd_curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05),
        eur_curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.03),
    },
    vols={vol_id: FlatVolSurface(sigma=0.08)},
)

# Create option
option = EuropeanFxForwardOption(
    option_type="call",
    notional=1_000_000,      # EUR 1M
    strike=1.12,             # Strike forward rate
    expiry=1.0,              # 1 year option
    forward_expiry=1.0,      # Same as option expiry
    spot_id=spot_id,
    vol_id=vol_id,
    domestic_curve_id=usd_curve_id,
    foreign_curve_id=eur_curve_id,
)

# Price
pricer = FxForwardOptionBlack76Pricer()
pv = pricer.price(option, market)
greeks = pricer.greeks(option, market)

print(f"PV: ${pv:,.2f}")
print(f"Forward Delta: {greeks['delta_forward']:,.0f}")
print(f"Spot Delta: {greeks['delta_spot']:,.0f}")
```

#### Greeks

| Greek | Description |
|-------|-------------|
| `delta_forward` | Sensitivity to forward rate (dPV/dF) |
| `delta_spot` | Sensitivity to spot rate (dPV/dS) |
| `gamma` | Second derivative w.r.t. forward |
| `vega` | Sensitivity to volatility |
| `theta` | Time decay |
| `rho_domestic` | Sensitivity to domestic rate |
| `rho_foreign` | Sensitivity to foreign rate |

#### Spot Delta vs Forward Delta

The relationship between spot and forward delta:

$$\Delta_{spot} = \Delta_{forward} \cdot \frac{DF_{foreign}}{DF_{domestic}}$$

For FX, this equals:
$$\Delta_{spot} = \Delta_{forward} \cdot e^{(r_d - r_f) \cdot T}$$

---

## Planned Asset Classes

### Rates Forward Options

**Status:** Phase 3 (Planned)

Forward rate agreements (FRAs) and options on forward rates will use Black76 with the forward rate as the underlying.

### Commodity Forward Options

**Status:** Future Phase

Options on commodity forwards (oil, gas, metals) will follow the same Black76 framework.

---

## Key Considerations

### 1. Forward Expiry vs Option Expiry

The `forward_expiry` can be greater than or equal to `expiry`:
- **Equal:** Standard European option
- **Greater:** Option on a forward contract with later delivery

### 2. Volatility Surface

Forward options typically use:
- **Forward volatility:** Volatility of the forward price
- **Strike convention:** Often absolute strike (not delta-based)

### 3. Hedging

Hedging a forward option:
1. **Delta hedge with forwards:** No financing cost (unlike spot hedge)
2. **Roll risk:** May need to roll hedges as contracts expire
3. **Basis risk:** Forward vs spot hedge basis

---

## Related Documentation

- [Black76 Model](../../reference/models/black76.md) - Mathematical details
- [Volatility Surfaces](../market-data/volatility_surfaces.md) - Vol surface conventions
- [FX Options](vanilla_options.md) - Spot-based FX options

---

*Document Version: 1.0 | QuantStrata Phase 3.1 | January 2026*
