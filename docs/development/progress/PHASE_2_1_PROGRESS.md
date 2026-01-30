# Phase 2.1 Implementation Progress

**Last Updated:** January 28, 2026  
**Status:** ✅ COMPLETE

## Overview

Phase 2.1 focuses on **Equity Core Instruments** - establishing the foundation for equity derivatives with the same quality standards as the FX implementation.

**Objectives:**
1. **Equity Spot & Forward** - Foundation instruments with dividend handling
2. **European Equity Vanilla Options** - BSM with dividend yield (`b = r - q`)
3. **American Equity Vanilla Options** - Early exercise via FD (PSOR)
4. **Pricers** - BSM, MC, FD adapters for equity

**Key Difference from FX:**
- FX: Two curves (domestic, foreign) → `b = r_d - r_f`
- Equity: One curve (risk-free) + dividend yield → `b = r - q`

---

## Components Implemented

| Component | Description | Status | Tests | Docs |
|-----------|-------------|--------|-------|------|
| `EquitySpot` | Spot equity instrument | ✅ | ✅ | ✅ |
| `EquityForward` | Forward with dividends | ✅ | ✅ | ✅ |
| `EuropeanEquityVanillaOption` | European call/put | ✅ | ✅ | ✅ |
| `AmericanEquityVanillaOption` | American call/put | ✅ | ✅ | ✅ |
| `EquityEuropeanVanillaBsmPricer` | BSM pricer | ✅ | ✅ | ✅ |
| `EquityEuropeanVanillaMcPricer` | Monte Carlo pricer | ✅ | ✅ | ✅ |
| `EquityEuropeanVanillaFdPricer` | Finite Difference pricer | ✅ | ✅ | ✅ |
| `EquityAmericanVanillaFdPricer` | American FD (PSOR) pricer | ✅ | ✅ | ✅ |
| `EquitySpotPricer` | Spot position pricer | ✅ | ✅ | ✅ |
| `EquityForwardPricer` | Forward pricer | ✅ | ✅ | ✅ |

---

## 1. Equity Instruments

### 1.1 Design Decisions

**Market ID Convention:**
```python
# Equity spot
MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")

# Equity vol surface
MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")

# Discount curve (risk-free rate)
MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
```

**Dividend Modeling:**
- Phase 2.1: Continuous dividend yield `q` (stored in instrument)
- Phase 2.3: Discrete dividends (future enhancement)

**Cost-of-Carry:**
- Equity: `b = r - q` where `r` = risk-free rate, `q` = dividend yield
- Forward price: `F = S × exp((r - q) × T)`

### 1.2 Instrument Dataclasses

```python
@dataclass(frozen=True, slots=True)
class EuropeanEquityVanillaOption:
    ticker: str                    # e.g., "AAPL"
    option_type: OptionType        # "call" or "put"
    strike: float
    expiry: float                  # Year fraction
    notional: float                # Number of shares
    dividend_yield: float          # Continuous dividend yield q
    spot_id: MarketId
    vol_id: MarketId
    curve_id: MarketId             # Single discount curve (not two like FX)
```

---

## 2. Equity Pricers

### 2.1 BSM Pricer

Reuse existing `BlackScholesMertonVanilla` engine with:
- `discount_rate = r` (from curve)
- `carry = r - q` (risk-free minus dividend yield)

### 2.2 Monte Carlo Pricer

Simulate GBM with drift `(r - q)`:
```
dS = (r - q) S dt + σ S dW
```

### 2.3 Finite Difference Pricer

Solve BS PDE with `b = r - q`:
```
∂V/∂t + (r-q)S ∂V/∂S + ½σ²S² ∂²V/∂S² = rV
```

### 2.4 American FD Pricer

Use PSOR for early exercise constraint:
```
V(S,t) ≥ max(±(S - K), 0)
```

---

## 3. Testing Strategy

### Unit Tests
- Instrument construction and validation
- Pricer accuracy vs known values
- Put-call parity for European options
- Early exercise premium for American options
- BSM vs MC vs FD agreement

### Parity Tests
- Put-Call Parity: `C - P = S × exp(-qT) - K × exp(-rT)`
- American ≥ European (early exercise premium)
- High dividend call may exercise early

---

## 4. Files to Create

```
src/instruments/equity/
├── __init__.py
├── linear/
│   ├── __init__.py
│   ├── spot.py          # EquitySpot
│   └── forward.py       # EquityForward
└── options/
    ├── __init__.py
    └── vanilla.py       # European/American equity vanilla

src/pricers/equity/
├── __init__.py
├── european_bsm.py      # BSM pricer
├── european_mc.py       # Monte Carlo pricer
├── european_fd.py       # Finite Difference pricer
└── american_fd.py       # American FD (PSOR) pricer

tests/unit/instruments/equity/
├── __init__.py
└── test_vanilla.py

tests/unit/pricers/equity/
├── __init__.py
├── test_european_bsm.py
├── test_european_mc.py
├── test_european_fd.py
└── test_american_fd.py
```

---

## Files Created

### Instruments
```
src/instruments/equity/
├── __init__.py
├── linear/
│   ├── __init__.py
│   ├── spot.py           # EquitySpot
│   └── forward.py        # EquityForward
└── options/
    ├── __init__.py
    └── vanilla.py        # European/American equity vanilla
```

### Pricers
```
src/pricers/equity/
├── __init__.py
├── spot.py               # EquitySpotPricer
├── forward.py            # EquityForwardPricer
├── european_bsm.py       # EquityEuropeanVanillaBsmPricer
├── european_mc.py        # EquityEuropeanVanillaMcPricer
├── european_fd.py        # EquityEuropeanVanillaFdPricer
└── american_fd.py        # EquityAmericanVanillaFdPricer
```

### Unit Tests
```
tests/unit/instruments/equity/
├── __init__.py
└── test_vanilla.py       # 22 tests

tests/unit/pricers/equity/
├── __init__.py
├── test_european_bsm.py  # 15 tests
├── test_european_mc.py   # 14 tests
├── test_european_fd.py   # 14 tests
└── test_american_fd.py   # 15 tests
```

---

## Test Summary

**Total Tests: 80**
- All tests passing ✅

### Test Coverage:
- **Instrument Validation**: Construction, validation, immutability
- **Pricing Accuracy**: BSM vs MC vs FD convergence
- **Put-Call Parity**: With and without dividends
- **Greeks**: Delta, Gamma, Vega, Rho signs and scaling
- **Early Exercise**: American ≥ European, intrinsic floor
- **Edge Cases**: Zero expiry, notional scaling, reproducibility

---

## Key Implementation Details

### Cost-of-Carry Mapping
- **FX**: `b = r_d - r_f` (two curves)
- **Equity**: `b = r - q` (one curve + dividend yield)

### Forward Price
```
F = S × exp((r - q) × T)
```

### Put-Call Parity (Equity)
```
C - P = S × exp(-q×T) - K × exp(-r×T)
```

### American Early Exercise
- **Put**: Always has early exercise premium
- **Call (no dividend)**: Never exercise early → American = European
- **Call (with dividend)**: May exercise early to capture dividend

---

## Progress Log

### January 28, 2026
- Created Phase 2.1 progress document
- Created directory structure for equity instruments and pricers
- Implemented `EquitySpot`, `EquityForward` instruments
- Implemented `EuropeanEquityVanillaOption`, `AmericanEquityVanillaOption`
- Implemented `EquitySpotPricer`, `EquityForwardPricer`
- Implemented `EquityEuropeanVanillaBsmPricer` with dividend yield support
- Implemented `EquityEuropeanVanillaMcPricer` with GBM simulation
- Implemented `EquityEuropeanVanillaFdPricer` with Crank-Nicolson
- Implemented `EquityAmericanVanillaFdPricer` with PSOR
- Updated payoff factory to support equity instruments
- Created comprehensive unit tests (80 tests, all passing)

### January 28, 2026 (Documentation Update)
- Added theta to BSM engine Greeks (`src/models/analytic/black_scholes_merton/vanilla.py`)
- Updated FX BSM pricer to include theta in Greeks
- Updated Equity BSM pricer to include theta in Greeks
- Verified equity forward pricer theta implementation (correct sign)
- Added "Asset Class Specifics: FX vs Equity" section to `docs/mathematics/vanilla_options.md`:
  - Parameter mapping (r, q, b) for FX vs Equity
  - Forward price formulas
  - Put-call parity differences
  - Greeks decomposition (dual rho for FX, single rho for equity)
  - Early exercise considerations
  - Implementation code examples
- Updated `docs/notebooks/vanilla_options_analysis.ipynb` with new sections:
  - Section 9: FX vs Equity Comparison (side-by-side pricing)
  - Section 10: Equity Dividend Impact Analysis
  - Section 11: Theta Analysis for FX and Equity

---

*Phase 2.1 completed on January 28, 2026.*
