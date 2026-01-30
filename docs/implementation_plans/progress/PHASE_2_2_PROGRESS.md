# Phase 2.2 Implementation Progress

**Last Updated:** January 28, 2026  
**Status:** 🚧 IN PROGRESS

## Overview

Phase 2.2 focuses on **Equity Exotic Options** - extending the equity derivatives suite with barrier, digital, Asian, and lookback options.

**Objectives:**
1. **Equity Barrier Options** - Knock-in/out barriers (MC pricer)
2. **Equity Digital Options** - Cash-or-nothing payouts
3. **Equity Asian Options** - Average price options
4. **Equity Lookback Options** - Path-dependent exotics

**Key Design Principle:** Reuse existing FX payoff infrastructure with equity-specific adapters.

---

## Components to Implement

| Component | Description | Status | Tests | Docs |
|-----------|-------------|--------|-------|------|
| `EuropeanEquityBarrierOption` | Barrier option instrument | ✅ | ✅ | 🔲 |
| `EquityBarrierMcPricer` | Monte Carlo barrier pricer | ✅ | ✅ | 🔲 |
| `EuropeanEquityDigitalOption` | Digital option instrument | ✅ | 🔲 | 🔲 |
| `EquityDigitalBsmPricer` | BSM digital pricer | ✅ | 🔲 | 🔲 |
| `EuropeanEquityAsianOption` | Asian option instrument | ✅ | ✅ | 🔲 |
| `EquityAsianMcPricer` | Monte Carlo Asian pricer | ✅ | ✅ | 🔲 |
| `EuropeanEquityLookbackOption` | Lookback option instrument | ✅ | ✅ | 🔲 |
| `EquityLookbackMcPricer` | Monte Carlo lookback pricer | ✅ | ✅ | 🔲 |

---

## 1. Equity Barrier Options

### 1.1 Instrument Design

```python
@dataclass(frozen=True, slots=True)
class EuropeanEquityBarrierOption:
    ticker: str
    option_type: OptionType           # "call" or "put"
    barrier_type: BarrierType         # "up_and_out", "down_and_in", etc.
    strike: float
    barrier: float
    expiry: float
    notional: float
    dividend_yield: float
    spot_id: MarketId
    vol_id: MarketId
    curve_id: MarketId
    rebate: float = 0.0               # Rebate if knocked out
```

### 1.2 Barrier Types

| Barrier Type | Knock Condition | Payoff |
|--------------|-----------------|--------|
| Up-and-Out | S > B at any time → knockout | 0 (or rebate) |
| Up-and-In | S > B at any time → knock-in | Vanilla payoff |
| Down-and-Out | S < B at any time → knockout | 0 (or rebate) |
| Down-and-In | S < B at any time → knock-in | Vanilla payoff |

### 1.3 Pricing Method

Monte Carlo with path simulation:
1. Simulate GBM paths: `dS = (r-q)S dt + σS dW`
2. Check barrier crossing at each time step
3. Apply knock-in/out logic
4. Discount expected payoff

---

## 2. Equity Digital Options

### 2.1 Instrument Design

```python
@dataclass(frozen=True, slots=True)
class EuropeanEquityDigitalOption:
    ticker: str
    option_type: OptionType           # "call" or "put"
    digital_type: DigitalType         # "cash" or "asset"
    strike: float
    expiry: float
    notional: float
    payout: float                     # Cash amount (for cash digital)
    dividend_yield: float
    spot_id: MarketId
    vol_id: MarketId
    curve_id: MarketId
```

### 2.2 Payoff Structure

**Cash-or-Nothing:**
- Call: Pays `payout` if S_T > K, else 0
- Put: Pays `payout` if S_T < K, else 0

**Asset-or-Nothing:**
- Call: Pays `S_T` if S_T > K, else 0
- Put: Pays `S_T` if S_T < K, else 0

### 2.3 BSM Closed-Form

Cash-or-Nothing Call:
```
V = payout × exp(-rT) × N(d2)
```

Asset-or-Nothing Call:
```
V = S × exp(-qT) × N(d1)
```

---

## 3. Equity Asian Options

### 3.1 Instrument Design

```python
@dataclass(frozen=True, slots=True)
class EuropeanEquityAsianOption:
    ticker: str
    option_type: OptionType
    strike: float
    expiry: float
    notional: float
    averaging_type: AveragingType     # "arithmetic" or "geometric"
    averaging_frequency: int           # Number of averaging points
    dividend_yield: float
    spot_id: MarketId
    vol_id: MarketId
    curve_id: MarketId
```

### 3.2 Average Types

- **Arithmetic:** A = (1/n) × Σ S_i
- **Geometric:** A = (∏ S_i)^(1/n)

### 3.3 Pricing

- **Geometric:** Closed-form adjustment (log-average is normal)
- **Arithmetic:** Monte Carlo simulation

---

## 4. Equity Lookback Options

### 4.1 Instrument Design

```python
@dataclass(frozen=True, slots=True)
class EuropeanEquityLookbackOption:
    ticker: str
    option_type: OptionType
    lookback_type: LookbackType       # "fixed_strike" or "floating_strike"
    strike: float                     # For fixed strike
    expiry: float
    notional: float
    dividend_yield: float
    spot_id: MarketId
    vol_id: MarketId
    curve_id: MarketId
```

### 4.2 Payoff Structure

**Fixed Strike:**
- Call: max(S_max - K, 0)
- Put: max(K - S_min, 0)

**Floating Strike:**
- Call: S_T - S_min
- Put: S_max - S_T

---

## Files to Create

```
src/instruments/equity/options/
├── barrier.py      # EuropeanEquityBarrierOption
├── digital.py      # EuropeanEquityDigitalOption
├── asian.py        # EuropeanEquityAsianOption
└── lookback.py     # EuropeanEquityLookbackOption

src/pricers/equity/
├── barrier_mc.py   # EquityBarrierMcPricer
├── digital_bsm.py  # EquityDigitalBsmPricer
├── asian_mc.py     # EquityAsianMcPricer
└── lookback_mc.py  # EquityLookbackMcPricer

tests/unit/instruments/equity/
├── test_barrier.py
├── test_digital.py
├── test_asian.py
└── test_lookback.py

tests/unit/pricers/equity/
├── test_barrier_mc.py
├── test_digital_bsm.py
├── test_asian_mc.py
└── test_lookback_mc.py
```

---

## Progress Log

### January 28, 2026
- Created Phase 2.2 progress document
- Outlined instrument and pricer designs
- Implemented all four equity exotic instruments:
  - `EuropeanEquityBarrierOption` - Single barrier with knock-in/out
  - `EuropeanEquityDigitalOption` - Cash/asset-or-nothing
  - `EuropeanEquityAsianOption` - Arithmetic/geometric averaging
  - `EuropeanEquityLookbackOption` - Fixed/floating strike
- Implemented all four equity exotic pricers (consolidated into existing files):
  - `EquityEuropeanBarrierMcPricer` - MC with path monitoring → `european_mc.py`
  - `EquityEuropeanDigitalBsmPricer` - BSM closed-form for digital options → `european_bsm.py`
  - `EquityEuropeanAsianMcPricer` - MC with averaging → `european_mc.py`
  - `EquityEuropeanLookbackMcPricer` - MC with path extremum → `european_mc.py`
- Updated `PayoffFactory` with equity exotic routing
- Updated `__init__.py` files for instruments and pricers
- **Restructured `european_mc.py`** to exactly match FX pattern:
  - All simulation artifact dataclasses grouped at top
  - All pricer classes grouped below
  - Exported simulation artifacts via `__init__.py`
- **Added unit tests** for exotic MC pricers to `test_equity_european_mc_pricer.py`:
  - `TestBarrierMcPricerBasic` - barrier option basic pricing tests
  - `TestBarrierMcSimulationArtifact` - barrier simulation artifact tests
  - `TestAsianMcPricerBasic` - Asian option basic pricing tests
  - `TestAsianMcSimulationArtifact` - Asian simulation artifact tests
  - `TestLookbackMcPricerBasic` - lookback option basic pricing tests
  - `TestLookbackMcSimulationArtifact` - lookback simulation artifact tests
- Next: Add unit tests for digital BSM pricer

### January 28, 2026 (Later)
- **Updated PayoffFactory** with equity exotic routing:
  - `EuropeanEquityDigitalOption` → `DigitalCashPayoff` / `DigitalAssetPayoff`
  - `EuropeanEquityBarrierOption` → `SingleBarrierPayoff`
  - `EuropeanEquityAsianOption` → `AsianPayoff`
  - `EuropeanEquityLookbackOption` → `LookbackPayoff`
- **Registry** already complete (verified)
- **Reviewed BSM model structure** (`src/models/analytic/black_scholes_merton/`):
  - `base.py` - **KEEP** - Core helpers (`d1_d2`, `CarryDiscountTerms`, `validate_bsm_inputs`)
  - `vanilla.py` - **KEEP** - `BlackScholesMertonVanilla` used by FX and Equity pricers
  - `digital.py` - **KEEP** - `BlackScholesMertonDigitalCash/Asset` used by FX pricer
  - `barrier.py` - **DELETED** - Was empty file
- **Updated ROADMAP** with new Phase 2.3 for Black76 and Bachelier models

---

*This document is updated as implementation progresses.*
