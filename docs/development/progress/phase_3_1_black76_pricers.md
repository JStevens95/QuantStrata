# Phase 3.1: Black76 Pricers - Implementation Progress

**Started:** January 27, 2026  
**Target Completion:** Week 12  
**Status:** In Progress

---

## Overview

Phase 3.1 implements Black76 pricers for forward/futures options across asset classes, building on the Black76 model foundation established in Phase 2.3.

The Black76 model is used when the underlying is a forward or futures price rather than spot:
- **No drift**: Forward/futures prices are martingales under the risk-neutral measure
- **Formula**: Same as BSM but with forward price F instead of spot S and no cost-of-carry term

---

## Implementation Tasks

### 3.1a: FX Forward Options (Black76) ✅

**Status:** COMPLETE

| Component | File | Status |
|-----------|------|--------|
| Instrument | `src/instruments/fx/options/forward.py` | ✅ Complete |
| Pricer | `src/pricers/fx/european_b76.py` | ✅ Complete |
| Unit Tests | `tests/unit/pricers/fx/test_fx_european_b76_pricer.py` | ✅ Complete |
| Documentation | `docs/guides/instruments/forward_options.md` | ✅ Complete |
| Tutorial | `docs/tutorials/instruments/forward_options.ipynb` | ✅ Complete |

**Classes Implemented:**
- `EuropeanFxForwardOption` - Full market data lookup
- `EuropeanFxForwardOptionSimple` - Direct parameter input
- `FxForwardOptionBlack76Pricer` - Full pricer with market data
- `FxForwardOptionBlack76PricerSimple` - Simple pricer

**Greeks Supported:**
- `delta_forward` - Sensitivity to forward rate
- `delta_spot` - Sensitivity to spot rate (= delta_forward × exp((r_d - r_f)×T))
- `gamma`, `vega`, `theta`, `rho_domestic`, `rho_foreign`

**Tests Verified:**
- Put-call parity
- Finite difference checks for delta and vega
- Simple vs full pricer consistency
- Edge cases (deep ITM/OTM)

---

### 3.1b: Equity Index Futures Options (Black76) ✅

**Status:** COMPLETE

| Component | File | Status |
|-----------|------|--------|
| Instrument | `src/instruments/equity/options/futures.py` | ✅ Complete |
| Pricer | `src/pricers/equity/european_b76.py` | ✅ Complete |
| Unit Tests | `tests/unit/pricers/equity/test_equity_european_b76_pricer.py` | ✅ Complete |
| Documentation | `docs/guides/instruments/futures_options.md` | ✅ Complete |
| Tutorial | `docs/tutorials/instruments/futures_options.ipynb` | ✅ Complete |

**Classes Implemented:**
- `EuropeanEquityFuturesOption` - Full market data lookup
- `EuropeanEquityFuturesOptionSimple` - Direct parameter input
- `EquityFuturesOptionBlack76Pricer` - Full pricer with market data
- `EquityFuturesOptionBlack76PricerSimple` - Simple pricer

**Greeks Supported:**
- `delta_futures` - Sensitivity to futures price
- `delta_spot` - Sensitivity to spot price (= delta_futures × exp((r - q)×T))
- `gamma`, `vega`, `theta`, `rho`

**Tests Verified:**
- Put-call parity
- Notional scaling
- Finite difference checks for delta and vega
- Simple vs full pricer consistency
- Edge cases

---

### 3.1c: Interest Rate Caps & Floors (Black76) ✅

**Status:** COMPLETE

| Component | File | Status |
|-----------|------|--------|
| Instrument | `src/instruments/ir/options/capfloor.py` | ✅ Complete |
| Pricer | `src/pricers/ir/european_b76.py` | ✅ Complete |
| Unit Tests | `tests/unit/pricers/ir/test_ir_european_b76_pricer.py` | ✅ Complete |
| Documentation | TBD | ⏳ Pending |

**Classes Implemented:**
- `Caplet`, `CapletSimple` - Single caplet instruments
- `Floorlet`, `FloorletSimple` - Single floorlet instruments
- `Cap`, `CapSimple` - Multi-period cap instruments
- `Floor`, `FloorSimple` - Multi-period floor instruments
- `CapletBlack76Pricer`, `CapletBlack76PricerSimple` - Caplet pricers
- `FloorletBlack76Pricer`, `FloorletBlack76PricerSimple` - Floorlet pricers
- `CapBlack76Pricer`, `CapBlack76PricerSimple` - Cap pricers
- `FloorBlack76Pricer`, `FloorBlack76PricerSimple` - Floor pricers

**Key Features:**
- Day count conventions: ACT/360, ACT/365, 30/360
- Forward rate calculation from discount curve
- Automatic caplet/floorlet generation for caps/floors
- Full Greeks: delta, gamma, vega, theta, rho

**Tests Verified (29 tests):**
- Instrument validation
- Basic pricing (ITM/OTM behavior)
- Caplet-floorlet parity
- Greeks computation
- Finite difference validation (delta, vega)
- Cap/floor as sum of caplets/floorlets
- Market data integration
- Edge cases (ATM, deep ITM/OTM, zero vol)

---

## Additional Deliverables

### Tutorial Notebooks ✅

| Notebook | Status |
|----------|--------|
| `tutorials/pricing/fx_options_pricing.ipynb` | ✅ Complete |
| `tutorials/instruments/forward_options.ipynb` | ✅ Complete |
| `tutorials/instruments/futures_options.ipynb` | ✅ Complete |

---

## Technical Design Decisions

### 1. Pricer File Organization
- All Black76 European-style pricers consolidated in `european_b76.py`
- Mirrors pattern of `european_bsm.py` for BSM pricers
- Separate files per asset class: `fx/`, `equity/`, `ir/`

### 2. Delta Decomposition
- Forward/futures options have two delta measures:
  - `delta_forward` / `delta_futures`: Sensitivity to forward/futures price
  - `delta_spot`: Sensitivity to spot price for hedging with spot
- Relationship: `delta_spot = delta_forward × exp((r - carry)×T)`

### 3. Simple Pricers
- Each pricer has a "Simple" variant accepting direct parameters
- Useful when forward price is directly observable (no model risk)
- Enables unit testing without full market data infrastructure

---

## Dependencies

Phase 3.1 builds on:
- **Phase 2.3**: Black76/Bachelier model foundation (`src/models/analytic/black76/`)
- **Phase 2.4**: Equity market data infrastructure
- **Phase 1**: FX market data infrastructure

---

## Next Steps After Phase 3.1

1. **Phase 3.2**: Bachelier pricers (swaptions, spread options)
2. **Phase 3.3**: Core rate instruments (FRA, IRS)
3. **Phase 3.4**: Rate models (Hull-White, etc.)

---

## Change Log

| Date | Change |
|------|--------|
| 2026-01-27 | Phase 3.1a complete - FX Forward Options |
| 2026-01-27 | Phase 3.1b complete - Equity Futures Options |
| 2026-01-27 | Tutorial notebooks created |
| 2026-01-27 | Phase 3.1c complete - Caps & Floors (29 tests passing) |
| 2026-01-27 | **Phase 3.1 COMPLETE** - All Black76 pricers implemented |
