# Phase 3.2: Linear IR Instruments (FRAs & IRS) - Implementation Progress

**Started:** January 27, 2026  
**Completed:** January 27, 2026  
**Status:** COMPLETE

---

## Overview

Phase 3.2 implements linear (non-optional) interest rate instruments: Forward Rate Agreements (FRAs) and Interest Rate Swaps (IRS). These are the foundational building blocks for interest rate derivatives, analogous to FX/Equity Spot and Forward instruments.

---

## Implementation Summary

### Forward Rate Agreements (FRA) ✅

| Component | File | Status |
|-----------|------|--------|
| Instrument | `src/instruments/ir/linear/fra.py` | ✅ Complete |
| Pricer | `src/pricers/ir/linear.py` | ✅ Complete |
| Unit Tests | `tests/unit/pricers/ir/test_ir_linear_pricer.py` | ✅ Complete |

**Classes Implemented:**
- `ForwardRateAgreement` - Market data lookup version
- `ForwardRateAgreementSimple` - Direct parameter version
- `FRAPricer` - Market data pricer
- `FRAPricerSimple` - Simple pricer

**Features:**
- Payer/Receiver direction
- Par rate calculation
- ITM/OTM detection
- Tenor description (e.g., "3x6")

**Greeks:**
- `delta` - dPV/dF (sensitivity to forward rate)
- `dv01` - Dollar value of 1 basis point
- `pv01` - Present value of 1 bp

---

### Interest Rate Swaps (IRS) ✅

| Component | File | Status |
|-----------|------|--------|
| Instrument | `src/instruments/ir/linear/swap.py` | ✅ Complete |
| Pricer | `src/pricers/ir/linear.py` | ✅ Complete |
| Unit Tests | `tests/unit/pricers/ir/test_ir_linear_pricer.py` | ✅ Complete |

**Classes Implemented:**
- `InterestRateSwap` - Market data lookup version
- `InterestRateSwapSimple` - Direct parameter version
- `SwapLeg`, `FixedLeg`, `FloatingLeg` - Leg cashflow types
- `IRSwapPricer` - Market data pricer
- `IRSwapPricerSimple` - Simple pricer

**Features:**
- Payer/Receiver direction
- Configurable fixed/floating frequencies
- Configurable day count conventions
- Floating leg spread support
- Automatic schedule generation

**Key Calculations:**
- Par swap rate
- Annuity (PV01 factor)
- Fixed leg PV
- Floating leg PV

**Greeks:**
- `delta` - dPV/dF (aggregate rate sensitivity)
- `dv01` - Dollar value of 1 basis point (signed)
- `pv01` - Present value of 1 bp (absolute)

---

## Test Summary

**Total Tests:** 36  
**Passing:** 36  
**Coverage:**

| Test Category | Count |
|---------------|-------|
| FRA Validation | 5 |
| FRA Pricing | 6 |
| FRA Greeks | 5 |
| Swap Validation | 3 |
| Swap Pricing | 4 |
| Swap Greeks | 4 |
| Market Data Pricers | 3 |
| Schedule Generation | 3 |
| Edge Cases | 3 |

---

## Mathematical Framework

### FRA Pricing

```
PV = N × τ × DF(T_end) × (F - K)  [payer]
PV = N × τ × DF(T_end) × (K - F)  [receiver]

Where:
- N = notional principal
- τ = day count fraction
- DF = discount factor
- F = forward rate
- K = contract rate
```

### IRS Pricing

```
PV_fixed = N × K × Σ[τ_i × DF_i]
PV_float = N × Σ[τ_i × DF_i × (F_i + spread)]

Receiver: PV = PV_fixed - PV_float
Payer:    PV = PV_float - PV_fixed

Par Rate: K_par = Σ[τ_i × DF_i × F_i] / Annuity
Annuity:  A = Σ[τ_i × DF_i]
```

---

## Files Created/Modified

**New Files:**
- `src/instruments/ir/linear/__init__.py`
- `src/instruments/ir/linear/fra.py`
- `src/instruments/ir/linear/swap.py`
- `src/pricers/ir/linear.py`
- `tests/unit/pricers/ir/test_ir_linear_pricer.py`

**Modified Files:**
- `src/instruments/ir/__init__.py` - Added linear instrument exports
- `src/pricers/ir/__init__.py` - Added linear pricer exports

---

## Design Decisions

### 1. Separate Simple and Market Data Versions
Following the pattern established in FX/Equity, each instrument has:
- **Simple version**: Direct parameter input, useful for testing
- **Market data version**: Lookup from Market object

### 2. Schedule Generation
Automatic schedule generation for swaps based on:
- Start/end times
- Fixed/floating frequencies
- Day count conventions

### 3. Leg Structure
Swaps are built from explicit leg cashflows (`FixedLeg`, `FloatingLeg`) for:
- Transparency in PV decomposition
- Support for non-standard swaps (e.g., amortizing)

---

## Relationship to Other Phases

| Phase | Dependency |
|-------|------------|
| 3.1 (Black76) | Uses same day count infrastructure from caps/floors |
| 3.3 (Bachelier/Swaptions) | Swaptions require IRS as underlying |
| 3.5 (Hull-White) | Enables MC/FD for rates using IRS as calibration instruments |

---

## Change Log

| Date | Change |
|------|--------|
| 2026-01-27 | Phase 3.2 started |
| 2026-01-27 | FRA instrument and pricer implemented |
| 2026-01-27 | IRS instrument and pricer implemented |
| 2026-01-27 | 36 unit tests passing |
| 2026-01-27 | **Phase 3.2 COMPLETE** |
