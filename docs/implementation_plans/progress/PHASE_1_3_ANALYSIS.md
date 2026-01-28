# Phase 1.3: Detailed Code Analysis

**Date:** January 27, 2026  
**Purpose:** Analyze existing calibration and bootstrapping infrastructure

---

## 1. Volatility Surface Calibration Analysis

### 1.1 Current Implementation (`src/marketdata/surfaces/fx/calibration.py`)

**What Works Well:**
- ✅ `calibrate_fx_smile_to_grid_surface()` - Converts ATM/RR/BF quotes to GridVolSurface
- ✅ `extract_fx_smile_from_grid_surface()` - Extracts quotes from surface
- ✅ Delta-to-strike conversion with fixed-point iteration
- ✅ Handles vol dependency correctly (strike depends on vol, vol depends on strike)
- ✅ Uses log-moneyness interpolation (appropriate for FX)

**What Needs Enhancement:**
- ⚠️ **Interpolation**: Currently uses flat extrapolation and linear interpolation
  - Could add cubic spline for smoother surfaces
  - Could add SABR/Vanna-Volga parametric fitting
- ⚠️ **Term Structure**: No explicit term structure fitting
  - Could add interpolation/extrapolation for missing expiries
- ⚠️ **Validation**: Basic arbitrage checks exist, but could be more comprehensive
- ⚠️ **Error Handling**: Could be more robust for edge cases

### 1.2 Quote Structures (`src/marketdata/surfaces/fx/quotes.py`)

**What Exists:**
- ✅ `FxSmileSliceQuotes` - Single expiry quotes (ATM, RR, BF)
- ✅ `FxSmileQuotes` - Multi-expiry container
- ✅ Helper methods: `vol_call()`, `vol_put()`, `deltas()`

**What's Good:**
- Clean separation of concerns
- Supports multiple delta conventions
- Immutable dataclasses

**Potential Enhancements:**
- Add validation for quote consistency
- Add helper for creating test quotes

---

## 2. Curve Bootstrapping Analysis

### 2.1 Current Implementation (`src/marketdata/curves/bootstrapper.py`)

**What Works Well:**
- ✅ `DepositQuote` - Simple/continuous compounding
- ✅ `ParSwapQuote` - Par swap bootstrapping
- ✅ Native Python implementation (no QuantLib dependency)
- ✅ QuantLib backend available as alternative
- ✅ Validation for discount factors (monotonicity, bounds)

**What Needs Enhancement:**
- ⚠️ **FRA Support**: Not implemented
  - Need to add `FraQuote` dataclass
  - Need to add FRA bootstrapping logic
- ⚠️ **Interpolation Methods**: Limited options
  - Currently uses ZeroRateCurve's interpolation
  - Could add explicit log-linear, cubic spline options
- ⚠️ **Error Messages**: Could be more descriptive
- ⚠️ **Day Count Conventions**: Hardcoded (could be configurable)

### 2.2 Bootstrapping Algorithm

**Native Implementation:**
```python
_bootstrap_discount_curve_native()
  - Sorts instruments by maturity
  - Processes deposits first (direct DF calculation)
  - Processes swaps (iterative bootstrapping)
  - Validates results
```

**Strengths:**
- Simple and deterministic
- No external dependencies
- Easy to understand and debug

**Limitations:**
- Assumes instruments are in correct order
- No handling of overlapping maturities
- Limited interpolation options

---

## 3. Arbitrage Validation Analysis

### 3.1 Current Checks (`src/marketdata/surfaces/validation/arbitrage.py`)

**What Exists:**
- ✅ Calendar arbitrage check (total variance non-decreasing)
- ✅ Call price convexity check
- ✅ Decreasing call price check

**What's Missing:**
- ⚠️ Butterfly arbitrage check (more complex)
- ⚠️ Strike arbitrage check (call spread, put spread)
- ⚠️ Forward arbitrage check

---

## 4. Implementation Plan

### Phase 1.3.1: Enhance Volatility Calibration

1. **Add SABR Parametric Fitting** (Optional)
   - Fit SABR parameters to market quotes
   - Generate smooth vol surface from SABR

2. **Improve Interpolation**
   - Add cubic spline option
   - Better extrapolation handling

3. **Term Structure Enhancement**
   - Interpolate missing expiries
   - Extrapolate beyond last expiry

4. **Enhanced Validation**
   - More comprehensive arbitrage checks
   - Better error messages

### Phase 1.3.2: Enhance Curve Bootstrapping

1. **Add FRA Support**
   - `FraQuote` dataclass
   - FRA bootstrapping logic

2. **Multiple Interpolation Methods**
   - Log-linear (common for rates)
   - Cubic spline (smooth but may introduce arbitrage)
   - Linear in discount factors

3. **Better Error Handling**
   - More descriptive error messages
   - Suggestions for fixing issues

4. **Day Count Conventions**
   - Make configurable
   - Support common conventions (ACT/365, 30/360, etc.)

---

## 5. Testing Strategy

### Unit Tests Needed

**Calibration:**
- Test delta-to-strike conversion accuracy
- Test smile calibration with known inputs
- Test term structure interpolation
- Test arbitrage detection

**Bootstrapping:**
- Test deposit bootstrapping (simple/continuous)
- Test swap bootstrapping (various frequencies)
- Test FRA bootstrapping (when implemented)
- Test interpolation methods
- Test arbitrage detection

### Integration Tests

- Full workflow: Quotes → Surface → Pricing
- Full workflow: Rates → Curve → Pricing
- End-to-end calibration → pricing pipeline

---

## 6. Documentation Plan

1. **Mathematical Documentation**
   - Volatility calibration theory
   - Curve bootstrapping theory
   - Arbitrage conditions

2. **Interactive Notebooks**
   - Calibration workflow demonstration
   - Bootstrapping workflow demonstration
   - Arbitrage detection examples

---

## 7. Priority Order

1. **High Priority:**
   - Fix any bugs in existing code
   - Add comprehensive unit tests
   - Enhance error messages

2. **Medium Priority:**
   - Add FRA support to bootstrapper
   - Improve interpolation options
   - Add more arbitrage checks

3. **Low Priority:**
   - SABR parametric fitting
   - Advanced interpolation methods
   - Day count convention flexibility

---

*This analysis will guide Phase 1.3 implementation.*
