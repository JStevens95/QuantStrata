# Phase 3.5: Hull-White Model - Progress Report

**Completion Date:** January 27, 2026  
**Status:** ✅ Complete

---

## Overview

Phase 3.5 implemented the Hull-White one-factor short rate model - the industry-standard model for interest rate derivative pricing. This phase establishes the foundation for short rate modeling in QuantStrata, with analytic, Monte Carlo, and Finite Difference pricing capabilities.

---

## Implemented Components

### 1. Hull-White Model Core

**File:** `src/models/short_rate/hull_white.py`

| Component | Description |
|-----------|-------------|
| `HullWhiteParameters` | Model parameters (a, σ, r0, θ) with validation |
| `HullWhiteDynamics` | Path simulator with exact/Euler schemes |
| `HullWhiteSimulation` | Output container with statistics |
| `hw_b_factor()` | B(t,T) factor computation |
| `hw_zc_bond_price()` | Analytic ZC bond pricing |
| `hw_zc_bond_option_price()` | Analytic bond option pricing |
| `hw_caplet_price()` | Caplet pricing via bond option |
| `hw_floorlet_price()` | Floorlet pricing via bond option |
| `hw_swaption_price_jamshidian()` | Swaption pricing via Jamshidian decomposition |

**Key Features:**
- Exact OU transition distribution for simulation
- Euler scheme for comparison
- Antithetic variates for variance reduction
- Path-wise discount factor computation

### 2. Hull-White Analytic Pricers

**File:** `src/pricers/ir/european_hw.py`

| Pricer | Products |
|--------|----------|
| `IrBondZeroCouponHWPricerSimple` | Zero coupon bonds |
| `IrBondZeroCouponHWPricer` | ZC bonds with market data |
| `IrBondEuropeanOptionHWPricerSimple` | Bond options |
| `IrBondEuropeanOptionHWPricer` | Bond options with market data |
| `IrCapletEuropeanOptionHWPricerSimple` | Caplets |
| `IrFloorletEuropeanOptionHWPricerSimple` | Floorlets |
| `IrCapEuropeanOptionHWPricerSimple` | Caps (portfolio of caplets) |
| `IrFloorEuropeanOptionHWPricerSimple` | Floors (portfolio of floorlets) |
| `IrSwaptionEuropeanOptionHWPricerSimple` | Swaptions (Jamshidian) |

### 3. Hull-White Monte Carlo Pricers

**File:** `src/pricers/ir/european_mc.py`

| Pricer | Products |
|--------|----------|
| `MCConfig` | MC configuration (paths, steps, seed) |
| `IrBondZeroCouponMCPricerSimple` | ZC bonds via MC |
| `IrBondEuropeanOptionMCPricerSimple` | Bond options via MC |
| `IrCapletEuropeanOptionMCPricerSimple` | Caplets via MC |
| `IrFloorletEuropeanOptionMCPricerSimple` | Floorlets via MC |
| `IrSwaptionEuropeanOptionMCPricerSimple` | Swaptions via MC |

**Features:**
- Configurable path count and time steps
- Antithetic variates
- Standard error estimation
- Greeks via finite difference bumping

### 4. Hull-White Finite Difference Pricers

**File:** `src/pricers/ir/european_fde.py`

| Pricer | Products |
|--------|----------|
| `FDConfig` | FD configuration (grid size, scheme) |
| `HWGrid` | Grid builder for HW PDE |
| `IrBondZeroCouponFDPricerSimple` | ZC bonds via FDE |
| `IrBondEuropeanOptionFDPricerSimple` | Bond options via FDE |
| `IrCapletEuropeanOptionFDPricerSimple` | Caplets via FDE |
| `IrFloorletEuropeanOptionFDPricerSimple` | Floorlets via FDE |

**Features:**
- Configurable grid resolution
- Crank-Nicolson scheme (θ = 0.5)
- Support for explicit and implicit schemes
- Thomas algorithm for tridiagonal solves

---

## Test Summary

### Model Tests (`test_hull_white.py`)

| Category | Tests | Status |
|----------|-------|--------|
| Parameter validation | 13 | ✅ Pass |
| Dynamics simulation | 8 | ✅ Pass |
| Analytic functions | 9 | ✅ Pass |
| Edge cases | 7 | ✅ Pass |
| **Total** | **37** | ✅ **All Pass** |

### Pricer Tests (`test_ir_hw_pricer.py`)

| Category | Tests | Status |
|----------|-------|--------|
| ZC bond analytic | 4 | ✅ Pass |
| Bond option analytic | 5 | ✅ Pass |
| Caplet/Floorlet | 3 | ✅ Pass |
| Swaption | 2 | ✅ Pass |
| MC vs Analytic | 2 | ✅ Pass |
| FD vs Analytic | 2 | ✅ Pass |
| Greeks FD verification | 1 | ✅ Pass |
| Edge cases | 4 | ✅ Pass |
| **Total** | **23** | ✅ **All Pass** |

**Combined Total: 60 tests passing**

---

## Architecture Decisions

### 1. Consistent Naming Convention

Following the established pattern:
- `european_hw.py` - Hull-White analytic pricers
- `european_mc.py` - Monte Carlo pricers (HW and future models)
- `european_fde.py` - Finite Difference pricers (HW and future models)

This matches the existing `european_bsm.py`, `european_b76.py`, `european_bch.py` pattern.

### 2. Model-Specific Pricers

Hull-White pricers require `HullWhiteParameters`, unlike model-agnostic pricers (BSM, B76) that extract parameters from market data. This is consistent with how `HestonDynamics` is handled - not in the registry, but instantiated directly.

### 3. Short Rate Module

Created `src/models/short_rate/` module for short rate models:
- `hull_white.py` - Hull-White 1-factor
- Future: `black_karasinski.py`, `cir.py`, etc.

This parallels the `src/models/stochastic_volatility/` structure.

---

## Files Created/Modified

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/models/short_rate/__init__.py` | 35 | Package exports |
| `src/models/short_rate/hull_white.py` | 570 | HW model core |
| `src/pricers/ir/european_hw.py` | 650 | HW analytic pricers |
| `src/pricers/ir/european_mc.py` | 580 | HW MC pricers |
| `src/pricers/ir/european_fde.py` | 520 | HW FDE pricers |
| `tests/unit/models/short_rate/test_hull_white.py` | 340 | Model tests |
| `tests/unit/pricers/ir/test_ir_hw_pricer.py` | 350 | Pricer tests |
| `docs/guides/models/hull_white.md` | 340 | Technical guide |

### Modified Files

| File | Changes |
|------|---------|
| `src/pricers/ir/__init__.py` | Added HW pricer exports |

---

## Key Mathematical Results

### Verified Properties

1. **Bond Option Put-Call Parity**: C - P = P(0,T) - K×P(0,S)
2. **Caplet-Floorlet Parity**: Caplet - Floorlet = FRA value
3. **Simulation Mean Convergence**: E[r(T)] matches theoretical
4. **Simulation Variance Convergence**: Var[r(T)] matches theoretical
5. **Exact vs Euler**: Both schemes produce consistent results

### Pricing Consistency

| Method | vs Analytic | Tolerance |
|--------|-------------|-----------|
| Monte Carlo (20k paths) | ✅ | < 2% (bonds), < 20% (options) |
| Finite Difference (100×100) | ✅ | < 2% (bonds), < 10% (options) |

---

## Next Steps (Phase 3.6)

1. **Black-Karasinski Model** (3.6.1)
   - Log-normal short rate (positive rates only)
   - MC and FDE pricers

2. **Rate Market Data Infrastructure** (3.6.2)
   - Swaption volatility surface generation
   - Cap/floor volatility surfaces

3. **Rate Curve Bootstrapping** (3.6.3)
   - Multi-instrument bootstrapping
   - Smoothness validation

---

## Summary

Phase 3.5 successfully implemented the Hull-White one-factor short rate model with:
- ✅ Complete model implementation (parameters, dynamics, simulation)
- ✅ Analytic pricers for ZC bonds, bond options, caps/floors, swaptions
- ✅ Monte Carlo pricers with variance reduction
- ✅ Finite Difference pricers with Crank-Nicolson scheme
- ✅ Comprehensive test coverage (60 tests)
- ✅ Technical documentation

This establishes the foundation for interest rate modeling in QuantStrata, enabling both production pricing and educational exploration of short rate dynamics.
