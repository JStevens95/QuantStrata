# Phase 3.6: Black-Karasinski Model - Progress Report

## Status: COMPLETE ✅

**Completion Date:** January 2026

## Summary

Phase 3.6 implements the Black-Karasinski one-factor short rate model, providing a log-normal alternative to Hull-White for positive-rate environments.

## Deliverables

### 1. Black-Karasinski Model Core

**File:** `src/models/short_rate/black_karasinski.py`

Implemented components:
- `BlackKarasinskiParameters`: Model parameters with validation
  - Mean reversion speed (a)
  - Volatility of log-rate (sigma)
  - Initial rate (r0, must be positive)
  - Long-term log-rate (theta)
  - Derived properties: x0, half_life, long_term_vol, long_term_rate
- `BlackKarasinskiDynamics`: Path simulation
  - Exact OU transition scheme
  - Euler-Maruyama scheme
  - Antithetic variates support
  - Discount factor computation
- `BlackKarasinskiSimulation`: Result container
  - Rate paths (always positive)
  - Log-rate paths
  - Statistical properties

### 2. Monte Carlo Pricers

**File:** `src/pricers/ir/european_bk.py`

Implemented pricers:
- `IrBondZeroCouponBKMCPricerSimple`: ZC bond MC pricer
- `IrBondEuropeanOptionBKMCPricerSimple`: Bond option MC pricer
- `IrCapletEuropeanOptionBKMCPricerSimple`: Caplet MC pricer
- `IrFloorletEuropeanOptionBKMCPricerSimple`: Floorlet MC pricer

Features:
- `BKMCConfig`: Configuration for MC parameters
- Full `MonteCarloEstimate` integration
- Greeks via finite difference bumping
- Intrinsic value handling for expired options

### 3. Standalone MC Functions

Utility functions in model file:
- `bk_zc_bond_price_mc()`: Quick ZC bond pricing
- `bk_zc_bond_option_price_mc()`: Quick bond option pricing

### 4. Unit Tests

**Files:**
- `tests/unit/models/short_rate/test_black_karasinski.py` (37 tests)
- `tests/unit/pricers/ir/test_ir_bk_pricer.py` (21 tests)

Total: **58 tests passing**

Test coverage:
- Parameter validation
- Parameter properties (x0, half_life, long_term_vol, etc.)
- Simulation correctness (exact and Euler schemes)
- Rate positivity guarantee
- Mean/variance convergence
- MC pricing (bonds, options, caplets, floorlets)
- Greeks computation
- Edge cases

### 5. Documentation

- `docs/guides/models/black_karasinski.md`: Complete technical guide
  - Mathematical framework
  - Comparison with Hull-White
  - Usage examples
  - Numerical considerations
  - Interview key points

## Key Design Decisions

### 1. Log-Rate Simulation

We simulate $x(t) = \ln r(t)$ using the exact OU transition, then compute $r(t) = e^{x(t)}$. This ensures:
- Numerical stability
- Guaranteed positive rates
- Correct distribution properties

### 2. MC-Only Pricing

Black-Karasinski is non-affine, so we use Monte Carlo for all pricing:
- No closed-form bond prices possible
- FDE support could be added in future phases
- Antithetic variates for variance reduction

### 3. Parameter Validation

r0 must be strictly positive (unlike Hull-White which allows negative rates):
```python
if self.r0 <= 0.0:
    raise ValueError(f"Initial rate r0 must be > 0; got {self.r0}.")
```

### 4. Base Infrastructure Integration

Leverages existing infrastructure:
- `NormalRng` from `src/models/numeric/monte_carlo/rng.py`
- `MonteCarloEstimate` from `src/models/numeric/monte_carlo/base.py`
- `estimate_from_samples` from `src/models/numeric/monte_carlo/estimators.py`

## Architecture

```
src/models/short_rate/
├── __init__.py           # Updated to export BK
├── hull_white.py         # Phase 3.5
└── black_karasinski.py   # NEW

src/pricers/ir/
├── __init__.py           # Updated to export BK pricers
├── european_hw.py        # Hull-White analytic
├── european_mc.py        # Hull-White MC
├── european_fde.py       # Hull-White FDE
└── european_bk.py        # NEW - Black-Karasinski MC
```

## Test Results

```
tests/unit/models/short_rate/test_black_karasinski.py .......... 37 passed
tests/unit/pricers/ir/test_ir_bk_pricer.py .................... 21 passed
============================== 58 passed ==============================
```

## Comparison: Hull-White vs Black-Karasinski

| Feature | Hull-White | Black-Karasinski |
|---------|------------|------------------|
| Rate distribution | Gaussian | Log-normal |
| Negative rates | Allowed | Not possible |
| Bond pricing | Analytic | MC only |
| Volatility | Additive | Proportional |
| Implementation | 3 pricers (HW, MC, FDE) | 1 pricer (MC) |
| Tests | 60+ | 58 |

## Future Enhancements

1. **FDE Pricer**: Implement finite difference for BK
2. **Calibration**: Fit θ(t) to match yield curve
3. **Swaption Pricing**: Add BK swaption support
4. **Tree Methods**: Alternative to MC

## Impact

Phase 3.6 provides:
1. Complete alternative short rate model to Hull-White
2. Guaranteed positive rate dynamics
3. Professional-grade MC implementation
4. Comprehensive test coverage
5. Clear educational documentation

This completes the short rate model suite for Phase 3, preparing for Phase 4's advanced models.
