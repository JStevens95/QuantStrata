# Model Layer Refactoring - Progress

**Last Updated:** January 27, 2026  
**Status:** ✅ COMPLETE

## Overview

Major architectural refactoring of the `src/models/analytic/` layer to create a clean, modular, asset-agnostic design using **pure functions**.

## Objectives

1. **Generic Model Layer**: All formulas in `src/models/*` are asset-class agnostic
2. **Pure Functions**: No classes, stateless, fully composable
3. **Consistent Pattern**: Same structure across BSM, Black76, and Bachelier
4. **Clean Separation**: Pricers handle asset-specific mappings

---

## Changes Made

### 1. Black-Scholes-Merton Refactoring

**Before:**
```
src/models/analytic/black_scholes_merton/
├── base.py       # Helpers (d1_d2, validation)
├── vanilla.py    # BlackScholesMertonVanilla class
├── digital.py    # BlackScholesMertonDigitalCash/Asset classes
└── README.md
```

**After:**
```
src/models/analytic/black_scholes_merton/
├── __init__.py   # Exports all functions
├── base.py       # ALL formulas as pure functions
└── README.md     # Updated documentation
```

**Deleted Files:**
- `vanilla.py` - merged into `base.py`
- `digital.py` - merged into `base.py`

**New API:**
```python
from src.models.analytic.black_scholes_merton import (
    # Vanilla
    vanilla_price, vanilla_greeks,
    vanilla_delta, vanilla_gamma, vanilla_vega,
    vanilla_theta, vanilla_rho_discount, vanilla_rho_carry,
    
    # Digital cash-or-nothing
    digital_cash_price, digital_cash_greeks,
    digital_cash_delta, digital_cash_gamma, digital_cash_vega,
    
    # Digital asset-or-nothing
    digital_asset_price, digital_asset_greeks,
    
    # Helpers
    validate_inputs, d1_d2, forward_factor, discount_factor, intrinsic,
)
```

### 2. Black76 Implementation

**New module:**
```
src/models/analytic/black76/
├── __init__.py   # Exports all functions
├── base.py       # All Black76 formulas
└── README.md     # Documentation
```

**API:**
```python
from src.models.analytic.black76 import (
    vanilla_price, vanilla_greeks,
    vanilla_delta, vanilla_gamma, vanilla_vega,
    vanilla_theta, vanilla_rho,
    validate_inputs, d1_d2, intrinsic,
)
```

### 3. Bachelier Implementation

**New module:**
```
src/models/analytic/bachelier/
├── __init__.py   # Exports all functions
├── base.py       # All Bachelier formulas
└── README.md     # Documentation
```

**API:**
```python
from src.models.analytic.bachelier import (
    vanilla_price, vanilla_greeks,
    vanilla_delta, vanilla_gamma, vanilla_vega,
    vanilla_theta, vanilla_rho,
    validate_inputs, d_moneyness, intrinsic,
)
```

### 4. Pricer Updates

Updated both FX and Equity BSM pricers to use the new pure functions:

**FX Pricer (`src/pricers/fx/european_bsm.py`):**
```python
# Before
from src.models.analytic.black_scholes_merton.vanilla import BlackScholesMertonVanilla
engine = BlackScholesMertonVanilla()
pv = engine.price(...)

# After
from src.models.analytic.black_scholes_merton import vanilla_price
pv = vanilla_price(option_type=..., spot=..., strike=..., ...)
```

**Equity Pricer (`src/pricers/equity/european_bsm.py`):**
- Same pattern applied
- Removed class instances
- Direct function calls

---

## Architecture Summary

### Model Layer (Generic, Asset-Agnostic)

```
src/models/analytic/
├── __init__.py
├── black_scholes_merton/    # Spot-based with cost-of-carry
│   ├── __init__.py
│   ├── base.py              # All formulas
│   └── README.md
├── black76/                 # Forward-based (futures/forwards)
│   ├── __init__.py
│   ├── base.py              # All formulas
│   └── README.md
└── bachelier/               # Normal model (negative rates/spreads)
    ├── __init__.py
    ├── base.py              # All formulas
    └── README.md
```

### Pricer Layer (Asset-Specific)

```
src/pricers/
├── fx/
│   └── european_bsm.py      # Maps FX → BSM (r=r_d, b=r_d-r_f)
├── equity/
│   └── european_bsm.py      # Maps Equity → BSM (r=r, b=r-q)
└── (future)
    ├── futures/             # Uses Black76
    └── rates/               # Uses Bachelier
```

---

## Greek Naming Convention

### Generic (Model Layer)

| Greek | Definition | Description |
|-------|------------|-------------|
| `delta` | dPV/dS or dPV/dF | Underlying sensitivity |
| `gamma` | d²PV/dS² | Convexity |
| `vega` | dPV/dσ | Vol sensitivity (per 1.0) |
| `theta` | -dPV/dT | Time decay (per year) |
| `rho_discount` | dPV/dr (b fixed) | Discount rate sensitivity |
| `rho_carry` | dPV/db (r fixed) | Carry sensitivity |

### Asset-Specific (Pricer Layer)

**FX:**
```python
rho_domestic = rho_discount + rho_carry  # dPV/d(r_d)
rho_foreign = -rho_carry                 # dPV/d(r_f)
```

**Equity:**
```python
rho = rho_discount + rho_carry  # Total rate sensitivity
```

**Black76/Bachelier:**
```python
rho = -T * PV  # Single rho (simple)
```

---

## Model Comparison

| Model | Underlying | Distribution | Key Parameters |
|-------|------------|--------------|----------------|
| **BSM** | Spot S | Log-normal | `spot, strike, expiry, discount_rate, carry, vol` |
| **Black76** | Forward F | Log-normal | `forward, strike, expiry, discount_factor, vol` |
| **Bachelier** | Forward F | Normal | `forward, strike, expiry, discount_factor, vol` |

### When to Use Each

| Use Case | Model |
|----------|-------|
| Equity options | BSM |
| FX options | BSM |
| Options on futures | Black76 |
| Interest rate caps/floors | Black76 |
| Swaptions | Black76 |
| Negative rate options | Bachelier |
| Spread options | Bachelier |

---

## Benefits of New Architecture

1. **Composability**: Pure functions compose easily
2. **Testability**: No state, easy to unit test
3. **Performance**: No object overhead
4. **Clarity**: Single place for all formulas
5. **Maintainability**: Clear separation of concerns
6. **Extensibility**: Easy to add new models with same pattern

---

## Files Modified

- `src/models/analytic/black_scholes_merton/base.py` - Complete rewrite
- `src/models/analytic/black_scholes_merton/__init__.py` - Updated exports
- `src/models/analytic/black_scholes_merton/README.md` - Updated documentation
- `src/models/analytic/black76/base.py` - New file
- `src/models/analytic/black76/__init__.py` - New file
- `src/models/analytic/black76/README.md` - New file
- `src/models/analytic/bachelier/base.py` - New file
- `src/models/analytic/bachelier/__init__.py` - New file
- `src/models/analytic/bachelier/README.md` - New file
- `src/models/analytic/__init__.py` - New file
- `src/pricers/fx/european_bsm.py` - Updated to use pure functions
- `src/pricers/equity/european_bsm.py` - Updated to use pure functions

## Files Deleted

- `src/models/analytic/black_scholes_merton/vanilla.py`
- `src/models/analytic/black_scholes_merton/digital.py`

---

## Next Steps

1. Add unit tests for all three model modules
2. Implement Black76 pricers for futures options
3. Implement Bachelier pricers for interest rate products
4. Update roadmap with Phase 2.3 products

---

*This document records the model layer refactoring completed on January 27, 2026.*
