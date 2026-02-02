# Phase 3.7: Rate Infrastructure Enhancement - Progress Report

## Status: COMPLETE ✅

**Completion Date:** January 2026

## Summary

Phase 3.7 enhances the interest rate market data infrastructure with swaption volatility cubes, cap/floor volatility surfaces, and improved curve validation.

## Deliverables

### 1. Swaption Volatility Cube

**File:** `src/marketdata/surfaces/vol_surface.py` (consolidated)

Implemented components:
- `SwaptionVolCube`: 3D volatility surface (expiry × tenor × strike)
  - Trilinear interpolation
  - Support for both absolute and relative-to-ATM strikes
  - Normal (Bachelier) and log-normal (Black) vol types
  - Flat extrapolation beyond grid boundaries
  - ATM vol retrieval
  - Smile extraction at any (expiry, tenor) point
- `FlatSwaptionVolCube`: Constant volatility for testing/baseline

**Key Features:**
- Industry-standard representation for swaption volatilities
- Post-2015 market convention support (normal vol in bp)
- Efficient trilinear interpolation

### 2. Cap/Floor Volatility Surface

**File:** `src/marketdata/surfaces/vol_surface.py` (consolidated)

Implemented components:
- `CapFloorVolSurface`: 2D volatility surface (expiry × strike)
  - Bilinear interpolation
  - Normal and log-normal vol types
  - Smile extraction
- `FlatCapFloorVolSurface`: Constant volatility for testing

### 3. Factory Functions

**File:** `src/marketdata/surfaces/vol_surface.py` (consolidated)

- `create_atm_swaption_vol_cube()`: Build cube from ATM vols with parameterized smile
- `create_cap_vol_surface_from_term_structure()`: Build surface from ATM term structure

### 4. Module Integration

**File:** `src/marketdata/surfaces/__init__.py`

Exports all vol surface classes from consolidated `vol_surface.py`.

### 5. Unit Tests

**File:** `tests/unit/marketdata/surfaces/test_surfaces_vol_surface.py`

27 tests covering (4 existing + 23 new):
- SwaptionVolCube creation and validation
- Interpolation (on-grid, between points, extrapolation)
- ATM and smile retrieval
- CapFloorVolSurface functionality
- Factory function behavior
- Integration with pricer interfaces

## Architecture

```
src/marketdata/surfaces/
├── __init__.py              # Module exports
├── vol_surface.py           # All vol surfaces (FX/Equity + IR)
│   ├── FlatVolSurface
│   ├── GridVolSurface
│   ├── SwaptionVolCube      # 3D: expiry × tenor × strike
│   ├── FlatSwaptionVolCube
│   ├── CapFloorVolSurface   # 2D: expiry × strike
│   ├── FlatCapFloorVolSurface
│   ├── create_atm_swaption_vol_cube()
│   └── create_cap_vol_surface_from_term_structure()
└── local_vol_surface.py     # Local vol surfaces
```

## Test Results

```
tests/unit/marketdata/surfaces/test_surfaces_vol_surface.py .......... 27 passed
============================== 27 passed ==============================
```

## Key Design Decisions

### 1. 3D Cube vs 2D Surfaces

Swaptions require a 3D structure (expiry × tenor × strike) because the forward swap rate depends on both option expiry and underlying swap tenor. This differs from FX/equity options which only need (expiry × strike).

### 2. Relative vs Absolute Strikes

Supported both:
- **Absolute strikes**: Direct strike rates (e.g., 2.5%, 3.0%)
- **Relative strikes**: Offset from ATM (e.g., -100bp, ATM, +100bp)

Market data is often quoted relative to ATM.

### 3. Vol Type Enum

Used `Literal["normal", "lognormal"]` to clearly indicate volatility convention:
- **Normal (Bachelier)**: Post-2015 standard, works with negative rates
- **Log-normal (Black)**: Pre-2015 standard

### 4. Flat Surfaces for Testing

Included flat versions of each surface for:
- Unit testing
- Baseline pricing
- Quick prototyping

## Existing Infrastructure Noted

The following already existed and was noted as adequate:
- Curve bootstrapping from deposits, FRAs, swaps (`curves/bootstrapper.py`)
- OIS support in quote types (`quotes/rates.py`)
- Arbitrage validation for vol surfaces (`surfaces/validation/arbitrage.py`)

## Usage Examples

### Swaption Vol Cube

```python
from src.marketdata.surfaces import SwaptionVolCube

cube = SwaptionVolCube(
    expiries=np.array([1.0, 2.0, 5.0, 10.0]),
    tenors=np.array([2.0, 5.0, 10.0, 30.0]),
    strikes=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
    vols=vol_data,  # Shape (4, 4, 5)
    vol_type="normal",
)

# Get vol for 5Y x 10Y swaption at 3% strike
vol = cube.implied_vol(expiry=5.0, tenor=10.0, strike=0.03)
```

### Cap/Floor Vol Surface

```python
from src.marketdata.surfaces import CapFloorVolSurface

surface = CapFloorVolSurface(
    expiries=np.array([1.0, 2.0, 5.0, 10.0]),
    strikes=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
    vols=vol_data,  # Shape (4, 5)
    vol_type="normal",
)

# Get vol for 5Y cap at 3% strike
vol = surface.implied_vol(expiry=5.0, strike=0.03)
```

## Impact

Phase 3.7 completes the rate infrastructure needed for:
- Swaption pricing with realistic vol surfaces
- Cap/floor pricing with market-calibrated vols
- Model calibration to IR vol markets
- Risk management with proper vol dependencies

## Next Steps

Phase 3.8 (Future): LIBOR Market Model
- Multi-factor forward rate model
- Correlation structures
- Advanced smile dynamics
