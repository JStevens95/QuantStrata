# Phase 5.1: Calibration Framework

**Status:** ✅ COMPLETE  
**Date Completed:** January 27, 2026  
**Total Tests:** 95 calibration tests passing

---

## Overview

Phase 5.1 implemented a **unified calibration framework** that provides a consistent interface for calibrating model parameters to market data. The framework supports multiple objective functions, optimizers, and model-specific calibrators.

---

## Architecture

```
src/calibration/
├── core/
│   ├── __init__.py           # Public exports
│   ├── engine.py             # CalibrationEngine, CalibrationResult
│   ├── objectives.py         # WeightedLeastSquares, PenalizedObjective, MaxLikelihood
│   └── optimizers.py         # LBFGSBConfig, DifferentialEvolutionConfig, LevenbergMarquardtConfig
├── stochastic_volatility/
│   ├── __init__.py
│   └── heston.py             # Heston calibration to vol surface
├── short_rate/
│   ├── __init__.py
│   └── hull_white.py         # Hull-White calibration to swaptions/caps
└── volatility_surface/
    ├── sabr.py               # Extended: SABR swaption smile calibration
    └── dupire.py             # Existing: local vol extraction
```

---

## Components Implemented

### 1. Calibration Core (`src/calibration/core/`)

#### CalibrationEngine (`engine.py`)

| Component | Description |
|-----------|-------------|
| `CalibrationEngine` | Generic optimizer orchestration with retry logic |
| `CalibrationResult` | Standardized result container (params, objective, convergence, timing) |
| `CalibrationConfig` | Global settings (verbose, retry, perturbation scale) |
| `calibrate()` | Convenience function for quick calibration |

**Key Features:**
- Pluggable objectives and optimizers
- Automatic retry with perturbed initial guess
- Comprehensive diagnostics (iterations, function evals, timing)
- Safe objective wrapper (handles NaN/Inf gracefully)

#### Objective Functions (`objectives.py`)

| Class | Use Case |
|-------|----------|
| `WeightedLeastSquares` | Standard vol/price fitting: Σ wᵢ(model - market)² |
| `PenalizedObjective` | Soft constraints (e.g., Feller condition) |
| `MaxLikelihood` | Probabilistic calibration: -Σ log(L) |
| `CombinedObjective` | Multi-criteria calibration |

**Factory Functions:**
- `create_vol_fitting_objective()` - Vega-weighted vol calibration
- `create_price_fitting_objective()` - Price calibration with relative errors

#### Optimizer Configurations (`optimizers.py`)

| Optimizer | Best For |
|-----------|----------|
| `LBFGSBConfig` | Local optimization with box constraints (default) |
| `DifferentialEvolutionConfig` | Global search for multi-modal objectives |
| `LevenbergMarquardtConfig` | Least-squares problems |

**Factory Functions:**
- `get_default_optimizer(problem_type)` - Returns appropriate config
- `create_global_then_local_optimizer()` - DE + L-BFGS-B polish

---

### 2. Heston Calibration (`src/calibration/stochastic_volatility/heston.py`)

**Parameters Calibrated:**
- κ (kappa): Mean reversion speed
- θ (theta): Long-term variance
- ξ (xi): Vol-of-vol
- V₀ (v0): Initial variance (optionally fixed to ATM)
- ρ (rho): Spot-variance correlation

**Functions:**
```python
calibrate_heston_to_vols(market_vols, strikes, expiries, spot, r, q, config) -> HestonCalibrationResult
calibrate_heston_to_surface(surface, spot, r, q, config) -> HestonCalibrationResult
```

**Configuration Options:**
- `fix_v0_to_atm=True` - Reduce to 4-parameter calibration
- `enforce_feller=True` - Penalize Feller condition violation
- `use_global_optimizer=True` - Use DE for robust search

**Pricing Method:** Carr-Madan FFT using Heston characteristic function

**Heston Pricing Functions Added:**
- `heston_characteristic_function()` - For FFT-based pricing
- `heston_call_price()`, `heston_put_price()` - Numerical integration
- `heston_implied_vol()` - IV extraction via root-finding
- `heston_implied_vol_surface()` - Vectorized surface computation

---

### 3. Hull-White Calibration (`src/calibration/short_rate/hull_white.py`)

**Parameters Calibrated:**
- a: Mean reversion speed
- σ: Short rate volatility

**Functions:**
```python
calibrate_hull_white_to_swaptions(swaption_vols, expiries, tenors, yield_curve_df, r0, config) -> HullWhiteCalibrationResult
calibrate_hull_white_to_caps(cap_vols, expiries, yield_curve_df, r0, config) -> HullWhiteCalibrationResult
```

**Configuration Options:**
- `vol_type="normal"/"lognormal"` - Market vol convention
- `use_atm_only=True` - ATM-only calibration (typical for HW)

**Pricing Method:** Jamshidian decomposition for swaptions, analytic caplet formula

---

### 4. SABR IR Extension (`src/calibration/volatility_surface/sabr.py`)

**New Functions:**
```python
calibrate_sabr_to_swaption_smile(strikes, market_vols, forward_swap_rate, expiry, tenor, vol_type, config) -> SabrParameters
calibrate_sabr_swaption_cube(expiries, tenors, strikes_by_point, vols_by_point, forward_by_point, vol_type, config) -> dict
```

**Key Features:**
- Supports normal vols (β=0) for rates markets
- Supports lognormal vols (β=1) for FX-style calibration
- Full cube calibration for swaption matrices

---

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/unit/calibration/core/test_engine.py` | 16 | ✅ All passing |
| `tests/unit/calibration/stochastic_volatility/test_heston_calibration.py` | 12 | ✅ All passing |
| `tests/unit/calibration/short_rate/test_hull_white_calibration.py` | 8 | ✅ All passing |
| `tests/unit/calibration/volatility_surface/test_sabr_ir.py` | 8 | ✅ All passing |
| `tests/unit/calibration/volatility_surface/test_vol_surface_sabr.py` | 27 | ✅ All passing |
| `tests/unit/calibration/volatility_surface/test_vol_surface_dupire.py` | 24 | ✅ All passing |
| **Total** | **95** | ✅ |

---

## Documentation

| Type | File |
|------|------|
| Technical Reference | `docs/reference/calibration/calibration_framework.md` |
| User Guide | `docs/guides/calibration/calibration_framework.md` |
| Tutorial | `docs/tutorials/calibration/calibration_framework.ipynb` |

**Tutorial Includes:**
- Toy calibration example with objective landscape visualization
- Heston calibration with surface heatmaps and smile overlays
- Hull-White swaption/cap calibration with error diagnostics
- SABR IR smile calibration with residual analysis

---

## Usage Examples

### Basic Calibration

```python
from src.calibration.core import CalibrationEngine, LBFGSBConfig
from src.calibration.core.objectives import WeightedLeastSquares

objective = WeightedLeastSquares(model_func=my_model, market_values=market_data)
engine = CalibrationEngine(optimizer=LBFGSBConfig(max_iter=500))
result = engine.calibrate(objective, initial_params, bounds)
```

### Heston Calibration

```python
from src.calibration.stochastic_volatility import calibrate_heston_to_surface

result = calibrate_heston_to_surface(
    surface=market_vol_surface,
    spot=100.0, r=0.05, q=0.02,
)
print(result)  # Shows κ, θ, ξ, V₀, ρ + diagnostics
```

### Hull-White Calibration

```python
from src.calibration.short_rate import calibrate_hull_white_to_swaptions

result = calibrate_hull_white_to_swaptions(
    swaption_vols=market_vols,
    expiries=expiries, tenors=tenors,
    yield_curve_df=df, r0=0.03,
)
print(f"a={result.params.a:.4f}, σ={result.params.sigma:.4f}")
```

---

## Design Decisions

1. **Reuse existing SABR pattern**: The existing SABR calibration was well-designed; we extracted the common pattern into `CalibrationEngine`

2. **Heston FFT pricing**: For calibration speed, use Carr-Madan numerical integration rather than Monte Carlo

3. **Hull-White uses existing pricing**: Leverages `hw_swaption_price_jamshidian` already in the codebase

4. **Global + local optimization**: For difficult surfaces (Heston), use `DifferentialEvolutionConfig(polish=True)` for DE followed by L-BFGS-B refinement

5. **Backward compatible**: Existing `calibrate_sabr_to_smile` continues to work; new engine is optional

---

## Next Steps (Phase 5.2+)

- Backtesting infrastructure
- Historical data integration
- Performance attribution
- VaR/CVaR implementation
- Greeks aggregation enhancements

---

*Phase 5.1 Complete | QuantStrata Calibration Framework | January 2026*
