# Volatility Surfaces

**Complete Technical Specification for Implied and Local Volatility Surfaces**

This document provides a comprehensive guide to the volatility surface implementations in QuantStrata, including implied vol surfaces (flat and grid-based) and local volatility surfaces for exotic pricing.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Volatility Types and Use Cases](#2-volatility-types-and-use-cases)
3. [FlatVolSurface](#3-flatvolsurface)
4. [GridVolSurface](#4-gridvolsurface)
5. [LocalVolSurface](#5-localvolsurface)
6. [Strike Space Conventions](#6-strike-space-conventions)
7. [Interpolation Methods](#7-interpolation-methods)
8. [Factory Pattern](#8-factory-pattern)
9. [FX vs Equity Conventions](#9-fx-vs-equity-conventions)
10. [Calibration Integration](#10-calibration-integration)

---

## 1. Executive Summary

### 1.1 Overview

QuantStrata provides three volatility surface types:

| Surface Type | Function | Use Case |
|--------------|----------|----------|
| `FlatVolSurface` | $\sigma(T, K) = \sigma_0$ | Testing, baseline pricing |
| `GridVolSurface` | $\sigma(T, K)$ on 2D grid | Real-world implied vol |
| `LocalVolSurface` | $\sigma_{LV}(S, t)$ | Exotic pricing, Dupire model |

### 1.2 Key Distinction

**Implied Volatility** vs **Local Volatility**:

| Aspect | Implied Vol $\sigma_{BS}(T, K)$ | Local Vol $\sigma_{LV}(S, t)$ |
|--------|--------------------------------|------------------------------|
| **Axes** | Expiry × Strike | Spot × Time |
| **Meaning** | BS input to match market prices | Instantaneous diffusion |
| **Source** | Market quotes | Calibrated from implied vol |
| **Use** | Vanilla pricing | Exotic pricing (barriers, etc.) |

### 1.3 Module Location

```
src/marketdata/surfaces/
├── vol_surface.py        # FlatVolSurface, GridVolSurface
├── local_vol_surface.py  # LocalVolSurface, FlatLocalVolSurface
├── factory.py            # FlatVolFactory, GridVolFactory
└── validation/
    └── arbitrage.py      # Surface validation utilities
```

---

## 2. Volatility Types and Use Cases

### 2.1 Implied Volatility

**Definition:** The volatility $\sigma$ that, when input to Black-Scholes, reproduces a market option price.

$$C_{market} = C_{BS}(S, K, T, r, q, \sigma_{impl})$$

**Characteristics:**
- Varies with strike and expiry
- Observable from option markets
- Not directly usable in path-dependent pricing

**Use Cases:**
- Vanilla option pricing
- Delta hedging
- Vol surface visualization

### 2.2 Local Volatility

**Definition:** The instantaneous volatility as a function of spot and time.

$$dS_t = (r - q) S_t \, dt + \sigma_{LV}(S_t, t) S_t \, dW_t$$

**Characteristics:**
- Deterministic function of $(S, t)$
- Uniquely determined by implied vol surface (Dupire)
- Provides exact fit to all vanilla prices

**Use Cases:**
- Barrier options
- Asian options
- Path-dependent exotics
- Monte Carlo simulation

### 2.3 Relationship: Implied ↔ Local

**Dupire's Formula:**

$$\sigma_{LV}^2(K, T) = \frac{\frac{\partial C}{\partial T} + (r-q)K \frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2 \frac{\partial^2 C}{\partial K^2}}$$

**Interpretation:**
- Local vol is *derived* from implied vol (or market prices)
- The calibration is unique (given complete market data)
- Local vol at $S = K$ equals "instantaneous" forward vol

---

## 3. FlatVolSurface

### 3.1 Purpose

The simplest volatility surface: constant vol everywhere.

$$\sigma(T, K) = \sigma_0 \quad \forall \, T, K$$

**Use Cases:**
- Unit testing
- Baseline comparisons
- Quick prototyping
- Black-Scholes pricing

### 3.2 Implementation

```python
@dataclass(frozen=True, slots=True)
class FlatVolSurface:
    sigma: float  # Constant volatility (e.g., 0.20 for 20%)
```

### 3.3 API

```python
from src.marketdata.surfaces.vol_surface import FlatVolSurface

# Create a 20% flat vol surface
surface = FlatVolSurface(sigma=0.20)

# Query vol (strike is ignored)
vol = surface.implied_vol(expiry=1.0, strike=100.0)  # → 0.20
vol = surface.vol(expiry=0.5, strike=120.0)          # → 0.20 (alias)
```

### 3.4 Validation

On construction:
- `sigma` must be finite
- `sigma` must be strictly positive

On query:
- `expiry` must be finite and ≥ 0
- `strike` is accepted but ignored

---

## 4. GridVolSurface

### 4.1 Purpose

A 2D implied volatility surface on a grid of (expiry, strike) with bilinear interpolation.

$$\sigma(T, K) = \text{interp}(T, K \,|\, \text{grid})$$

**Use Cases:**
- Real-world FX and equity vol surfaces
- Production pricing
- Vol surface visualization

### 4.2 Implementation

```python
@dataclass(frozen=True, slots=True)
class GridVolSurface:
    expiries: np.ndarray       # Shape (n_exp,), strictly increasing
    strikes: np.ndarray        # Shape (n_k,), strictly increasing
    implied_vols: np.ndarray   # Shape (n_exp, n_k)
    extrapolation: str = "flat"  # "flat" or "error"
    strike_space: str = "absolute"  # Metadata
    surface_id: str | None = None   # Optional identifier
```

### 4.3 Grid Structure

```
               strikes (K)
        ┌──────────────────────────┐
        │ K₁    K₂    K₃    K₄    │
        ├──────────────────────────┤
  T₁    │ σ₁₁   σ₁₂   σ₁₃   σ₁₄  │
expiries│                          │
  T₂    │ σ₂₁   σ₂₂   σ₂₃   σ₂₄  │
  (T)   │                          │
  T₃    │ σ₃₁   σ₃₂   σ₃₃   σ₃₄  │
        └──────────────────────────┘
```

### 4.4 API

```python
import numpy as np
from src.marketdata.surfaces.vol_surface import GridVolSurface

# Define grid
expiries = np.array([0.25, 0.5, 1.0, 2.0])
strikes = np.array([80, 90, 100, 110, 120])

# Create vol grid (4 expiries × 5 strikes)
implied_vols = np.array([
    [0.25, 0.22, 0.20, 0.21, 0.23],  # T=0.25
    [0.24, 0.21, 0.19, 0.20, 0.22],  # T=0.5
    [0.23, 0.20, 0.18, 0.19, 0.21],  # T=1.0
    [0.22, 0.19, 0.17, 0.18, 0.20],  # T=2.0
])

surface = GridVolSurface(
    expiries=expiries,
    strikes=strikes,
    implied_vols=implied_vols,
    extrapolation="flat",
)

# Query at grid point
vol = surface.implied_vol(expiry=0.5, strike=100)  # → 0.19

# Query between grid points (interpolated)
vol = surface.implied_vol(expiry=0.75, strike=95)  # → bilinear interp

# Query outside grid (extrapolated flat)
vol = surface.implied_vol(expiry=3.0, strike=150)  # → edge value
```

### 4.5 Validation

On construction:
- `expiries`: non-empty, finite, ≥ 0, strictly increasing
- `strikes`: non-empty, finite, strictly increasing
- `implied_vols`: shape must be `(n_exp, n_strikes)`
- `implied_vols`: must be finite and strictly positive
- `extrapolation`: must be "flat" or "error"

### 4.6 Extrapolation Modes

| Mode | Behavior |
|------|----------|
| `"flat"` | Clamp queries to grid edges (recommended) |
| `"error"` | Raise `ValueError` if outside grid |

---

## 5. LocalVolSurface

### 5.1 Purpose

A 2D local volatility surface $\sigma_{LV}(S, t)$ for use in Dupire's model.

$$dS_t = (r - q) S_t \, dt + \sigma_{LV}(S_t, t) S_t \, dW_t$$

**Use Cases:**
- Monte Carlo simulation with vol smile
- Finite difference PDE pricing
- Barrier and exotic option pricing

### 5.2 Implementation

```python
@dataclass(frozen=True, slots=True)
class LocalVolSurface:
    times: np.ndarray       # Shape (n_times,), time axis
    spots: np.ndarray       # Shape (n_spots,), spot axis
    local_vols: np.ndarray  # Shape (n_times, n_spots)
    extrapolation: str = "flat"
    surface_id: str | None = None
```

### 5.3 Grid Structure

```
                 spots (S)
        ┌──────────────────────────┐
        │ S₁    S₂    S₃    S₄    │
        ├──────────────────────────┤
  t₁    │ σ₁₁   σ₁₂   σ₁₃   σ₁₄  │
 times  │                          │
  t₂    │ σ₂₁   σ₂₂   σ₂₃   σ₂₄  │
  (t)   │                          │
  t₃    │ σ₃₁   σ₃₂   σ₃₃   σ₃₄  │
        └──────────────────────────┘
```

**Note:** The axes are (time, spot), different from GridVolSurface's (expiry, strike).

### 5.4 API

```python
import numpy as np
from src.marketdata.surfaces.local_vol_surface import LocalVolSurface

# Define grid
times = np.array([0.0, 0.5, 1.0])
spots = np.array([80.0, 100.0, 120.0])

# Create local vol grid (equity skew pattern)
local_vols = np.array([
    [0.25, 0.20, 0.18],  # t=0.0
    [0.24, 0.19, 0.17],  # t=0.5
    [0.23, 0.18, 0.16],  # t=1.0
])

surface = LocalVolSurface(times=times, spots=spots, local_vols=local_vols)

# Query local vol
vol = surface.local_vol(spot=100.0, time=0.5)  # → 0.19

# Callable interface
vol = surface(spot=100.0, time=0.5)  # → same as above

# Properties
print(surface.time_range)  # → (0.0, 1.0)
print(surface.spot_range)  # → (80.0, 120.0)
print(surface.shape)       # → (3, 3)
```

### 5.5 FlatLocalVolSurface

For constant local volatility (Black-Scholes equivalent):

```python
from src.marketdata.surfaces.local_vol_surface import FlatLocalVolSurface

surface = FlatLocalVolSurface(sigma=0.20)
vol = surface.local_vol(spot=100.0, time=0.5)  # → 0.20 always
```

---

## 6. Strike Space Conventions

### 6.1 Available Strike Spaces

```python
StrikeSpace = Literal[
    "absolute",              # K in price units (canonical)
    "spot_moneyness",        # K = m × S₀
    "forward_moneyness",     # K = m × F₀(T)
    "log_forward_moneyness", # log(K/F₀(T))
]
```

### 6.2 Recommendations

| Strike Space | Storage | Pricing | Notes |
|--------------|---------|---------|-------|
| `"absolute"` | ✅ Recommended | ✅ Required | Standard for production |
| `"spot_moneyness"` | ⚠️ Avoid | ❌ No | Needs spot context |
| `"forward_moneyness"` | ⚠️ Avoid | ❌ No | Needs forward context |
| `"log_forward_moneyness"` | ⚠️ Avoid | ❌ No | FX convention only |

### 6.3 Conversion

If you receive data in moneyness space, convert to absolute before storage:

```python
# From FX delta-based quotes
K = F * np.exp(log_moneyness)

# From spot moneyness
K = S_0 * moneyness
```

---

## 7. Interpolation Methods

### 7.1 Bilinear Interpolation

Both `GridVolSurface` and `LocalVolSurface` use bilinear interpolation:

$$f(x_q, y_q) = (1-t_x)(1-t_y) f_{00} + t_x(1-t_y) f_{10} + (1-t_x)t_y f_{01} + t_x t_y f_{11}$$

Where:
- $(x_0, y_0)$ and $(x_1, y_1)$ are the bounding grid points
- $t_x = \frac{x_q - x_0}{x_1 - x_0}$
- $t_y = \frac{y_q - y_0}{y_1 - y_0}$

### 7.2 Edge Cases

| Scenario | Behavior |
|----------|----------|
| Single expiry | Linear interpolation in strike |
| Single strike | Linear interpolation in expiry |
| Single point | Return that point |
| Outside grid | Flat extrapolation (clamp) |

### 7.3 Interpolation Quality

For production surfaces, consider:
- Denser grid near ATM
- More expiry points at short maturities
- Smooth grid spacing to reduce interpolation error

---

## 8. Factory Pattern

### 8.1 Purpose

Factories reconstruct vol surfaces from parameter arrays stored in `MarketDataset`:

```
MarketDataset
    │
    ├── vol_params[mid] = Panel(data=[T,S,n_exp,n_k], ...)
    └── vol_factories[mid] = GridVolFactory(expiries, strikes)
           │
           │  factory.build(params[t, s, :, :])
           ▼
       GridVolSurface
```

### 8.2 FlatVolFactory

```python
from src.marketdata.surfaces.factory import FlatVolFactory

factory = FlatVolFactory()
surface = factory.build(params=np.array([0.20]))  # → FlatVolSurface(sigma=0.20)
```

### 8.3 GridVolFactory

```python
from src.marketdata.surfaces.factory import GridVolFactory

factory = GridVolFactory(
    expiries=np.array([0.25, 0.5, 1.0]),
    strikes=np.array([80, 100, 120]),
    extrapolation="flat",
)

# params shape: (n_exp × n_strikes) or (n_exp, n_strikes)
params = vol_cube[time_idx, scenario_idx, :, :]  # Shape: (3, 3)
surface = factory.build(params)  # → GridVolSurface
```

### 8.4 Factory in MarketDataset

```python
# During snapshot extraction
for mkt_id, panel in dataset.vol_params.items():
    params_block = panel.data[time_idx, scenario_idx, ...]
    factory = dataset.vol_factories[mkt_id]
    vol_surface = factory.build(params_block)
    # Store in Market.vols
```

---

## 9. FX vs Equity Conventions

### 9.1 FX Volatility Surfaces

**Quoting Convention:** Delta-based
- 25Δ Call, 25Δ Put
- 10Δ Call, 10Δ Put
- ATM (typically DNS or Forward Delta Neutral Straddle)

**Surface Parameterization:**
- Forward moneyness: $m = \ln(K/F)$
- Smile symmetric around ATM
- Risk reversals and butterflies

**Example FX Vol Grid:**
```python
# Expiries in years
expiries = np.array([0.25, 0.5, 1.0, 2.0])

# Strikes from delta-to-strike conversion
strikes = np.array([1.05, 1.08, 1.10, 1.12, 1.15])  # EURUSD strikes

# Vol grid (smile shape)
implied_vols = np.array([
    [0.12, 0.11, 0.10, 0.11, 0.12],  # Smile at T=0.25
    ...
])
```

### 9.2 Equity Volatility Surfaces

**Quoting Convention:** Strike-based
- Fixed strike grid (e.g., 80%, 90%, 100%, 110%, 120% of spot)
- Or fixed delta grid converted to strikes

**Surface Parameterization:**
- Spot moneyness: $m = (K - S)/S$
- Negative skew (put wing higher than call wing)
- "Smirk" shape rather than symmetric smile

**Example Equity Vol Grid:**
```python
# Expiries in years
expiries = np.array([0.25, 0.5, 1.0, 2.0])

# Strikes in absolute terms
strikes = np.array([80, 90, 100, 110, 120])

# Vol grid (negative skew)
implied_vols = np.array([
    [0.30, 0.25, 0.20, 0.22, 0.25],  # T=0.25: higher at low strikes
    [0.28, 0.23, 0.19, 0.21, 0.24],  # T=0.5
    [0.26, 0.21, 0.18, 0.20, 0.23],  # T=1.0
    [0.24, 0.19, 0.17, 0.19, 0.22],  # T=2.0
])
```

### 9.3 Comparison Table

| Aspect | FX | Equity |
|--------|----|----|
| **Quoting** | Delta-based | Strike-based |
| **Moneyness** | Forward ($K/F$) | Spot ($(K-S)/S$) |
| **Shape** | Smile (symmetric) | Smirk (negative skew) |
| **Storage** | Absolute strikes | Absolute strikes |
| **Typical ATM** | 8-15% | 15-30% |

---

## 10. Calibration Integration

### 10.1 Dupire Calibration

Convert implied vol surface to local vol:

```python
from src.calibration.volatility_surface.dupire import DupireCalibrator, DupireConfig

# Create calibrator
calibrator = DupireCalibrator(config=DupireConfig())

# Calibrate at a single point
local_vol = calibrator.local_vol_at_point(
    implied_surface=grid_vol_surface,
    spot=100.0,
    strike=100.0,
    expiry=1.0,
    r=0.05,  # Risk-free rate
    q=0.02,  # Dividend yield (or foreign rate for FX)
)

# Calibrate full grid
local_vol_surface = calibrator.calibrate_grid(
    implied_surface=grid_vol_surface,
    spot=100.0,
    r=0.05,
    q=0.02,
    times=np.array([0.0, 0.25, 0.5, 1.0]),
    spots=np.array([80, 90, 100, 110, 120]),
)
```

### 10.2 SABR Calibration

For parametric vol surface fitting (FX):

```python
from src.calibration.volatility_surface.sabr import SABRCalibrator

# Calibrate SABR parameters from market quotes
sabr_params = calibrator.fit(
    forward=1.10,
    expiry=1.0,
    strikes=market_strikes,
    market_vols=market_vols,
)

# Generate vol surface from SABR
vol = sabr_vol(strike=1.12, forward=1.10, expiry=1.0, **sabr_params)
```

### 10.3 Workflow

```
Market Quotes
    │
    ├── FX: Delta-based quotes
    │       → Convert to strikes
    │       → GridVolSurface
    │
    └── Equity: Strike-based quotes
            → GridVolSurface
    │
    │  Dupire Calibration
    ▼
LocalVolSurface
    │
    │  Use in pricing
    ▼
Monte Carlo / Finite Difference
```

---

## References

- `src/marketdata/surfaces/vol_surface.py` - Implied vol surfaces
- `src/marketdata/surfaces/local_vol_surface.py` - Local vol surfaces
- `src/marketdata/surfaces/factory.py` - Factory implementations
- `src/calibration/volatility_surface/dupire.py` - Dupire calibration
- `docs/mathematics/local_volatility.md` - Mathematical theory
- `docs/mathematics/volatility_calibration.md` - Calibration methods
