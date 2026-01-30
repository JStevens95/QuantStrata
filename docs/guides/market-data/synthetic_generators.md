# Synthetic Market Data Generators

**Complete Technical Specification for Deterministic Market Data Generation**

This document provides a comprehensive guide to the synthetic market data generators, including mathematical models, configuration options, and usage patterns for FX, Interest Rates, and Equity asset classes.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture](#2-architecture)
3. [FX Generators](#3-fx-generators)
4. [Interest Rate Generators](#4-interest-rate-generators)
5. [Equity Generators](#5-equity-generators)
6. [Configuration](#6-configuration)
7. [Dependency Resolution](#7-dependency-resolution)
8. [Determinism and Reproducibility](#8-determinism-and-reproducibility)
9. [Extending with New Generators](#9-extending-with-new-generators)
10. [Best Practices](#10-best-practices)

---

## 1. Executive Summary

### 1.1 Purpose

Synthetic generators produce deterministic, market-like data for:
- **Testing** - Reproducible unit and integration tests
- **Examples** - Documentation and demos
- **Development** - Work without live data feeds
- **Backtesting** - Monte Carlo scenario generation

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Determinism** | Same seed + MarketId always produces identical data |
| **Dependency Closure** | VOL surfaces auto-request SPOT and curves |
| **Asset-Class Specific** | FX (delta-based), Equity (strike-based), IR (curves) |
| **Configurable** | Per-MarketId specification overrides |

### 1.3 Supported Asset Classes

| Asset Class | SPOT | FIXING | CURVE | VOL |
|-------------|------|--------|-------|-----|
| **FX** | ✅ GBM | ✅ Reuse SPOT | - | ✅ Forward-moneyness smile |
| **IR** | - | - | ✅ Zero rates | - |
| **EQUITY** | ✅ GBM | ✅ Reuse SPOT | - | ✅ Strike-based smile |

---

## 2. Architecture

### 2.1 Component Diagram

```
SyntheticProvider
        │
        ▼
SyntheticRegistry ──────────────────────────────────────────────┐
   │                                                            │
   │  register(asset_class, mkt_type, generator, requirements)  │
   │                                                            │
   │  (FX, SPOT)     → FxGenerators.generate_spot              │
   │  (FX, VOL)      → FxGenerators.generate_vol + reqs        │
   │  (IR, CURVE)    → IrGenerators.generate_curve             │
   │  (EQUITY, SPOT) → EquityGenerators.generate_spot          │
   │  (EQUITY, VOL)  → EquityGenerators.generate_vol + reqs    │
   └────────────────────────────────────────────────────────────┘
        │
        ▼
SyntheticMarketEngine
   │
   │  1. Expand universe (dependency closure)
   │  2. Topological sort (SPOT before VOL)
   │  3. Generate each MarketId
   │  4. Assemble MarketDataset
   │
        ▼
   MarketDataset
```

### 2.2 Generator Protocol

All generators follow this signature:

```python
def generate(market_id: MarketId, state: SyntheticGenerationState) -> None:
    """
    Generate data for market_id and store in state.
    
    State containers:
    - state.quote_panels[market_id] = Panel(...)      # For SPOT/FIXING
    - state.curve_param_panels[market_id] = Panel(...) # For CURVE
    - state.curve_factories[market_id] = Factory(...)  # For CURVE
    - state.vol_param_panels[market_id] = Panel(...)   # For VOL
    - state.vol_factories[market_id] = Factory(...)    # For VOL
    """
```

### 2.3 Module Location

```
src/marketdata/providers/synthetic/
├── provider.py         # SyntheticProvider facade
├── engine.py           # SyntheticMarketEngine
├── registry.py         # SyntheticRegistry
├── context.py          # SyntheticGenerationState
├── config.py           # SyntheticProviderConfig
├── specs.py            # SpotGbmSpec, CurveZeroSpec, VolGridSmileSpec
└── generators/
    ├── base.py           # Generator protocol
    ├── foreign_exchange.py  # FX generators
    ├── interest_rate.py     # IR generators
    └── equity.py            # Equity generators
```

---

## 3. FX Generators

### 3.1 Overview

FX generators produce:
- **SPOT**: Exchange rate time series via GBM
- **FIXING**: Reuses SPOT (or constant fallback)
- **VOL**: Forward-moneyness smile surface

### 3.2 FX SPOT Generator

**Model: Geometric Brownian Motion**

$$S_{t+dt} = S_t \cdot \exp\left[(\mu - \frac{1}{2}\sigma^2)dt + \sigma\sqrt{dt} \cdot Z\right]$$

Where:
- $S_t$: Spot rate at time $t$
- $\mu$: Drift parameter
- $\sigma$: Volatility
- $dt$: Time step
- $Z \sim \mathcal{N}(0,1)$

**Parameters:**
```python
@dataclass
class SpotGbmSpec:
    initial_level: float      # Starting spot (e.g., 1.10 for EURUSD)
    drift: float = 0.0        # μ parameter
    vol: float = 0.10         # σ parameter (10% annualized)
    dt: float = 1/252         # Time step (1 trading day)
    initial_dispersion: float = 0.0  # Cross-scenario dispersion at t=0
```

**Output:**
```python
# Panel shape: [n_time, n_scenarios]
# Axis names: ("time", "scenario")
state.quote_panels[market_id] = Panel(data=spot_array, axis_names=("time", "scenario"))
```

**Example:**
```python
from src.marketdata.core.ids import MarketId

spot_id = MarketId(
    asset_class="FX",
    mkt_type="SPOT",
    name="EURUSD",
    qualifiers=(("dom", "USD"), ("for", "EUR")),
)
# → "FX.SPOT.EURUSD|dom=USD|for=EUR"
```

### 3.3 FX VOL Generator

**Model: Forward-Moneyness Smile**

FX vol surfaces use forward-moneyness parameterization:

$$m = \ln\left(\frac{K}{F(T)}\right)$$

Where:
- $K$: Strike
- $F(T) = S \cdot e^{(r_{dom} - r_{for}) \cdot T}$: Forward price

**Smile Formula:**
$$\sigma(T, K) = \text{ATM}(T) \cdot \left(1 + \text{skew} \cdot m + \text{smile} \cdot m^2\right)$$

**Term Structure:**
$$\text{ATM}(T) = \sigma_{ATM} \cdot \left(1 + \text{term} \cdot (\sqrt{T} - \sqrt{T_{ref}})\right)$$

**Parameters:**
```python
@dataclass
class VolGridSmileSpec:
    expiries: np.ndarray      # [0.25, 0.5, 1.0] years
    strikes: np.ndarray       # [0.9, 1.0, 1.1] absolute strikes
    atm_vol: float = 0.12     # 12% ATM vol
    skew: float = -0.15       # Skew coefficient
    smile: float = 0.20       # Smile (convexity) coefficient
    term: float = 0.10        # Term structure coefficient
    noise_scale: float = 0.002  # Random noise
```

**Output:**
```python
# Panel shape: [n_time, n_scenarios, n_expiries, n_strikes]
# Axis names: ("time", "scenario", "expiry", "strike")
state.vol_param_panels[market_id] = Panel(data=vol_cube, ...)
state.vol_factories[market_id] = GridVolFactory(expiries, strikes)
```

**Dependencies:**
FX VOL requires:
1. Corresponding FX SPOT (same pair)
2. Domestic IR CURVE (if `dom` qualifier present)
3. Foreign IR CURVE (if `for` qualifier present)

```python
# Automatically resolved via requirements_for_vol()
def requirements_for_vol(market_id: MarketId) -> Tuple[MarketId, ...]:
    spot_id = MarketId("FX", "SPOT", market_id.name, market_id.qualifiers)
    dom_curve_id = MarketId("IR", "CURVE", f"{dom_ccy}.OIS", (("ccy", dom_ccy),))
    for_curve_id = MarketId("IR", "CURVE", f"{for_ccy}.OIS", (("ccy", for_ccy),))
    return (spot_id, dom_curve_id, for_curve_id)
```

---

## 4. Interest Rate Generators

### 4.1 Overview

IR generators produce zero rate curves with configurable term structure.

### 4.2 IR CURVE Generator

**Model: Parametric Zero Curve**

$$r(\tau) = r_{base} + \text{slope} \cdot \tau + \text{curvature} \cdot e^{-\tau} + \varepsilon$$

Where:
- $r(\tau)$: Zero rate at tenor $\tau$
- $r_{base}$: Base rate level
- $\text{slope}$: Term premium
- $\text{curvature}$: Short-end convexity
- $\varepsilon \sim \mathcal{N}(0, \sigma^2)$: Optional noise

**Parameters:**
```python
@dataclass
class CurveZeroSpec:
    tenors: np.ndarray        # [0.25, 0.5, 1.0, 2.0, 5.0, 10.0] years
    base_rate: float = 0.02   # 2% base rate
    slope: float = 0.00       # Flat by default
    curvature: float = 0.00   # No curvature by default
    noise_scale: float = 0.0005  # 5bp noise
    extrapolation: str = "flat"  # Extrapolation method
```

**Output:**
```python
# Panel shape: [n_time, n_scenarios, n_tenors, 2]
# Last dimension: [tenor, zero_rate]
state.curve_param_panels[market_id] = Panel(
    data=params,
    axis_names=("time", "scenario", "tenor", "cols"),
)
state.curve_factories[market_id] = ZeroRateCurveFactory(extrapolation="flat")
```

**Example:**
```python
curve_id = MarketId(
    asset_class="IR",
    mkt_type="CURVE",
    name="USD.OIS",
    qualifiers=(("ccy", "USD"),),
)
# → "IR.CURVE.USD.OIS|ccy=USD"
```

### 4.3 Curve Shapes

**Normal (Upward Sloping):**
```python
CurveZeroSpec(
    tenors=np.array([0.25, 0.5, 1, 2, 5, 10]),
    base_rate=0.02,
    slope=0.005,  # 50bp per year
    curvature=0.0,
)
```

**Inverted:**
```python
CurveZeroSpec(
    tenors=np.array([0.25, 0.5, 1, 2, 5, 10]),
    base_rate=0.05,
    slope=-0.003,  # -30bp per year
    curvature=0.0,
)
```

**Humped:**
```python
CurveZeroSpec(
    tenors=np.array([0.25, 0.5, 1, 2, 5, 10]),
    base_rate=0.02,
    slope=0.002,
    curvature=0.01,  # Short-end bump
)
```

---

## 5. Equity Generators

### 5.1 Overview

Equity generators produce:
- **SPOT**: Stock price time series via GBM
- **FIXING**: Reuses SPOT (or constant fallback)
- **VOL**: Strike-based smile surface (equity convention)

### 5.2 Equity SPOT Generator

**Model: GBM with Dividend Yield**

$$S_{t+dt} = S_t \cdot \exp\left[(\mu - q - \frac{1}{2}\sigma^2)dt + \sigma\sqrt{dt} \cdot Z\right]$$

Where:
- $q$: Continuous dividend yield (embedded in drift)

**Parameters:** Same as FX `SpotGbmSpec` with drift adjusted for dividends.

**Example:**
```python
spot_id = MarketId(
    asset_class="EQUITY",
    mkt_type="SPOT",
    name="AAPL",
    qualifiers=(("ccy", "USD"),),
)
# → "EQUITY.SPOT.AAPL|ccy=USD"
```

### 5.3 Equity VOL Generator

**Model: Strike-Based Smile**

Unlike FX (forward-moneyness), equity uses spot moneyness:

$$m = \frac{K - S_0}{S_0}$$

**Smile Formula:**
$$\sigma(T, K) = \text{ATM}(T) \cdot \left(1 + \text{skew} \cdot m + \text{smile} \cdot m^2\right)$$

**Key Difference: Negative Skew**

Equity markets typically exhibit **negative skew**:
- Lower strikes → Higher vol (crash protection)
- Higher strikes → Lower vol

```python
# Typical equity skew
VolGridSmileSpec(
    expiries=np.array([0.25, 0.5, 1.0]),
    strikes=np.array([80, 90, 100, 110, 120]),
    atm_vol=0.20,
    skew=-0.10,  # Negative! (vol increases as K decreases)
    smile=0.05,
)
```

**Comparison: FX vs Equity Vol Convention**

| Aspect | FX | Equity |
|--------|----|----|
| **Moneyness** | Forward: $\ln(K/F)$ | Spot: $(K-S)/S$ |
| **Typical Skew** | Can be positive or negative | Typically negative |
| **Smile** | "Smile" shape (both wings up) | "Smirk" shape (put wing up) |
| **Quoting** | Delta-based (25Δ, 10Δ) | Strike-based |

**Dependencies:**
Equity VOL requires:
1. Corresponding EQUITY SPOT (same ticker)
2. Discount CURVE (if `ccy` qualifier present)

### 5.4 Dividend Models

Equity generators include dividend handling utilities:

**Discrete Dividend Adjustment:**
```python
from src.marketdata.providers.synthetic.generators.equity import (
    adjust_spot_for_discrete_dividend,
    compute_forward_with_dividends,
)

# Adjust spot for future dividend
adjusted = adjust_spot_for_discrete_dividend(
    spot=100.0,
    dividend_amount=2.0,
    ex_date_fraction=0.25,  # 3 months
    current_time=0.0,
)
# → 98.0 (spot minus dividend)

# Compute forward with both continuous and discrete dividends
forward = compute_forward_with_dividends(
    spot=100.0,
    discount_rate=0.05,
    dividend_yield=0.02,
    expiry=1.0,
    discrete_dividends=[(0.25, 1.0), (0.75, 1.0)],  # Two $1 dividends
)
```

---

## 6. Configuration

### 6.1 SyntheticProviderConfig

The config object controls per-MarketId behavior:

```python
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.specs import SpotGbmSpec, CurveZeroSpec, VolGridSmileSpec

config = SyntheticProviderConfig(
    # Default spot spec (used for FX and Equity SPOT)
    spot=SpotGbmSpec(
        initial_level=100.0,
        drift=0.05,
        vol=0.20,
    ),
    
    # Default curve spec
    curve_method="zeros",
    curve_zero=CurveZeroSpec(
        tenors=np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0]),
        base_rate=0.03,
    ),
    
    # Default vol spec
    vol=VolGridSmileSpec(
        expiries=np.array([0.25, 0.5, 1.0]),
        strikes=np.array([0.9, 1.0, 1.1]),
        atm_vol=0.15,
    ),
)

provider = SyntheticProvider(seed=42, config=config)
```

### 6.2 Per-MarketId Overrides

Config can provide per-MarketId specs via override methods:

```python
# Config internally routes:
config.spot_spec(market_id)        # → SpotGbmSpec
config.curve_zero_spec(market_id)  # → CurveZeroSpec
config.vol_spec(market_id)         # → VolGridSmileSpec
```

---

## 7. Dependency Resolution

### 7.1 How Dependencies Work

1. **Registration with Requirements**
```python
registry.register(
    asset_class="FX",
    mkt_type="VOL",
    generator=fx.generate_vol,
    requirements=fx.requirements_for_vol,  # Returns prerequisite MarketIds
)
```

2. **Dependency Closure**
When you request FX VOL, the engine automatically adds SPOT and curves:
```python
# Request
universe = [MarketId("FX", "VOL", "EURUSD", (("dom", "USD"), ("for", "EUR")))]

# Expanded (after closure)
expanded = [
    MarketId("FX", "SPOT", "EURUSD", (("dom", "USD"), ("for", "EUR"))),
    MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),)),
    MarketId("IR", "CURVE", "EUR.OIS", (("ccy", "EUR"),)),
    MarketId("FX", "VOL", "EURUSD", (("dom", "USD"), ("for", "EUR"))),
]
```

3. **Topological Sort**
Engine generates in dependency order:
- SPOT before VOL (VOL needs spot for forward-moneyness)
- CURVE before VOL (VOL needs curves for carry)

### 7.2 Dependency Graph

```
FX VOL (EURUSD)
    │
    ├── FX SPOT (EURUSD)
    ├── IR CURVE (USD.OIS)
    └── IR CURVE (EUR.OIS)

EQUITY VOL (AAPL)
    │
    ├── EQUITY SPOT (AAPL)
    └── IR CURVE (USD.OIS)  [if ccy=USD qualifier]
```

---

## 8. Determinism and Reproducibility

### 8.1 RNG Derivation

Each MarketId gets a unique, deterministic RNG stream:

```python
def rng_for_market_id(base_seed: int, market_id: MarketId) -> np.random.Generator:
    """Derive deterministic RNG from seed + MarketId.key()."""
    key_hash = hash(market_id.key()) & 0xFFFFFFFF
    combined_seed = (base_seed + key_hash) & 0xFFFFFFFF
    return np.random.default_rng(combined_seed)
```

### 8.2 Guarantees

| Property | Guarantee |
|----------|-----------|
| **Same seed + same request** | Identical output |
| **Different MarketIds** | Independent RNG streams |
| **Generation order** | Does not affect output |
| **Request order** | Does not affect output |

### 8.3 Example

```python
provider1 = SyntheticProvider(seed=42)
provider2 = SyntheticProvider(seed=42)

ds1 = provider1.get_timeseries(request)
ds2 = provider2.get_timeseries(request)

# ds1 and ds2 are identical
```

---

## 9. Extending with New Generators

### 9.1 Adding a New Asset Class

**Step 1: Create generator module**
```python
# src/marketdata/providers/synthetic/generators/commodity.py

from dataclasses import dataclass
from src.marketdata.providers.synthetic.registry import SyntheticRegistry

def register_commodity_generators(
    *,
    registry: SyntheticRegistry,
    base_seed: int,
    config: SyntheticProviderConfig,
) -> None:
    comm = _CommodityGenerators(base_seed=base_seed, config=config)
    registry.register(asset_class="COMMODITY", mkt_type="SPOT", generator=comm.generate_spot)

@dataclass(frozen=True, slots=True)
class _CommodityGenerators:
    base_seed: int
    config: SyntheticProviderConfig
    
    def generate_spot(self, market_id: MarketId, state: SyntheticGenerationState) -> None:
        # Implementation...
```

**Step 2: Register in provider**
```python
# src/marketdata/providers/synthetic/provider.py

from src.marketdata.providers.synthetic.generators.commodity import register_commodity_generators

def __post_init__(self) -> None:
    registry = SyntheticRegistry()
    register_fx_generators(registry=registry, ...)
    register_ir_generators(registry=registry, ...)
    register_equity_generators(registry=registry, ...)
    register_commodity_generators(registry=registry, ...)  # Add this
```

### 9.2 Adding a New Market Type

**Example: Adding DIVIDEND generator**
```python
def generate_dividend(self, market_id: MarketId, state: SyntheticGenerationState) -> None:
    """Generate dividend schedule panel."""
    # Create dividend schedule...
    state.quote_panels[market_id] = Panel(
        data=dividend_schedule,
        axis_names=("time", "scenario", "ex_date"),
    )
```

---

## 10. Best Practices

### 10.1 Choosing Seeds

**Do:**
- Use fixed seeds for reproducible tests
- Document seed values in test files
- Use different seeds for different test scenarios

**Don't:**
- Use random seeds in production code
- Use seed=0 (less random bits)

### 10.2 Requesting Data

**Do:**
- Request only what you need
- Let dependency closure add prerequisites
- Use qualifiers for disambiguation

**Don't:**
- Manually request all dependencies
- Over-specify qualifiers unnecessarily

### 10.3 Config Management

**Do:**
- Define configs at module/test level
- Override specs for specific test cases
- Document non-default configs

**Don't:**
- Modify configs during generation
- Share mutable config objects

### 10.4 Testing

**Do:**
- Test generator output shapes
- Test determinism with same seed
- Test independence with different MarketIds

```python
def test_equity_spot_determinism():
    cfg = SyntheticProviderConfig()
    state1 = make_state(n_time=5, n_scenarios=10)
    state2 = make_state(n_time=5, n_scenarios=10)
    
    eq1 = _EquityGenerators(base_seed=42, config=cfg)
    eq2 = _EquityGenerators(base_seed=42, config=cfg)
    
    eq1.generate_spot(spot_id, state1)
    eq2.generate_spot(spot_id, state2)
    
    np.testing.assert_array_equal(
        state1.quote_panels[spot_id].data,
        state2.quote_panels[spot_id].data,
    )
```

---

## References

- `src/marketdata/providers/synthetic/generators/foreign_exchange.py` - FX implementation
- `src/marketdata/providers/synthetic/generators/interest_rate.py` - IR implementation
- `src/marketdata/providers/synthetic/generators/equity.py` - Equity implementation
- `src/marketdata/providers/synthetic/specs.py` - Specification dataclasses
- `docs/marketdata/architecture.md` - Core data structures
