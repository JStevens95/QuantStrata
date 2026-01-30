# Market Data Architecture

**Complete Technical Specification for QuantStrata Market Data Layer**

This document provides a comprehensive guide to the market data architecture, including the core data structures, design patterns, and API contracts that enable pricing, risk, and analytics.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Philosophy](#2-design-philosophy)
3. [Core Data Structures](#3-core-data-structures)
4. [MarketId: Universal Identifier](#4-marketid-universal-identifier)
5. [Panel: Multi-Dimensional Data Container](#5-panel-multi-dimensional-data-container)
6. [MarketDataset: Time Series Container](#6-marketdataset-time-series-container)
7. [Market: Pricing Snapshot](#7-market-pricing-snapshot)
8. [Provider Pattern](#8-provider-pattern)
9. [Data Flow Architecture](#9-data-flow-architecture)
10. [Best Practices](#10-best-practices)

---

## 1. Executive Summary

### 1.1 Overview

The QuantStrata market data architecture provides a clean separation between:

1. **Data Sources** (providers) - Generate or retrieve market data
2. **Storage Containers** (datasets, panels) - Hold multi-dimensional data
3. **Pricing Interfaces** (Market) - Immutable snapshots for pricing

This separation enables:
- **Reproducible pricing** - Same Market snapshot always produces same price
- **Flexible sourcing** - Switch between synthetic, static, or live data
- **Efficient backtesting** - Store once, snapshot many times

### 1.2 Key Components

| Component | Purpose | Immutable |
|-----------|---------|-----------|
| `MarketId` | Universal identifier for any market object | ✅ Yes |
| `Panel` | N-dimensional numpy array with named axes | ✅ Yes |
| `MarketDataset` | Time-series container for quotes, curves, vols | ✅ Yes |
| `Market` | Immutable pricing snapshot | ✅ Yes |
| `MarketDataProvider` | Interface for data generation/retrieval | Protocol |

### 1.3 Data Flow

```
Provider.get_timeseries(request)
           ↓
    MarketDataset
    ┌──────────────────────────────────────────┐
    │ dates: ["2026-01-01", "2026-01-02", ...] │
    │ n_scenarios: 1000                        │
    │ panels: {MarketId → Panel}               │
    │ curve_params: {MarketId → Panel}         │
    │ vol_params: {MarketId → Panel}           │
    └──────────────────────────────────────────┘
           ↓
    dataset.snapshot(time_idx=5, scenario_idx=42)
           ↓
       Market
    ┌──────────────────────────────────────────┐
    │ asof: "2026-01-06"                       │
    │ quotes: {MarketId → Quote}               │
    │ curves: {MarketId → Curve}               │
    │ vols: {MarketId → VolSurface}            │
    └──────────────────────────────────────────┘
           ↓
    pricer.price(instrument, market)
           ↓
       float (price)
```

---

## 2. Design Philosophy

### 2.1 Core Principles

**Principle 1: Immutability**
All market data objects are frozen dataclasses. Once created, they cannot be modified. This ensures:
- Thread safety
- Reproducible results
- Clear data provenance

**Principle 2: Separation of Concerns**
- Providers are responsible for data sourcing
- Datasets are responsible for storage
- Markets are responsible for pricing interfaces
- Pricers never interact with providers directly

**Principle 3: Factory Pattern for Rich Objects**
Curves and vol surfaces are stored as parameter arrays, then reconstructed at snapshot time via factories. This allows:
- Compact storage (just numpy arrays)
- Flexible reconstruction (different interpolation methods)
- Efficient slicing (no object overhead)

### 2.2 Type Safety

All core objects use:
- `@dataclass(frozen=True, slots=True)` for immutability and memory efficiency
- Strong typing with `Mapping`, `Tuple`, `Protocol` for interface contracts
- Validation in `__post_init__` for fail-fast error detection

---

## 3. Core Data Structures

### 3.1 Hierarchy

```
MarketDataProvider (protocol)
        │
        ├── get_market(request) → Market
        │
        └── get_timeseries(request) → MarketDataset
                                           │
                                           └── snapshot(t, s) → Market
                                                                  │
                                                                  ├── quotes: {MarketId → Quote}
                                                                  ├── curves: {MarketId → Curve}
                                                                  └── vols: {MarketId → VolSurface}
```

### 3.2 Module Location

```
src/marketdata/
├── core/
│   ├── ids.py           # MarketId
│   ├── panel.py         # Panel
│   ├── dataset.py       # MarketDataset
│   ├── market.py        # Market
│   └── interfaces.py    # Quote, Curve, VolSurface protocols
├── providers/
│   ├── interfaces.py    # MarketDataProvider protocol
│   ├── synthetic/       # SyntheticProvider
│   ├── static/          # StaticProvider
│   └── hybrid/          # HybridProvider
├── curves/
│   └── factory.py       # ZeroRateCurveFactory
└── surfaces/
    └── factory.py       # GridVolFactory
```

---

## 4. MarketId: Universal Identifier

### 4.1 Purpose

`MarketId` is the canonical identifier for any market object in QuantStrata. It provides:
- **Unique identification** across asset classes and data types
- **Routing** for generators and factories
- **Persistence** via a stable string representation

### 4.2 Structure

```python
@dataclass(frozen=True, slots=True)
class MarketId:
    asset_class: str   # "FX", "IR", "EQUITY", "COMMODITY"
    mkt_type: str      # "SPOT", "CURVE", "VOL", "FIXING"
    name: str          # "EURUSD", "USD.OIS", "SPX"
    qualifiers: tuple  # (("dom", "USD"), ("for", "EUR"))
```

### 4.3 Canonical String Format

```
ASSET_CLASS.MKT_TYPE.NAME|key1=value1|key2=value2
```

**Examples:**
```python
# FX spot rate
MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
# → "FX.SPOT.EURUSD"

# IR curve with currency qualifier
MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", 
         qualifiers=(("ccy", "USD"),))
# → "IR.CURVE.USD.OIS|ccy=USD"

# FX vol with dom/for currencies
MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD",
         qualifiers=(("dom", "USD"), ("for", "EUR")))
# → "FX.VOL.EURUSD|dom=USD|for=EUR"

# Equity spot
MarketId(asset_class="EQUITY", mkt_type="SPOT", name="AAPL")
# → "EQUITY.SPOT.AAPL"
```

### 4.4 Usage Patterns

**Creating MarketIds:**
```python
from src.marketdata.core.ids import MarketId

# Direct construction
spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")

# Parsing from string
parsed = MarketId.parse("FX.SPOT.EURUSD|dom=USD|for=EUR")

# Adding qualifiers
enriched = spot_id.with_qualifiers((("dom", "USD"), ("for", "EUR")))
```

**Using in lookups:**
```python
# Dataset lookup
panel = dataset.panels[spot_id]

# Market lookup
spot_value = market.quote(spot_id)
curve = market.curve(curve_id)
vol_surface = market.vol_surface(vol_id)
```

### 4.5 Asset Class Conventions

| Asset Class | SPOT | CURVE | VOL | FIXING |
|-------------|------|-------|-----|--------|
| **FX** | Exchange rate | - | Implied vol surface | Settlement rate |
| **IR** | - | Zero rate curve | Swaption vol | - |
| **EQUITY** | Stock price | - | Implied vol surface | Settlement price |

---

## 5. Panel: Multi-Dimensional Data Container

### 5.1 Purpose

`Panel` wraps a numpy array with named axes, providing:
- **Self-documenting dimensions** via axis_names
- **Consistent slicing** across the codebase
- **Lightweight storage** (just data + names)

### 5.2 Structure

```python
@dataclass(frozen=True, slots=True)
class Panel:
    data: np.ndarray
    axis_names: Tuple[str, ...]
```

### 5.3 Common Shapes

**Quote Panels (1D or 2D):**
```python
# Time series only (single scenario)
Panel(data=np.array([1.10, 1.11, 1.12]), axis_names=("time",))
# Shape: (3,)

# Time × Scenarios
Panel(data=np.random.randn(100, 1000), axis_names=("time", "scenario"))
# Shape: (100, 1000)
```

**Curve Parameter Panels (3D or 4D):**
```python
# Shape: (n_time, n_scenarios, n_tenors, 2)
# Last dimension: [tenor, zero_rate]
Panel(data=curve_params, axis_names=("time", "scenario", "tenor_idx", "param"))
```

**Vol Parameter Panels (4D):**
```python
# Shape: (n_time, n_scenarios, n_expiries, n_strikes)
Panel(data=vol_cube, axis_names=("time", "scenario", "expiry", "strike"))
```

### 5.4 Extracting Values

```python
# Scalar at (time, scenario)
value = panel.scalar_at(time_idx=5, scenario_idx=42)

# For block parameters, use direct slicing
block = panel.data[time_idx, scenario_idx]  # Shape: (n_tenors, 2) for curves
```

---

## 6. MarketDataset: Time Series Container

### 6.1 Purpose

`MarketDataset` is the central storage container for multi-day, multi-scenario market data. It bridges:
- **Generation** (from providers)
- **Storage** (panels with factories)
- **Pricing** (via snapshot extraction)

### 6.2 Structure

```python
@dataclass(frozen=True, slots=True)
class MarketDataset:
    dates: List[str]                              # ["2026-01-01", ...]
    n_scenarios: int                              # Number of scenarios
    panels: Mapping[MarketId, Panel]              # Quote data
    curve_params: Mapping[MarketId, Panel]        # Curve parameters
    curve_factories: Mapping[MarketId, CurveFactory]
    vol_params: Mapping[MarketId, Panel]          # Vol parameters
    vol_factories: Mapping[MarketId, VolSurfaceFactory]
    meta: Mapping[str, Any] | None = None         # Optional metadata
```

### 6.3 Key Invariants

The dataset validates on construction:

1. **Time axis alignment** - All panels must have compatible time dimension
2. **Scenario axis alignment** - All scenario-aware panels must match `n_scenarios`
3. **Factory completeness** - Every param panel must have a corresponding factory

### 6.4 Snapshot Extraction

```python
# Get a pricing snapshot
market = dataset.snapshot(time_idx=10, scenario_idx=42)

# The snapshot contains:
# - quotes: Scalar values from quote panels
# - curves: Reconstructed from curve_params via curve_factories
# - vols: Reconstructed from vol_params via vol_factories
```

### 6.5 Example Usage

```python
from datetime import date
from src.marketdata.core.requests import TimeseriesRequest
from src.marketdata.providers.synthetic.provider import SyntheticProvider

# Create provider
provider = SyntheticProvider(seed=42)

# Request time series
request = TimeseriesRequest(
    start=date(2026, 1, 1),
    end=date(2026, 3, 31),
    freq="D",
    universe=[
        MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
        MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS"),
    ],
    scenarios=100,
)

# Generate dataset
dataset = provider.get_timeseries(request)

# Extract snapshots for pricing
for t in range(len(dataset.dates)):
    for s in range(dataset.n_scenarios):
        market = dataset.snapshot(time_idx=t, scenario_idx=s)
        # Use market for pricing...
```

---

## 7. Market: Pricing Snapshot

### 7.1 Purpose

`Market` is the immutable snapshot consumed by pricers. It provides a clean API for:
- **Quotes** (spot rates, fixings)
- **Curves** (discount factors, zero rates)
- **Vol surfaces** (implied volatilities)

### 7.2 Structure

```python
@dataclass(frozen=True, slots=True)
class Market:
    asof: str                              # As-of date
    quotes: Mapping[MarketId, Quote]       # Scalar quotes
    curves: Mapping[MarketId, Curve]       # Curve objects
    vols: Mapping[MarketId, VolSurface]    # Vol surface objects
    meta: Mapping[str, Any] | None = None
```

### 7.3 API Methods

```python
# Get scalar quote
spot = market.quote(MarketId("FX", "SPOT", "EURUSD"))
# → float (e.g., 1.1234)

# Get curve
curve = market.curve(MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),)))
df = curve.df(1.0)  # Discount factor to 1 year
# → float (e.g., 0.9512)

# Get vol surface
vol_surface = market.vol_surface(MarketId("FX", "VOL", "EURUSD"))
sigma = vol_surface.vol(expiry=1.0, strike=1.10)
# → float (e.g., 0.12)

# Check existence
if market.has(some_id):
    ...
```

### 7.4 Design Rule

**Pricers depend ONLY on Market, never on providers or datasets.**

This ensures:
- Pricers are testable in isolation
- Same market always produces same price
- Data sourcing is completely decoupled from pricing logic

---

## 8. Provider Pattern

### 8.1 MarketDataProvider Protocol

```python
class MarketDataProvider(Protocol):
    @property
    def name(self) -> str: ...
    
    def get_market(self, request: MarketRequest) -> Market: ...
    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset: ...
```

### 8.2 Available Providers

| Provider | Purpose | Use Case |
|----------|---------|----------|
| `SyntheticProvider` | Deterministic generation | Testing, examples, demos |
| `StaticProvider` | Replay frozen data | Backtesting, production replay |
| `HybridProvider` | Primary + fallback chain | Production with resilience |

### 8.3 SyntheticProvider

The primary provider for testing and development:

```python
from src.marketdata.providers.synthetic.provider import SyntheticProvider

provider = SyntheticProvider(seed=42)  # Deterministic

# Single snapshot
market = provider.get_market(MarketRequest(
    asof=date(2026, 1, 15),
    universe=[MarketId("FX", "SPOT", "EURUSD")],
))

# Time series
dataset = provider.get_timeseries(TimeseriesRequest(
    start=date(2026, 1, 1),
    end=date(2026, 12, 31),
    freq="D",
    universe=[...],
    scenarios=1000,
))
```

### 8.4 Request Objects

```python
@dataclass(frozen=True, slots=True)
class MarketRequest:
    asof: date                    # As-of date
    universe: List[MarketId]      # Required market objects
    scenario: int | None = None   # Specific scenario (optional)

@dataclass(frozen=True, slots=True)
class TimeseriesRequest:
    start: date
    end: date
    freq: str                     # "D" (daily), "W" (weekly), etc.
    universe: List[MarketId]
    scenarios: int = 1            # Number of Monte Carlo scenarios
```

---

## 9. Data Flow Architecture

### 9.1 Generation Flow

```
TimeseriesRequest
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   SyntheticProvider                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │               SyntheticRegistry                    │  │
│  │   (FX.SPOT) → FxGenerators.generate_spot          │  │
│  │   (FX.VOL)  → FxGenerators.generate_vol           │  │
│  │   (IR.CURVE)→ IrGenerators.generate_curve         │  │
│  │   (EQUITY.SPOT) → EquityGenerators.generate_spot  │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │            SyntheticMarketEngine                   │  │
│  │   1. Expand universe (dependency closure)          │  │
│  │   2. Order by dependencies (SPOT before VOL)       │  │
│  │   3. Generate each MarketId with deterministic RNG │  │
│  │   4. Assemble into MarketDataset                   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   MarketDataset
```

### 9.2 Snapshot Flow

```
   MarketDataset
        │
        │  snapshot(time_idx=t, scenario_idx=s)
        ▼
┌─────────────────────────────────────────────────────────┐
│  For each quote panel:                                  │
│    value = panel.scalar_at(t, s) → Quote               │
│                                                         │
│  For each curve_params + curve_factory:                │
│    params = curve_params.data[t, s, ...]               │
│    curve = factory.build(params) → Curve               │
│                                                         │
│  For each vol_params + vol_factory:                    │
│    params = vol_params.data[t, s, ...]                 │
│    surface = factory.build(params) → VolSurface        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
      Market
```

### 9.3 Pricing Flow

```
      Market
        │
        │  pricer.price(instrument, market)
        ▼
┌─────────────────────────────────────────────────────────┐
│  1. Extract required market data:                       │
│     spot = market.quote(spot_id)                       │
│     curve = market.curve(curve_id)                     │
│     vol = market.vol_surface(vol_id).vol(T, K)         │
│                                                         │
│  2. Apply pricing model:                                │
│     BSM, Monte Carlo, Finite Difference, etc.          │
│                                                         │
│  3. Return result                                       │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   float (price)
```

---

## 10. Best Practices

### 10.1 MarketId Conventions

**Do:**
- Use uppercase for asset_class and mkt_type
- Use consistent naming (e.g., "EURUSD" not "EUR/USD")
- Include qualifiers for disambiguation

**Don't:**
- Use spaces in names or qualifiers
- Mix conventions within a project

### 10.2 Panel Design

**Do:**
- Always include "time" as first axis for time series
- Include "scenario" as second axis when applicable
- Use descriptive axis names

**Don't:**
- Create panels with inconsistent shapes
- Assume axis order without checking axis_names

### 10.3 Provider Usage

**Do:**
- Use `SyntheticProvider` with fixed seed for reproducible tests
- Request only the MarketIds you need
- Handle missing data gracefully

**Don't:**
- Call providers from within pricers
- Assume providers are thread-safe without checking

### 10.4 Snapshot Usage

**Do:**
- Extract snapshots immediately before pricing
- Pass Market objects to pricers, not datasets
- Cache snapshots if repricing frequently

**Don't:**
- Modify Market objects (they're immutable anyway)
- Store references to panels across snapshot calls

---

## References

- `src/marketdata/core/ids.py` - MarketId implementation
- `src/marketdata/core/panel.py` - Panel implementation
- `src/marketdata/core/dataset.py` - MarketDataset implementation
- `src/marketdata/core/market.py` - Market implementation
- `src/marketdata/providers/interfaces.py` - Provider protocol
- `docs/interfaces.md` - API contracts
