# Phase 4.3: Multi-Asset Products - Progress Report

**Status:** COMPLETE  
**Completed:** January 27, 2026  
**Architecture Update:** January 27, 2026 (Naming Convention Refactor)

---

## Overview

Phase 4.3 implements multi-asset derivative products including basket options, spread options, exchange options, and rainbow options (best-of/worst-of).

The architecture follows the same naming conventions as FX instruments and pricers for consistency across the library.

---

## Deliverables

### 1. Multi-Asset Simulation Infrastructure

**Location:** `src/models/numeric/monte_carlo/multi_asset.py`

**Components:**
- `CorrelationMatrix`: Validated correlation structure with Cholesky decomposition
  - Factory methods: `from_flat()`, `from_pairs()`
  - Validation: symmetry, PSD, diagonal=1
- `MultiAssetGBM`: Multi-dimensional GBM simulator
  - Methods: `simulate()` (full paths), `simulate_terminal()` (efficient for Europeans)
  - Features: Antithetic variates, Cholesky correlation

### 2. Basket Options

**Instruments:** `src/instruments/multi_asset/basket.py`
- `MultiAssetBasketEuropeanOption` (with `option_type: "call" | "put"`)

**Pricers:** `src/pricers/multi_asset/basket_european_mc.py`
- `MultiAssetBasketEuropeanOptionMcPricer`
- `MultiAssetBasketEuropeanOptionMcSimulation` (simulation artifact)

**Payoff:**
- Call: max(Σ wᵢSᵢ(T) - K, 0)
- Put: max(K - Σ wᵢSᵢ(T), 0)

**Tests:** 13 unit tests passing

### 3. Spread Options

**Instruments:** `src/instruments/multi_asset/spread.py`
- `MultiAssetSpreadEuropeanOption` (with `option_type: "call" | "put"`)
- `MultiAssetExchangeEuropeanOption` (spread with K=0)

**Pricers:** `src/pricers/multi_asset/spread_european_mc.py`
- `MultiAssetSpreadEuropeanOptionMcPricer` (Monte Carlo)
- `MultiAssetSpreadEuropeanOptionKirkPricer` (Kirk's approximation)
- `MultiAssetExchangeEuropeanOptionMargrabePricer` (exact closed-form)
- `MultiAssetSpreadEuropeanOptionMcSimulation` (simulation artifact)

**Payoff:**
- Call: max(S₁(T) - S₂(T) - K, 0)
- Put: max(K - (S₁(T) - S₂(T)), 0)
- Exchange: max(S₁(T) - S₂(T), 0) (K=0)

**Tests:** 10 unit tests passing

### 4. Rainbow Options (Best-of / Worst-of)

**Instruments:** `src/instruments/multi_asset/rainbow.py`
- `MultiAssetBestOfEuropeanOption` (with `option_type: "call" | "put"`)
- `MultiAssetWorstOfEuropeanOption` (with `option_type: "call" | "put"`)

**Pricers:** `src/pricers/multi_asset/rainbow_european_mc.py`
- `MultiAssetBestOfEuropeanOptionMcPricer`
- `MultiAssetWorstOfEuropeanOptionMcPricer`
- `MultiAssetBestOfEuropeanOptionMcSimulation` (simulation artifact)
- `MultiAssetWorstOfEuropeanOptionMcSimulation` (simulation artifact)

**Payoffs:**
- Best-of Call: max(max(S₁, S₂, ..., Sₙ) - K, 0)
- Best-of Put: max(K - max(S₁, S₂, ..., Sₙ), 0)
- Worst-of Call: max(min(S₁, S₂, ..., Sₙ) - K, 0)
- Worst-of Put: max(K - min(S₁, S₂, ..., Sₙ), 0)

**Tests:** 10 unit tests passing

---

## Test Summary

**Total Tests:** 41 passing

| Component | Tests |
|-----------|-------|
| Instrument Definitions | 24 |
| Pricer Tests | 17 |
| **Total** | 41 |

---

## Architecture

### Directory Structure

```
src/instruments/multi_asset/
├── __init__.py
├── basket.py          # MultiAssetBasketEuropeanOption
├── spread.py          # MultiAssetSpreadEuropeanOption, MultiAssetExchangeEuropeanOption
└── rainbow.py         # MultiAssetBestOfEuropeanOption, MultiAssetWorstOfEuropeanOption

src/pricers/multi_asset/
├── __init__.py
├── basket_european_mc.py    # MultiAssetBasketEuropeanOptionMcPricer
├── spread_european_mc.py    # MultiAssetSpreadEuropeanOptionMcPricer, Kirk, Margrabe
└── rainbow_european_mc.py   # MultiAssetBestOfEuropeanOptionMcPricer, MultiAssetWorstOfEuropeanOptionMcPricer

src/models/numeric/monte_carlo/
└── multi_asset.py           # CorrelationMatrix, MultiAssetGBM
```

### Naming Conventions

Following FX pattern for consistency:

| Component | FX Example | Multi-Asset Example |
|-----------|------------|---------------------|
| Instrument | `FxVanillaEuropeanOption` | `MultiAssetBasketEuropeanOption` |
| MC Pricer | `FxVanillaEuropeanOptionMcPricer` | `MultiAssetBasketEuropeanOptionMcPricer` |
| Simulation | `FxVanillaOptionMcSimulation` | `MultiAssetBasketEuropeanOptionMcSimulation` |

### Key Design Decisions

1. **Unified Option Type**: All instruments use `option_type: "call" | "put"` field instead of separate classes.

2. **Pricer Classes**: Pricers are classes with `price()`, `price_with_std_error()`, and `run()` methods (consistent with FX pricers).

3. **Simulation Artifacts**: Each pricer returns a typed simulation dataclass containing all inputs, settings, and outputs.

4. **Separation of Concerns**: Instruments define contracts; pricers implement pricing logic.

---

## Usage Examples

### Basket Call

```python
import numpy as np
from src.marketdata.core.ids import MarketId
from src.instruments.multi_asset import MultiAssetBasketEuropeanOption
from src.pricers.multi_asset import MultiAssetBasketEuropeanOptionMcPricer
from src.models.numeric.monte_carlo.multi_asset import CorrelationMatrix

def make_id(name: str) -> MarketId:
    return MarketId(asset_class="EQ", mkt_type="SPOT", name=name)

# Create instrument
basket = MultiAssetBasketEuropeanOption(
    option_type="call",
    underlyings=(make_id("AAPL"), make_id("GOOGL"), make_id("MSFT")),
    weights=(0.4, 0.35, 0.25),
    strike=100.0,
    expiry=1.0,
)

# Create pricer and price
pricer = MultiAssetBasketEuropeanOptionMcPricer(n_paths=100000, seed=42)
price, std = pricer.price_with_std_error(
    basket,
    spots=np.array([100.0, 100.0, 100.0]),
    r=0.05,
    dividends=np.array([0.02, 0.02, 0.02]),
    volatilities=np.array([0.2, 0.25, 0.3]),
    correlation=CorrelationMatrix.from_flat(0.5, n=3),
)
print(f"Basket Call: {price:.4f} ± {std:.4f}")
```

### Spread Option with Kirk's Approximation

```python
from src.instruments.multi_asset import MultiAssetSpreadEuropeanOption
from src.pricers.multi_asset import (
    MultiAssetSpreadEuropeanOptionMcPricer,
    MultiAssetSpreadEuropeanOptionKirkPricer,
)

spread = MultiAssetSpreadEuropeanOption(
    option_type="call",
    underlying1=make_id("CL"),
    underlying2=make_id("HO"),
    strike=5.0,
    expiry=0.5,
)

mc_pricer = MultiAssetSpreadEuropeanOptionMcPricer(n_paths=100000, seed=42)
kirk_pricer = MultiAssetSpreadEuropeanOptionKirkPricer()

mc_price = mc_pricer.price(
    spread, spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.01,
    sigma1=0.2, sigma2=0.25, rho=0.6
)
kirk_price = kirk_pricer.price(
    spread, spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.01,
    sigma1=0.2, sigma2=0.25, rho=0.6
)

print(f"MC Price: {mc_price:.4f}")
print(f"Kirk's Approximation: {kirk_price:.4f}")
```

### Best-of / Worst-of Options

```python
from src.instruments.multi_asset import (
    MultiAssetBestOfEuropeanOption,
    MultiAssetWorstOfEuropeanOption,
)
from src.pricers.multi_asset import (
    MultiAssetBestOfEuropeanOptionMcPricer,
    MultiAssetWorstOfEuropeanOptionMcPricer,
)

best_of = MultiAssetBestOfEuropeanOption(
    option_type="call",
    underlyings=(make_id("A"), make_id("B")),
    strike=100.0,
    expiry=1.0,
)

worst_of = MultiAssetWorstOfEuropeanOption(
    option_type="call",
    underlyings=(make_id("A"), make_id("B")),
    strike=100.0,
    expiry=1.0,
)

params = {
    'spots': np.array([100.0, 100.0]),
    'r': 0.05,
    'dividends': np.array([0.02, 0.02]),
    'volatilities': np.array([0.25, 0.3]),
    'correlation': CorrelationMatrix.from_flat(0.5, n=2),
}

best_pricer = MultiAssetBestOfEuropeanOptionMcPricer(n_paths=100000, seed=42)
worst_pricer = MultiAssetWorstOfEuropeanOptionMcPricer(n_paths=100000, seed=42)

best_price = best_pricer.price(best_of, **params)
worst_price = worst_pricer.price(worst_of, **params)

print(f"Best-of Call: {best_price:.4f}")
print(f"Worst-of Call: {worst_price:.4f}")
print(f"Best-of >= Worst-of: {best_price >= worst_price}")
```

---

## Key Relationships

### Correlation Effects

| Product | Higher Correlation Effect |
|---------|--------------------------|
| Basket Call | Lower price (less diversification) |
| Spread Option | Lower price (assets move together) |
| Best-of Call | Lower price (best less likely to excel) |
| Worst-of Call | Higher price (worst less likely to crash) |

### Price Ordering

For same parameters:
- Best-of Call ≥ Single-asset Call ≥ Worst-of Call
- Worst-of Put ≥ Single-asset Put ≥ Best-of Put

---

## Documentation

### User Guides

- `docs/guides/multi_asset/basket_options.md`
- `docs/guides/multi_asset/spread_options.md`
- `docs/guides/multi_asset/rainbow_options.md`

All guides updated to reflect new naming conventions.

---

*Document Version: 2.0 | QuantStrata Phase 4.3 | January 2026*
