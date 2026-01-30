# Market Data Documentation

This directory contains technical documentation for the QuantStrata market data layer.

## Contents

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Core data structures: `MarketId`, `Panel`, `MarketDataset`, `Market` |
| [synthetic_generators.md](synthetic_generators.md) | Deterministic data generation for FX, IR, and Equity |
| [volatility_surfaces.md](volatility_surfaces.md) | Implied and local volatility surface implementations |

## Quick Reference

### Core Classes

| Class | Purpose | Module |
|-------|---------|--------|
| `MarketId` | Universal market object identifier | `src.marketdata.core.ids` |
| `Panel` | N-dimensional numpy array with named axes | `src.marketdata.core.panel` |
| `MarketDataset` | Time-series container for multi-scenario data | `src.marketdata.core.dataset` |
| `Market` | Immutable pricing snapshot | `src.marketdata.core.market` |

### Providers

| Provider | Purpose | Module |
|----------|---------|--------|
| `SyntheticProvider` | Deterministic data generation | `src.marketdata.providers.synthetic` |
| `StaticProvider` | Replay frozen datasets | `src.marketdata.providers.static` |
| `HybridProvider` | Primary + fallback chains | `src.marketdata.providers.hybrid` |

### Surfaces

| Surface | Axes | Use Case |
|---------|------|----------|
| `FlatVolSurface` | None (constant) | Testing, baseline |
| `GridVolSurface` | Expiry × Strike | Real-world implied vol |
| `LocalVolSurface` | Spot × Time | Exotic pricing |

## Usage Example

```python
from datetime import date
from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import TimeseriesRequest
from src.marketdata.providers.synthetic.provider import SyntheticProvider

# Create provider
provider = SyntheticProvider(seed=42)

# Define universe
universe = [
    MarketId("FX", "SPOT", "EURUSD", (("dom", "USD"), ("for", "EUR"))),
    MarketId("FX", "VOL", "EURUSD", (("dom", "USD"), ("for", "EUR"))),
    MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),)),
]

# Generate time series
dataset = provider.get_timeseries(TimeseriesRequest(
    start=date(2026, 1, 1),
    end=date(2026, 12, 31),
    freq="D",
    universe=universe,
    scenarios=100,
))

# Extract snapshot for pricing
market = dataset.snapshot(time_idx=10, scenario_idx=0)

# Use in pricing
spot = market.quote(universe[0])
vol_surface = market.vol_surface(universe[1])
curve = market.curve(universe[2])
```

## Related Documentation

- [interfaces.md](../interfaces.md) - API contracts and stability guarantees
- [mathematics/local_volatility.md](../mathematics/local_volatility.md) - Dupire model theory
- [mathematics/volatility_calibration.md](../mathematics/volatility_calibration.md) - Calibration methods
