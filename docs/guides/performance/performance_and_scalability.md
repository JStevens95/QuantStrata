# Performance and Scalability - User Guide

This guide shows how to use QuantStrata's performance and scalability features: backend selection (NumPy/Numba/JAX), parallel portfolio pricing, and caching.

---

## Quick Start

### Backend selection

```python
from src.core.performance.backend import get_backend, Backend, jax_available

# Auto-select best available: NUMBA > JAX > NUMPY
backend = get_backend("auto")

# Or force a backend
backend = get_backend("numpy")   # Always NumPy
backend = get_backend("numba")   # Numba if installed
backend = get_backend("jax")     # JAX if installed (CPU/GPU)
```

### Parallel portfolio pricing

```python
from src.portfolio.portfolio import PortfolioPricer
from src.portfolio.parallel import ParallelPortfolioPricer

base_pricer = PortfolioPricer(pricer_registry=reg)
parallel_pricer = ParallelPortfolioPricer(
    portfolio_pricer=base_pricer,
    max_workers=4,
)
result = parallel_pricer.price(portfolio, market)
```

### Pricer result cache

```python
from src.portfolio.caching import CachingPortfolioPricer

caching_pricer = CachingPortfolioPricer(
    portfolio_pricer=base_pricer,
    max_size=1024,
    ttl_seconds=60.0,
)
result = caching_pricer.price(portfolio, market)
```

### Market data cache

```python
from src.marketdata.cache import CachingMarketDataProvider

caching_provider = CachingMarketDataProvider(
    provider=my_provider,
    max_size=512,
    ttl_seconds=300.0,
)
market = caching_provider.get_market(request)
```

---

## Backend Selection

| Backend | When to use | Notes |
|---------|--------------|--------|
| **NumPy** | Default fallback, debugging | Always available |
| **Numba** | MC/FD kernels, benchmarks | JIT; first call has compile overhead |
| **JAX** | Optional GPU/CPU MC | Install: `pip install jax jaxlib`; GPU automatic with CUDA/ROCm |

Set the default backend:

```python
from src.core.performance.backend import set_default_backend

set_default_backend("numba")  # Prefer Numba everywhere
```

---

## JAX MC Pricer (Optional)

When JAX is installed, you can price FX European vanilla options with the JAX MC pricer for CPU/GPU acceleration. Use `pricer_id="jax_mc"` when resolving:

```python
from src.pricers.registry import DefaultPricerRegistry
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption

reg = DefaultPricerRegistry().build()
# Only registered when JAX is available
pricer = reg.resolve(vanilla_option, pricer_id="jax_mc")
pv = pricer.price(vanilla_option, market)
```

GPU setup: JAX uses GPU automatically when `jaxlib` is installed with CUDA or ROCm support. No code changes are required; device is determined by JAX’s default backend.

---

## Parallel Portfolio Pricing

- **When to use:** Portfolios with many positions; pricer and market are read-only per call.
- **Thread safety:** Do not use with pricers or markets that mutate shared state. If unsure, use `max_workers=1`.
- **API:** Same `PortfolioResult` as sequential; `max_workers=None` uses a default (e.g. CPU count).

---

## Caching

| Cache | Key | Eviction |
|-------|-----|----------|
| **Market data** | (asof, universe ids, scenario) | LRU by `max_size`; optional TTL |
| **Pricer result** | (portfolio_key, market_key) | LRU by `max_size`; optional TTL |

Custom key functions for the pricer cache:

```python
def portfolio_key(p):
    return (len(p.positions), tuple(id(pos) for pos in p.positions))

def market_key(m):
    return getattr(m, "asof", id(m))

CachingPortfolioPricer(
    portfolio_pricer=base_pricer,
    max_size=1024,
    portfolio_key_fn=portfolio_key,
    market_key_fn=market_key,
)
```

---

*See also: [Performance Optimization (Reference)](../../reference/performance_optimisation.md) | [Tutorial](../tutorials/performance/performance_and_scalability.ipynb)*
