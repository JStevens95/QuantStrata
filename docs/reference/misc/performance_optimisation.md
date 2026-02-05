# Performance Optimization

**High-Performance Numerical Computing in QuantStrata**

This document covers the theory and implementation of performance optimizations for Monte Carlo simulation and Finite Difference methods.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Numba JIT Compilation](#2-numba-jit-compilation)
3. [Monte Carlo Optimization](#3-monte-carlo-optimization)
4. [Finite Difference Optimization](#4-finite-difference-optimization)
5. [Parallelization Strategies](#5-parallelization-strategies)
6. [Benchmarking](#6-benchmarking)
7. [JAX Backend (Optional)](#7-jax-backend-optional)
8. [Parallel Portfolio Pricing](#8-parallel-portfolio-pricing)
9. [Caching (Market Data and Pricer Results)](#9-caching-market-data-and-pricer-results)
10. [Interview Key Points](#10-interview-key-points)

---

## 1. Introduction

### Why Performance Matters

In quantitative finance, computational performance is critical:
- **Risk calculations**: CVA, PFE require millions of paths
- **Real-time pricing**: Market makers need sub-millisecond responses
- **Calibration**: Iterative optimization over many evaluations
- **Greeks**: Bump-and-reprice requires repeated calculations

### Performance Bottlenecks

| Component | Bottleneck | Typical Time |
|-----------|------------|--------------|
| Monte Carlo | Path generation | 70-80% |
| Monte Carlo | Payoff evaluation | 10-20% |
| Finite Difference | Tridiagonal solve | 60-70% |
| Finite Difference | PSOR iterations | 80-90% (American) |

### Optimization Strategy

```
Pure Python → NumPy Vectorization → Numba JIT → Parallelization
     ↓              ↓                  ↓              ↓
  1-10 ms       0.1-1 ms           0.01-0.1 ms    0.001-0.01 ms
```

---

## 2. Numba JIT Compilation

### What is Numba?

Numba is a Just-In-Time (JIT) compiler that translates Python and NumPy code into optimized machine code using LLVM.

### Key Decorators

```python
from numba import njit, prange

@njit                        # No-Python mode (fastest)
@njit(parallel=True)         # Enable parallelization
@njit(cache=True)            # Cache compiled code
@njit(fastmath=True)         # Allow unsafe math optimizations
```

### How JIT Works

1. **First call**: Python bytecode → LLVM IR → Machine code
2. **Subsequent calls**: Execute cached machine code directly

```
First call:  Python → Type inference → LLVM IR → Optimization → Machine code
                                                                    ↓
Later calls: ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

### JIT Limitations

1. **No Python objects**: Only basic types (int, float, arrays)
2. **No dynamic typing**: Types must be inferrable at compile time
3. **Limited NumPy**: Some advanced NumPy features not supported
4. **First-call overhead**: Compilation adds ~100ms per function

---

## 3. Monte Carlo Optimization

### 3.1 Path Generation

**GBM Exact Discretization:**
$$
S_{t+\Delta t} = S_t \cdot \exp\left((\mu - \frac{\sigma^2}{2})\Delta t + \sigma\sqrt{\Delta t} Z\right)
$$

**NumPy Implementation:**
```python
def gbm_step_numpy(spot, drift, vol, dt, sqrt_dt, z):
    log_increment = (drift - 0.5 * vol * vol) * dt + vol * sqrt_dt * z
    return spot * np.exp(log_increment)
```

**Numba Implementation:**
```python
@njit(parallel=True, fastmath=True)
def gbm_paths_numba(spot0, drift, vol, dt, sqrt_dt, z, out):
    n_paths = z.shape[1]
    n_steps = z.shape[0]
    drift_adj = drift - 0.5 * vol * vol
    
    for p in prange(n_paths):  # Parallel over paths
        spot = spot0
        for t in range(n_steps):
            log_inc = drift_adj * dt + vol * sqrt_dt * z[t, p]
            spot = spot * np.exp(log_inc)
        out[p] = spot
```

### 3.2 Performance Comparison

| Implementation | 100K paths × 252 steps | Speedup |
|----------------|------------------------|---------|
| Pure Python | ~10,000 ms | 1x |
| NumPy vectorized | ~100 ms | 100x |
| Numba serial | ~10 ms | 1,000x |
| Numba parallel | ~2 ms | 5,000x |

### 3.3 Payoff Optimization

**Vanilla Payoff (JIT):**
```python
@njit(parallel=True, fastmath=True)
def vanilla_call_jit(spots, strike, out):
    for i in prange(spots.shape[0]):
        diff = spots[i] - strike
        out[i] = diff if diff > 0.0 else 0.0
```

**Why faster than NumPy?**
- NumPy `np.maximum(spots - strike, 0)` creates intermediate arrays
- JIT eliminates temporaries, fuses operations
- Parallel execution over paths

---

## 4. Finite Difference Optimization

### 4.1 Thomas Algorithm

The Thomas algorithm solves tridiagonal systems $Ax = b$ in $O(n)$:

**Forward Elimination:**
$$
c'_0 = \frac{c_0}{b_0}, \quad d'_0 = \frac{d_0}{b_0}
$$
$$
c'_i = \frac{c_i}{b_i - a_i c'_{i-1}}, \quad d'_i = \frac{d_i - a_i d'_{i-1}}{b_i - a_i c'_{i-1}}
$$

**Back Substitution:**
$$
x_n = d'_n, \quad x_i = d'_i - c'_i x_{i+1}
$$

**Numba Implementation:**
```python
@njit(cache=True, fastmath=True)
def thomas_solve_jit(lower, diag, upper, rhs, out):
    n = diag.shape[0]
    c_prime = np.empty(n - 1)
    
    # Forward elimination
    c_prime[0] = upper[0] / diag[0]
    out[0] = rhs[0] / diag[0]
    
    for i in range(1, n):
        denom = diag[i] - lower[i-1] * c_prime[i-1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        out[i] = (rhs[i] - lower[i-1] * out[i-1]) / denom
    
    # Back substitution
    for i in range(n - 2, -1, -1):
        out[i] = out[i] - c_prime[i] * out[i+1]
```

### 4.2 PSOR Solver

**Projected SOR for American Options:**

Solve $Ax = b$ subject to $x \geq \text{floor}$ (early exercise constraint).

**SOR Iteration:**
$$
x_i^{(k+1)} = (1-\omega)x_i^{(k)} + \omega \cdot \text{GS update}
$$

**Projection:**
$$
x_i^{(k+1)} = \max(x_i^{(k+1)}, \text{floor}_i)
$$

**Numba Implementation:**
```python
@njit(cache=True)
def psor_solve_jit(lower, diag, upper, rhs, floor, omega, max_iter, tol, out):
    for iteration in range(max_iter):
        for i in range(n):
            left = lower[i-1] * out[i-1] if i > 0 else 0.0
            right = upper[i] * out_old[i+1] if i < n-1 else 0.0
            
            x_gs = (rhs[i] - left - right) / diag[i]
            x_sor = (1 - omega) * out_old[i] + omega * x_gs
            out[i] = max(x_sor, floor[i])  # Projection
        
        if converged:
            break
```

### 4.3 Performance Comparison

| Solver | Grid size n=500 | Speedup |
|--------|-----------------|---------|
| NumPy loops | ~5 ms | 1x |
| Numba JIT | ~0.2 ms | 25x |
| PSOR (NumPy) | ~50 ms | 1x |
| PSOR (Numba) | ~2 ms | 25x |

---

## 5. Parallelization Strategies

### 5.1 Monte Carlo Parallelization

**Embarrassingly Parallel:**
- Each path is independent
- Use `prange` for path loop
- Scales linearly with cores

```python
@njit(parallel=True)
def simulate_paths(spot0, drift, vol, z, out):
    for p in prange(n_paths):  # Parallel
        spot = spot0
        for t in range(n_steps):  # Sequential within path
            spot = step(spot, ...)
        out[p] = spot
```

### 5.2 Finite Difference Parallelization

**Less Parallelizable:**
- Time stepping is sequential (CFL condition)
- Tridiagonal solve is sequential
- Greeks computation can be parallel (batch solves)

```python
@njit(parallel=True)
def batch_solve(lower, diag, upper, rhs_batch, out_batch):
    for j in prange(batch_size):  # Parallel over batch
        thomas_solve(lower, diag, upper, rhs_batch[:, j], out_batch[:, j])
```

### 5.3 Thread Safety

Numba's `prange`:
- Automatic thread-safe iteration
- Each thread gets exclusive index range
- No race conditions for embarrassingly parallel code

---

## 6. Benchmarking

### 6.1 Methodology

**Best Practices:**
1. **Warm-up runs**: Trigger JIT compilation before timing
2. **Multiple runs**: Average over 10+ iterations
3. **Garbage collection**: Call `gc.collect()` before timing
4. **Consistent input**: Use same random seed

```python
def benchmark(func, args, n_runs=10, n_warmup=3):
    # Warm-up
    for _ in range(n_warmup):
        func(*args)
    
    # Timed runs
    gc.collect()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)
```

### 6.2 Key Metrics

| Metric | Definition | Use |
|--------|------------|-----|
| **Latency** | Time per operation | Real-time pricing |
| **Throughput** | Operations/second | Batch processing |
| **Speedup** | T_baseline / T_optimized | Improvement measure |
| **Efficiency** | Speedup / n_cores | Parallel scaling |

### 6.3 Typical Results

| Operation | NumPy | Numba | Speedup |
|-----------|-------|-------|---------|
| GBM 100K paths × 252 steps | 120 ms | 3 ms | 40x |
| Vanilla payoff 1M paths | 2 ms | 0.1 ms | 20x |
| Tridiagonal n=1000 | 0.5 ms | 0.02 ms | 25x |
| PSOR n=500 | 50 ms | 2 ms | 25x |

---

## 7. JAX Backend (Optional)

### Overview

JAX is an optional backend for CPU/GPU-accelerated Monte Carlo. When installed, it is used automatically when the backend is set to `"jax"` or `"auto"` (and Numba is not available). JAX runs on GPU if `jaxlib` is installed with CUDA or ROCm support; no code change is required.

### Backend Selection

```python
from src.core.performance.backend import get_backend, Backend, jax_available, get_jax_version

# Check if JAX is available
if jax_available():
    print("JAX version:", get_jax_version())

# Select backend: "numpy", "numba", "jax", or "auto"
backend = get_backend("auto")  # Prefer NUMBA > JAX > NUMPY
```

### JAX MC Pricer (FX Vanilla)

A JAX-based Monte Carlo pricer for FX European vanilla options is available when JAX is installed. Use `pricer_id="jax_mc"` when resolving the pricer:

```python
from src.pricers.registry import DefaultPricerRegistry

reg = DefaultPricerRegistry().build()
# Resolve JAX MC pricer (raises if JAX not installed and pricer not registered)
pricer = reg.resolve(vanilla_option, pricer_id="jax_mc")
pv = pricer.price(vanilla_option, market)
```

### JAX Kernels

The module `src.core.performance.jax_kernels` provides JAX implementations of GBM terminal spot generation and vanilla/digital payoffs. These are used internally by the JAX MC pricer; device (CPU/GPU) is determined by JAX’s default backend.

---

## 8. Parallel Portfolio Pricing

### Design

Portfolio pricing can run positions in parallel via a wrapper that uses `ThreadPoolExecutor`. The same `PortfolioResult` is returned as for sequential pricing, so callers need not change when not using parallel.

### Thread Safety

Parallel pricing assumes that for each call to `price(instrument, market)` the pricer and market are **read-only** and do not mutate shared state. If a pricer or market is not thread-safe, use `max_workers=1` or the sequential pricer.

### Usage

```python
from src.portfolio.portfolio import PortfolioPricer
from src.portfolio.parallel import ParallelPortfolioPricer

base = PortfolioPricer(pricer_registry=reg)
parallel_pricer = ParallelPortfolioPricer(portfolio_pricer=base, max_workers=4)
result = parallel_pricer.price(portfolio, market)
```

`max_workers=None` uses a default (e.g. CPU count); `max_workers=1` delegates to the underlying sequential pricer.

---

## 9. Caching (Market Data and Pricer Results)

### Market Data Cache

A wrapper around any `MarketDataProvider` caches `get_market(request)` by a stable key derived from the request (e.g. asof, universe ids). Eviction is LRU by `max_size`; optional TTL (seconds) makes entries expire.

```python
from src.marketdata.cache import CachingMarketDataProvider

caching_provider = CachingMarketDataProvider(
    provider=my_provider,
    max_size=512,
    ttl_seconds=300.0,
)
market = caching_provider.get_market(request)
```

### Pricer Result Cache

A wrapper around a portfolio pricer caches `PortfolioResult` by `(portfolio_key, market_key)`. Eviction is LRU by `max_size`; optional TTL. Key functions can be customized (default: `id(portfolio)`, `id(market)`).

```python
from src.portfolio.caching import CachingPortfolioPricer

caching_pricer = CachingPortfolioPricer(
    portfolio_pricer=base_pricer,
    max_size=1024,
    ttl_seconds=60.0,
)
result = caching_pricer.price(portfolio, market)
```

Cache hit: second call with same key returns the cached result without calling the underlying provider/pricer. Cache miss: first call or different key computes and stores the result.

---

## 10. Interview Key Points

### Performance Questions

**Q: How would you speed up a Monte Carlo simulation?**

A: Progressive optimization:
1. **Vectorize** with NumPy (10-100x)
2. **JIT compile** with Numba (10-100x)
3. **Parallelize** over paths (Nx cores)
4. **GPU** with CuPy/CUDA (10-100x more)

**Q: What's the bottleneck in MC pricing?**

A: Path generation (70-80%), specifically:
- Random number generation
- Exponential function calls
- Memory allocation

**Q: Why can't you parallelize the Thomas algorithm?**

A: The forward elimination has a **data dependency**: each $c'_i$ depends on $c'_{i-1}$. This is an inherently sequential algorithm.

**Q: How do you benchmark fairly?**

A:
1. Warm-up to trigger JIT
2. Multiple runs for statistics
3. GC before timing
4. Same random seed
5. Report mean ± std

### Numba Questions

**Q: What does `@njit(parallel=True)` do?**

A: 
- `@njit`: No-Python mode (full compilation, no Python interpreter)
- `parallel=True`: Enable automatic parallelization of `prange` loops

**Q: What are Numba's limitations?**

A:
1. Only supports subset of Python/NumPy
2. First-call compilation overhead
3. No dynamic typing
4. No Python objects in JIT functions

**Q: When is Numba NOT helpful?**

A:
- Already vectorized NumPy code
- I/O bound operations
- Very small arrays (overhead dominates)
- Code using unsupported features

---

## References

1. Numba Documentation: https://numba.pydata.org/
2. "High Performance Python" by Gorelick & Ozsvald
3. Glasserman, P. "Monte Carlo Methods in Financial Engineering"

---

*Document Version: 1.0 | Last Updated: January 2026*
