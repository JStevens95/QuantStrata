"""
Performance Benchmarking Framework.

This module provides tools for systematic performance measurement:
- Timing utilities with warm-up and multiple runs
- Memory profiling
- Comparison across backends
- Reporting and visualization

Usage
-----
>>> from src.core.performance.benchmark import benchmark_function, BenchmarkResult
>>> result = benchmark_function(my_function, args, n_runs=10)
>>> print(result)

Author: QuantStrata Team
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.core.performance.backend import Backend, get_backend, numba_available


# =============================================================================
# Benchmark Result Container
# =============================================================================

@dataclass
class BenchmarkResult:
    """
    Container for benchmark results.
    
    Attributes
    ----------
    name:
        Benchmark name/description.
    backend:
        Backend used ("numpy" or "numba").
    n_runs:
        Number of timing runs.
    times:
        Individual run times in seconds.
    mean_time:
        Mean execution time.
    std_time:
        Standard deviation of times.
    min_time:
        Minimum execution time.
    max_time:
        Maximum execution time.
    throughput:
        Operations per second (if applicable).
    memory_mb:
        Peak memory usage in MB (if measured).
    metadata:
        Additional metadata.
    """
    name: str
    backend: str
    n_runs: int
    times: List[float]
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: Optional[float] = None
    memory_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        lines = [
            f"Benchmark: {self.name}",
            f"  Backend: {self.backend}",
            f"  Runs: {self.n_runs}",
            f"  Mean: {self.mean_time * 1000:.3f} ms",
            f"  Std:  {self.std_time * 1000:.3f} ms",
            f"  Min:  {self.min_time * 1000:.3f} ms",
            f"  Max:  {self.max_time * 1000:.3f} ms",
        ]
        if self.throughput:
            lines.append(f"  Throughput: {self.throughput:.2e} ops/sec")
        if self.memory_mb:
            lines.append(f"  Memory: {self.memory_mb:.1f} MB")
        return "\n".join(lines)
    
    def speedup_vs(self, other: "BenchmarkResult") -> float:
        """Compute speedup factor vs another result."""
        return other.mean_time / self.mean_time


# =============================================================================
# Timing Utilities
# =============================================================================

def benchmark_function(
    func: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict] = None,
    n_runs: int = 10,
    n_warmup: int = 2,
    name: Optional[str] = None,
    backend: str = "auto",
    n_ops: Optional[int] = None,
) -> BenchmarkResult:
    """
    Benchmark a function with warm-up and multiple runs.
    
    Parameters
    ----------
    func:
        Function to benchmark.
    args:
        Positional arguments for func.
    kwargs:
        Keyword arguments for func.
    n_runs:
        Number of timed runs.
    n_warmup:
        Number of warm-up runs (not timed).
    name:
        Benchmark name (defaults to func.__name__).
    backend:
        Backend identifier for reporting.
    n_ops:
        Number of operations (for throughput calculation).
        
    Returns
    -------
    BenchmarkResult
        Benchmark results.
        
    Examples
    --------
    >>> def matrix_multiply(a, b):
    ...     return np.dot(a, b)
    >>> a = np.random.randn(1000, 1000)
    >>> b = np.random.randn(1000, 1000)
    >>> result = benchmark_function(matrix_multiply, (a, b), n_runs=5)
    >>> print(result)
    """
    kwargs = kwargs or {}
    name = name or getattr(func, "__name__", "unknown")
    
    # Warm-up runs (trigger JIT compilation, cache warming)
    for _ in range(n_warmup):
        func(*args, **kwargs)
    
    # Force garbage collection before timing
    gc.collect()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    times_arr = np.array(times)
    
    throughput = None
    if n_ops is not None:
        throughput = n_ops / np.mean(times_arr)
    
    return BenchmarkResult(
        name=name,
        backend=backend,
        n_runs=n_runs,
        times=times,
        mean_time=float(np.mean(times_arr)),
        std_time=float(np.std(times_arr)),
        min_time=float(np.min(times_arr)),
        max_time=float(np.max(times_arr)),
        throughput=throughput,
    )


def compare_backends(
    numpy_func: Callable,
    numba_func: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict] = None,
    n_runs: int = 10,
    n_warmup: int = 3,
    name: str = "comparison",
    n_ops: Optional[int] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Compare NumPy and Numba implementations.
    
    Parameters
    ----------
    numpy_func:
        NumPy-based implementation.
    numba_func:
        Numba-based implementation.
    args:
        Arguments for both functions.
    kwargs:
        Keyword arguments for both functions.
    n_runs:
        Number of timed runs per backend.
    n_warmup:
        Warm-up runs.
    name:
        Benchmark name.
    n_ops:
        Number of operations (for throughput).
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by "numpy" and "numba".
    """
    results = {}
    
    # NumPy benchmark
    results["numpy"] = benchmark_function(
        numpy_func, args, kwargs, n_runs, n_warmup,
        f"{name} (NumPy)", "numpy", n_ops
    )
    
    # Numba benchmark (skip if not available)
    if numba_available():
        results["numba"] = benchmark_function(
            numba_func, args, kwargs, n_runs, n_warmup,
            f"{name} (Numba)", "numba", n_ops
        )
    
    return results


# =============================================================================
# Predefined Benchmarks
# =============================================================================

def benchmark_gbm_simulation(
    n_paths: int = 100000,
    n_steps: int = 252,
    n_runs: int = 5,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark GBM path simulation.
    
    Parameters
    ----------
    n_paths:
        Number of Monte Carlo paths.
    n_steps:
        Number of time steps.
    n_runs:
        Number of benchmark runs.
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results for each backend.
    """
    from src.core.performance.mc_kernels import (
        _simulate_gbm_paths_numpy,
        _simulate_gbm_paths_numba,
    )
    
    # Generate random numbers once (shared)
    np.random.seed(42)
    z = np.random.randn(n_steps, n_paths)
    
    spot0 = 100.0
    drift = 0.05
    vol = 0.20
    T = 1.0
    
    results = {}
    
    # NumPy
    def run_numpy():
        return _simulate_gbm_paths_numpy(
            spot0, drift, vol, T, n_steps, n_paths, z, "exact", False
        )
    
    results["numpy"] = benchmark_function(
        run_numpy, (), None, n_runs, 2,
        f"GBM {n_paths:,} paths × {n_steps} steps (NumPy)",
        "numpy", n_paths * n_steps
    )
    
    # Numba
    if numba_available():
        def run_numba():
            return _simulate_gbm_paths_numba(
                spot0, drift, vol, T, n_steps, n_paths, z, "exact", False
            )
        
        results["numba"] = benchmark_function(
            run_numba, (), None, n_runs, 3,
            f"GBM {n_paths:,} paths × {n_steps} steps (Numba)",
            "numba", n_paths * n_steps
        )
    
    return results


def benchmark_vanilla_payoff(
    n_paths: int = 1000000,
    n_runs: int = 10,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark vanilla payoff evaluation.
    """
    from src.core.performance.payoff_kernels import (
        vanilla_payoff_numpy,
        vanilla_payoff,
    )
    
    np.random.seed(42)
    spots = np.random.lognormal(mean=np.log(100), sigma=0.2, size=n_paths)
    strike = 100.0
    
    results = {}
    
    # NumPy
    results["numpy"] = benchmark_function(
        lambda: vanilla_payoff_numpy(spots, strike, "call"),
        (), None, n_runs, 2,
        f"Vanilla payoff {n_paths:,} paths (NumPy)",
        "numpy", n_paths
    )
    
    # Numba
    if numba_available():
        results["numba"] = benchmark_function(
            lambda: vanilla_payoff(spots, strike, "call", backend="numba"),
            (), None, n_runs, 3,
            f"Vanilla payoff {n_paths:,} paths (Numba)",
            "numba", n_paths
        )
    
    return results


def benchmark_tridiagonal_solve(
    n: int = 1000,
    n_runs: int = 20,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark tridiagonal solver.
    """
    from src.core.performance.fd_kernels import (
        solve_tridiagonal_numpy,
        solve_tridiagonal,
    )
    
    # Create diagonally dominant tridiagonal system
    np.random.seed(42)
    lower = -np.ones(n - 1)
    diag = 3.0 * np.ones(n)
    upper = -np.ones(n - 1)
    rhs = np.random.randn(n)
    
    results = {}
    
    # NumPy
    results["numpy"] = benchmark_function(
        lambda: solve_tridiagonal_numpy(lower, diag, upper, rhs),
        (), None, n_runs, 2,
        f"Tridiagonal solve n={n} (NumPy)",
        "numpy", n
    )
    
    # Numba
    if numba_available():
        results["numba"] = benchmark_function(
            lambda: solve_tridiagonal(lower, diag, upper, rhs, backend="numba"),
            (), None, n_runs, 3,
            f"Tridiagonal solve n={n} (Numba)",
            "numba", n
        )
    
    return results


def benchmark_psor_solve(
    n: int = 500,
    n_runs: int = 10,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark PSOR solver.
    """
    from src.core.performance.fd_kernels import (
        solve_tridiagonal_psor_numpy,
        solve_tridiagonal_psor,
    )
    
    # Create typical American option PDE system
    np.random.seed(42)
    lower = -0.5 * np.ones(n - 1)
    diag = 1.5 * np.ones(n)
    upper = -0.5 * np.ones(n - 1)
    rhs = np.linspace(10, -10, n)  # Typical PDE RHS
    floor = np.maximum(np.linspace(100, 80, n) - 90, 0)  # Intrinsic value
    
    results = {}
    
    # NumPy
    results["numpy"] = benchmark_function(
        lambda: solve_tridiagonal_psor_numpy(lower, diag, upper, rhs, floor),
        (), None, n_runs, 2,
        f"PSOR solve n={n} (NumPy)",
        "numpy"
    )
    
    # Numba
    if numba_available():
        results["numba"] = benchmark_function(
            lambda: solve_tridiagonal_psor(lower, diag, upper, rhs, floor, backend="numba"),
            (), None, n_runs, 3,
            f"PSOR solve n={n} (Numba)",
            "numba"
        )
    
    return results


# =============================================================================
# Full Benchmark Suite
# =============================================================================

def run_all_benchmarks(verbose: bool = True) -> Dict[str, Dict[str, BenchmarkResult]]:
    """
    Run complete benchmark suite.
    
    Returns
    -------
    Dict[str, Dict[str, BenchmarkResult]]
        Nested dict: benchmark_name -> backend -> result
    """
    all_results = {}
    
    benchmarks = [
        ("GBM Simulation (100K paths)", benchmark_gbm_simulation),
        ("Vanilla Payoff (1M paths)", benchmark_vanilla_payoff),
        ("Tridiagonal Solve (n=1000)", benchmark_tridiagonal_solve),
        ("PSOR Solve (n=500)", benchmark_psor_solve),
    ]
    
    for name, bench_func in benchmarks:
        if verbose:
            print(f"\nRunning: {name}")
            print("-" * 50)
        
        results = bench_func()
        all_results[name] = results
        
        if verbose:
            for backend, result in results.items():
                print(f"\n{result}")
            
            # Print speedup if both backends available
            if "numpy" in results and "numba" in results:
                speedup = results["numba"].speedup_vs(results["numpy"])
                print(f"\n  Speedup (Numba vs NumPy): {speedup:.1f}x")
    
    return all_results


def print_benchmark_summary(results: Dict[str, Dict[str, BenchmarkResult]]) -> None:
    """Print formatted benchmark summary table."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Benchmark':<35} {'NumPy (ms)':<12} {'Numba (ms)':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for name, backends in results.items():
        numpy_time = backends.get("numpy")
        numba_time = backends.get("numba")
        
        numpy_ms = f"{numpy_time.mean_time * 1000:.2f}" if numpy_time else "N/A"
        numba_ms = f"{numba_time.mean_time * 1000:.2f}" if numba_time else "N/A"
        
        if numpy_time and numba_time:
            speedup = f"{numba_time.speedup_vs(numpy_time):.1f}x"
        else:
            speedup = "N/A"
        
        print(f"{name:<35} {numpy_ms:<12} {numba_ms:<12} {speedup:<10}")
    
    print("=" * 70)
