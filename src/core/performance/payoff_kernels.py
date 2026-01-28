"""
Numba-Optimized Payoff Evaluation Kernels.

This module provides JIT-compiled kernels for payoff evaluation:
- Vanilla options (call/put)
- Digital options (cash-or-nothing)
- Barrier options (path-dependent)
- Asian options (arithmetic/geometric averaging)
- Lookback options (floating/fixed strike)

Performance Characteristics
---------------------------
- Vanilla payoffs: ~100x speedup vs pure Python
- Path-dependent payoffs: ~20-50x speedup with parallel reduction

Author: QuantStrata Team
"""
from __future__ import annotations

import numpy as np
from typing import Literal, Optional

from src.core.performance.backend import Backend, get_backend, numba_available


# =============================================================================
# Type Aliases
# =============================================================================

OptionTypeStr = Literal["call", "put"]
BarrierTypeStr = Literal["up_and_out", "up_and_in", "down_and_out", "down_and_in"]
AsianTypeStr = Literal["arithmetic", "geometric"]
LookbackTypeStr = Literal["floating", "fixed"]


# =============================================================================
# NumPy Implementations (Baseline)
# =============================================================================

def vanilla_payoff_numpy(
    spots: np.ndarray,
    strike: float,
    option_type: OptionTypeStr,
) -> np.ndarray:
    """
    Compute vanilla option payoff using NumPy.
    
    Parameters
    ----------
    spots:
        Terminal spot prices, shape (n_paths,).
    strike:
        Strike price K.
    option_type:
        "call" for max(S-K, 0), "put" for max(K-S, 0).
        
    Returns
    -------
    np.ndarray
        Payoffs, shape (n_paths,).
    """
    if option_type == "call":
        return np.maximum(spots - strike, 0.0)
    else:
        return np.maximum(strike - spots, 0.0)


def digital_payoff_numpy(
    spots: np.ndarray,
    strike: float,
    option_type: OptionTypeStr,
    payout: float = 1.0,
) -> np.ndarray:
    """
    Compute digital (cash-or-nothing) option payoff using NumPy.
    
    Parameters
    ----------
    spots:
        Terminal spot prices.
    strike:
        Strike price K.
    option_type:
        "call" pays if S > K, "put" pays if S < K.
    payout:
        Cash payout amount.
        
    Returns
    -------
    np.ndarray
        Payoffs (0 or payout).
    """
    if option_type == "call":
        return np.where(spots > strike, payout, 0.0)
    else:
        return np.where(spots < strike, payout, 0.0)


def barrier_payoff_numpy(
    paths: np.ndarray,
    strike: float,
    barrier: float,
    option_type: OptionTypeStr,
    barrier_type: BarrierTypeStr,
) -> np.ndarray:
    """
    Compute barrier option payoff using NumPy.
    
    Parameters
    ----------
    paths:
        Full price paths, shape (n_steps+1, n_paths).
    strike:
        Strike price K.
    barrier:
        Barrier level B.
    option_type:
        "call" or "put".
    barrier_type:
        One of "up_and_out", "up_and_in", "down_and_out", "down_and_in".
        
    Returns
    -------
    np.ndarray
        Payoffs, shape (n_paths,).
    """
    n_paths = paths.shape[1]
    terminal_spots = paths[-1, :]
    
    # Check barrier breach
    if barrier_type.startswith("up"):
        breached = np.any(paths >= barrier, axis=0)
    else:  # down
        breached = np.any(paths <= barrier, axis=0)
    
    # Compute vanilla payoff
    if option_type == "call":
        vanilla = np.maximum(terminal_spots - strike, 0.0)
    else:
        vanilla = np.maximum(strike - terminal_spots, 0.0)
    
    # Apply barrier logic
    if barrier_type.endswith("out"):
        # Knock-out: pay only if NOT breached
        return np.where(breached, 0.0, vanilla)
    else:  # in
        # Knock-in: pay only if breached
        return np.where(breached, vanilla, 0.0)


def asian_payoff_numpy(
    paths: np.ndarray,
    strike: float,
    option_type: OptionTypeStr,
    asian_type: AsianTypeStr = "arithmetic",
) -> np.ndarray:
    """
    Compute Asian option payoff using NumPy.
    
    Parameters
    ----------
    paths:
        Full price paths, shape (n_steps+1, n_paths).
    strike:
        Strike price K.
    option_type:
        "call" or "put".
    asian_type:
        "arithmetic" for arithmetic average, "geometric" for geometric.
        
    Returns
    -------
    np.ndarray
        Payoffs, shape (n_paths,).
    """
    if asian_type == "arithmetic":
        avg = np.mean(paths, axis=0)
    else:  # geometric
        avg = np.exp(np.mean(np.log(paths), axis=0))
    
    if option_type == "call":
        return np.maximum(avg - strike, 0.0)
    else:
        return np.maximum(strike - avg, 0.0)


def lookback_payoff_numpy(
    paths: np.ndarray,
    strike: Optional[float],
    option_type: OptionTypeStr,
    lookback_type: LookbackTypeStr = "floating",
) -> np.ndarray:
    """
    Compute lookback option payoff using NumPy.
    
    Parameters
    ----------
    paths:
        Full price paths, shape (n_steps+1, n_paths).
    strike:
        Strike price K (only used for fixed strike lookback).
    option_type:
        "call" or "put".
    lookback_type:
        "floating" (strike set at min/max) or "fixed".
        
    Returns
    -------
    np.ndarray
        Payoffs, shape (n_paths,).
    """
    terminal = paths[-1, :]
    
    if lookback_type == "floating":
        if option_type == "call":
            # Call: S_T - min(S)
            min_s = np.min(paths, axis=0)
            return terminal - min_s
        else:
            # Put: max(S) - S_T
            max_s = np.max(paths, axis=0)
            return max_s - terminal
    else:  # fixed
        if strike is None:
            raise ValueError("strike required for fixed lookback")
        if option_type == "call":
            # Call: max(max(S) - K, 0)
            max_s = np.max(paths, axis=0)
            return np.maximum(max_s - strike, 0.0)
        else:
            # Put: max(K - min(S), 0)
            min_s = np.min(paths, axis=0)
            return np.maximum(strike - min_s, 0.0)


# =============================================================================
# Numba Implementations (High Performance)
# =============================================================================

_PAYOFF_KERNELS_COMPILED = False
_vanilla_call_numba = None
_vanilla_put_numba = None
_digital_call_numba = None
_digital_put_numba = None
_barrier_monitor_numba = None
_asian_arithmetic_numba = None
_asian_geometric_numba = None
_lookback_floating_call_numba = None
_lookback_floating_put_numba = None


def _compile_payoff_kernels() -> None:
    """Compile Numba payoff kernels on first use."""
    global _PAYOFF_KERNELS_COMPILED
    global _vanilla_call_numba, _vanilla_put_numba
    global _digital_call_numba, _digital_put_numba
    global _barrier_monitor_numba
    global _asian_arithmetic_numba, _asian_geometric_numba
    global _lookback_floating_call_numba, _lookback_floating_put_numba
    
    if _PAYOFF_KERNELS_COMPILED:
        return
        
    if not numba_available():
        raise ImportError("Numba is required for JIT-compiled kernels.")
        
    from numba import njit, prange
    
    # -------------------------------------------------------------------------
    # Vanilla Payoffs (JIT)
    # -------------------------------------------------------------------------
    
    @njit(parallel=True, cache=True, fastmath=True)
    def vanilla_call_jit(spots: np.ndarray, strike: float, out: np.ndarray) -> None:
        """JIT-compiled call payoff: max(S - K, 0)."""
        n = spots.shape[0]
        for i in prange(n):
            diff = spots[i] - strike
            out[i] = diff if diff > 0.0 else 0.0
    
    @njit(parallel=True, cache=True, fastmath=True)
    def vanilla_put_jit(spots: np.ndarray, strike: float, out: np.ndarray) -> None:
        """JIT-compiled put payoff: max(K - S, 0)."""
        n = spots.shape[0]
        for i in prange(n):
            diff = strike - spots[i]
            out[i] = diff if diff > 0.0 else 0.0
    
    # -------------------------------------------------------------------------
    # Digital Payoffs (JIT)
    # -------------------------------------------------------------------------
    
    @njit(parallel=True, cache=True, fastmath=True)
    def digital_call_jit(spots: np.ndarray, strike: float, payout: float, out: np.ndarray) -> None:
        """JIT-compiled digital call: payout if S > K."""
        n = spots.shape[0]
        for i in prange(n):
            out[i] = payout if spots[i] > strike else 0.0
    
    @njit(parallel=True, cache=True, fastmath=True)
    def digital_put_jit(spots: np.ndarray, strike: float, payout: float, out: np.ndarray) -> None:
        """JIT-compiled digital put: payout if S < K."""
        n = spots.shape[0]
        for i in prange(n):
            out[i] = payout if spots[i] < strike else 0.0
    
    # -------------------------------------------------------------------------
    # Barrier Monitoring (JIT)
    # -------------------------------------------------------------------------
    
    @njit(parallel=True, cache=True)
    def barrier_up_breached_jit(paths: np.ndarray, barrier: float, breached: np.ndarray) -> None:
        """Check if paths breach upper barrier."""
        n_steps = paths.shape[0]
        n_paths = paths.shape[1]
        for p in prange(n_paths):
            hit = False
            for t in range(n_steps):
                if paths[t, p] >= barrier:
                    hit = True
                    break
            breached[p] = hit
    
    @njit(parallel=True, cache=True)
    def barrier_down_breached_jit(paths: np.ndarray, barrier: float, breached: np.ndarray) -> None:
        """Check if paths breach lower barrier."""
        n_steps = paths.shape[0]
        n_paths = paths.shape[1]
        for p in prange(n_paths):
            hit = False
            for t in range(n_steps):
                if paths[t, p] <= barrier:
                    hit = True
                    break
            breached[p] = hit
    
    # -------------------------------------------------------------------------
    # Asian Averaging (JIT)
    # -------------------------------------------------------------------------
    
    @njit(parallel=True, cache=True, fastmath=True)
    def asian_arithmetic_avg_jit(paths: np.ndarray, out: np.ndarray) -> None:
        """Compute arithmetic average along paths."""
        n_steps = paths.shape[0]
        n_paths = paths.shape[1]
        for p in prange(n_paths):
            total = 0.0
            for t in range(n_steps):
                total += paths[t, p]
            out[p] = total / n_steps
    
    @njit(parallel=True, cache=True, fastmath=True)
    def asian_geometric_avg_jit(paths: np.ndarray, out: np.ndarray) -> None:
        """Compute geometric average along paths."""
        n_steps = paths.shape[0]
        n_paths = paths.shape[1]
        for p in prange(n_paths):
            log_sum = 0.0
            for t in range(n_steps):
                log_sum += np.log(paths[t, p])
            out[p] = np.exp(log_sum / n_steps)
    
    # -------------------------------------------------------------------------
    # Lookback Extrema (JIT)
    # -------------------------------------------------------------------------
    
    @njit(parallel=True, cache=True, fastmath=True)
    def lookback_floating_call_jit(paths: np.ndarray, out: np.ndarray) -> None:
        """Lookback floating call: S_T - min(S)."""
        n_steps = paths.shape[0]
        n_paths = paths.shape[1]
        for p in prange(n_paths):
            min_s = paths[0, p]
            for t in range(1, n_steps):
                if paths[t, p] < min_s:
                    min_s = paths[t, p]
            out[p] = paths[n_steps - 1, p] - min_s
    
    @njit(parallel=True, cache=True, fastmath=True)
    def lookback_floating_put_jit(paths: np.ndarray, out: np.ndarray) -> None:
        """Lookback floating put: max(S) - S_T."""
        n_steps = paths.shape[0]
        n_paths = paths.shape[1]
        for p in prange(n_paths):
            max_s = paths[0, p]
            for t in range(1, n_steps):
                if paths[t, p] > max_s:
                    max_s = paths[t, p]
            out[p] = max_s - paths[n_steps - 1, p]
    
    # Store compiled functions
    _vanilla_call_numba = vanilla_call_jit
    _vanilla_put_numba = vanilla_put_jit
    _digital_call_numba = digital_call_jit
    _digital_put_numba = digital_put_jit
    _barrier_monitor_numba = (barrier_up_breached_jit, barrier_down_breached_jit)
    _asian_arithmetic_numba = asian_arithmetic_avg_jit
    _asian_geometric_numba = asian_geometric_avg_jit
    _lookback_floating_call_numba = lookback_floating_call_jit
    _lookback_floating_put_numba = lookback_floating_put_jit
    
    _PAYOFF_KERNELS_COMPILED = True


# =============================================================================
# Unified API
# =============================================================================

def vanilla_payoff(
    spots: np.ndarray,
    strike: float,
    option_type: OptionTypeStr,
    backend: str = "auto",
) -> np.ndarray:
    """
    Compute vanilla option payoff with automatic backend selection.
    
    Parameters
    ----------
    spots:
        Terminal spot prices, shape (n_paths,).
    strike:
        Strike price K.
    option_type:
        "call" for max(S-K, 0), "put" for max(K-S, 0).
    backend:
        Computational backend: "numpy", "numba", or "auto".
        
    Returns
    -------
    np.ndarray
        Payoffs, shape (n_paths,).
    """
    selected = get_backend(backend)
    
    if selected == Backend.NUMBA:
        _compile_payoff_kernels()
        out = np.empty(spots.shape[0], dtype=np.float64)
        spots = np.ascontiguousarray(spots, dtype=np.float64)
        if option_type == "call":
            _vanilla_call_numba(spots, float(strike), out)
        else:
            _vanilla_put_numba(spots, float(strike), out)
        return out
    else:
        return vanilla_payoff_numpy(spots, strike, option_type)


def digital_payoff(
    spots: np.ndarray,
    strike: float,
    option_type: OptionTypeStr,
    payout: float = 1.0,
    backend: str = "auto",
) -> np.ndarray:
    """
    Compute digital option payoff with automatic backend selection.
    """
    selected = get_backend(backend)
    
    if selected == Backend.NUMBA:
        _compile_payoff_kernels()
        out = np.empty(spots.shape[0], dtype=np.float64)
        spots = np.ascontiguousarray(spots, dtype=np.float64)
        if option_type == "call":
            _digital_call_numba(spots, float(strike), float(payout), out)
        else:
            _digital_put_numba(spots, float(strike), float(payout), out)
        return out
    else:
        return digital_payoff_numpy(spots, strike, option_type, payout)


def barrier_payoff(
    paths: np.ndarray,
    strike: float,
    barrier: float,
    option_type: OptionTypeStr,
    barrier_type: BarrierTypeStr,
    backend: str = "auto",
) -> np.ndarray:
    """
    Compute barrier option payoff with automatic backend selection.
    """
    selected = get_backend(backend)
    
    if selected == Backend.NUMBA:
        _compile_payoff_kernels()
        paths = np.ascontiguousarray(paths, dtype=np.float64)
        n_paths = paths.shape[1]
        
        # Check barrier breach
        breached = np.empty(n_paths, dtype=np.bool_)
        up_breach_fn, down_breach_fn = _barrier_monitor_numba
        if barrier_type.startswith("up"):
            up_breach_fn(paths, float(barrier), breached)
        else:
            down_breach_fn(paths, float(barrier), breached)
        
        # Compute vanilla payoff
        terminal = paths[-1, :]
        vanilla = vanilla_payoff(terminal, strike, option_type, backend="numba")
        
        # Apply barrier logic
        if barrier_type.endswith("out"):
            return np.where(breached, 0.0, vanilla)
        else:
            return np.where(breached, vanilla, 0.0)
    else:
        return barrier_payoff_numpy(paths, strike, barrier, option_type, barrier_type)


def asian_payoff(
    paths: np.ndarray,
    strike: float,
    option_type: OptionTypeStr,
    asian_type: AsianTypeStr = "arithmetic",
    backend: str = "auto",
) -> np.ndarray:
    """
    Compute Asian option payoff with automatic backend selection.
    """
    selected = get_backend(backend)
    
    if selected == Backend.NUMBA:
        _compile_payoff_kernels()
        paths = np.ascontiguousarray(paths, dtype=np.float64)
        n_paths = paths.shape[1]
        
        # Compute average
        avg = np.empty(n_paths, dtype=np.float64)
        if asian_type == "arithmetic":
            _asian_arithmetic_numba(paths, avg)
        else:
            _asian_geometric_numba(paths, avg)
        
        # Compute payoff
        return vanilla_payoff(avg, strike, option_type, backend="numba")
    else:
        return asian_payoff_numpy(paths, strike, option_type, asian_type)


def lookback_payoff(
    paths: np.ndarray,
    strike: Optional[float],
    option_type: OptionTypeStr,
    lookback_type: LookbackTypeStr = "floating",
    backend: str = "auto",
) -> np.ndarray:
    """
    Compute lookback option payoff with automatic backend selection.
    """
    selected = get_backend(backend)
    
    if selected == Backend.NUMBA and lookback_type == "floating":
        _compile_payoff_kernels()
        paths = np.ascontiguousarray(paths, dtype=np.float64)
        n_paths = paths.shape[1]
        out = np.empty(n_paths, dtype=np.float64)
        
        if option_type == "call":
            _lookback_floating_call_numba(paths, out)
        else:
            _lookback_floating_put_numba(paths, out)
        return out
    else:
        return lookback_payoff_numpy(paths, strike, option_type, lookback_type)
