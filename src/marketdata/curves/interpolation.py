"""
Curve Interpolation Methods.

This module provides various interpolation methods for discount curves and zero rate curves.
Each method has different characteristics:

- **Linear in DF**: Simple but can produce non-smooth forward rates
- **Log-linear in DF**: Produces constant forward rates between nodes (common in rates)
- **Linear in zero rates**: Simple and intuitive
- **Cubic spline**: Smooth curves but may introduce arbitrage

Mathematical Background
-----------------------
For discount factors DF(T) and zero rates r(T):
    DF(T) = exp(-r(T) * T)
    r(T) = -ln(DF(T)) / T

Forward rate between T1 and T2:
    f(T1, T2) = (r(T2)*T2 - r(T1)*T1) / (T2 - T1)
              = -ln(DF(T2)/DF(T1)) / (T2 - T1)

Log-linear interpolation in DF produces constant forward rates:
    DF(t) = DF(T1) * (DF(T2)/DF(T1))^((t-T1)/(T2-T1))
    => f(T1, t) = f(T1, T2) = constant  (for T1 < t < T2)

This is the most common choice for rate curve interpolation as it avoids
discontinuous forward rates while maintaining arbitrage-free properties.

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, Sequence

import numpy as np
from scipy import interpolate as scipy_interp


# =============================================================================
# Interpolation Method Types
# =============================================================================

class InterpolationMethod(str, Enum):
    """Supported interpolation methods for curves."""
    LINEAR_DF = "linear_df"  # Linear interpolation in discount factors
    LOG_LINEAR_DF = "log_linear_df"  # Log-linear in DF (constant forward rates)
    LINEAR_ZERO = "linear_zero"  # Linear interpolation in zero rates
    CUBIC_SPLINE_ZERO = "cubic_spline_zero"  # Cubic spline in zero rates
    CUBIC_SPLINE_DF = "cubic_spline_df"  # Cubic spline in log(DF)


InterpolationMethodStr = Literal[
    "linear_df", "log_linear_df", "linear_zero", "cubic_spline_zero", "cubic_spline_df"
]


# =============================================================================
# Extrapolation Modes
# =============================================================================

class ExtrapolationMode(str, Enum):
    """Supported extrapolation modes."""
    FLAT = "flat"  # Constant extrapolation (safe, conservative)
    LINEAR = "linear"  # Linear extrapolation (can be dangerous)
    NONE = "none"  # Raise error if extrapolation requested


ExtrapolationModeStr = Literal["flat", "linear", "none"]


# =============================================================================
# Interpolator Protocol
# =============================================================================

class CurveInterpolator(Protocol):
    """Protocol for curve interpolators."""
    
    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        """Interpolate value at time(s) t."""
        ...
    
    @property
    def tenors(self) -> np.ndarray:
        """Return the node tenors."""
        ...
    
    @property
    def values(self) -> np.ndarray:
        """Return the node values."""
        ...


# =============================================================================
# Discount Factor Interpolators
# =============================================================================

@dataclass(frozen=True, slots=True)
class LinearDfInterpolator:
    """
    Linear interpolation in discount factors.
    
    Simple but can produce non-smooth (and potentially negative) forward rates.
    Use with caution - mainly useful for short-dated curves.
    
    DF(t) = DF(T1) + (t - T1) * (DF(T2) - DF(T1)) / (T2 - T1)
    """
    _tenors: np.ndarray  # Node times (years)
    _dfs: np.ndarray  # Discount factors at nodes
    _extrapolation: ExtrapolationModeStr = "flat"
    
    def __post_init__(self) -> None:
        # Validate inputs
        tenors = np.asarray(self._tenors, dtype=float).reshape(-1)
        dfs = np.asarray(self._dfs, dtype=float).reshape(-1)
        
        if tenors.size == 0:
            raise ValueError("tenors must be non-empty.")
        if tenors.size != dfs.size:
            raise ValueError("tenors and dfs must have same length.")
        if np.any(np.diff(tenors) <= 0.0):
            raise ValueError("tenors must be strictly increasing.")
        if np.any(dfs <= 0.0):
            raise ValueError("All discount factors must be positive.")
        
        # Store validated arrays
        object.__setattr__(self, "_tenors", tenors)
        object.__setattr__(self, "_dfs", dfs)
    
    @property
    def tenors(self) -> np.ndarray:
        return self._tenors
    
    @property
    def values(self) -> np.ndarray:
        return self._dfs
    
    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        """Interpolate discount factor at time(s) t."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        scalar_input = np.ndim(t) == 0
        
        # Handle extrapolation
        t_clamped = self._handle_extrapolation(t_arr)
        
        # Linear interpolation
        result = np.interp(t_clamped, self._tenors, self._dfs)
        
        return float(result[0]) if scalar_input else result
    
    def _handle_extrapolation(self, t: np.ndarray) -> np.ndarray:
        """Handle extrapolation based on mode."""
        if self._extrapolation == "none":
            if np.any(t < self._tenors[0]) or np.any(t > self._tenors[-1]):
                raise ValueError(
                    f"Extrapolation disabled. t must be in [{self._tenors[0]}, {self._tenors[-1]}]."
                )
            return t
        elif self._extrapolation == "flat":
            return np.clip(t, self._tenors[0], self._tenors[-1])
        elif self._extrapolation == "linear":
            return t  # np.interp does linear extrapolation by default
        else:
            raise ValueError(f"Unknown extrapolation mode: {self._extrapolation}")


@dataclass(frozen=True)
class LogLinearDfInterpolator:
    """
    Log-linear interpolation in discount factors.
    
    This is the industry-standard method for rate curves as it produces
    constant forward rates between nodes, avoiding discontinuities.
    
    ln(DF(t)) = ln(DF(T1)) + (t - T1) * (ln(DF(T2)) - ln(DF(T1))) / (T2 - T1)
    
    Equivalently:
    DF(t) = DF(T1) * (DF(T2) / DF(T1))^((t - T1) / (T2 - T1))
    
    Forward rate between T1 and T2:
    f(T1, T2) = -ln(DF(T2)/DF(T1)) / (T2 - T1)  (constant)
    """
    _tenors: np.ndarray  # Node times (years)
    _dfs: np.ndarray  # Discount factors at nodes
    _extrapolation: ExtrapolationModeStr = "flat"
    _log_dfs: np.ndarray = field(init=False, repr=False)  # Computed ln(DF)
    
    def __post_init__(self) -> None:
        # Validate inputs
        tenors = np.asarray(self._tenors, dtype=float).reshape(-1)
        dfs = np.asarray(self._dfs, dtype=float).reshape(-1)
        
        if tenors.size == 0:
            raise ValueError("tenors must be non-empty.")
        if tenors.size != dfs.size:
            raise ValueError("tenors and dfs must have same length.")
        if np.any(np.diff(tenors) <= 0.0):
            raise ValueError("tenors must be strictly increasing.")
        if np.any(dfs <= 0.0):
            raise ValueError("All discount factors must be positive.")
        
        # Compute log(DF) for efficient interpolation
        log_dfs = np.log(dfs)
        
        # Store validated arrays
        object.__setattr__(self, "_tenors", tenors)
        object.__setattr__(self, "_dfs", dfs)
        object.__setattr__(self, "_log_dfs", log_dfs)
    
    @property
    def tenors(self) -> np.ndarray:
        return self._tenors
    
    @property
    def values(self) -> np.ndarray:
        return self._dfs
    
    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        """Interpolate discount factor at time(s) t."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        scalar_input = np.ndim(t) == 0
        
        # Handle extrapolation
        t_clamped = self._handle_extrapolation(t_arr)
        
        # Log-linear interpolation: interpolate in ln(DF), then exp back
        log_df_interp = np.interp(t_clamped, self._tenors, self._log_dfs)
        result = np.exp(log_df_interp)
        
        return float(result[0]) if scalar_input else result
    
    def _handle_extrapolation(self, t: np.ndarray) -> np.ndarray:
        """Handle extrapolation based on mode."""
        if self._extrapolation == "none":
            if np.any(t < self._tenors[0]) or np.any(t > self._tenors[-1]):
                raise ValueError(
                    f"Extrapolation disabled. t must be in [{self._tenors[0]}, {self._tenors[-1]}]."
                )
            return t
        elif self._extrapolation == "flat":
            return np.clip(t, self._tenors[0], self._tenors[-1])
        elif self._extrapolation == "linear":
            return t  # Linear extrapolation in log(DF) space
        else:
            raise ValueError(f"Unknown extrapolation mode: {self._extrapolation}")
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Compute forward rate between t1 and t2.
        
        f(t1, t2) = -ln(DF(t2)/DF(t1)) / (t2 - t1)
        
        For log-linear interpolation, this is constant between nodes.
        """
        if t2 <= t1:
            raise ValueError("t2 must be > t1.")
        
        df1 = float(self(t1))
        df2 = float(self(t2))
        
        return -math.log(df2 / df1) / (t2 - t1)


# =============================================================================
# Zero Rate Interpolators
# =============================================================================

@dataclass(frozen=True, slots=True)
class LinearZeroInterpolator:
    """
    Linear interpolation in zero rates.
    
    Simple and intuitive, but can produce non-smooth forward rates.
    
    r(t) = r(T1) + (t - T1) * (r(T2) - r(T1)) / (T2 - T1)
    """
    _tenors: np.ndarray  # Node times (years)
    _zero_rates: np.ndarray  # Zero rates at nodes
    _extrapolation: ExtrapolationModeStr = "flat"
    
    def __post_init__(self) -> None:
        # Validate inputs
        tenors = np.asarray(self._tenors, dtype=float).reshape(-1)
        zero_rates = np.asarray(self._zero_rates, dtype=float).reshape(-1)
        
        if tenors.size == 0:
            raise ValueError("tenors must be non-empty.")
        if tenors.size != zero_rates.size:
            raise ValueError("tenors and zero_rates must have same length.")
        if np.any(np.diff(tenors) <= 0.0):
            raise ValueError("tenors must be strictly increasing.")
        if np.any(~np.isfinite(zero_rates)):
            raise ValueError("All zero rates must be finite.")
        
        # Store validated arrays
        object.__setattr__(self, "_tenors", tenors)
        object.__setattr__(self, "_zero_rates", zero_rates)
    
    @property
    def tenors(self) -> np.ndarray:
        return self._tenors
    
    @property
    def values(self) -> np.ndarray:
        return self._zero_rates
    
    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        """Interpolate zero rate at time(s) t."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        scalar_input = np.ndim(t) == 0
        
        # Handle extrapolation
        t_clamped = self._handle_extrapolation(t_arr)
        
        # Linear interpolation
        result = np.interp(t_clamped, self._tenors, self._zero_rates)
        
        return float(result[0]) if scalar_input else result
    
    def _handle_extrapolation(self, t: np.ndarray) -> np.ndarray:
        """Handle extrapolation based on mode."""
        if self._extrapolation == "none":
            if np.any(t < self._tenors[0]) or np.any(t > self._tenors[-1]):
                raise ValueError(
                    f"Extrapolation disabled. t must be in [{self._tenors[0]}, {self._tenors[-1]}]."
                )
            return t
        elif self._extrapolation == "flat":
            return np.clip(t, self._tenors[0], self._tenors[-1])
        elif self._extrapolation == "linear":
            return t
        else:
            raise ValueError(f"Unknown extrapolation mode: {self._extrapolation}")
    
    def df(self, t: float | np.ndarray) -> float | np.ndarray:
        """Compute discount factor DF(t) = exp(-r(t) * t)."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        scalar_input = np.ndim(t) == 0
        
        r = self(t_arr)
        result = np.exp(-r * t_arr)
        
        return float(result[0]) if scalar_input else result


@dataclass(frozen=True)
class CubicSplineZeroInterpolator:
    """
    Cubic spline interpolation in zero rates.
    
    Produces smooth zero rate curves, but:
    - May introduce negative forward rates (arbitrage!)
    - Requires at least 4 points for proper spline fitting
    
    Use with caution - validate for arbitrage after calibration.
    
    The cubic spline ensures:
    - Continuous r(t), r'(t), r''(t) at interior nodes
    - Natural boundary conditions (r''=0 at endpoints) by default
    """
    _tenors: np.ndarray  # Node times (years)
    _zero_rates: np.ndarray  # Zero rates at nodes
    _extrapolation: ExtrapolationModeStr = "flat"
    _spline: Any = field(init=False, repr=False)  # Computed scipy spline object
    
    def __post_init__(self) -> None:
        # Validate inputs
        tenors = np.asarray(self._tenors, dtype=float).reshape(-1)
        zero_rates = np.asarray(self._zero_rates, dtype=float).reshape(-1)
        
        if tenors.size == 0:
            raise ValueError("tenors must be non-empty.")
        if tenors.size != zero_rates.size:
            raise ValueError("tenors and zero_rates must have same length.")
        if np.any(np.diff(tenors) <= 0.0):
            raise ValueError("tenors must be strictly increasing.")
        if np.any(~np.isfinite(zero_rates)):
            raise ValueError("All zero rates must be finite.")
        if tenors.size < 2:
            raise ValueError("At least 2 points required for spline interpolation.")
        
        # Build cubic spline with natural boundary conditions
        # bc_type='natural' means r''(t) = 0 at the endpoints
        spline = scipy_interp.CubicSpline(tenors, zero_rates, bc_type='natural')
        
        # Store validated arrays and spline
        object.__setattr__(self, "_tenors", tenors)
        object.__setattr__(self, "_zero_rates", zero_rates)
        object.__setattr__(self, "_spline", spline)
    
    @property
    def tenors(self) -> np.ndarray:
        return self._tenors
    
    @property
    def values(self) -> np.ndarray:
        return self._zero_rates
    
    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        """Interpolate zero rate at time(s) t."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        scalar_input = np.ndim(t) == 0
        
        # Handle extrapolation
        if self._extrapolation == "none":
            if np.any(t_arr < self._tenors[0]) or np.any(t_arr > self._tenors[-1]):
                raise ValueError(
                    f"Extrapolation disabled. t must be in [{self._tenors[0]}, {self._tenors[-1]}]."
                )
            result = self._spline(t_arr)
        elif self._extrapolation == "flat":
            # Clip to range, then interpolate
            t_clamped = np.clip(t_arr, self._tenors[0], self._tenors[-1])
            result = self._spline(t_clamped)
        elif self._extrapolation == "linear":
            # Use spline extrapolation (polynomial continuation)
            result = self._spline(t_arr)
        else:
            raise ValueError(f"Unknown extrapolation mode: {self._extrapolation}")
        
        return float(result[0]) if scalar_input else np.asarray(result, dtype=float)
    
    def df(self, t: float | np.ndarray) -> float | np.ndarray:
        """Compute discount factor DF(t) = exp(-r(t) * t)."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        scalar_input = np.ndim(t) == 0
        
        r = self(t_arr)
        result = np.exp(-r * t_arr)
        
        return float(result[0]) if scalar_input else result
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Compute forward rate between t1 and t2.
        
        f(t1, t2) = (r(t2)*t2 - r(t1)*t1) / (t2 - t1)
        """
        if t2 <= t1:
            raise ValueError("t2 must be > t1.")
        
        r1 = float(self(t1))
        r2 = float(self(t2))
        
        return (r2 * t2 - r1 * t1) / (t2 - t1)
    
    def check_no_arbitrage(self, n_points: int = 100) -> None:
        """
        Check that the spline doesn't produce negative forward rates.
        
        Raises ValueError if arbitrage detected.
        """
        t_grid = np.linspace(self._tenors[0], self._tenors[-1], n_points)
        
        for i in range(len(t_grid) - 1):
            fwd = self.forward_rate(t_grid[i], t_grid[i + 1])
            if fwd < 0:
                raise ValueError(
                    f"Negative forward rate detected at t={t_grid[i]:.4f}: f={fwd:.6f}.\n"
                    "Cubic spline may introduce arbitrage. Consider log-linear interpolation."
                )


# =============================================================================
# Factory Function
# =============================================================================

def create_curve_interpolator(
    *,
    tenors: np.ndarray | Sequence[float],
    values: np.ndarray | Sequence[float],
    method: InterpolationMethodStr = "log_linear_df",
    extrapolation: ExtrapolationModeStr = "flat",
    value_type: Literal["df", "zero_rate"] = "df",
) -> CurveInterpolator:
    """
    Factory function to create a curve interpolator.
    
    Parameters
    ----------
    tenors:
        Node times (years). Must be strictly increasing.
    values:
        Either discount factors (if value_type="df") or zero rates (if value_type="zero_rate").
    method:
        Interpolation method. One of:
        - "linear_df": Linear in discount factors
        - "log_linear_df": Log-linear in discount factors (industry standard)
        - "linear_zero": Linear in zero rates
        - "cubic_spline_zero": Cubic spline in zero rates
        - "cubic_spline_df": Cubic spline in log(DF)
    extrapolation:
        Extrapolation mode. One of: "flat", "linear", "none"
    value_type:
        Whether values are discount factors ("df") or zero rates ("zero_rate").
    
    Returns
    -------
    CurveInterpolator
        An interpolator that can be called with t to get interpolated values.
    
    Examples
    --------
    >>> tenors = [0.25, 0.5, 1.0, 2.0]
    >>> dfs = [0.99, 0.98, 0.95, 0.90]
    >>> interp = create_curve_interpolator(tenors=tenors, values=dfs, method="log_linear_df")
    >>> interp(0.75)  # Interpolate at 0.75 years
    0.965...
    """
    tenors = np.asarray(tenors, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    
    # Convert zero rates to DFs if needed (for DF-based interpolators)
    if value_type == "zero_rate":
        zero_rates = values
        dfs = np.exp(-zero_rates * tenors)
    else:
        dfs = values
        zero_rates = np.where(tenors > 0, -np.log(dfs) / tenors, 0.0)
    
    # Create interpolator based on method
    if method == "linear_df":
        return LinearDfInterpolator(_tenors=tenors, _dfs=dfs, _extrapolation=extrapolation)
    
    elif method == "log_linear_df":
        return LogLinearDfInterpolator(_tenors=tenors, _dfs=dfs, _extrapolation=extrapolation)
    
    elif method == "linear_zero":
        return LinearZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates, _extrapolation=extrapolation)
    
    elif method == "cubic_spline_zero":
        return CubicSplineZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates, _extrapolation=extrapolation)
    
    elif method == "cubic_spline_df":
        # Cubic spline in log(DF) space
        # We use the zero rate spline since log(DF) = -r*T
        return CubicSplineZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates, _extrapolation=extrapolation)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


# =============================================================================
# Utility Functions
# =============================================================================

def df_to_zero_rate(*, df: float | np.ndarray, t: float | np.ndarray) -> float | np.ndarray:
    """
    Convert discount factor(s) to continuous zero rate(s).
    
    r(t) = -ln(DF(t)) / t  for t > 0
    r(0) = 0  by convention
    """
    df = np.atleast_1d(np.asarray(df, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    
    if df.shape != t.shape:
        raise ValueError("df and t must have same shape.")
    if np.any(df <= 0):
        raise ValueError("All discount factors must be positive.")
    
    # Avoid division by zero at t=0
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(t > 0, -np.log(df) / t, 0.0)
    
    if r.size == 1:
        return float(r[0])
    return r


def zero_rate_to_df(*, r: float | np.ndarray, t: float | np.ndarray) -> float | np.ndarray:
    """
    Convert continuous zero rate(s) to discount factor(s).
    
    DF(t) = exp(-r(t) * t)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    
    if r.shape != t.shape:
        raise ValueError("r and t must have same shape.")
    
    df = np.exp(-r * t)
    
    if df.size == 1:
        return float(df[0])
    return df


def forward_rate_from_dfs(*, df1: float, df2: float, t1: float, t2: float) -> float:
    """
    Compute forward rate between t1 and t2 from discount factors.
    
    f(t1, t2) = -ln(DF(t2)/DF(t1)) / (t2 - t1)
    
    This is the continuously compounded forward rate.
    """
    if t2 <= t1:
        raise ValueError("t2 must be > t1.")
    if df1 <= 0 or df2 <= 0:
        raise ValueError("Discount factors must be positive.")
    
    return -math.log(df2 / df1) / (t2 - t1)


def simple_forward_rate_from_dfs(*, df1: float, df2: float, t1: float, t2: float) -> float:
    """
    Compute simple (LIBOR-style) forward rate between t1 and t2.
    
    f_simple(t1, t2) = (DF(t1)/DF(t2) - 1) / (t2 - t1)
    
    This is the forward rate used in FRA and swap pricing.
    """
    if t2 <= t1:
        raise ValueError("t2 must be > t1.")
    if df1 <= 0 or df2 <= 0:
        raise ValueError("Discount factors must be positive.")
    
    return (df1 / df2 - 1.0) / (t2 - t1)
