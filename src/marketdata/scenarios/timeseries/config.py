"""
Configuration dataclasses for time series generation.

This module defines the configuration schema for specifying risk factors,
their dynamics, and the overall time series generation parameters.

Mathematical Background
-----------------------
Each risk factor follows a stochastic process:

**GBM (Geometric Brownian Motion)**:
    dS_t = μ S_t dt + σ S_t dW_t

**Heston (Stochastic Volatility)**:
    dS_t = μ S_t dt + √V_t S_t dW_t^S
    dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V
    Corr(dW_t^S, dW_t^V) = ρ

**Ornstein-Uhlenbeck (Mean-Reverting)**:
    dX_t = κ(θ - X_t) dt + σ dW_t

Correlation between different risk factors is handled via Cholesky decomposition.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Union

from src.marketdata.core.ids import MarketId


# =============================================================================
# Dynamics Specifications
# =============================================================================

@dataclass(frozen=True, slots=True)
class GBMDynamicsSpec:
    """
    Specification for Geometric Brownian Motion dynamics.

    Mathematical Model
    ------------------
    dS_t = μ S_t dt + σ S_t dW_t

    Solution: S_t = S_0 exp((μ - σ²/2)t + σ W_t)

    Parameters
    ----------
    drift : float
        Drift parameter μ (annual). For risk-neutral: μ = r - q.
        Typical values: -0.05 to 0.10.
    vol : float
        Volatility parameter σ (annual). Must be >= 0.
        Typical values: 0.05 to 0.50.

    Examples
    --------
    >>> # FX spot with 8% annual vol, zero drift (martingale under domestic measure)
    >>> fx_dynamics = GBMDynamicsSpec(drift=0.0, vol=0.08)
    >>>
    >>> # Equity with 5% drift and 20% vol
    >>> eq_dynamics = GBMDynamicsSpec(drift=0.05, vol=0.20)
    """

    drift: float
    vol: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.drift):
            raise ValueError("GBMDynamicsSpec.drift must be finite.")
        if not np.isfinite(self.vol):
            raise ValueError("GBMDynamicsSpec.vol must be finite.")
        if self.vol < 0.0:
            raise ValueError("GBMDynamicsSpec.vol must be >= 0.")


@dataclass(frozen=True, slots=True)
class HestonDynamicsSpec:
    """
    Specification for Heston stochastic volatility dynamics.

    Mathematical Model
    ------------------
    dS_t = μ S_t dt + √V_t S_t dW_t^S
    dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V
    Corr(dW_t^S, dW_t^V) = ρ_internal

    Parameters
    ----------
    drift : float
        Risk-neutral drift μ = r - q.
    kappa : float
        Mean reversion speed κ > 0.
    theta : float
        Long-term variance θ > 0.
    xi : float
        Volatility of variance ξ > 0 (vol-of-vol).
    v0 : float
        Initial variance V_0 > 0.
    rho_internal : float
        Spot-variance correlation ρ ∈ (-1, 1).
        Note: This is the internal Heston correlation, separate from
        cross-factor correlation handled by TimeseriesGenerator.

    Notes
    -----
    **Feller Condition**: If 2κθ > ξ², variance stays positive a.s.

    Examples
    --------
    >>> # Typical equity Heston parameters
    >>> heston = HestonDynamicsSpec(
    ...     drift=0.03,
    ...     kappa=2.0,
    ...     theta=0.04,  # 20% long-term vol
    ...     xi=0.3,
    ...     v0=0.04,
    ...     rho_internal=-0.7,  # Negative leverage effect
    ... )
    """

    drift: float
    kappa: float
    theta: float
    xi: float
    v0: float
    rho_internal: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.drift):
            raise ValueError("HestonDynamicsSpec.drift must be finite.")
        if self.kappa <= 0.0:
            raise ValueError("HestonDynamicsSpec.kappa must be > 0.")
        if self.theta <= 0.0:
            raise ValueError("HestonDynamicsSpec.theta must be > 0.")
        if self.xi <= 0.0:
            raise ValueError("HestonDynamicsSpec.xi must be > 0.")
        if self.v0 <= 0.0:
            raise ValueError("HestonDynamicsSpec.v0 must be > 0.")
        if not (-1.0 < self.rho_internal < 1.0):
            raise ValueError("HestonDynamicsSpec.rho_internal must be in (-1, 1).")

    @property
    def feller_ratio(self) -> float:
        """2κθ/ξ². If > 1, Feller condition is satisfied."""
        return 2.0 * self.kappa * self.theta / (self.xi ** 2)

    @property
    def feller_satisfied(self) -> bool:
        """Check if Feller condition 2κθ > ξ² holds."""
        return self.feller_ratio > 1.0


@dataclass(frozen=True, slots=True)
class OUDynamicsSpec:
    """
    Specification for Ornstein-Uhlenbeck (mean-reverting) dynamics.

    Mathematical Model
    ------------------
    dX_t = κ(θ - X_t) dt + σ dW_t

    Solution: X_t = θ + (X_0 - θ)e^(-κt) + σ∫₀ᵗ e^(-κ(t-s)) dW_s

    Parameters
    ----------
    mean : float
        Long-term mean θ.
    kappa : float
        Mean reversion speed κ > 0.
    vol : float
        Volatility parameter σ >= 0.

    Notes
    -----
    **Half-life**: t_{1/2} = ln(2)/κ

    Useful for:
    - Interest rate levels
    - Credit spreads
    - Volatility factors
    - Basis/spread trading

    Examples
    --------
    >>> # Rate level with 5% mean, 50% annual mean reversion, 50bp vol
    >>> rate_dynamics = OUDynamicsSpec(mean=0.05, kappa=0.5, vol=0.005)
    >>>
    >>> # Vol factor with zero mean (shock process), 20% mean reversion
    >>> vol_factor = OUDynamicsSpec(mean=0.0, kappa=0.2, vol=0.05)
    """

    mean: float
    kappa: float
    vol: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("OUDynamicsSpec.mean must be finite.")
        if not np.isfinite(self.kappa):
            raise ValueError("OUDynamicsSpec.kappa must be finite.")
        if self.kappa <= 0.0:
            raise ValueError("OUDynamicsSpec.kappa must be > 0.")
        if not np.isfinite(self.vol):
            raise ValueError("OUDynamicsSpec.vol must be finite.")
        if self.vol < 0.0:
            raise ValueError("OUDynamicsSpec.vol must be >= 0.")

    @property
    def half_life(self) -> float:
        """Half-life of mean reversion: ln(2)/κ."""
        return np.log(2.0) / self.kappa

    @property
    def stationary_variance(self) -> float:
        """Long-run variance: σ²/(2κ)."""
        return (self.vol ** 2) / (2.0 * self.kappa)


@dataclass(frozen=True, slots=True)
class FactorDynamicsSpec:
    """
    Specification for factor model dynamics (e.g., PCA-based rate curve factors).

    Mathematical Model
    ------------------
    dF_t = κ(θ - F_t) dt + σ dW_t

    The factor is then transformed via factor loadings to drive curve/surface shifts.

    Parameters
    ----------
    mean : float
        Long-term mean θ (typically 0 for shock factors).
    kappa : float
        Mean reversion speed κ > 0.
    vol : float
        Factor volatility σ >= 0.
    loadings : dict[str, float], optional
        Mapping from tenor/point to loading. E.g., {"1Y": 0.8, "5Y": 1.0, "10Y": 0.9}.

    Notes
    -----
    Factor models are common for:
    - Yield curve: Level (parallel), Slope (twist), Curvature (butterfly)
    - Vol surface: ATM level, Skew, Smile

    Examples
    --------
    >>> # Level factor (parallel shift)
    >>> level_factor = FactorDynamicsSpec(
    ...     mean=0.0,
    ...     kappa=0.1,
    ...     vol=0.005,
    ...     loadings={"1Y": 1.0, "2Y": 1.0, "5Y": 1.0, "10Y": 1.0, "30Y": 1.0},
    ... )
    >>>
    >>> # Slope factor (2s10s steepener)
    >>> slope_factor = FactorDynamicsSpec(
    ...     mean=0.0,
    ...     kappa=0.2,
    ...     vol=0.002,
    ...     loadings={"1Y": -0.5, "2Y": -0.3, "5Y": 0.0, "10Y": 0.3, "30Y": 0.5},
    ... )
    """

    mean: float
    kappa: float
    vol: float
    loadings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("FactorDynamicsSpec.mean must be finite.")
        if self.kappa <= 0.0:
            raise ValueError("FactorDynamicsSpec.kappa must be > 0.")
        if self.vol < 0.0:
            raise ValueError("FactorDynamicsSpec.vol must be >= 0.")


# Type alias for all dynamics specifications
DynamicsSpec = Union[GBMDynamicsSpec, HestonDynamicsSpec, OUDynamicsSpec, FactorDynamicsSpec]


# =============================================================================
# Risk Factor Specification
# =============================================================================

@dataclass(frozen=True, slots=True)
class RiskFactorSpec:
    """
    Specification for a single risk factor.

    Parameters
    ----------
    market_id : MarketId
        Market identifier for this risk factor.
        Examples: MarketId("FX", "SPOT", "EURUSD"), MarketId("IR", "LEVEL", "USD").
    initial_value : float
        Starting value for the simulation.
    dynamics : DynamicsSpec
        Dynamics specification (GBM, Heston, OU, or Factor).
    name : str, optional
        Human-readable name. Defaults to market_id.key().

    Examples
    --------
    >>> from src.marketdata.core.ids import MarketId
    >>>
    >>> # FX spot factor
    >>> fx_factor = RiskFactorSpec(
    ...     market_id=MarketId("FX", "SPOT", "EURUSD"),
    ...     initial_value=1.08,
    ...     dynamics=GBMDynamicsSpec(drift=0.0, vol=0.08),
    ...     name="EUR/USD Spot",
    ... )
    >>>
    >>> # Equity index
    >>> spx_factor = RiskFactorSpec(
    ...     market_id=MarketId("EQ", "SPOT", "SPX"),
    ...     initial_value=4500.0,
    ...     dynamics=HestonDynamicsSpec(
    ...         drift=0.05, kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho_internal=-0.7
    ...     ),
    ...     name="S&P 500",
    ... )
    """

    market_id: MarketId
    initial_value: float
    dynamics: DynamicsSpec
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_value):
            raise ValueError("RiskFactorSpec.initial_value must be finite.")

    @property
    def display_name(self) -> str:
        """Human-readable name for this factor."""
        return self.name if self.name else self.market_id.key()

    @property
    def dynamics_type(self) -> str:
        """String identifier for the dynamics type."""
        if isinstance(self.dynamics, GBMDynamicsSpec):
            return "gbm"
        elif isinstance(self.dynamics, HestonDynamicsSpec):
            return "heston"
        elif isinstance(self.dynamics, OUDynamicsSpec):
            return "ou"
        elif isinstance(self.dynamics, FactorDynamicsSpec):
            return "factor"
        else:
            raise ValueError(f"Unknown dynamics type: {type(self.dynamics)}")


# =============================================================================
# Time Series Configuration
# =============================================================================

@dataclass(slots=True)
class TimeseriesConfig:
    """
    Top-level configuration for time series generation.

    Parameters
    ----------
    factors : Sequence[RiskFactorSpec]
        List of risk factors to simulate.
    correlation : np.ndarray
        Correlation matrix, shape (n_factors, n_factors).
        Must be symmetric, positive semi-definite, with ones on diagonal.
    start_date : str
        Start date in ISO format "YYYY-MM-DD".
    end_date : str
        End date in ISO format "YYYY-MM-DD".
    freq : str
        Frequency: "D" (daily), "W" (weekly), "M" (monthly), "B" (business day).
    n_scenarios : int
        Number of Monte Carlo scenarios. Default 1000.
    dt : float, optional
        Override time step in years. If None, inferred from freq.
        D=1/252, W=1/52, M=1/12, B=1/252.

    Notes
    -----
    **Correlation Handling**:
    - For GBM factors: correlation is applied directly to Brownian increments
    - For Heston factors: cross-factor correlation is applied to spot Brownians;
      internal spot-vol correlation uses HestonDynamicsSpec.rho_internal
    - For OU/Factor: correlation is applied to driving Brownians

    Examples
    --------
    >>> import numpy as np
    >>> from src.marketdata.core.ids import MarketId
    >>>
    >>> # Define two correlated FX pairs
    >>> factors = [
    ...     RiskFactorSpec(
    ...         market_id=MarketId("FX", "SPOT", "EURUSD"),
    ...         initial_value=1.08,
    ...         dynamics=GBMDynamicsSpec(drift=0.0, vol=0.08),
    ...     ),
    ...     RiskFactorSpec(
    ...         market_id=MarketId("FX", "SPOT", "GBPUSD"),
    ...         initial_value=1.26,
    ...         dynamics=GBMDynamicsSpec(drift=0.0, vol=0.09),
    ...     ),
    ... ]
    >>>
    >>> # 60% correlation between EUR/USD and GBP/USD
    >>> correlation = np.array([
    ...     [1.0, 0.6],
    ...     [0.6, 1.0],
    ... ])
    >>>
    >>> config = TimeseriesConfig(
    ...     factors=factors,
    ...     correlation=correlation,
    ...     start_date="2024-01-01",
    ...     end_date="2024-12-31",
    ...     freq="D",
    ...     n_scenarios=10000,
    ... )
    """

    factors: Sequence[RiskFactorSpec]
    correlation: np.ndarray
    start_date: str
    end_date: str
    freq: Literal["D", "W", "M", "B"] = "D"
    n_scenarios: int = 1000
    dt: Optional[float] = None

    # Computed fields (set in __post_init__)
    _cholesky: np.ndarray = field(default=None, repr=False)
    _dates: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # Convert to numpy array if needed
        self.correlation = np.asarray(self.correlation, dtype=np.float64)

        # Validate factors
        if len(self.factors) == 0:
            raise ValueError("TimeseriesConfig.factors must not be empty.")

        # Validate correlation matrix
        n_factors = len(self.factors)
        self._validate_correlation(n_factors)

        # Compute Cholesky decomposition
        self._cholesky = np.linalg.cholesky(self.correlation)

        # Generate date grid
        self._dates = self._generate_dates()

        # Validate scenarios
        if self.n_scenarios < 1:
            raise ValueError("TimeseriesConfig.n_scenarios must be >= 1.")

    def _validate_correlation(self, n_factors: int) -> None:
        """Validate correlation matrix properties."""
        corr = self.correlation

        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError("TimeseriesConfig.correlation must be a square 2D matrix.")

        if corr.shape[0] != n_factors:
            raise ValueError(
                f"TimeseriesConfig.correlation has shape {corr.shape} "
                f"but {n_factors} factors specified."
            )

        if not np.allclose(corr, corr.T, atol=1e-10):
            raise ValueError("TimeseriesConfig.correlation must be symmetric.")

        if not np.allclose(np.diag(corr), 1.0, atol=1e-10):
            raise ValueError("TimeseriesConfig.correlation diagonal must be 1.")

        if np.any(corr < -1.0 - 1e-10) or np.any(corr > 1.0 + 1e-10):
            raise ValueError("TimeseriesConfig.correlation values must be in [-1, 1].")

        # Check positive semi-definite (Cholesky will fail if not)
        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "TimeseriesConfig.correlation must be positive definite (Cholesky failed)."
            ) from e

    def _generate_dates(self) -> List[str]:
        """Generate date grid from start_date to end_date with given frequency."""
        import pandas as pd

        freq_map = {
            "D": "D",
            "W": "W-FRI",
            "M": "ME",
            "B": "B",
        }

        pd_freq = freq_map.get(self.freq, "D")
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq=pd_freq)

        return [d.strftime("%Y-%m-%d") for d in dates]

    @property
    def n_factors(self) -> int:
        """Number of risk factors."""
        return len(self.factors)

    @property
    def n_time_steps(self) -> int:
        """Number of time steps (excluding initial point)."""
        return len(self._dates) - 1

    @property
    def dates(self) -> List[str]:
        """List of simulation dates."""
        return self._dates

    @property
    def cholesky(self) -> np.ndarray:
        """Cholesky factor L where Σ = LL^T."""
        return self._cholesky

    @property
    def time_step(self) -> float:
        """Time step in years."""
        if self.dt is not None:
            return self.dt

        dt_map = {
            "D": 1.0 / 252.0,
            "W": 1.0 / 52.0,
            "M": 1.0 / 12.0,
            "B": 1.0 / 252.0,
        }
        return dt_map.get(self.freq, 1.0 / 252.0)

    @property
    def maturity(self) -> float:
        """Total time horizon in years."""
        return self.n_time_steps * self.time_step

    def factor_names(self) -> List[str]:
        """List of factor display names."""
        return [f.display_name for f in self.factors]

    def market_ids(self) -> List[MarketId]:
        """List of market IDs for all factors."""
        return [f.market_id for f in self.factors]
