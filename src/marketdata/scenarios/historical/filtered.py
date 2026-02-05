"""
Filtered Historical Simulation.

Volatility-adjusts historical returns to reflect current market conditions.
This addresses a key limitation of standard historical simulation: it treats
all historical periods as equally likely, ignoring current volatility regime.

Mathematical Framework
----------------------
Standard historical simulation uses raw returns:
    r̃_t = r_{hist,i}  (randomly selected)

Filtered historical simulation (Hull-White, 1998):
    1. Estimate historical volatility at each time: σ_{hist,t}
    2. Standardize historical returns: z_t = r_t / σ_{hist,t}
    3. Re-scale by current volatility: r̃_t = z_t × σ_current

This ensures VaR estimates reflect current volatility levels while
preserving the empirical distribution of standardized returns.

GARCH Extension
---------------
For more accurate volatility dynamics, we can use GARCH filtering:
    r_t = σ_t × z_t
    σ²_t = ω + α × r²_{t-1} + β × σ²_{t-1}

The standardized residuals z_t are then resampled.

References
----------
- Hull, J. & White, A. (1998). "Incorporating Volatility Updating into
  the Historical Simulation Method for Value at Risk."
  Journal of Risk.
- Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). "VaR without
  correlations for portfolios of derivative securities."
  Journal of Futures Markets.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple


@dataclass(frozen=True, slots=True)
class FilteredConfig:
    """
    Configuration for Filtered Historical Simulation.

    Parameters
    ----------
    historical_returns : np.ndarray
        Historical log-returns, shape (n_assets, n_observations).
    current_volatility : np.ndarray
        Current annualized volatility for each asset, shape (n_assets,).
    volatility_method : str
        Method to estimate historical volatility:
        - "rolling": Simple rolling standard deviation
        - "ewma": Exponentially weighted moving average
        - "garch": GARCH(1,1) filtering
    window : int
        Lookback window for rolling/ewma methods.
    ewma_lambda : float
        Decay factor for EWMA (typical: 0.94 for daily data).
    garch_params : tuple, optional
        GARCH(1,1) parameters (omega, alpha, beta). If None, estimated from data.
    annualization_factor : float
        Factor to annualize volatility (sqrt(252) for daily data).

    Examples
    --------
    >>> returns = np.random.randn(5, 1000) * 0.01  # 5 assets, 1000 days
    >>> current_vol = np.array([0.15, 0.20, 0.25, 0.18, 0.22])  # Annualized
    >>>
    >>> config = FilteredConfig(
    ...     historical_returns=returns,
    ...     current_volatility=current_vol,
    ...     volatility_method="ewma",
    ...     ewma_lambda=0.94,
    ... )
    """

    historical_returns: np.ndarray
    current_volatility: np.ndarray
    volatility_method: Literal["rolling", "ewma", "garch"] = "ewma"
    window: int = 60
    ewma_lambda: float = 0.94
    garch_params: Optional[Tuple[float, float, float]] = None
    annualization_factor: float = np.sqrt(252)

    def __post_init__(self) -> None:
        if self.historical_returns.ndim != 2:
            raise ValueError(
                "historical_returns must be 2D: (n_assets, n_observations)"
            )
        if self.current_volatility.ndim != 1:
            raise ValueError("current_volatility must be 1D: (n_assets,)")
        if len(self.current_volatility) != self.historical_returns.shape[0]:
            raise ValueError(
                "current_volatility length must match number of assets"
            )
        if self.volatility_method not in ("rolling", "ewma", "garch"):
            raise ValueError(
                f"volatility_method must be 'rolling', 'ewma', or 'garch'"
            )

    @property
    def n_assets(self) -> int:
        return self.historical_returns.shape[0]

    @property
    def n_observations(self) -> int:
        return self.historical_returns.shape[1]


@dataclass(slots=True)
class FilteredHistorical:
    """
    Filtered Historical Simulation.

    Standardizes historical returns by their contemporaneous volatility,
    then rescales by current volatility.

    Parameters
    ----------
    config : FilteredConfig
        Configuration parameters.

    Attributes
    ----------
    standardized_returns : np.ndarray
        Historical returns divided by historical volatility.
    historical_volatility : np.ndarray
        Estimated historical volatility at each time.

    Examples
    --------
    >>> returns = np.random.randn(3, 500) * 0.01  # 3 assets
    >>> current_vol = np.array([0.15, 0.20, 0.18])
    >>>
    >>> config = FilteredConfig(
    ...     historical_returns=returns,
    ...     current_volatility=current_vol,
    ...     volatility_method="garch",
    ... )
    >>> filtered = FilteredHistorical(config)
    >>>
    >>> # Generate scenarios adjusted for current volatility
    >>> scenarios = filtered.sample(n_scenarios=10000, horizon=252, seed=42)
    """

    config: FilteredConfig
    standardized_returns: np.ndarray = field(init=False)
    historical_volatility: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Compute standardized returns on initialization."""
        self._estimate_volatility()
        self._standardize_returns()

    def _estimate_volatility(self) -> None:
        """Estimate historical volatility using configured method."""
        returns = self.config.historical_returns
        n_assets, n_obs = returns.shape

        if self.config.volatility_method == "rolling":
            self.historical_volatility = self._rolling_volatility(returns)
        elif self.config.volatility_method == "ewma":
            self.historical_volatility = self._ewma_volatility(returns)
        else:  # garch
            self.historical_volatility = self._garch_volatility(returns)

    def _rolling_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling window volatility."""
        n_assets, n_obs = returns.shape
        window = self.config.window
        ann_factor = self.config.annualization_factor

        vol = np.full((n_assets, n_obs), np.nan, dtype=np.float64)

        for a in range(n_assets):
            for t in range(window - 1, n_obs):
                vol[a, t] = np.std(returns[a, t - window + 1 : t + 1], ddof=1)

        # Fill early observations with first valid estimate
        for a in range(n_assets):
            first_valid = vol[a, window - 1]
            vol[a, : window - 1] = first_valid

        return vol * ann_factor

    def _ewma_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Compute EWMA (RiskMetrics-style) volatility."""
        n_assets, n_obs = returns.shape
        lam = self.config.ewma_lambda
        ann_factor = self.config.annualization_factor

        # Initialize with sample variance of first window
        window = min(self.config.window, n_obs)
        var = np.var(returns[:, :window], axis=1, ddof=1)

        vol = np.empty((n_assets, n_obs), dtype=np.float64)
        vol[:, 0] = np.sqrt(var)

        # EWMA recursion: σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}
        for t in range(1, n_obs):
            var = lam * var + (1 - lam) * returns[:, t - 1] ** 2
            vol[:, t] = np.sqrt(var)

        return vol * ann_factor

    def _garch_volatility(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute GARCH(1,1) volatility.

        Model: σ²_t = ω + α r²_{t-1} + β σ²_{t-1}
        """
        n_assets, n_obs = returns.shape
        ann_factor = self.config.annualization_factor

        vol = np.empty((n_assets, n_obs), dtype=np.float64)

        for a in range(n_assets):
            r = returns[a, :]

            if self.config.garch_params is not None:
                omega, alpha, beta = self.config.garch_params
            else:
                # Simple moment estimation
                omega, alpha, beta = self._estimate_garch_params(r)

            # Initial variance
            var = np.var(r, ddof=1)
            vol[a, 0] = np.sqrt(var)

            for t in range(1, n_obs):
                var = omega + alpha * r[t - 1] ** 2 + beta * var
                vol[a, t] = np.sqrt(var)

        return vol * ann_factor

    def _estimate_garch_params(
        self, returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Simple GARCH(1,1) parameter estimation using moment matching.

        For production, consider maximum likelihood or use arch package.
        """
        # Typical starting values / simple estimation
        var = np.var(returns, ddof=1)

        # Target unconditional variance = omega / (1 - alpha - beta)
        # Typical values: alpha ~ 0.05-0.15, beta ~ 0.80-0.90
        alpha = 0.08
        beta = 0.88
        omega = var * (1 - alpha - beta)

        return omega, alpha, beta

    def _standardize_returns(self) -> None:
        """Divide historical returns by their contemporaneous volatility."""
        returns = self.config.historical_returns
        # Convert annualized vol to daily for standardization
        daily_vol = self.historical_volatility / self.config.annualization_factor

        # Avoid division by zero
        daily_vol = np.maximum(daily_vol, 1e-10)

        self.standardized_returns = returns / daily_vol

    def sample(
        self,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate filtered historical scenarios.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios.
        horizon : int
            Length of each scenario.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Filtered returns, shape (n_assets, horizon, n_scenarios).
            Returns are scaled to current volatility levels.
        """
        self._rng = np.random.default_rng(seed)

        n_assets = self.config.n_assets
        n_obs = self.config.n_observations
        current_vol = self.config.current_volatility

        # Convert current annual vol to daily
        current_daily_vol = current_vol / self.config.annualization_factor

        scenarios = np.empty((n_assets, horizon, n_scenarios), dtype=np.float64)

        for s in range(n_scenarios):
            # Randomly select time indices
            indices = self._rng.integers(0, n_obs, size=horizon)

            for t, idx in enumerate(indices):
                # Get standardized returns and rescale by current vol
                scenarios[:, t, s] = (
                    self.standardized_returns[:, idx] * current_daily_vol
                )

        return scenarios

    def sample_block(
        self,
        n_scenarios: int,
        horizon: int,
        block_length: int = 20,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate scenarios using block resampling of standardized returns.

        Combines filtered historical simulation with block bootstrap
        to preserve autocorrelation structure.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios.
        horizon : int
            Length of each scenario.
        block_length : int
            Block length for bootstrap.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Filtered returns, shape (n_assets, horizon, n_scenarios).
        """
        self._rng = np.random.default_rng(seed)

        n_assets = self.config.n_assets
        n_obs = self.config.n_observations
        current_vol = self.config.current_volatility
        current_daily_vol = current_vol / self.config.annualization_factor

        scenarios = np.empty((n_assets, horizon, n_scenarios), dtype=np.float64)

        for s in range(n_scenarios):
            t = 0
            while t < horizon:
                # Start new block
                start = self._rng.integers(0, n_obs - block_length + 1)

                for b in range(block_length):
                    if t >= horizon:
                        break
                    idx = start + b
                    scenarios[:, t, s] = (
                        self.standardized_returns[:, idx] * current_daily_vol
                    )
                    t += 1

        return scenarios

    def sample_paths(
        self,
        initial_values: np.ndarray,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate price paths from filtered returns.

        Parameters
        ----------
        initial_values : np.ndarray
            Initial prices, shape (n_assets,).
        n_scenarios : int
            Number of scenarios.
        horizon : int
            Path length.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Price paths, shape (n_assets, horizon + 1, n_scenarios).
        """
        returns = self.sample(n_scenarios, horizon, seed)

        n_assets = self.config.n_assets
        paths = np.empty((n_assets, horizon + 1, n_scenarios), dtype=np.float64)
        paths[:, 0, :] = initial_values[:, np.newaxis]

        for t in range(horizon):
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(returns[:, t, :])

        return paths
