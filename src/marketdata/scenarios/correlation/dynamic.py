"""
Dynamic Correlation Models.

Provides time-varying correlation structures:

1. **DCC-GARCH** (Dynamic Conditional Correlation)
   - Engle (2002) model
   - Correlation evolves based on standardized residuals

2. **Regime-Switching Correlation**
   - Different correlation matrices in different market regimes
   - Markov-switching dynamics

Mathematical Framework
----------------------
DCC-GARCH:
    Q_t = (1 - α - β)Q̄ + α(z_{t-1}z'_{t-1}) + βQ_{t-1}
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

where:
    Q̄ = unconditional covariance of standardized residuals
    z_t = standardized residuals (r_t / σ_t)
    α + β < 1 for stationarity

Typical parameters: α ≈ 0.01-0.05, β ≈ 0.90-0.98

References
----------
- Engle, R. (2002). "Dynamic Conditional Correlation: A Simple Class of
  Multivariate Generalized Autoregressive Conditional Heteroskedasticity
  Models." Journal of Business & Economic Statistics.
- Engle, R. & Sheppard, K. (2001). "Theoretical and Empirical Properties
  of Dynamic Conditional Correlation Multivariate GARCH."
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple


@dataclass(frozen=True, slots=True)
class DCCConfig:
    """
    Configuration for DCC-GARCH model.

    Parameters
    ----------
    historical_returns : np.ndarray
        Historical returns, shape (n_assets, n_observations).
    alpha : float
        News coefficient (response to shocks). Typical: 0.01-0.05.
    beta : float
        Persistence coefficient. Typical: 0.90-0.98.
    garch_omega : float
        GARCH(1,1) omega for univariate volatility.
    garch_alpha : float
        GARCH(1,1) alpha for univariate volatility.
    garch_beta : float
        GARCH(1,1) beta for univariate volatility.

    Examples
    --------
    >>> returns = np.random.randn(3, 500) * 0.01
    >>> config = DCCConfig(
    ...     historical_returns=returns,
    ...     alpha=0.02,
    ...     beta=0.95,
    ... )
    """

    historical_returns: np.ndarray
    alpha: float = 0.02
    beta: float = 0.95
    garch_omega: float = 0.00001
    garch_alpha: float = 0.08
    garch_beta: float = 0.90

    def __post_init__(self) -> None:
        if self.alpha + self.beta >= 1:
            raise ValueError("alpha + beta must be < 1 for stationarity")
        if self.garch_alpha + self.garch_beta >= 1:
            raise ValueError("garch_alpha + garch_beta must be < 1")
        if self.historical_returns.ndim != 2:
            raise ValueError("historical_returns must be 2D")


@dataclass(slots=True)
class DynamicCorrelation:
    """
    DCC-GARCH Dynamic Correlation Model.

    Estimates time-varying correlation and provides forecasting
    for scenario generation.

    Parameters
    ----------
    config : DCCConfig
        Model configuration.

    Attributes
    ----------
    volatilities : np.ndarray
        GARCH volatilities, shape (n_assets, n_observations).
    correlations : np.ndarray
        DCC correlations, shape (n_assets, n_assets, n_observations).
    standardized_residuals : np.ndarray
        Residuals divided by volatility, shape (n_assets, n_observations).

    Examples
    --------
    >>> config = DCCConfig(historical_returns=returns, alpha=0.02, beta=0.95)
    >>> dcc = DynamicCorrelation(config)
    >>>
    >>> # Get current correlation estimate
    >>> current_corr = dcc.current_correlation
    >>> print(current_corr)
    >>>
    >>> # Forecast correlation h steps ahead
    >>> forecast_corr = dcc.forecast_correlation(h=20)
    >>>
    >>> # Generate scenarios with dynamic correlation
    >>> scenarios = dcc.simulate_paths(
    ...     initial_values=np.array([100, 50, 1.10]),
    ...     n_scenarios=10000,
    ...     horizon=252,
    ... )
    """

    config: DCCConfig
    volatilities: np.ndarray = field(init=False)
    correlations: np.ndarray = field(init=False)
    standardized_residuals: np.ndarray = field(init=False)
    _Q_bar: np.ndarray = field(init=False, repr=False)
    _Q_t: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Estimate DCC model on historical data."""
        self._estimate_univariate_garch()
        self._estimate_dcc()

    def _estimate_univariate_garch(self) -> None:
        """Estimate univariate GARCH(1,1) for each asset."""
        returns = self.config.historical_returns
        n_assets, n_obs = returns.shape

        omega = self.config.garch_omega
        alpha = self.config.garch_alpha
        beta = self.config.garch_beta

        self.volatilities = np.empty((n_assets, n_obs), dtype=np.float64)
        self.standardized_residuals = np.empty_like(returns)

        for a in range(n_assets):
            r = returns[a, :]
            var = np.var(r)  # Initial variance

            for t in range(n_obs):
                self.volatilities[a, t] = np.sqrt(var)
                self.standardized_residuals[a, t] = r[t] / np.sqrt(var)
                var = omega + alpha * r[t] ** 2 + beta * var

    def _estimate_dcc(self) -> None:
        """Estimate DCC correlation dynamics."""
        z = self.standardized_residuals
        n_assets, n_obs = z.shape
        alpha = self.config.alpha
        beta = self.config.beta

        # Unconditional covariance of standardized residuals
        self._Q_bar = np.cov(z)

        # Initialize Q_t with Q_bar
        Q_t = self._Q_bar.copy()

        # Store all correlations
        self.correlations = np.empty((n_assets, n_assets, n_obs), dtype=np.float64)

        for t in range(n_obs):
            # Convert Q to correlation R
            diag_Q = np.sqrt(np.diag(Q_t))
            R_t = Q_t / np.outer(diag_Q, diag_Q)
            self.correlations[:, :, t] = R_t

            # Update Q for next period
            z_t = z[:, t:t+1]  # Column vector
            Q_t = (1 - alpha - beta) * self._Q_bar + alpha * (z_t @ z_t.T) + beta * Q_t

        # Store final Q_t for forecasting
        self._Q_t = Q_t

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.config.historical_returns.shape[0]

    @property
    def current_correlation(self) -> np.ndarray:
        """
        Current (most recent) correlation estimate.

        Returns
        -------
        np.ndarray
            Correlation matrix, shape (n_assets, n_assets).
        """
        return self.correlations[:, :, -1]

    @property
    def current_volatility(self) -> np.ndarray:
        """
        Current (most recent) volatility estimate (daily).

        Returns
        -------
        np.ndarray
            Volatilities, shape (n_assets,).
        """
        return self.volatilities[:, -1]

    def forecast_correlation(self, h: int = 1) -> np.ndarray:
        """
        Forecast correlation h steps ahead.

        Uses the DCC recursion with unconditional standardized residuals.

        Parameters
        ----------
        h : int
            Forecast horizon.

        Returns
        -------
        np.ndarray
            Forecasted correlation, shape (n_assets, n_assets).
        """
        alpha = self.config.alpha
        beta = self.config.beta

        # Mean-reverting to Q_bar
        # Q_{t+h} = Q_bar + (alpha + beta)^h * (Q_t - Q_bar)
        persistence = alpha + beta
        Q_h = self._Q_bar + (persistence ** h) * (self._Q_t - self._Q_bar)

        # Convert to correlation
        diag_Q = np.sqrt(np.diag(Q_h))
        R_h = Q_h / np.outer(diag_Q, diag_Q)

        return R_h

    def forecast_volatility(self, h: int = 1) -> np.ndarray:
        """
        Forecast volatility h steps ahead (annualized).

        Parameters
        ----------
        h : int
            Forecast horizon.

        Returns
        -------
        np.ndarray
            Forecasted annualized volatilities, shape (n_assets,).
        """
        omega = self.config.garch_omega
        alpha = self.config.garch_alpha
        beta = self.config.garch_beta

        # Unconditional variance
        var_unc = omega / (1 - alpha - beta)

        # Mean-reverting forecast
        current_var = self.volatilities[:, -1] ** 2
        persistence = alpha + beta
        forecast_var = var_unc + (persistence ** h) * (current_var - var_unc)

        # Annualize (assuming daily data)
        return np.sqrt(forecast_var * 252)

    def simulate_paths(
        self,
        initial_values: np.ndarray,
        n_scenarios: int,
        horizon: int,
        seed: Optional[int] = None,
        use_forecast_corr: bool = True,
    ) -> np.ndarray:
        """
        Simulate price paths with dynamic correlation.

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
        use_forecast_corr : bool
            If True, use multi-step forecasted correlation.
            If False, use current correlation throughout.

        Returns
        -------
        np.ndarray
            Price paths, shape (n_assets, horizon + 1, n_scenarios).
        """
        rng = np.random.default_rng(seed)

        n_assets = self.n_assets
        paths = np.empty((n_assets, horizon + 1, n_scenarios), dtype=np.float64)
        paths[:, 0, :] = initial_values[:, np.newaxis]

        # Get initial correlation and volatility
        current_corr = self.current_correlation
        current_vol = self.current_volatility

        # DCC parameters for simulation
        alpha = self.config.alpha
        beta = self.config.beta
        Q_t = self._Q_t.copy()

        for t in range(horizon):
            # Get correlation for this time step
            if use_forecast_corr:
                # Forecast correlation
                diag_Q = np.sqrt(np.diag(Q_t))
                R_t = Q_t / np.outer(diag_Q, diag_Q)
            else:
                R_t = current_corr

            # Cholesky for correlated normals
            chol = np.linalg.cholesky(R_t)

            # Generate correlated shocks
            z_indep = rng.standard_normal((n_scenarios, n_assets))
            z_corr = z_indep @ chol.T

            # Scale by volatility (using current vol for simplicity)
            daily_vol = current_vol
            returns = z_corr * daily_vol

            # Update paths
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(returns.T)

            # Update Q_t for next period (using average z)
            if use_forecast_corr:
                z_mean = z_corr.mean(axis=0)[:, np.newaxis]
                Q_t = (1 - alpha - beta) * self._Q_bar + alpha * (z_mean @ z_mean.T) + beta * Q_t

        return paths

    def correlation_half_life(self) -> float:
        """
        Compute correlation half-life (mean reversion speed).

        Returns
        -------
        float
            Half-life in time periods.
        """
        persistence = self.config.alpha + self.config.beta
        if persistence >= 1:
            return float("inf")
        return -np.log(2) / np.log(persistence)
