"""
Monte Carlo pricer for Cliquet options under GBM dynamics.

Prices cliquet options by simulating GBM paths and computing
the path-dependent payoff with capped/floored returns.

Example:
    from src.pricers.equity.cliquet_gbm_mc import EquityCliquetGbmMcPricer
    from src.instruments.equity.options.cliquet import EquityCliquetOption
    
    pricer = EquityCliquetGbmMcPricer(n_paths=100_000, seed=42)
    
    cliquet = EquityCliquetOption(
        underlying_id="SPY",
        notional=1_000_000,
        start_date=date(2024, 1, 1),
        end_date=date(2025, 1, 1),
        reset_dates=[...],
        local_cap=0.03,
        local_floor=-0.01,
        global_floor=0.0,
    )
    
    result = pricer.price(cliquet, market_data)
    print(f"Price: {result.price:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.instruments.equity.options.cliquet import EquityCliquetOption, FxCliquetOption
from src.models.payoffs.cliquet import CliquetPayoff


# =============================================================================
# Pricer Configuration
# =============================================================================


@dataclass
class CliquetMcConfig:
    """Configuration for cliquet Monte Carlo pricer."""
    
    n_paths: int = 100_000
    seed: Optional[int] = None
    antithetic: bool = True
    
    # Greeks computation
    compute_greeks: bool = True
    bump_size_delta: float = 0.01  # 1% bump for delta
    bump_size_vega: float = 0.01  # 1 vol point bump for vega


# =============================================================================
# Pricing Result
# =============================================================================


@dataclass
class CliquetPricingResult:
    """Result from cliquet pricing."""
    
    price: float
    standard_error: float
    
    # Greeks (optional)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    theta: Optional[float] = None
    
    # Diagnostics
    n_paths: int = 0
    elapsed_seconds: float = 0.0
    
    # Additional info
    expected_global_return: float = 0.0
    prob_positive_return: float = 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "price": self.price,
            "standard_error": self.standard_error,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "rho": self.rho,
            "theta": self.theta,
            "expected_global_return": self.expected_global_return,
            "prob_positive_return": self.prob_positive_return,
        }


# =============================================================================
# Market Data Interface
# =============================================================================


@dataclass
class CliquetMarketData:
    """Market data for cliquet pricing."""
    
    spot: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    valuation_date: Optional[date] = None
    
    @classmethod
    def from_provider(
        cls,
        provider: Any,
        underlying_id: str,
        valuation_date: Optional[date] = None,
    ) -> "CliquetMarketData":
        """
        Create market data from a market data provider.
        
        Parameters
        ----------
        provider : MarketDataProvider
            Market data provider (e.g., StaticMarketDataProvider).
        underlying_id : str
            Underlying asset identifier.
        valuation_date : date, optional
            Valuation date.
        
        Returns
        -------
        CliquetMarketData
            Populated market data.
        """
        # This would integrate with the actual market data provider
        # For now, return a simple extraction
        return cls(
            spot=100.0,
            volatility=0.2,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            valuation_date=valuation_date,
        )


# =============================================================================
# Cliquet Pricer
# =============================================================================


class EquityCliquetGbmMcPricer:
    """
    Monte Carlo pricer for Equity Cliquet options under GBM.
    
    Simulates GBM paths at reset dates and computes the cliquet payoff
    with local and global caps/floors.
    
    Features:
    - Antithetic variance reduction
    - Greek computation via bump-and-reval
    - Detailed diagnostics
    
    Example:
        pricer = EquityCliquetGbmMcPricer(n_paths=100_000)
        
        cliquet = EquityCliquetOption(
            underlying_id="SPY",
            notional=1_000_000,
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            reset_dates=[date(2024, i, 1) for i in range(2, 13)],
            local_cap=0.03,
            local_floor=-0.01,
            global_floor=0.0,
        )
        
        market_data = CliquetMarketData(
            spot=100,
            volatility=0.2,
            risk_free_rate=0.05,
        )
        
        result = pricer.price(cliquet, market_data)
    """
    
    def __init__(
        self,
        n_paths: int = 100_000,
        seed: Optional[int] = None,
        antithetic: bool = True,
        compute_greeks: bool = True,
    ) -> None:
        """
        Initialize cliquet pricer.
        
        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            Random seed for reproducibility.
        antithetic : bool
            Use antithetic variance reduction.
        compute_greeks : bool
            Compute Greeks via bump-and-reval.
        """
        self.n_paths = n_paths
        self.seed = seed
        self.antithetic = antithetic
        self.compute_greeks = compute_greeks
        self._rng = np.random.default_rng(seed)
    
    def price(
        self,
        instrument: EquityCliquetOption,
        market_data: CliquetMarketData,
    ) -> CliquetPricingResult:
        """
        Price a cliquet option.
        
        Parameters
        ----------
        instrument : EquityCliquetOption
            Cliquet instrument to price.
        market_data : CliquetMarketData
            Market data for pricing.
        
        Returns
        -------
        CliquetPricingResult
            Pricing result with price, error, and Greeks.
        """
        import time
        start_time = time.time()
        
        # Get time fractions to each reset date
        valuation_date = market_data.valuation_date or instrument.start_date
        time_fractions = self._get_time_fractions(
            valuation_date,
            instrument.start_date,
            instrument.reset_dates,
        )
        
        # Total time to maturity
        T = (instrument.end_date - valuation_date).days / 365.0
        if T <= 0:
            # Expired
            return CliquetPricingResult(
                price=0.0,
                standard_error=0.0,
                n_paths=0,
            )
        
        # Create payoff
        payoff = CliquetPayoff(
            local_cap=instrument.local_cap,
            local_floor=instrument.local_floor,
            global_cap=instrument.global_cap,
            global_floor=instrument.global_floor,
            participation=instrument.participation,
        )
        
        # Simulate paths and compute payoffs
        paths = self._simulate_paths(
            spot=market_data.spot,
            volatility=market_data.volatility,
            rate=market_data.risk_free_rate,
            dividend=market_data.dividend_yield,
            time_fractions=time_fractions,
        )
        
        # Compute payoffs
        payoffs = payoff.terminal_from_paths(paths) * instrument.notional
        
        # Discount
        discount = np.exp(-market_data.risk_free_rate * T)
        discounted_payoffs = payoffs * discount
        
        # Statistics
        price = float(np.mean(discounted_payoffs))
        std_error = float(np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs)))
        
        # Compute diagnostics
        global_returns = self._compute_global_returns(paths, payoff)
        expected_return = float(np.mean(global_returns))
        prob_positive = float(np.mean(global_returns > 0))
        
        # Compute Greeks
        delta, gamma, vega, rho, theta = None, None, None, None, None
        if self.compute_greeks:
            delta, gamma = self._compute_delta_gamma(
                instrument, market_data, payoff, time_fractions, T
            )
            vega = self._compute_vega(
                instrument, market_data, payoff, time_fractions, T
            )
            rho = self._compute_rho(
                instrument, market_data, payoff, time_fractions, T
            )
        
        elapsed = time.time() - start_time
        
        return CliquetPricingResult(
            price=price,
            standard_error=std_error,
            delta=delta,
            gamma=gamma,
            vega=vega,
            rho=rho,
            theta=theta,
            n_paths=self.n_paths,
            elapsed_seconds=elapsed,
            expected_global_return=expected_return,
            prob_positive_return=prob_positive,
        )
    
    def _get_time_fractions(
        self,
        valuation_date: date,
        start_date: date,
        reset_dates: List[date],
    ) -> np.ndarray:
        """Get time fractions from valuation date to each observation."""
        all_dates = [start_date] + reset_dates
        
        # Filter dates >= valuation_date
        future_dates = [d for d in all_dates if d >= valuation_date]
        
        # Convert to year fractions
        fractions = [(d - valuation_date).days / 365.0 for d in future_dates]
        
        return np.array(fractions)
    
    def _simulate_paths(
        self,
        spot: float,
        volatility: float,
        rate: float,
        dividend: float,
        time_fractions: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate GBM paths at observation dates.
        
        Parameters
        ----------
        spot : float
            Current spot price.
        volatility : float
            Annualized volatility.
        rate : float
            Risk-free rate.
        dividend : float
            Dividend yield.
        time_fractions : ndarray
            Time fractions to each observation.
        
        Returns
        -------
        ndarray
            Simulated paths of shape (n_paths, n_observations).
        """
        n_obs = len(time_fractions)
        
        if n_obs == 0:
            return np.full((self.n_paths, 1), spot)
        
        # Time deltas between observations
        if n_obs == 1:
            dt = np.array([time_fractions[0]])
        else:
            dt = np.diff(np.concatenate([[0], time_fractions]))
        
        # Generate random increments
        n_paths = self.n_paths
        if self.antithetic:
            n_paths = n_paths // 2
        
        # Standard normal increments
        Z = self._rng.standard_normal((n_paths, n_obs))
        
        if self.antithetic:
            Z = np.vstack([Z, -Z])
        
        # Build paths using log returns
        drift = (rate - dividend - 0.5 * volatility**2)
        
        paths = np.zeros((len(Z), n_obs + 1))
        paths[:, 0] = spot
        
        for i, (dti, Zi) in enumerate(zip(dt, Z.T)):
            if dti > 0:
                log_return = drift * dti + volatility * np.sqrt(dti) * Zi
                paths[:, i + 1] = paths[:, i] * np.exp(log_return)
            else:
                paths[:, i + 1] = paths[:, i]
        
        return paths
    
    def _compute_global_returns(
        self,
        paths: np.ndarray,
        payoff: CliquetPayoff,
    ) -> np.ndarray:
        """Compute capped/floored global returns from paths."""
        n_paths, n_steps = paths.shape
        
        if n_steps < 2:
            return np.zeros(n_paths)
        
        # Period returns
        period_returns = paths[:, 1:] / paths[:, :-1] - 1
        
        # Local caps/floors
        capped_returns = np.clip(
            period_returns,
            payoff.local_floor,
            payoff.local_cap,
        )
        
        # Global return
        global_returns = np.sum(capped_returns, axis=1)
        
        # Apply global bounds
        global_returns = np.maximum(global_returns, payoff.global_floor)
        if payoff.global_cap is not None:
            global_returns = np.minimum(global_returns, payoff.global_cap)
        
        return global_returns
    
    def _compute_delta_gamma(
        self,
        instrument: EquityCliquetOption,
        market_data: CliquetMarketData,
        payoff: CliquetPayoff,
        time_fractions: np.ndarray,
        T: float,
    ) -> Tuple[float, float]:
        """Compute delta and gamma via bump-and-reval."""
        bump = market_data.spot * 0.01  # 1% bump
        
        # Base price
        paths_base = self._simulate_paths(
            market_data.spot, market_data.volatility,
            market_data.risk_free_rate, market_data.dividend_yield,
            time_fractions,
        )
        payoffs_base = payoff.terminal_from_paths(paths_base) * instrument.notional
        price_base = np.mean(payoffs_base) * np.exp(-market_data.risk_free_rate * T)
        
        # Up price
        paths_up = self._simulate_paths(
            market_data.spot + bump, market_data.volatility,
            market_data.risk_free_rate, market_data.dividend_yield,
            time_fractions,
        )
        payoffs_up = payoff.terminal_from_paths(paths_up) * instrument.notional
        price_up = np.mean(payoffs_up) * np.exp(-market_data.risk_free_rate * T)
        
        # Down price
        paths_down = self._simulate_paths(
            market_data.spot - bump, market_data.volatility,
            market_data.risk_free_rate, market_data.dividend_yield,
            time_fractions,
        )
        payoffs_down = payoff.terminal_from_paths(paths_down) * instrument.notional
        price_down = np.mean(payoffs_down) * np.exp(-market_data.risk_free_rate * T)
        
        # Delta and gamma
        delta = (price_up - price_down) / (2 * bump)
        gamma = (price_up - 2 * price_base + price_down) / (bump ** 2)
        
        return float(delta), float(gamma)
    
    def _compute_vega(
        self,
        instrument: EquityCliquetOption,
        market_data: CliquetMarketData,
        payoff: CliquetPayoff,
        time_fractions: np.ndarray,
        T: float,
    ) -> float:
        """Compute vega via bump-and-reval."""
        vol_bump = 0.01  # 1 vol point
        
        # Up vol
        paths_up = self._simulate_paths(
            market_data.spot, market_data.volatility + vol_bump,
            market_data.risk_free_rate, market_data.dividend_yield,
            time_fractions,
        )
        payoffs_up = payoff.terminal_from_paths(paths_up) * instrument.notional
        price_up = np.mean(payoffs_up) * np.exp(-market_data.risk_free_rate * T)
        
        # Down vol
        paths_down = self._simulate_paths(
            market_data.spot, market_data.volatility - vol_bump,
            market_data.risk_free_rate, market_data.dividend_yield,
            time_fractions,
        )
        payoffs_down = payoff.terminal_from_paths(paths_down) * instrument.notional
        price_down = np.mean(payoffs_down) * np.exp(-market_data.risk_free_rate * T)
        
        vega = (price_up - price_down) / (2 * vol_bump)
        
        return float(vega)
    
    def _compute_rho(
        self,
        instrument: EquityCliquetOption,
        market_data: CliquetMarketData,
        payoff: CliquetPayoff,
        time_fractions: np.ndarray,
        T: float,
    ) -> float:
        """Compute rho via bump-and-reval."""
        rate_bump = 0.0001  # 1 bp
        
        # Up rate
        paths_up = self._simulate_paths(
            market_data.spot, market_data.volatility,
            market_data.risk_free_rate + rate_bump, market_data.dividend_yield,
            time_fractions,
        )
        payoffs_up = payoff.terminal_from_paths(paths_up) * instrument.notional
        price_up = np.mean(payoffs_up) * np.exp(-(market_data.risk_free_rate + rate_bump) * T)
        
        # Down rate
        paths_down = self._simulate_paths(
            market_data.spot, market_data.volatility,
            market_data.risk_free_rate - rate_bump, market_data.dividend_yield,
            time_fractions,
        )
        payoffs_down = payoff.terminal_from_paths(paths_down) * instrument.notional
        price_down = np.mean(payoffs_down) * np.exp(-(market_data.risk_free_rate - rate_bump) * T)
        
        # Rho per 1% (100 bps)
        rho = (price_up - price_down) / (2 * rate_bump) * 0.01
        
        return float(rho)


class FxCliquetGbmMcPricer(EquityCliquetGbmMcPricer):
    """
    Monte Carlo pricer for FX Cliquet options under GBM.
    
    Extends the equity pricer for FX-specific handling.
    """
    
    def price(
        self,
        instrument: FxCliquetOption,
        market_data: CliquetMarketData,
    ) -> CliquetPricingResult:
        """
        Price an FX cliquet option.
        
        Parameters
        ----------
        instrument : FxCliquetOption
            FX cliquet to price.
        market_data : CliquetMarketData
            Market data with FX spot and vol.
        
        Returns
        -------
        CliquetPricingResult
            Pricing result.
        """
        # Convert FX cliquet to equity-like structure for pricing
        equity_like = EquityCliquetOption(
            underlying_id=instrument.pair_id,
            notional=instrument.notional,
            start_date=instrument.start_date,
            end_date=instrument.end_date,
            reset_dates=instrument.reset_dates,
            local_cap=instrument.local_cap,
            local_floor=instrument.local_floor,
            global_cap=instrument.global_cap,
            global_floor=instrument.global_floor,
            participation=instrument.participation,
        )
        
        return super().price(equity_like, market_data)


__all__ = [
    "EquityCliquetGbmMcPricer",
    "FxCliquetGbmMcPricer",
    "CliquetPricingResult",
    "CliquetMarketData",
    "CliquetMcConfig",
]
