"""
Historical Data Adapter for deep hedging environments.

Prepares historical market data for use with hedging environments,
enabling model-agnostic hedging on real data.

Example:
    from src.deep_hedging.adapters import HistoricalDataAdapter
    
    adapter = HistoricalDataAdapter()
    
    # Load from price series
    market_data = adapter.from_prices(
        prices=historical_prices,
        dates=dates,
        volatility_window=20,
    )
    
    # Use in hedging environment
    env = HistoricalHedgingEnv(market_data=market_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Historical Market Data
# =============================================================================


@dataclass
class HistoricalMarketData:
    """
    Historical market data prepared for hedging environments.
    
    Contains aligned price, volatility, and rate data for backtesting
    or training hedging agents on real data.
    
    Attributes
    ----------
    prices : ndarray
        Price series.
    volatilities : ndarray
        Volatility series (same length as prices).
    rates : ndarray
        Risk-free rate series.
    dates : list of date
        Dates corresponding to data.
    returns : ndarray
        Log returns computed from prices.
    realized_vol : ndarray
        Rolling realized volatility.
    """
    
    prices: np.ndarray
    volatilities: np.ndarray
    rates: np.ndarray
    dates: List[date]
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    realized_vol: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @property
    def n_steps(self) -> int:
        """Number of data points."""
        return len(self.prices)
    
    @property
    def time_span_years(self) -> float:
        """Total time span in years."""
        if len(self.dates) < 2:
            return 0.0
        return (self.dates[-1] - self.dates[0]).days / 365.0
    
    def get_window(
        self,
        start_idx: int,
        window_size: int,
    ) -> "HistoricalMarketData":
        """
        Get a window of data.
        
        Parameters
        ----------
        start_idx : int
            Starting index.
        window_size : int
            Window size in data points.
        
        Returns
        -------
        HistoricalMarketData
            Windowed data.
        """
        end_idx = start_idx + window_size
        
        return HistoricalMarketData(
            prices=self.prices[start_idx:end_idx].copy(),
            volatilities=self.volatilities[start_idx:end_idx].copy(),
            rates=self.rates[start_idx:end_idx].copy(),
            dates=self.dates[start_idx:end_idx],
            returns=self.returns[start_idx:end_idx].copy() if len(self.returns) > 0 else np.array([]),
            realized_vol=self.realized_vol[start_idx:end_idx].copy() if len(self.realized_vol) > 0 else np.array([]),
        )
    
    def get_episode_data(
        self,
        start_date: date,
        end_date: date,
    ) -> "HistoricalMarketData":
        """
        Get data for a specific date range.
        
        Parameters
        ----------
        start_date : date
            Start date.
        end_date : date
            End date.
        
        Returns
        -------
        HistoricalMarketData
            Filtered data.
        """
        mask = np.array([
            start_date <= d <= end_date
            for d in self.dates
        ])
        
        return HistoricalMarketData(
            prices=self.prices[mask].copy(),
            volatilities=self.volatilities[mask].copy(),
            rates=self.rates[mask].copy(),
            dates=[d for d, m in zip(self.dates, mask) if m],
            returns=self.returns[mask].copy() if len(self.returns) > 0 else np.array([]),
            realized_vol=self.realized_vol[mask].copy() if len(self.realized_vol) > 0 else np.array([]),
        )
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "n_steps": self.n_steps,
            "time_span_years": self.time_span_years,
            "start_date": self.dates[0] if self.dates else None,
            "end_date": self.dates[-1] if self.dates else None,
            "mean_price": float(np.mean(self.prices)),
            "mean_vol": float(np.mean(self.volatilities)),
            "mean_rate": float(np.mean(self.rates)),
            "annualized_return": float(np.mean(self.returns) * 252) if len(self.returns) > 0 else 0.0,
            "annualized_vol": float(np.std(self.returns) * np.sqrt(252)) if len(self.returns) > 0 else 0.0,
        }


# =============================================================================
# Historical Data Adapter
# =============================================================================


class HistoricalDataAdapter:
    """
    Adapter for preparing historical data for hedging environments.
    
    Handles:
    - Loading data from various sources
    - Computing derived quantities (returns, realized vol)
    - Aligning and validating data
    - Creating episodes for training/evaluation
    
    Example:
        adapter = HistoricalDataAdapter()
        
        # From price series
        market_data = adapter.from_prices(
            prices=historical_prices,
            dates=dates,
            volatility_window=20,
        )
        
        # Get random episodes for training
        episodes = adapter.create_episodes(
            market_data=market_data,
            episode_length=50,
            n_episodes=100,
        )
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        rate_default: float = 0.05,
    ) -> None:
        """
        Initialize adapter.
        
        Parameters
        ----------
        volatility_window : int
            Window for computing realized volatility.
        rate_default : float
            Default risk-free rate if not provided.
        """
        self.volatility_window = volatility_window
        self.rate_default = rate_default
    
    def from_prices(
        self,
        prices: np.ndarray,
        dates: Optional[Sequence[date]] = None,
        volatilities: Optional[np.ndarray] = None,
        rates: Optional[np.ndarray] = None,
        volatility_window: Optional[int] = None,
    ) -> HistoricalMarketData:
        """
        Create market data from price series.
        
        Parameters
        ----------
        prices : ndarray
            Price series.
        dates : sequence of date, optional
            Dates. If None, generates daily dates from today.
        volatilities : ndarray, optional
            Volatility series. If None, computes realized vol.
        rates : ndarray, optional
            Rate series. If None, uses default rate.
        volatility_window : int, optional
            Window for realized vol. Overrides instance default.
        
        Returns
        -------
        HistoricalMarketData
            Prepared market data.
        """
        prices = np.asarray(prices)
        n_steps = len(prices)
        
        # Generate dates if not provided
        if dates is None:
            base_date = date.today() - timedelta(days=n_steps - 1)
            dates = [base_date + timedelta(days=i) for i in range(n_steps)]
        else:
            dates = list(dates)
        
        # Compute returns
        returns = np.zeros(n_steps)
        returns[1:] = np.log(prices[1:] / prices[:-1])
        
        # Compute or use volatilities
        vol_window = volatility_window or self.volatility_window
        if volatilities is None:
            volatilities = self._compute_realized_vol(returns, vol_window)
        else:
            volatilities = np.asarray(volatilities)
        
        # Use or generate rates
        if rates is None:
            rates = np.full(n_steps, self.rate_default)
        else:
            rates = np.asarray(rates)
        
        # Compute rolling realized vol
        realized_vol = self._compute_realized_vol(returns, vol_window)
        
        return HistoricalMarketData(
            prices=prices,
            volatilities=volatilities,
            rates=rates,
            dates=dates,
            returns=returns,
            realized_vol=realized_vol,
        )
    
    def from_dataframe(
        self,
        df: Any,
        price_col: str = "close",
        vol_col: Optional[str] = None,
        rate_col: Optional[str] = None,
        date_col: str = "date",
    ) -> HistoricalMarketData:
        """
        Create market data from pandas DataFrame.
        
        Parameters
        ----------
        df : DataFrame
            Historical data.
        price_col : str
            Column name for prices.
        vol_col : str, optional
            Column name for volatility.
        rate_col : str, optional
            Column name for rates.
        date_col : str
            Column name for dates.
        
        Returns
        -------
        HistoricalMarketData
            Prepared market data.
        """
        prices = df[price_col].values
        
        dates = None
        if date_col in df.columns:
            dates = [d.date() if hasattr(d, "date") else d for d in df[date_col]]
        
        volatilities = None
        if vol_col and vol_col in df.columns:
            volatilities = df[vol_col].values
        
        rates = None
        if rate_col and rate_col in df.columns:
            rates = df[rate_col].values
        
        return self.from_prices(
            prices=prices,
            dates=dates,
            volatilities=volatilities,
            rates=rates,
        )
    
    def create_episodes(
        self,
        market_data: HistoricalMarketData,
        episode_length: int,
        n_episodes: int,
        overlap: bool = True,
        seed: Optional[int] = None,
    ) -> List[HistoricalMarketData]:
        """
        Create episodes for training/evaluation.
        
        Parameters
        ----------
        market_data : HistoricalMarketData
            Full market data.
        episode_length : int
            Length of each episode in data points.
        n_episodes : int
            Number of episodes to create.
        overlap : bool
            Allow overlapping episodes.
        seed : int, optional
            Random seed.
        
        Returns
        -------
        list of HistoricalMarketData
            List of episode data.
        """
        rng = np.random.default_rng(seed)
        
        max_start = market_data.n_steps - episode_length
        if max_start <= 0:
            raise ValueError(
                f"Episode length ({episode_length}) exceeds data length ({market_data.n_steps})"
            )
        
        if overlap:
            # Random starts with potential overlap
            starts = rng.integers(0, max_start, size=n_episodes)
        else:
            # Sequential non-overlapping
            n_possible = max_start // episode_length
            if n_episodes > n_possible:
                raise ValueError(
                    f"Cannot create {n_episodes} non-overlapping episodes "
                    f"(max {n_possible})"
                )
            all_starts = list(range(0, max_start, episode_length))
            starts = rng.choice(all_starts, size=n_episodes, replace=False)
        
        episodes = []
        for start in starts:
            episodes.append(market_data.get_window(int(start), episode_length))
        
        return episodes
    
    def create_train_test_split(
        self,
        market_data: HistoricalMarketData,
        test_ratio: float = 0.2,
        by_date: bool = True,
    ) -> Tuple[HistoricalMarketData, HistoricalMarketData]:
        """
        Split data into train and test sets.
        
        Parameters
        ----------
        market_data : HistoricalMarketData
            Full market data.
        test_ratio : float
            Fraction of data for testing.
        by_date : bool
            If True, split by date (chronological). If False, random.
        
        Returns
        -------
        train_data, test_data : tuple
            Train and test market data.
        """
        n_test = int(market_data.n_steps * test_ratio)
        n_train = market_data.n_steps - n_test
        
        if by_date:
            # Chronological split
            train_data = market_data.get_window(0, n_train)
            test_data = market_data.get_window(n_train, n_test)
        else:
            # Random split (not recommended for time series)
            raise NotImplementedError(
                "Random split not recommended for time series data"
            )
        
        return train_data, test_data
    
    def _compute_realized_vol(
        self,
        returns: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Compute rolling realized volatility."""
        n = len(returns)
        vol = np.zeros(n)
        
        for i in range(n):
            start = max(0, i - window + 1)
            if i > 0:
                window_returns = returns[start:i + 1]
                vol[i] = np.std(window_returns) * np.sqrt(252)  # Annualized
            else:
                vol[i] = 0.20  # Default
        
        # Fill early values with first computed vol
        if window < n:
            vol[:window] = vol[window]
        
        return vol


__all__ = [
    "HistoricalDataAdapter",
    "HistoricalMarketData",
]
