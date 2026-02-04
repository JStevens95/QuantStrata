"""
Unit tests for historical data adapter module.

Tests HistoricalDataAdapter and HistoricalMarketData.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.deep_hedging.adapters.historical_data import (
    HistoricalDataAdapter,
    HistoricalMarketData,
)


class TestHistoricalMarketData:
    """Tests for HistoricalMarketData dataclass."""
    
    def test_data_creation(self) -> None:
        """Test data creation."""
        n = 100
        data = HistoricalMarketData(
            prices=np.random.randn(n) + 100,
            volatilities=np.ones(n) * 0.2,
            rates=np.ones(n) * 0.05,
            dates=[date.today() - timedelta(days=i) for i in range(n)],
        )
        
        assert len(data.prices) == n
        assert len(data.volatilities) == n
        assert len(data.rates) == n
    
    def test_data_without_dates(self) -> None:
        """Test data without dates."""
        n = 50
        data = HistoricalMarketData(
            prices=np.random.randn(n) + 100,
            volatilities=np.ones(n) * 0.2,
            rates=np.ones(n) * 0.05,
        )
        
        assert data.dates is None or len(data.dates) == 0


class TestHistoricalDataAdapter:
    """Tests for HistoricalDataAdapter."""
    
    def test_adapter_creation(self) -> None:
        """Test adapter creation."""
        adapter = HistoricalDataAdapter(volatility_window=20)
        
        assert adapter.volatility_window == 20
    
    def test_from_prices_basic(self) -> None:
        """Test creating data from price array."""
        adapter = HistoricalDataAdapter()
        
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(100) * 0.01)
        
        data = adapter.from_prices(prices)
        
        assert isinstance(data, HistoricalMarketData)
        assert len(data.prices) == len(prices)
    
    def test_from_prices_with_volatilities(self) -> None:
        """Test with provided volatilities."""
        adapter = HistoricalDataAdapter()
        
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(100) * 0.01)
        vols = np.ones(100) * 0.25
        
        data = adapter.from_prices(prices, volatilities=vols)
        
        np.testing.assert_array_equal(data.volatilities, vols)
    
    def test_from_prices_with_rates(self) -> None:
        """Test with provided interest rates."""
        adapter = HistoricalDataAdapter()
        
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(100) * 0.01)
        rates = np.linspace(0.03, 0.05, 100)
        
        data = adapter.from_prices(prices, rates=rates)
        
        np.testing.assert_array_equal(data.rates, rates)
    
    def test_from_prices_computes_volatility(self) -> None:
        """Test that volatility is computed if not provided."""
        adapter = HistoricalDataAdapter(volatility_window=20)
        
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(100) * 0.01)
        
        data = adapter.from_prices(prices)
        
        # Volatility should be computed
        assert data.volatilities is not None
        assert len(data.volatilities) == len(prices)
        
        # Volatility should be positive
        assert all(data.volatilities >= 0)
    
    def test_from_prices_with_dates(self) -> None:
        """Test with provided dates."""
        adapter = HistoricalDataAdapter()
        
        n = 100
        prices = np.linspace(100, 110, n)
        dates = [date.today() - timedelta(days=i) for i in range(n)]
        
        data = adapter.from_prices(prices, dates=dates)
        
        assert data.dates is not None
        assert len(data.dates) == n
    
    def test_from_dataframe(self) -> None:
        """Test creating data from DataFrame."""
        adapter = HistoricalDataAdapter()
        
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "price": 100 * np.cumprod(1 + np.random.randn(n) * 0.01),
            "vol": np.ones(n) * 0.20,
            "rate": np.ones(n) * 0.05,
        })
        
        data = adapter.from_dataframe(
            df,
            price_col="price",
            vol_col="vol",
            rate_col="rate",
        )
        
        assert len(data.prices) == n
        assert len(data.volatilities) == n
    
    def test_create_episodes(self) -> None:
        """Test creating episodes for training."""
        adapter = HistoricalDataAdapter()
        
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.01)
        
        data = adapter.from_prices(prices)
        
        episodes = adapter.create_episodes(
            data,
            episode_length=50,
            n_episodes=5,
        )
        
        assert len(episodes) == 5
        
        for episode in episodes:
            assert len(episode.prices) == 50
    
    def test_create_episodes_with_overlap(self) -> None:
        """Test creating overlapping episodes."""
        adapter = HistoricalDataAdapter()
        
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(200) * 0.01)
        
        data = adapter.from_prices(prices)
        
        episodes = adapter.create_episodes(
            data,
            episode_length=50,
            n_episodes=10,
            overlap=True,
        )
        
        assert len(episodes) == 10
    
    def test_default_rate(self) -> None:
        """Test default rate is applied."""
        adapter = HistoricalDataAdapter(rate_default=0.03)
        
        prices = np.linspace(100, 110, 50)
        
        data = adapter.from_prices(prices)
        
        # Rates should be the default
        assert all(data.rates == 0.03)
    
    def test_volatility_window_effect(self) -> None:
        """Test effect of volatility window size."""
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(200) * 0.01)
        
        adapter_short = HistoricalDataAdapter(volatility_window=10)
        adapter_long = HistoricalDataAdapter(volatility_window=60)
        
        data_short = adapter_short.from_prices(prices)
        data_long = adapter_long.from_prices(prices)
        
        # Short window should have more volatile vol estimates
        vol_of_vol_short = np.std(data_short.volatilities[60:])
        vol_of_vol_long = np.std(data_long.volatilities[60:])
        
        assert vol_of_vol_short >= vol_of_vol_long * 0.5
