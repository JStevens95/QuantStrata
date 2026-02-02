"""
Unit tests for historical data providers.

Tests the data provider implementations:
- DictDataProvider
- CsvDataProvider
- SimpleMarketSnapshot
"""

import numpy as np
import pytest
import tempfile
from datetime import date
from pathlib import Path

from src.backtesting.data.providers import (
    HistoricalDataProvider,
    DictDataProvider,
    CsvDataProvider,
    SimpleMarketSnapshot,
    create_data_provider,
)


class TestSimpleMarketSnapshot:
    """Tests for SimpleMarketSnapshot."""
    
    def test_basic_creation(self):
        """Test basic snapshot creation."""
        snapshot = SimpleMarketSnapshot(
            asof=date(2024, 1, 1),
            prices={"AAPL": 150.0, "GOOGL": 140.0},
        )
        
        assert snapshot.asof == date(2024, 1, 1)
        assert snapshot.get_price("AAPL") == 150.0
        assert snapshot.get_price("GOOGL") == 140.0
    
    def test_missing_price_raises(self):
        """Test that missing price raises KeyError."""
        snapshot = SimpleMarketSnapshot(
            asof=date(2024, 1, 1),
            prices={"AAPL": 150.0},
        )
        
        with pytest.raises(KeyError):
            snapshot.get_price("GOOGL")
    
    def test_contains(self):
        """Test __contains__ method."""
        snapshot = SimpleMarketSnapshot(
            asof=date(2024, 1, 1),
            prices={"AAPL": 150.0},
        )
        
        assert "AAPL" in snapshot
        assert "GOOGL" not in snapshot
    
    def test_additional_data(self):
        """Test additional data storage."""
        snapshot = SimpleMarketSnapshot(
            asof=date(2024, 1, 1),
            prices={"AAPL": 150.0},
            data={"volume": 1000000, "high": 152.0},
        )
        
        assert snapshot.get("volume") == 1000000
        assert snapshot.get("high") == 152.0
        assert snapshot.get("missing", "default") == "default"


class TestDictDataProvider:
    """Tests for DictDataProvider."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            date(2024, 1, 1): {"AAPL": 150.0, "GOOGL": 140.0},
            date(2024, 1, 2): {"AAPL": 152.0, "GOOGL": 142.0},
            date(2024, 1, 3): {"AAPL": 151.0, "GOOGL": 145.0},
        }
    
    def test_basic_creation(self, sample_data):
        """Test basic provider creation."""
        provider = DictDataProvider(sample_data)
        
        assert provider.num_dates == 3
        assert provider.start_date == date(2024, 1, 1)
        assert provider.end_date == date(2024, 1, 3)
    
    def test_get_dates_ordered(self, sample_data):
        """Test dates are returned in order."""
        provider = DictDataProvider(sample_data)
        dates = provider.get_dates()
        
        assert dates == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    
    def test_get_snapshot(self, sample_data):
        """Test getting snapshot for a date."""
        provider = DictDataProvider(sample_data)
        snapshot = provider.get_snapshot(date(2024, 1, 1))
        
        assert snapshot.asof == date(2024, 1, 1)
        assert snapshot.get_price("AAPL") == 150.0
    
    def test_missing_date_raises(self, sample_data):
        """Test that missing date raises KeyError."""
        provider = DictDataProvider(sample_data)
        
        with pytest.raises(KeyError):
            provider.get_snapshot(date(2024, 1, 10))
    
    def test_get_instruments(self, sample_data):
        """Test getting list of instruments."""
        provider = DictDataProvider(sample_data)
        instruments = provider.get_instruments()
        
        assert "AAPL" in instruments
        assert "GOOGL" in instruments
    
    def test_accepts_snapshots(self):
        """Test accepting pre-built snapshots (prices are used; extra .data not preserved via marketdata)."""
        data = {
            date(2024, 1, 1): SimpleMarketSnapshot(
                asof=date(2024, 1, 1),
                prices={"AAPL": 150.0},
                data={"volume": 1000000},
            ),
        }
        provider = DictDataProvider(data)
        snapshot = provider.get_snapshot(date(2024, 1, 1))
        # DictDataProvider uses marketdata HistoricalProvider: snapshot has get_price and asof
        assert snapshot.get_price("AAPL") == 150.0
        assert snapshot.asof == date(2024, 1, 1)


class TestCsvDataProvider:
    """Tests for CsvDataProvider."""
    
    def test_wide_format(self):
        """Test loading wide-format CSV."""
        csv_content = """date,AAPL,GOOGL
2024-01-01,150.0,140.0
2024-01-02,152.0,142.0
2024-01-03,151.0,145.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            provider = CsvDataProvider(f.name, format="wide")
            
            assert provider.num_dates == 3
            snapshot = provider.get_snapshot(date(2024, 1, 1))
            assert snapshot.get_price("AAPL") == 150.0
            assert snapshot.get_price("GOOGL") == 140.0
    
    def test_long_format(self):
        """Test loading long-format CSV."""
        csv_content = """date,instrument,price
2024-01-01,AAPL,150.0
2024-01-01,GOOGL,140.0
2024-01-02,AAPL,152.0
2024-01-02,GOOGL,142.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            provider = CsvDataProvider(f.name, format="long")
            
            assert provider.num_dates == 2
            snapshot = provider.get_snapshot(date(2024, 1, 1))
            assert snapshot.get_price("AAPL") == 150.0
    
    def test_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            CsvDataProvider("nonexistent.csv")


class TestCreateDataProvider:
    """Tests for create_data_provider factory."""
    
    def test_create_from_dict(self):
        """Test creating provider from dict."""
        data = {date(2024, 1, 1): {"AAPL": 150.0}}
        provider = create_data_provider(data)
        
        assert isinstance(provider, DictDataProvider)
    
    def test_create_from_csv(self):
        """Test creating provider from CSV path."""
        csv_content = """date,AAPL
2024-01-01,150.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            provider = create_data_provider(f.name, format="wide")
            assert isinstance(provider, CsvDataProvider)
    
    def test_unknown_source_raises(self):
        """Test error on unknown source type."""
        with pytest.raises(ValueError):
            create_data_provider("data.unknown")
