"""
Unit tests for HistoricalProvider.

Tests loading from dict and CSV and the MarketDataProvider API.
"""

import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest, Universe
from src.marketdata.providers.historical import HistoricalProvider, HistoricalProviderConfig


class TestHistoricalProviderFromDict:
    """Tests for HistoricalProvider with dict input."""

    @pytest.fixture
    def sample_data(self):
        return {
            date(2024, 1, 1): {"AAPL": 150.0, "GOOGL": 140.0},
            date(2024, 1, 2): {"AAPL": 152.0, "GOOGL": 142.0},
            date(2024, 1, 3): {"AAPL": 151.0, "GOOGL": 145.0},
        }

    def test_basic_creation(self, sample_data):
        """Test creation from dict."""
        provider = HistoricalProvider(data=sample_data)
        assert provider.name == "HistoricalProvider"
        assert len(provider.dates) == 3
        assert provider.dates[0] == "2024-01-01"

    def test_universe(self, sample_data):
        """Test universe is built from symbols."""
        provider = HistoricalProvider(data=sample_data)
        u = provider.universe
        assert len(u.ids) == 2
        names = {mid.name for mid in u.ids}
        assert names == {"AAPL", "GOOGL"}
        assert all(mid.asset_class == "EQUITY" and mid.mkt_type == "SPOT" for mid in u.ids)

    def test_get_market(self, sample_data):
        """Test get_market returns quote-only Market."""
        provider = HistoricalProvider(data=sample_data)
        u = provider.universe
        req = MarketRequest(asof="2024-01-01", universe=u)
        market = provider.get_market(req)
        assert market.asof == "2024-01-01"
        assert len(market.quotes) == 2
        assert market.quote(MarketId("EQUITY", "SPOT", "AAPL")) == 150.0
        assert market.quote(MarketId("EQUITY", "SPOT", "GOOGL")) == 140.0
        assert len(market.curves) == 0
        assert len(market.vols) == 0

    def test_get_market_missing_date_raises(self, sample_data):
        """Test get_market raises for missing date."""
        provider = HistoricalProvider(data=sample_data)
        u = provider.universe
        req = MarketRequest(asof="2024-01-10", universe=u)
        with pytest.raises(ValueError, match="no data"):
            provider.get_market(req)

    def test_get_timeseries(self, sample_data):
        """Test get_timeseries returns MarketDataset."""
        provider = HistoricalProvider(data=sample_data)
        u = provider.universe
        req = TimeseriesRequest(
            start="2024-01-01",
            end="2024-01-03",
            freq="D",
            universe=u,
            scenarios=1,
        )
        ds = provider.get_timeseries(req)
        assert len(ds.dates) == 3
        assert ds.n_scenarios == 1
        market = ds.snapshot(time_idx=0, scenario_idx=0)
        assert market.quote(MarketId("EQUITY", "SPOT", "AAPL")) == 150.0

    def test_config_asset_class(self, sample_data):
        """Test custom asset_class in config."""
        config = HistoricalProviderConfig(asset_class="FX")
        provider = HistoricalProvider(data=sample_data, config=config)
        u = provider.universe
        assert all(mid.asset_class == "FX" for mid in u.ids)


class TestHistoricalProviderFromCsv:
    """Tests for HistoricalProvider with CSV input."""

    def test_wide_format(self):
        """Test loading wide-format CSV."""
        csv_content = """date,AAPL,GOOGL
2024-01-01,150.0,140.0
2024-01-02,152.0,142.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            provider = HistoricalProvider(data=Path(f.name), format="wide")
            assert len(provider.dates) == 2
            u = provider.universe
            req = MarketRequest(asof="2024-01-01", universe=u)
            market = provider.get_market(req)
            assert market.quote(MarketId("EQUITY", "SPOT", "AAPL")) == 150.0

    def test_long_format(self):
        """Test loading long-format CSV."""
        csv_content = """date,instrument,price
2024-01-01,AAPL,150.0
2024-01-01,GOOGL,140.0
2024-01-02,AAPL,152.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            provider = HistoricalProvider(
                data=Path(f.name),
                format="long",
                instrument_column="instrument",
                price_column="price",
            )
            assert len(provider.dates) == 2
            u = provider.universe
            req = MarketRequest(asof="2024-01-01", universe=u)
            market = provider.get_market(req)
            assert market.quote(MarketId("EQUITY", "SPOT", "AAPL")) == 150.0

    def test_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            HistoricalProvider(data=Path("nonexistent.csv"))
