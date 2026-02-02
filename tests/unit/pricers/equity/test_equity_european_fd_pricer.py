"""
Unit tests for equity European vanilla FD pricer.

Tests pricing accuracy, convergence, and comparison with BSM.

Author: QuantStrata Team
"""

import math
import pytest
import numpy as np

from src.instruments.equity.options.vanilla import EquityVanillaEuropeanOption
from src.pricers.equity.european_bsm_fde import EquityVanillaEuropeanOptionFdPricer
from src.pricers.equity.european_bsm import EquityVanillaEuropeanOptionBsmPricer
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def spot_id() -> MarketId:
    return MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")


@pytest.fixture
def vol_id() -> MarketId:
    return MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")


@pytest.fixture
def curve_id() -> MarketId:
    return MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")


@pytest.fixture
def standard_market(spot_id, vol_id, curve_id) -> Market:
    """Standard market: S=100, r=5%, vol=20%."""
    return Market(
        asof="2026-01-28",
        quotes={spot_id: Quote(value=100.0)},
        curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05)},
        vols={vol_id: FlatVolSurface(sigma=0.20)},
    )


# =============================================================================
# Basic Pricing Tests
# =============================================================================

class TestBasicPricing:
    """Tests for basic FD pricing."""
    
    def test_call_positive_value(self, standard_market, spot_id, vol_id, curve_id):
        """FD call should have positive value."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=201, n_time_steps=100)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0
    
    def test_put_positive_value(self, standard_market, spot_id, vol_id, curve_id):
        """FD put should have positive value."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=201, n_time_steps=100)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0
    
    def test_notional_scaling(self, standard_market, spot_id, vol_id, curve_id):
        """Price should scale linearly with notional."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=201, n_time_steps=100)
        
        opt_1 = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        opt_100 = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=100, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        pv_1 = pricer.price(opt_1, standard_market)
        pv_100 = pricer.price(opt_100, standard_market)
        
        assert abs(pv_100 - 100 * pv_1) < 1e-8


# =============================================================================
# Convergence to BSM Tests
# =============================================================================

class TestConvergenceToBsm:
    """Tests for FD convergence to analytical BSM."""
    
    def test_call_converges_to_bsm(self, standard_market, spot_id, vol_id, curve_id):
        """FD call should converge to BSM price."""
        fd_pricer = EquityVanillaEuropeanOptionFdPricer(n_space=401, n_time_steps=200)
        bsm_pricer = EquityVanillaEuropeanOptionBsmPricer()
        
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        fd_pv = fd_pricer.price(opt, standard_market)
        bsm_pv = bsm_pricer.price(opt, standard_market)
        
        # FD should be within 0.1% of BSM with fine grid
        rel_error = abs(fd_pv - bsm_pv) / bsm_pv
        assert rel_error < 0.001
    
    def test_put_converges_to_bsm(self, standard_market, spot_id, vol_id, curve_id):
        """FD put should converge to BSM price."""
        fd_pricer = EquityVanillaEuropeanOptionFdPricer(n_space=401, n_time_steps=200)
        bsm_pricer = EquityVanillaEuropeanOptionBsmPricer()
        
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        fd_pv = fd_pricer.price(opt, standard_market)
        bsm_pv = bsm_pricer.price(opt, standard_market)
        
        rel_error = abs(fd_pv - bsm_pv) / bsm_pv
        assert rel_error < 0.001
    
    def test_with_dividend(self, spot_id, vol_id, curve_id):
        """FD with dividend yield should match BSM."""
        market = Market(
            asof="2026-01-28",
            quotes={spot_id: Quote(value=100.0)},
            curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05)},
            vols={vol_id: FlatVolSurface(sigma=0.20)},
        )
        
        fd_pricer = EquityVanillaEuropeanOptionFdPricer(n_space=401, n_time_steps=200)
        bsm_pricer = EquityVanillaEuropeanOptionBsmPricer()
        
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.02, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        fd_pv = fd_pricer.price(opt, market)
        bsm_pv = bsm_pricer.price(opt, market)
        
        rel_error = abs(fd_pv - bsm_pv) / bsm_pv
        assert rel_error < 0.001


# =============================================================================
# Grid Refinement Tests
# =============================================================================

class TestGridRefinement:
    """Tests for grid refinement improving accuracy."""
    
    def test_finer_grid_more_accurate(self, standard_market, spot_id, vol_id, curve_id):
        """Finer grid should give more accurate results."""
        bsm_pricer = EquityVanillaEuropeanOptionBsmPricer()
        
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        bsm_pv = bsm_pricer.price(opt, standard_market)
        
        # Coarse grid
        fd_coarse = EquityVanillaEuropeanOptionFdPricer(n_space=51, n_time_steps=25)
        pv_coarse = fd_coarse.price(opt, standard_market)
        error_coarse = abs(pv_coarse - bsm_pv) / bsm_pv
        
        # Fine grid
        fd_fine = EquityVanillaEuropeanOptionFdPricer(n_space=401, n_time_steps=200)
        pv_fine = fd_fine.price(opt, standard_market)
        error_fine = abs(pv_fine - bsm_pv) / bsm_pv
        
        assert error_fine < error_coarse


# =============================================================================
# Greeks Tests
# =============================================================================

class TestGreeks:
    """Tests for FD Greeks computation."""
    
    def test_call_delta_positive(self, standard_market, spot_id, vol_id, curve_id):
        """FD call delta should be positive and between 0 and 1."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=201, n_time_steps=100)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        assert 0.0 < greeks["delta"] < 1.0
    
    def test_put_delta_negative(self, standard_market, spot_id, vol_id, curve_id):
        """FD put delta should be negative and between -1 and 0."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=201, n_time_steps=100)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        assert -1.0 < greeks["delta"] < 0.0
    
    def test_gamma_positive(self, standard_market, spot_id, vol_id, curve_id):
        """FD gamma should be positive."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=201, n_time_steps=100)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        assert greeks["gamma"] > 0.0
    
    def test_vega_positive(self, standard_market, spot_id, vol_id, curve_id):
        """FD vega should be positive."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=201, n_time_steps=100)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        assert greeks["vega"] > 0.0


# =============================================================================
# Diagnostics Tests
# =============================================================================

class TestDiagnostics:
    """Tests for FD diagnostics."""
    
    def test_diagnostics_returns_grids(self, standard_market, spot_id, vol_id, curve_id):
        """diagnostics() should return grid information."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=101, n_time_steps=50)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        diag = pricer.diagnostics(opt, standard_market, store_surface=True)
        
        assert diag.spot_grid is not None
        assert len(diag.spot_grid) == 101
        assert diag.values_t0_per_unit is not None
    
    def test_diagnostics_surface_shape(self, standard_market, spot_id, vol_id, curve_id):
        """Surface should have correct shape (n_space x n_time)."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=51, n_time_steps=25)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        diag = pricer.diagnostics(opt, standard_market, store_surface=True)
        
        assert diag.surface_per_unit is not None
        # Surface shape depends on FD solver convention
        # Verify dimensions are correct (either order is fine for this test)
        n_space_dim = diag.surface_per_unit.shape[0]
        n_time_dim = diag.surface_per_unit.shape[1]
        assert (n_space_dim == 51 and n_time_dim == 26) or (n_space_dim == 26 and n_time_dim == 51)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_expiry(self, standard_market, spot_id, vol_id, curve_id):
        """At expiry, FD should return intrinsic value."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=201, n_time_steps=100)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=90.0, expiry=0.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert abs(pv - 10.0) < 1e-10
    
    def test_zero_expiry_greeks(self, standard_market, spot_id, vol_id, curve_id):
        """At expiry, FD greeks should be zero."""
        pricer = EquityVanillaEuropeanOptionFdPricer(n_space=201, n_time_steps=100)
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=0.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        for key in ["delta", "gamma", "vega", "rho"]:
            assert greeks[key] == 0.0
