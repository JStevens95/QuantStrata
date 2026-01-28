"""
Unit tests for equity European vanilla Monte Carlo pricer.

Tests pricing accuracy, convergence, and comparison with BSM.

Author: QuantStrata Team
"""

import math
import pytest
import numpy as np

from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption
from src.pricers.equity.european_mc import EquityEuropeanVanillaMcPricer
from src.pricers.equity.european_bsm import EquityEuropeanVanillaBsmPricer
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
    """Tests for basic Monte Carlo pricing."""
    
    def test_call_positive_value(self, standard_market, spot_id, vol_id, curve_id):
        """MC call should have positive value."""
        pricer = EquityEuropeanVanillaMcPricer(n_paths=50_000, seed=42)
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0
    
    def test_put_positive_value(self, standard_market, spot_id, vol_id, curve_id):
        """MC put should have positive value."""
        pricer = EquityEuropeanVanillaMcPricer(n_paths=50_000, seed=42)
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0
    
    def test_notional_scaling(self, standard_market, spot_id, vol_id, curve_id):
        """Price should scale linearly with notional."""
        pricer = EquityEuropeanVanillaMcPricer(n_paths=50_000, seed=42)
        
        opt_1 = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        opt_100 = EuropeanEquityVanillaOption(
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
    """Tests for MC convergence to analytical BSM."""
    
    def test_call_converges_to_bsm(self, standard_market, spot_id, vol_id, curve_id):
        """MC call should converge to BSM price."""
        mc_pricer = EquityEuropeanVanillaMcPricer(n_paths=200_000, seed=42, antithetic=True)
        bsm_pricer = EquityEuropeanVanillaBsmPricer()
        
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        mc_pv = mc_pricer.price(opt, standard_market)
        bsm_pv = bsm_pricer.price(opt, standard_market)
        
        # MC should be within 1% of BSM
        rel_error = abs(mc_pv - bsm_pv) / bsm_pv
        assert rel_error < 0.01
    
    def test_put_converges_to_bsm(self, standard_market, spot_id, vol_id, curve_id):
        """MC put should converge to BSM price."""
        mc_pricer = EquityEuropeanVanillaMcPricer(n_paths=200_000, seed=42, antithetic=True)
        bsm_pricer = EquityEuropeanVanillaBsmPricer()
        
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        mc_pv = mc_pricer.price(opt, standard_market)
        bsm_pv = bsm_pricer.price(opt, standard_market)
        
        rel_error = abs(mc_pv - bsm_pv) / bsm_pv
        assert rel_error < 0.01
    
    def test_with_dividend(self, spot_id, vol_id, curve_id):
        """MC with dividend yield should match BSM."""
        market = Market(
            asof="2026-01-28",
            quotes={spot_id: Quote(value=100.0)},
            curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05)},
            vols={vol_id: FlatVolSurface(sigma=0.20)},
        )
        
        mc_pricer = EquityEuropeanVanillaMcPricer(n_paths=200_000, seed=42, antithetic=True)
        bsm_pricer = EquityEuropeanVanillaBsmPricer()
        
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.02, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        mc_pv = mc_pricer.price(opt, market)
        bsm_pv = bsm_pricer.price(opt, market)
        
        rel_error = abs(mc_pv - bsm_pv) / bsm_pv
        assert rel_error < 0.01


# =============================================================================
# Simulation Artifact Tests
# =============================================================================

class TestSimulationArtifact:
    """Tests for MC simulation artifact."""
    
    def test_run_returns_artifact(self, standard_market, spot_id, vol_id, curve_id):
        """run() should return EquityMcSimulation artifact."""
        pricer = EquityEuropeanVanillaMcPricer(n_paths=1_000, seed=42)
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        assert sim.spot0 == 100.0
        assert sim.strike == 100.0
        assert sim.maturity == 1.0
        assert sim.option_type == "call"
        assert sim.n_paths_requested == 1_000
    
    def test_terminal_spots_shape(self, standard_market, spot_id, vol_id, curve_id):
        """Terminal spots should have correct shape."""
        pricer = EquityEuropeanVanillaMcPricer(n_paths=1_000, seed=42, antithetic=True)
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        # With antithetic, effective paths = 2 * requested (rounded)
        assert sim.terminal_spots.shape[0] == sim.n_paths_effective
        assert sim.n_paths_effective >= 1_000
    
    def test_discounted_payoffs_shape(self, standard_market, spot_id, vol_id, curve_id):
        """Discounted payoffs should have same shape as terminal spots."""
        pricer = EquityEuropeanVanillaMcPricer(n_paths=1_000, seed=42)
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        assert sim.discounted_payoffs.shape == sim.terminal_spots.shape
    
    def test_store_paths(self, standard_market, spot_id, vol_id, curve_id):
        """With store_paths=True, paths should be stored."""
        pricer = EquityEuropeanVanillaMcPricer(n_paths=100, n_steps=10, seed=42)
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market, store_paths=True)
        
        assert sim.paths is not None
        assert sim.paths.shape[1] == 11  # n_steps + 1
    
    def test_paths_keep_limits(self, standard_market, spot_id, vol_id, curve_id):
        """paths_keep should limit stored paths."""
        pricer = EquityEuropeanVanillaMcPricer(n_paths=100, n_steps=10, seed=42, antithetic=False)
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market, store_paths=True, paths_keep=10)
        
        assert sim.paths is not None
        assert sim.paths.shape[0] == 10


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_expiry(self, standard_market, spot_id, vol_id, curve_id):
        """At expiry, MC should return intrinsic value."""
        pricer = EquityEuropeanVanillaMcPricer(n_paths=1_000, seed=42)
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=90.0, expiry=0.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert abs(pv - 10.0) < 1e-10  # S - K = 100 - 90
    
    def test_reproducible_with_seed(self, standard_market, spot_id, vol_id, curve_id):
        """Same seed should give same result."""
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        pricer1 = EquityEuropeanVanillaMcPricer(n_paths=10_000, seed=42)
        pricer2 = EquityEuropeanVanillaMcPricer(n_paths=10_000, seed=42)
        
        pv1 = pricer1.price(opt, standard_market)
        pv2 = pricer2.price(opt, standard_market)
        
        assert pv1 == pv2
    
    def test_different_seeds_different_results(self, standard_market, spot_id, vol_id, curve_id):
        """Different seeds should give different results."""
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        pricer1 = EquityEuropeanVanillaMcPricer(n_paths=10_000, seed=42)
        pricer2 = EquityEuropeanVanillaMcPricer(n_paths=10_000, seed=123)
        
        pv1 = pricer1.price(opt, standard_market)
        pv2 = pricer2.price(opt, standard_market)
        
        assert pv1 != pv2
