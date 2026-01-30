"""
Unit tests for equity European Monte Carlo pricers.

Tests pricing accuracy, convergence, and simulation artifacts for:
- Vanilla options (call/put)
- Barrier options (knock-in/out)
- Asian options (arithmetic/geometric)
- Lookback options (fixed/floating strike)

Author: QuantStrata Team
"""

import pytest
import numpy as np

from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption
from src.instruments.equity.options.barrier import EuropeanEquityBarrierOption
from src.instruments.equity.options.asian import EuropeanEquityAsianOption
from src.instruments.equity.options.lookback import EuropeanEquityLookbackOption
from src.pricers.equity.european_mc import (
    EquityEuropeanVanillaMcPricer,
    EquityEuropeanBarrierMcPricer,
    EquityEuropeanAsianMcPricer,
    EquityEuropeanLookbackMcPricer,
    EquityBarrierMcSimulation,
    EquityAsianMcSimulation,
    EquityLookbackMcSimulation,
)
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


# =============================================================================
# Barrier Option MC Pricer Tests
# =============================================================================


class TestBarrierMcPricerBasic:
    """Tests for equity barrier option MC pricer - basic functionality."""

    def test_up_and_out_call_positive(self, standard_market, spot_id, vol_id, curve_id):
        """Up-and-out call should have positive value when barrier not breached."""
        pricer = EquityEuropeanBarrierMcPricer(n_paths=50_000, n_steps=64, seed=42)
        opt = EuropeanEquityBarrierOption(
            ticker="AAPL", option_type="call",
            barrier_direction="up", barrier_style="knock_out",
            strike=100.0, barrier_level=150.0, expiry=1.0,
            notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv >= 0.0

    def test_down_and_out_put_positive(self, standard_market, spot_id, vol_id, curve_id):
        """Down-and-out put should have positive value when barrier not breached."""
        pricer = EquityEuropeanBarrierMcPricer(n_paths=50_000, n_steps=64, seed=42)
        opt = EuropeanEquityBarrierOption(
            ticker="AAPL", option_type="put",
            barrier_direction="down", barrier_style="knock_out",
            strike=100.0, barrier_level=60.0, expiry=1.0,
            notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv >= 0.0

    def test_barrier_less_than_vanilla(self, standard_market, spot_id, vol_id, curve_id):
        """Knock-out barrier should be worth less than equivalent vanilla."""
        barrier_pricer = EquityEuropeanBarrierMcPricer(n_paths=100_000, n_steps=64, seed=42)
        vanilla_pricer = EquityEuropeanVanillaMcPricer(n_paths=100_000, seed=42)
        
        barrier_opt = EuropeanEquityBarrierOption(
            ticker="AAPL", option_type="call",
            barrier_direction="up", barrier_style="knock_out",
            strike=100.0, barrier_level=130.0, expiry=1.0,
            notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        vanilla_opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        barrier_pv = barrier_pricer.price(barrier_opt, standard_market)
        vanilla_pv = vanilla_pricer.price(vanilla_opt, standard_market)
        
        assert barrier_pv < vanilla_pv

    def test_notional_scaling(self, standard_market, spot_id, vol_id, curve_id):
        """Barrier option price should scale with notional."""
        pricer = EquityEuropeanBarrierMcPricer(n_paths=50_000, n_steps=64, seed=42)
        
        opt_1 = EuropeanEquityBarrierOption(
            ticker="AAPL", option_type="call",
            barrier_direction="up", barrier_style="knock_out",
            strike=100.0, barrier_level=150.0, expiry=1.0,
            notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        opt_100 = EuropeanEquityBarrierOption(
            ticker="AAPL", option_type="call",
            barrier_direction="up", barrier_style="knock_out",
            strike=100.0, barrier_level=150.0, expiry=1.0,
            notional=100, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        pv_1 = pricer.price(opt_1, standard_market)
        pv_100 = pricer.price(opt_100, standard_market)
        
        assert abs(pv_100 - 100 * pv_1) < 1e-8


class TestBarrierMcSimulationArtifact:
    """Tests for barrier MC simulation artifact."""

    def test_run_returns_artifact(self, standard_market, spot_id, vol_id, curve_id):
        """run() should return EquityBarrierMcSimulation artifact."""
        pricer = EquityEuropeanBarrierMcPricer(n_paths=1_000, n_steps=32, seed=42)
        opt = EuropeanEquityBarrierOption(
            ticker="AAPL", option_type="call",
            barrier_direction="up", barrier_style="knock_out",
            strike=100.0, barrier_level=150.0, expiry=1.0,
            notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        assert isinstance(sim, EquityBarrierMcSimulation)
        assert sim.spot0 == 100.0
        assert sim.strike == 100.0
        assert sim.barrier_level == 150.0
        assert sim.barrier_direction == "up"
        assert sim.barrier_style == "knock_out"

    def test_terminal_spots_shape(self, standard_market, spot_id, vol_id, curve_id):
        """Terminal spots should have correct shape."""
        pricer = EquityEuropeanBarrierMcPricer(n_paths=1_000, n_steps=32, seed=42, antithetic=True)
        opt = EuropeanEquityBarrierOption(
            ticker="AAPL", option_type="call",
            barrier_direction="up", barrier_style="knock_out",
            strike=100.0, barrier_level=150.0, expiry=1.0,
            notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        assert sim.terminal_spots.shape[0] == sim.n_paths_effective
        assert sim.discounted_payoffs.shape[0] == sim.n_paths_effective


# =============================================================================
# Asian Option MC Pricer Tests
# =============================================================================


class TestAsianMcPricerBasic:
    """Tests for equity Asian option MC pricer - basic functionality."""

    def test_arithmetic_asian_call_positive(self, standard_market, spot_id, vol_id, curve_id):
        """Arithmetic Asian call should have positive value."""
        pricer = EquityEuropeanAsianMcPricer(n_paths=50_000, seed=42)
        opt = EuropeanEquityAsianOption(
            ticker="AAPL", option_type="call", averaging_type="arithmetic",
            strike=100.0, expiry=1.0, notional=1, n_averaging=12,
            dividend_yield=0.0, spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0

    def test_geometric_asian_call_positive(self, standard_market, spot_id, vol_id, curve_id):
        """Geometric Asian call should have positive value."""
        pricer = EquityEuropeanAsianMcPricer(n_paths=50_000, seed=42)
        opt = EuropeanEquityAsianOption(
            ticker="AAPL", option_type="call", averaging_type="geometric",
            strike=100.0, expiry=1.0, notional=1, n_averaging=12,
            dividend_yield=0.0, spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0

    def test_asian_less_than_vanilla(self, standard_market, spot_id, vol_id, curve_id):
        """Asian option should be worth less than equivalent vanilla (averaging reduces volatility)."""
        asian_pricer = EquityEuropeanAsianMcPricer(n_paths=100_000, seed=42)
        vanilla_pricer = EquityEuropeanVanillaMcPricer(n_paths=100_000, seed=42)
        
        asian_opt = EuropeanEquityAsianOption(
            ticker="AAPL", option_type="call", averaging_type="arithmetic",
            strike=100.0, expiry=1.0, notional=1, n_averaging=12,
            dividend_yield=0.0, spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        vanilla_opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        asian_pv = asian_pricer.price(asian_opt, standard_market)
        vanilla_pv = vanilla_pricer.price(vanilla_opt, standard_market)
        
        assert asian_pv < vanilla_pv

    def test_arithmetic_greater_than_geometric(self, standard_market, spot_id, vol_id, curve_id):
        """Arithmetic average Asian should be worth more than geometric (Jensen's inequality)."""
        pricer = EquityEuropeanAsianMcPricer(n_paths=100_000, seed=42)
        
        arith_opt = EuropeanEquityAsianOption(
            ticker="AAPL", option_type="call", averaging_type="arithmetic",
            strike=100.0, expiry=1.0, notional=1, n_averaging=12,
            dividend_yield=0.0, spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        geom_opt = EuropeanEquityAsianOption(
            ticker="AAPL", option_type="call", averaging_type="geometric",
            strike=100.0, expiry=1.0, notional=1, n_averaging=12,
            dividend_yield=0.0, spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        arith_pv = pricer.price(arith_opt, standard_market)
        geom_pv = pricer.price(geom_opt, standard_market)
        
        assert arith_pv >= geom_pv

    def test_notional_scaling(self, standard_market, spot_id, vol_id, curve_id):
        """Asian option price should scale with notional."""
        pricer = EquityEuropeanAsianMcPricer(n_paths=50_000, seed=42)
        
        opt_1 = EuropeanEquityAsianOption(
            ticker="AAPL", option_type="call", averaging_type="arithmetic",
            strike=100.0, expiry=1.0, notional=1, n_averaging=12,
            dividend_yield=0.0, spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        opt_100 = EuropeanEquityAsianOption(
            ticker="AAPL", option_type="call", averaging_type="arithmetic",
            strike=100.0, expiry=1.0, notional=100, n_averaging=12,
            dividend_yield=0.0, spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        pv_1 = pricer.price(opt_1, standard_market)
        pv_100 = pricer.price(opt_100, standard_market)
        
        assert abs(pv_100 - 100 * pv_1) < 1e-8


class TestAsianMcSimulationArtifact:
    """Tests for Asian MC simulation artifact."""

    def test_run_returns_artifact(self, standard_market, spot_id, vol_id, curve_id):
        """run() should return EquityAsianMcSimulation artifact."""
        pricer = EquityEuropeanAsianMcPricer(n_paths=1_000, seed=42)
        opt = EuropeanEquityAsianOption(
            ticker="AAPL", option_type="call", averaging_type="arithmetic",
            strike=100.0, expiry=1.0, notional=1, n_averaging=12,
            dividend_yield=0.0, spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        assert isinstance(sim, EquityAsianMcSimulation)
        assert sim.spot0 == 100.0
        assert sim.strike == 100.0
        assert sim.averaging_type == "arithmetic"

    def test_average_spots_shape(self, standard_market, spot_id, vol_id, curve_id):
        """Average spots should have correct shape."""
        pricer = EquityEuropeanAsianMcPricer(n_paths=1_000, seed=42, antithetic=True)
        opt = EuropeanEquityAsianOption(
            ticker="AAPL", option_type="call", averaging_type="arithmetic",
            strike=100.0, expiry=1.0, notional=1, n_averaging=12,
            dividend_yield=0.0, spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        assert sim.average_spots.shape[0] == sim.n_paths_effective
        assert sim.terminal_spots.shape[0] == sim.n_paths_effective


# =============================================================================
# Lookback Option MC Pricer Tests
# =============================================================================


class TestLookbackMcPricerBasic:
    """Tests for equity lookback option MC pricer - basic functionality."""

    def test_floating_strike_call_positive(self, standard_market, spot_id, vol_id, curve_id):
        """Floating strike lookback call should have positive value (always ITM)."""
        pricer = EquityEuropeanLookbackMcPricer(n_paths=50_000, n_steps=252, seed=42)
        opt = EuropeanEquityLookbackOption(
            ticker="AAPL", option_type="call", lookback_type="floating_strike",
            strike=100.0, expiry=1.0, notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0

    def test_fixed_strike_call_positive(self, standard_market, spot_id, vol_id, curve_id):
        """Fixed strike lookback call should have positive value."""
        pricer = EquityEuropeanLookbackMcPricer(n_paths=50_000, n_steps=252, seed=42)
        opt = EuropeanEquityLookbackOption(
            ticker="AAPL", option_type="call", lookback_type="fixed_strike",
            strike=100.0, expiry=1.0, notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0

    def test_lookback_greater_than_vanilla(self, standard_market, spot_id, vol_id, curve_id):
        """Lookback option should be worth more than equivalent vanilla (hindsight)."""
        lookback_pricer = EquityEuropeanLookbackMcPricer(n_paths=100_000, n_steps=252, seed=42)
        vanilla_pricer = EquityEuropeanVanillaMcPricer(n_paths=100_000, seed=42)
        
        lookback_opt = EuropeanEquityLookbackOption(
            ticker="AAPL", option_type="call", lookback_type="fixed_strike",
            strike=100.0, expiry=1.0, notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        vanilla_opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        lookback_pv = lookback_pricer.price(lookback_opt, standard_market)
        vanilla_pv = vanilla_pricer.price(vanilla_opt, standard_market)
        
        assert lookback_pv > vanilla_pv

    def test_floating_strike_always_itm(self, standard_market, spot_id, vol_id, curve_id):
        """Floating strike lookback should always have non-zero payoff."""
        pricer = EquityEuropeanLookbackMcPricer(n_paths=10_000, n_steps=100, seed=42)
        opt = EuropeanEquityLookbackOption(
            ticker="AAPL", option_type="call", lookback_type="floating_strike",
            strike=100.0, expiry=1.0, notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        # Floating strike call: S_T - S_min >= 0 always (equality only if no movement)
        assert np.all(sim.discounted_payoffs >= 0)

    def test_notional_scaling(self, standard_market, spot_id, vol_id, curve_id):
        """Lookback option price should scale with notional."""
        pricer = EquityEuropeanLookbackMcPricer(n_paths=50_000, n_steps=100, seed=42)
        
        opt_1 = EuropeanEquityLookbackOption(
            ticker="AAPL", option_type="call", lookback_type="floating_strike",
            strike=100.0, expiry=1.0, notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        opt_100 = EuropeanEquityLookbackOption(
            ticker="AAPL", option_type="call", lookback_type="floating_strike",
            strike=100.0, expiry=1.0, notional=100, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        pv_1 = pricer.price(opt_1, standard_market)
        pv_100 = pricer.price(opt_100, standard_market)
        
        assert abs(pv_100 - 100 * pv_1) < 1e-8


class TestLookbackMcSimulationArtifact:
    """Tests for lookback MC simulation artifact."""

    def test_run_returns_artifact(self, standard_market, spot_id, vol_id, curve_id):
        """run() should return EquityLookbackMcSimulation artifact."""
        pricer = EquityEuropeanLookbackMcPricer(n_paths=1_000, n_steps=50, seed=42)
        opt = EuropeanEquityLookbackOption(
            ticker="AAPL", option_type="call", lookback_type="floating_strike",
            strike=100.0, expiry=1.0, notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        assert isinstance(sim, EquityLookbackMcSimulation)
        assert sim.spot0 == 100.0
        assert sim.lookback_type == "floating_strike"

    def test_extrema_shapes(self, standard_market, spot_id, vol_id, curve_id):
        """Max and min spots should have correct shapes."""
        pricer = EquityEuropeanLookbackMcPricer(n_paths=1_000, n_steps=50, seed=42, antithetic=True)
        opt = EuropeanEquityLookbackOption(
            ticker="AAPL", option_type="call", lookback_type="floating_strike",
            strike=100.0, expiry=1.0, notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        assert sim.max_spots.shape[0] == sim.n_paths_effective
        assert sim.min_spots.shape[0] == sim.n_paths_effective
        assert sim.terminal_spots.shape[0] == sim.n_paths_effective

    def test_extrema_bounds(self, standard_market, spot_id, vol_id, curve_id):
        """Max should be >= terminal >= min."""
        pricer = EquityEuropeanLookbackMcPricer(n_paths=1_000, n_steps=50, seed=42)
        opt = EuropeanEquityLookbackOption(
            ticker="AAPL", option_type="call", lookback_type="floating_strike",
            strike=100.0, expiry=1.0, notional=1, dividend_yield=0.0,
            spot_id=spot_id, vol_id=vol_id, curve_id=curve_id,
        )
        
        sim = pricer.run(opt, standard_market)
        
        assert np.all(sim.max_spots >= sim.terminal_spots)
        assert np.all(sim.terminal_spots >= sim.min_spots)
