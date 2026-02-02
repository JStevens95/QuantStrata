# tests/unit/pricers/ir/test_ir_european_b76_pricer.py
"""
Unit tests for Interest Rate Black76 Pricers.

Tests cover:
- Caplet and floorlet pricing
- Cap and floor pricing
- Put-call parity (cap-floor = swap)
- Greeks computation
- Finite difference validation
- Edge cases

Author: QuantStrata Team
"""
from __future__ import annotations

import pytest

from src.instruments.ir.options.capfloor import (
    IrCapletEuropeanOptionSimple,
    IrFloorletEuropeanOptionSimple,
    IrCapEuropeanOptionSimple,
    IrFloorEuropeanOptionSimple,
    IrCapEuropeanOption,
    IrCapletEuropeanOption,
    compute_accrual_factor,
)
from src.pricers.ir.european_b76 import (
    IrCapletEuropeanOptionB76PricerSimple,
    IrFloorletEuropeanOptionB76PricerSimple,
    IrCapEuropeanOptionB76PricerSimple,
    IrFloorEuropeanOptionB76PricerSimple,
    IrCapletEuropeanOptionB76Pricer,
    IrCapEuropeanOptionB76Pricer,
)
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def base_params():
    """Base parameters for a single caplet/floorlet."""
    return {
        "notional": 1_000_000,
        "strike": 0.05,            # 5% strike
        "fixing_time": 1.0,        # 1 year to fixing
        "payment_time": 1.25,      # Payment 3 months after fixing
        "accrual_factor": 0.25,    # Quarterly
        "forward_rate": 0.055,     # 5.5% forward (slightly ITM for caplet)
        "vol": 0.20,               # 20% lognormal vol
        "discount_factor": 0.95,   # DF to payment date
    }


@pytest.fixture
def caplet(base_params):
    """Create a simple caplet."""
    return IrCapletEuropeanOptionSimple(**base_params)


@pytest.fixture
def floorlet(base_params):
    """Create a simple floorlet."""
    return IrFloorletEuropeanOptionSimple(**base_params)


@pytest.fixture
def market():
    """Create a simple market for testing."""
    curve_id = MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),))
    vol_id = MarketId("IR", "VOL", "USD.CAPS", (("ccy", "USD"),))
    
    return Market(
        asof="2026-01-15",
        quotes={},  # No quotes needed for cap/floor pricing
        curves={
            curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05),
        },
        vols={
            vol_id: FlatVolSurface(sigma=0.20),
        },
    )


@pytest.fixture
def curve_id():
    """Standard curve ID."""
    return MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),))


@pytest.fixture
def vol_id():
    """Standard vol ID."""
    return MarketId("IR", "VOL", "USD.CAPS", (("ccy", "USD"),))


# =============================================================================
# INSTRUMENT VALIDATION TESTS
# =============================================================================


class TestCapletValidation:
    """Tests for caplet instrument validation."""
    
    def test_valid_caplet(self, base_params):
        """Valid caplet should be created without errors."""
        caplet = IrCapletEuropeanOptionSimple(**base_params)
        assert caplet.notional == base_params["notional"]
        assert caplet.strike == base_params["strike"]
    
    def test_zero_notional_raises(self, base_params):
        """Zero notional should raise."""
        base_params["notional"] = 0.0
        with pytest.raises(ValueError, match="notional must be non-zero"):
            IrCapletEuropeanOptionSimple(**base_params)
    
    def test_negative_strike_raises(self, base_params):
        """Negative strike should raise."""
        base_params["strike"] = -0.01
        with pytest.raises(ValueError, match="strike must be > 0"):
            IrCapletEuropeanOptionSimple(**base_params)
    
    def test_payment_before_fixing_raises(self, base_params):
        """Payment time before fixing time should raise."""
        base_params["payment_time"] = 0.5  # Before fixing at 1.0
        with pytest.raises(ValueError, match="payment_time must be > fixing_time"):
            IrCapletEuropeanOptionSimple(**base_params)


class TestFloorletValidation:
    """Tests for floorlet instrument validation."""
    
    def test_valid_floorlet(self, base_params):
        """Valid floorlet should be created without errors."""
        floorlet = IrFloorletEuropeanOptionSimple(**base_params)
        assert floorlet.notional == base_params["notional"]
    
    def test_zero_notional_raises(self, base_params):
        """Zero notional should raise."""
        base_params["notional"] = 0.0
        with pytest.raises(ValueError, match="notional must be non-zero"):
            IrFloorletEuropeanOptionSimple(**base_params)


# =============================================================================
# BASIC PRICING TESTS
# =============================================================================


class TestCapletPricing:
    """Tests for caplet pricing."""
    
    def test_caplet_price_positive(self, caplet):
        """Caplet price should be positive."""
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        pv = pricer.price(caplet)
        assert pv > 0
    
    def test_itm_caplet_higher_than_otm(self, base_params):
        """ITM caplet should be more valuable than OTM."""
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        
        # ITM: forward > strike
        itm_params = {**base_params, "forward_rate": 0.07, "strike": 0.05}
        itm_caplet = IrCapletEuropeanOptionSimple(**itm_params)
        
        # OTM: forward < strike
        otm_params = {**base_params, "forward_rate": 0.03, "strike": 0.05}
        otm_caplet = IrCapletEuropeanOptionSimple(**otm_params)
        
        assert pricer.price(itm_caplet) > pricer.price(otm_caplet)
    
    def test_caplet_price_scales_with_notional(self, base_params):
        """Price should scale linearly with notional."""
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        
        caplet_1 = IrCapletEuropeanOptionSimple(**base_params)
        
        params_2 = {**base_params, "notional": 2_000_000}
        caplet_2 = IrCapletEuropeanOptionSimple(**params_2)
        
        pv_1 = pricer.price(caplet_1)
        pv_2 = pricer.price(caplet_2)
        
        assert abs(pv_2 - 2 * pv_1) < 1e-6


class TestFloorletPricing:
    """Tests for floorlet pricing."""
    
    def test_floorlet_price_positive(self, floorlet):
        """Floorlet price should be positive."""
        pricer = IrFloorletEuropeanOptionB76PricerSimple()
        pv = pricer.price(floorlet)
        assert pv > 0
    
    def test_itm_floorlet_higher_than_otm(self, base_params):
        """ITM floorlet should be more valuable than OTM."""
        pricer = IrFloorletEuropeanOptionB76PricerSimple()
        
        # ITM: forward < strike
        itm_params = {**base_params, "forward_rate": 0.03, "strike": 0.05}
        itm_floorlet = IrFloorletEuropeanOptionSimple(**itm_params)
        
        # OTM: forward > strike
        otm_params = {**base_params, "forward_rate": 0.07, "strike": 0.05}
        otm_floorlet = IrFloorletEuropeanOptionSimple(**otm_params)
        
        assert pricer.price(itm_floorlet) > pricer.price(otm_floorlet)


# =============================================================================
# PUT-CALL PARITY TESTS
# =============================================================================


class TestPutCallParity:
    """Tests for caplet-floorlet parity."""
    
    def test_caplet_floorlet_parity(self, base_params):
        """
        Caplet - Floorlet = DF × τ × N × (F - K)
        
        This is the cap-floor parity for a single period.
        """
        caplet = IrCapletEuropeanOptionSimple(**base_params)
        floorlet = IrFloorletEuropeanOptionSimple(**base_params)
        
        caplet_pricer = IrCapletEuropeanOptionB76PricerSimple()
        floorlet_pricer = IrFloorletEuropeanOptionB76PricerSimple()
        
        caplet_pv = caplet_pricer.price(caplet)
        floorlet_pv = floorlet_pricer.price(floorlet)
        
        # Expected: DF × τ × N × (F - K)
        N = base_params["notional"]
        tau = base_params["accrual_factor"]
        df = base_params["discount_factor"]
        F = base_params["forward_rate"]
        K = base_params["strike"]
        
        expected_diff = df * tau * N * (F - K)
        actual_diff = caplet_pv - floorlet_pv
        
        assert abs(actual_diff - expected_diff) < 1e-6, \
            f"Parity failed: {actual_diff} != {expected_diff}"


# =============================================================================
# GREEKS TESTS
# =============================================================================


class TestCapletGreeks:
    """Tests for caplet Greeks."""
    
    def test_greeks_exist(self, caplet):
        """Greeks should be computed."""
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(caplet)
        
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks
    
    def test_call_delta_positive(self, caplet):
        """Caplet delta should be positive (call on forward)."""
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(caplet)
        assert greeks["delta"] > 0
    
    def test_gamma_positive(self, caplet):
        """Gamma should be positive (option is convex)."""
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(caplet)
        assert greeks["gamma"] > 0
    
    def test_vega_positive(self, caplet):
        """Vega should be positive."""
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(caplet)
        assert greeks["vega"] > 0


class TestFloorletGreeks:
    """Tests for floorlet Greeks."""
    
    def test_put_delta_negative(self, floorlet):
        """Floorlet delta should be negative (put on forward)."""
        pricer = IrFloorletEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(floorlet)
        assert greeks["delta"] < 0


# =============================================================================
# FINITE DIFFERENCE TESTS
# =============================================================================


class TestFiniteDifferenceGreeks:
    """Finite difference validation for Greeks."""
    
    def test_delta_fd(self, base_params):
        """Delta should match finite difference approximation."""
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        
        # Bump forward rate.
        bump = 1e-4
        
        params_up = {**base_params, "forward_rate": base_params["forward_rate"] + bump}
        params_dn = {**base_params, "forward_rate": base_params["forward_rate"] - bump}
        
        caplet_up = IrCapletEuropeanOptionSimple(**params_up)
        caplet_dn = IrCapletEuropeanOptionSimple(**params_dn)
        caplet_mid = IrCapletEuropeanOptionSimple(**base_params)
        
        pv_up = pricer.price(caplet_up)
        pv_dn = pricer.price(caplet_dn)
        
        fd_delta = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(caplet_mid)
        analytic_delta = greeks["delta"]
        
        # Check within 1% relative error.
        assert abs(fd_delta - analytic_delta) / abs(analytic_delta) < 0.01, \
            f"Delta FD failed: {fd_delta} vs {analytic_delta}"
    
    def test_vega_fd(self, base_params):
        """Vega should match finite difference approximation."""
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        
        bump = 1e-4
        
        params_up = {**base_params, "vol": base_params["vol"] + bump}
        params_dn = {**base_params, "vol": base_params["vol"] - bump}
        
        caplet_up = IrCapletEuropeanOptionSimple(**params_up)
        caplet_dn = IrCapletEuropeanOptionSimple(**params_dn)
        caplet_mid = IrCapletEuropeanOptionSimple(**base_params)
        
        pv_up = pricer.price(caplet_up)
        pv_dn = pricer.price(caplet_dn)
        
        fd_vega = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(caplet_mid)
        analytic_vega = greeks["vega"]
        
        assert abs(fd_vega - analytic_vega) / abs(analytic_vega) < 0.01, \
            f"Vega FD failed: {fd_vega} vs {analytic_vega}"


# =============================================================================
# CAP AND FLOOR TESTS
# =============================================================================


class TestCapPricing:
    """Tests for cap pricing (portfolio of caplets)."""
    
    def test_cap_is_sum_of_caplets(self, base_params):
        """Cap price should be sum of caplet prices."""
        # Create individual caplets.
        caplet1 = IrCapletEuropeanOptionSimple(**base_params)
        
        params2 = {
            **base_params,
            "fixing_time": 1.25,
            "payment_time": 1.5,
        }
        caplet2 = IrCapletEuropeanOptionSimple(**params2)
        
        # Create cap.
        cap = IrCapEuropeanOptionSimple(
            notional=base_params["notional"],
            strike=base_params["strike"],
            caplets=(caplet1, caplet2),
        )
        
        # Price.
        caplet_pricer = IrCapletEuropeanOptionB76PricerSimple()
        cap_pricer = IrCapEuropeanOptionB76PricerSimple()
        
        caplet1_pv = caplet_pricer.price(caplet1)
        caplet2_pv = caplet_pricer.price(caplet2)
        cap_pv = cap_pricer.price(cap)
        
        assert abs(cap_pv - (caplet1_pv + caplet2_pv)) < 1e-6


class TestFloorPricing:
    """Tests for floor pricing (portfolio of floorlets)."""
    
    def test_floor_is_sum_of_floorlets(self, base_params):
        """Floor price should be sum of floorlet prices."""
        floorlet1 = IrFloorletEuropeanOptionSimple(**base_params)
        
        params2 = {**base_params, "fixing_time": 1.25, "payment_time": 1.5}
        floorlet2 = IrFloorletEuropeanOptionSimple(**params2)
        
        floor = IrFloorEuropeanOptionSimple(
            notional=base_params["notional"],
            strike=base_params["strike"],
            floorlets=(floorlet1, floorlet2),
        )
        
        floorlet_pricer = IrFloorletEuropeanOptionB76PricerSimple()
        floor_pricer = IrFloorEuropeanOptionB76PricerSimple()
        
        expected = floorlet_pricer.price(floorlet1) + floorlet_pricer.price(floorlet2)
        actual = floor_pricer.price(floor)
        
        assert abs(actual - expected) < 1e-6


# =============================================================================
# MARKET DATA PRICER TESTS
# =============================================================================


class TestMarketDataPricers:
    """Tests for pricers with market data lookup."""
    
    def test_caplet_market_pricer(self, market, curve_id, vol_id):
        """Market data pricer should produce reasonable price."""
        caplet = IrCapletEuropeanOption(
            notional=1_000_000,
            strike=0.05,
            fixing_time=1.0,
            payment_time=1.25,
            day_count="ACT/360",
            curve_id=curve_id,
            vol_id=vol_id,
        )
        
        pricer = IrCapletEuropeanOptionB76Pricer()
        pv = pricer.price(caplet, market)
        
        assert pv > 0
        assert pv < caplet.notional  # Option price bounded by notional
    
    def test_cap_market_pricer(self, market, curve_id, vol_id):
        """Cap market pricer should produce reasonable price."""
        cap = IrCapEuropeanOption(
            notional=1_000_000,
            strike=0.05,
            start_time=0.25,  # First reset in 3 months
            end_time=2.0,     # 2 year cap
            frequency=0.25,   # Quarterly
            day_count="ACT/360",
            curve_id=curve_id,
            vol_id=vol_id,
        )
        
        pricer = IrCapEuropeanOptionB76Pricer()
        pv = pricer.price(cap, market)
        
        assert pv > 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_atm_caplet(self, base_params):
        """ATM caplet (forward = strike) should have positive value."""
        params = {**base_params, "forward_rate": base_params["strike"]}
        caplet = IrCapletEuropeanOptionSimple(**params)
        
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        pv = pricer.price(caplet)
        
        assert pv > 0
    
    def test_deep_otm_caplet(self, base_params):
        """Deep OTM caplet should have small but non-negative value."""
        params = {**base_params, "forward_rate": 0.02, "strike": 0.10}
        caplet = IrCapletEuropeanOptionSimple(**params)
        
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        pv = pricer.price(caplet)
        
        # Allow for small floating-point errors near zero
        assert pv >= -1e-10
        assert pv < 100  # Very small
    
    def test_deep_itm_caplet(self, base_params):
        """Deep ITM caplet should be close to intrinsic."""
        params = {**base_params, "forward_rate": 0.10, "strike": 0.02}
        caplet = IrCapletEuropeanOptionSimple(**params)
        
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        pv = pricer.price(caplet)
        
        # Intrinsic = DF × τ × N × (F - K)
        intrinsic = params["discount_factor"] * params["accrual_factor"] * \
                    params["notional"] * (0.10 - 0.02)
        
        # Option value should be at least intrinsic (allowing small floating point tolerance)
        assert pv >= intrinsic - 1e-6
        # For deep ITM, price is close to intrinsic (within 20% due to time value)
        assert abs(pv - intrinsic) / intrinsic < 0.2
    
    def test_zero_vol_caplet(self, base_params):
        """Zero vol caplet should equal discounted intrinsic."""
        params = {**base_params, "vol": 0.0}
        caplet = IrCapletEuropeanOptionSimple(**params)
        
        pricer = IrCapletEuropeanOptionB76PricerSimple()
        pv = pricer.price(caplet)
        
        # With zero vol, price = intrinsic
        F = params["forward_rate"]
        K = params["strike"]
        intrinsic = params["discount_factor"] * params["accrual_factor"] * \
                    params["notional"] * max(F - K, 0)
        
        assert abs(pv - intrinsic) < 1e-6


# =============================================================================
# ACCRUAL FACTOR TESTS
# =============================================================================


class TestAccrualFactor:
    """Tests for accrual factor computation."""
    
    def test_act_360(self):
        """ACT/360 should scale period appropriately."""
        tau = compute_accrual_factor(0.0, 0.25, "ACT/360")
        # 0.25 years × (365/360) ≈ 0.2535
        assert abs(tau - 0.25 * 365 / 360) < 1e-6
    
    def test_act_365(self):
        """ACT/365 should return period directly."""
        tau = compute_accrual_factor(0.0, 0.25, "ACT/365")
        assert abs(tau - 0.25) < 1e-6
