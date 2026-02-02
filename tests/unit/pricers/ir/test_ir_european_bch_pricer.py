# tests/unit/pricers/ir/test_ir_european_bch_pricer.py
"""
Unit tests for IR European Bachelier Pricers (Swaptions).

Tests cover:
- Swaption pricing using Bachelier model
- Payer vs receiver swaptions
- Put-call parity
- Greeks computation
- Finite difference validation
- Edge cases

Author: QuantStrata Team
"""
from __future__ import annotations

import pytest

from src.instruments.ir.options.swaption import (
    IrSwaptionEuropeanOption,
    IrSwaptionEuropeanOptionSimple,
)
from src.pricers.ir.european_bch import (
    IrSwaptionEuropeanOptionBchPricer,
    IrSwaptionEuropeanOptionBchPricerSimple,
)
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def swaption_params():
    """Base parameters for a swaption."""
    return {
        "notional": 10_000_000,
        "strike": 0.04,               # 4% strike
        "option_expiry": 1.0,         # 1 year to expiry
        "swap_tenor": 5.0,            # 5 year underlying swap
        "forward_swap_rate": 0.042,   # 4.2% forward (slightly ITM for payer)
        "annuity": 4.5,               # PV01 of underlying
        "vol": 0.0060,                # 60bp normal vol
        "discount_factor": 0.95,
        "swaption_type": "payer",
    }


@pytest.fixture
def payer_swaption(swaption_params):
    """Create a payer swaption."""
    return IrSwaptionEuropeanOptionSimple(**swaption_params)


@pytest.fixture
def receiver_swaption(swaption_params):
    """Create a receiver swaption."""
    params = {**swaption_params, "swaption_type": "receiver"}
    return IrSwaptionEuropeanOptionSimple(**params)


@pytest.fixture
def curve_id():
    """Standard curve ID."""
    return MarketId("IR", "CURVE", "USD.SOFR", (("ccy", "USD"),))


@pytest.fixture
def vol_id():
    """Standard vol ID."""
    return MarketId("IR", "VOL", "USD.SWAPTION", (("ccy", "USD"),))


@pytest.fixture
def market(curve_id, vol_id):
    """Create a simple market for testing."""
    return Market(
        asof="2026-01-15",
        quotes={},
        curves={
            curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.04),
        },
        vols={
            vol_id: FlatVolSurface(sigma=0.0060),  # 60bp normal vol
        },
    )


# =============================================================================
# INSTRUMENT VALIDATION TESTS
# =============================================================================


class TestSwaptionValidation:
    """Tests for swaption instrument validation."""
    
    def test_valid_swaption(self, swaption_params):
        """Valid swaption should be created without errors."""
        swaption = IrSwaptionEuropeanOptionSimple(**swaption_params)
        assert swaption.notional == swaption_params["notional"]
        assert swaption.strike == swaption_params["strike"]
    
    def test_zero_notional_raises(self, swaption_params):
        """Zero notional should raise."""
        swaption_params["notional"] = 0.0
        with pytest.raises(ValueError, match="notional must be non-zero"):
            IrSwaptionEuropeanOptionSimple(**swaption_params)
    
    def test_negative_expiry_raises(self, swaption_params):
        """Negative expiry should raise."""
        swaption_params["option_expiry"] = -0.1
        with pytest.raises(ValueError, match="option_expiry must be >= 0"):
            IrSwaptionEuropeanOptionSimple(**swaption_params)
    
    def test_tenor_description(self, payer_swaption):
        """Test tenor description."""
        assert payer_swaption.tenor_description == "1Y5Y"
    
    def test_is_itm_payer(self, payer_swaption):
        """Payer is ITM when forward > strike."""
        assert payer_swaption.is_in_the_money  # 4.2% > 4%
    
    def test_is_otm_receiver(self, receiver_swaption):
        """Receiver is OTM when forward > strike."""
        assert not receiver_swaption.is_in_the_money


# =============================================================================
# BASIC PRICING TESTS
# =============================================================================


class TestSwaptionPricing:
    """Tests for swaption pricing."""
    
    def test_payer_price_positive(self, payer_swaption):
        """Payer swaption price should be positive."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        pv = pricer.price(payer_swaption)
        assert pv > 0
    
    def test_receiver_price_positive(self, receiver_swaption):
        """Receiver swaption price should be positive."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        pv = pricer.price(receiver_swaption)
        assert pv > 0
    
    def test_itm_higher_than_otm(self, swaption_params):
        """ITM swaption should be more valuable than OTM."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        
        # ITM payer: forward > strike
        itm_params = {**swaption_params, "forward_swap_rate": 0.06, "strike": 0.04}
        itm = IrSwaptionEuropeanOptionSimple(**itm_params)
        
        # OTM payer: forward < strike
        otm_params = {**swaption_params, "forward_swap_rate": 0.03, "strike": 0.04}
        otm = IrSwaptionEuropeanOptionSimple(**otm_params)
        
        assert pricer.price(itm) > pricer.price(otm)
    
    def test_price_scales_with_notional(self, swaption_params):
        """Price should scale linearly with notional."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        
        sw1 = IrSwaptionEuropeanOptionSimple(**swaption_params)
        
        params2 = {**swaption_params, "notional": 20_000_000}
        sw2 = IrSwaptionEuropeanOptionSimple(**params2)
        
        pv1 = pricer.price(sw1)
        pv2 = pricer.price(sw2)
        
        assert abs(pv2 - 2 * pv1) < 1e-6
    
    def test_price_scales_with_annuity(self, swaption_params):
        """Price should scale linearly with annuity."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        
        sw1 = IrSwaptionEuropeanOptionSimple(**swaption_params)
        
        params2 = {**swaption_params, "annuity": 9.0}  # Double annuity
        sw2 = IrSwaptionEuropeanOptionSimple(**params2)
        
        pv1 = pricer.price(sw1)
        pv2 = pricer.price(sw2)
        
        assert abs(pv2 - 2 * pv1) < 1e-6


# =============================================================================
# PUT-CALL PARITY TESTS
# =============================================================================


class TestPutCallParity:
    """Tests for payer-receiver parity."""
    
    def test_payer_receiver_parity(self, swaption_params):
        """
        Payer - Receiver = Forward value of swap.
        
        For swaptions:
        Payer - Receiver = A × N × (F - K)
        """
        payer = IrSwaptionEuropeanOptionSimple(**swaption_params)
        receiver_params = {**swaption_params, "swaption_type": "receiver"}
        receiver = IrSwaptionEuropeanOptionSimple(**receiver_params)
        
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        
        payer_pv = pricer.price(payer)
        receiver_pv = pricer.price(receiver)
        
        # Expected: A × N × (F - K)
        N = swaption_params["notional"]
        A = swaption_params["annuity"]
        F = swaption_params["forward_swap_rate"]
        K = swaption_params["strike"]
        
        expected_diff = A * N * (F - K)
        actual_diff = payer_pv - receiver_pv
        
        assert abs(actual_diff - expected_diff) < 1e-6, \
            f"Parity failed: {actual_diff} != {expected_diff}"


# =============================================================================
# GREEKS TESTS
# =============================================================================


class TestSwaptionGreeks:
    """Tests for swaption Greeks."""
    
    def test_greeks_exist(self, payer_swaption):
        """Greeks should be computed."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(payer_swaption)
        
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks
    
    def test_payer_delta_positive(self, payer_swaption):
        """Payer swaption delta should be positive."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(payer_swaption)
        assert greeks["delta"] > 0
    
    def test_receiver_delta_negative(self, receiver_swaption):
        """Receiver swaption delta should be negative."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(receiver_swaption)
        assert greeks["delta"] < 0
    
    def test_gamma_positive(self, payer_swaption):
        """Gamma should be positive."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(payer_swaption)
        assert greeks["gamma"] > 0
    
    def test_vega_positive(self, payer_swaption):
        """Vega should be positive."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(payer_swaption)
        assert greeks["vega"] > 0
    
    def test_vega_bp(self, payer_swaption):
        """Vega per bp should be reasonable."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        vega_bp = pricer.vega_bp(payer_swaption)
        
        # Vega per bp should be smaller than vega per 1.0 vol
        greeks = pricer.greeks(payer_swaption)
        assert vega_bp == greeks["vega"] * 0.0001


# =============================================================================
# FINITE DIFFERENCE TESTS
# =============================================================================


class TestFiniteDifferenceGreeks:
    """Finite difference validation for Greeks."""
    
    def test_delta_fd(self, swaption_params):
        """Delta should match finite difference approximation."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        
        bump = 1e-5
        
        params_up = {**swaption_params, "forward_swap_rate": swaption_params["forward_swap_rate"] + bump}
        params_dn = {**swaption_params, "forward_swap_rate": swaption_params["forward_swap_rate"] - bump}
        
        sw_up = IrSwaptionEuropeanOptionSimple(**params_up)
        sw_dn = IrSwaptionEuropeanOptionSimple(**params_dn)
        sw_mid = IrSwaptionEuropeanOptionSimple(**swaption_params)
        
        pv_up = pricer.price(sw_up)
        pv_dn = pricer.price(sw_dn)
        
        fd_delta = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(sw_mid)
        analytic_delta = greeks["delta"]
        
        assert abs(fd_delta - analytic_delta) / abs(analytic_delta) < 0.01
    
    def test_vega_fd(self, swaption_params):
        """Vega should match finite difference approximation."""
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        
        bump = 1e-6
        
        params_up = {**swaption_params, "vol": swaption_params["vol"] + bump}
        params_dn = {**swaption_params, "vol": swaption_params["vol"] - bump}
        
        sw_up = IrSwaptionEuropeanOptionSimple(**params_up)
        sw_dn = IrSwaptionEuropeanOptionSimple(**params_dn)
        sw_mid = IrSwaptionEuropeanOptionSimple(**swaption_params)
        
        pv_up = pricer.price(sw_up)
        pv_dn = pricer.price(sw_dn)
        
        fd_vega = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(sw_mid)
        analytic_vega = greeks["vega"]
        
        assert abs(fd_vega - analytic_vega) / abs(analytic_vega) < 0.01


# =============================================================================
# MARKET DATA PRICER TESTS
# =============================================================================


class TestMarketDataPricer:
    """Tests for pricer with market data lookup."""
    
    def test_swaption_market_pricer(self, market, curve_id, vol_id):
        """Market data pricer should produce reasonable price."""
        swaption = IrSwaptionEuropeanOption(
            notional=10_000_000,
            strike=0.04,
            option_expiry=1.0,
            swap_start=1.0,
            swap_end=6.0,
            swaption_type="payer",
            curve_id=curve_id,
            vol_id=vol_id,
        )
        
        pricer = IrSwaptionEuropeanOptionBchPricer()
        pv = pricer.price(swaption, market)
        
        assert pv > 0
        # Price should be reasonable (less than 10% of notional)
        assert pv < swaption.notional * 0.1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_atm_swaption(self, swaption_params):
        """ATM swaption should have positive value."""
        params = {**swaption_params, "forward_swap_rate": swaption_params["strike"]}
        swaption = IrSwaptionEuropeanOptionSimple(**params)
        
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        pv = pricer.price(swaption)
        
        assert pv > 0
    
    def test_zero_vol_swaption(self, swaption_params):
        """Zero vol swaption should equal intrinsic."""
        params = {**swaption_params, "vol": 0.0}
        swaption = IrSwaptionEuropeanOptionSimple(**params)
        
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        pv = pricer.price(swaption)
        
        # Intrinsic for payer = A × N × max(F - K, 0)
        A = params["annuity"]
        N = params["notional"]
        F = params["forward_swap_rate"]
        K = params["strike"]
        intrinsic = A * N * max(F - K, 0)
        
        assert abs(pv - intrinsic) < 1e-6
    
    def test_negative_strike(self, swaption_params):
        """Bachelier handles negative strikes (for negative rate environments)."""
        params = {**swaption_params, "strike": -0.01, "forward_swap_rate": -0.005}
        swaption = IrSwaptionEuropeanOptionSimple(**params)
        
        pricer = IrSwaptionEuropeanOptionBchPricerSimple()
        pv = pricer.price(swaption)
        
        # Should still produce valid positive price
        assert pv > 0
