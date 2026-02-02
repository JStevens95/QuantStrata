# tests/unit/pricers/ir/test_ir_bond_pricer.py
"""
Unit tests for IR Bond Instruments and Pricers.

Tests cover:
- Zero coupon bond pricing and risk measures
- Fixed rate coupon bond pricing and risk measures
- Bond option pricing using Black76
- Put-call parity for bond options
- Finite difference validation of Greeks

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import pytest

from src.instruments.ir.linear.bond import (
    IrBondZeroCoupon,
    IrBondZeroCouponSimple,
    IrBondFixedRate,
    IrBondFixedRateSimple,
)
from src.instruments.ir.options.bond import (
    IrBondEuropeanOption,
    IrBondEuropeanOptionSimple,
)
from src.pricers.ir.bond import (
    IrBondZeroCouponPricer,
    IrBondZeroCouponPricerSimple,
    IrBondFixedRatePricer,
    IrBondFixedRatePricerSimple,
)
from src.pricers.ir.european_b76 import (
    IrBondEuropeanOptionB76Pricer,
    IrBondEuropeanOptionB76PricerSimple,
)


# =============================================================================
# ZERO COUPON BOND FIXTURES
# =============================================================================


@pytest.fixture
def zc_bond_params():
    """Base parameters for a zero coupon bond."""
    return {
        "face_value": 100.0,
        "maturity": 5.0,
        "discount_factor": 0.85,
    }


@pytest.fixture
def zc_bond(zc_bond_params):
    """Create a zero coupon bond."""
    return IrBondZeroCouponSimple(**zc_bond_params)


# =============================================================================
# FIXED RATE BOND FIXTURES
# =============================================================================


@pytest.fixture
def fr_bond_params():
    """Base parameters for a fixed rate coupon bond."""
    return {
        "face_value": 100.0,
        "coupon_rate": 0.05,  # 5% annual coupon
        "coupon_times": (1.0, 2.0, 3.0, 4.0, 5.0),
        "coupon_dfs": (0.97, 0.94, 0.91, 0.88, 0.85),
        "accrued_interest": 0.0,
    }


@pytest.fixture
def fr_bond(fr_bond_params):
    """Create a fixed rate coupon bond."""
    return IrBondFixedRateSimple(**fr_bond_params)


# =============================================================================
# BOND OPTION FIXTURES
# =============================================================================


@pytest.fixture
def bond_option_params():
    """Base parameters for a bond option."""
    return {
        "notional": 1_000_000,
        "strike": 102.0,
        "expiry": 0.5,
        "forward_bond_price": 103.5,
        "vol": 0.08,
        "discount_factor": 0.975,
        "option_type": "call",
    }


@pytest.fixture
def call_bond_option(bond_option_params):
    """Create a bond call option."""
    return IrBondEuropeanOptionSimple(**bond_option_params)


@pytest.fixture
def put_bond_option(bond_option_params):
    """Create a bond put option."""
    params = {**bond_option_params, "option_type": "put"}
    return IrBondEuropeanOptionSimple(**params)


# =============================================================================
# ZERO COUPON BOND VALIDATION TESTS
# =============================================================================


class TestZeroCouponBondValidation:
    """Tests for zero coupon bond instrument validation."""
    
    def test_valid_zc_bond(self, zc_bond_params):
        """Valid zero coupon bond should be created without errors."""
        bond = IrBondZeroCouponSimple(**zc_bond_params)
        assert bond.face_value == zc_bond_params["face_value"]
        assert bond.maturity == zc_bond_params["maturity"]
    
    def test_negative_face_raises(self, zc_bond_params):
        """Negative face value should raise."""
        zc_bond_params["face_value"] = -100.0
        with pytest.raises(ValueError, match="face_value must be > 0"):
            IrBondZeroCouponSimple(**zc_bond_params)
    
    def test_negative_maturity_raises(self, zc_bond_params):
        """Negative maturity should raise."""
        zc_bond_params["maturity"] = -1.0
        with pytest.raises(ValueError, match="maturity must be >= 0"):
            IrBondZeroCouponSimple(**zc_bond_params)
    
    def test_negative_df_raises(self, zc_bond_params):
        """Negative discount factor should raise."""
        zc_bond_params["discount_factor"] = -0.1
        with pytest.raises(ValueError, match="discount_factor must be > 0"):
            IrBondZeroCouponSimple(**zc_bond_params)
    
    def test_implied_zero_rate(self, zc_bond):
        """Test implied zero rate calculation."""
        # DF = exp(-rT) => r = -ln(DF)/T
        expected_rate = -math.log(0.85) / 5.0
        assert abs(zc_bond.implied_zero_rate - expected_rate) < 1e-10


# =============================================================================
# ZERO COUPON BOND PRICING TESTS
# =============================================================================


class TestZeroCouponBondPricing:
    """Tests for zero coupon bond pricing."""
    
    def test_zc_bond_price(self, zc_bond):
        """Zero coupon bond PV = Face × DF."""
        pricer = IrBondZeroCouponPricerSimple()
        pv = pricer.price(zc_bond)
        
        expected = 100.0 * 0.85
        assert abs(pv - expected) < 1e-10
    
    def test_zc_bond_greeks(self, zc_bond):
        """Zero coupon bond risk measures."""
        pricer = IrBondZeroCouponPricerSimple()
        greeks = pricer.greeks(zc_bond)
        
        assert "dv01" in greeks
        assert "modified_duration" in greeks
        assert "macaulay_duration" in greeks
        assert "convexity" in greeks
        
        # Macaulay duration = maturity for zero coupon
        assert abs(greeks["macaulay_duration"] - 5.0) < 1e-10
        
        # DV01 > 0
        assert greeks["dv01"] > 0
    
    def test_shorter_maturity_lower_duration(self, zc_bond_params):
        """Shorter maturity should have lower duration."""
        pricer = IrBondZeroCouponPricerSimple()
        
        bond_long = IrBondZeroCouponSimple(**zc_bond_params)
        
        short_params = {**zc_bond_params, "maturity": 2.0, "discount_factor": 0.94}
        bond_short = IrBondZeroCouponSimple(**short_params)
        
        dur_long = pricer.greeks(bond_long)["macaulay_duration"]
        dur_short = pricer.greeks(bond_short)["macaulay_duration"]
        
        assert dur_short < dur_long


# =============================================================================
# FIXED RATE BOND VALIDATION TESTS
# =============================================================================


class TestFixedRateBondValidation:
    """Tests for fixed rate bond instrument validation."""
    
    def test_valid_fr_bond(self, fr_bond_params):
        """Valid fixed rate bond should be created without errors."""
        bond = IrBondFixedRateSimple(**fr_bond_params)
        assert bond.face_value == fr_bond_params["face_value"]
        assert bond.coupon_rate == fr_bond_params["coupon_rate"]
    
    def test_negative_coupon_raises(self, fr_bond_params):
        """Negative coupon rate should raise."""
        fr_bond_params["coupon_rate"] = -0.05
        with pytest.raises(ValueError, match="coupon_rate must be >= 0"):
            IrBondFixedRateSimple(**fr_bond_params)
    
    def test_empty_coupons_raises(self, fr_bond_params):
        """Empty coupon times should raise."""
        fr_bond_params["coupon_times"] = ()
        fr_bond_params["coupon_dfs"] = ()
        with pytest.raises(ValueError, match="coupon_times must have at least one"):
            IrBondFixedRateSimple(**fr_bond_params)
    
    def test_mismatched_times_dfs_raises(self, fr_bond_params):
        """Mismatched coupon times and dfs should raise."""
        fr_bond_params["coupon_dfs"] = (0.97, 0.94)  # Too few
        with pytest.raises(ValueError, match="same length"):
            IrBondFixedRateSimple(**fr_bond_params)
    
    def test_maturity_property(self, fr_bond):
        """Maturity should be last coupon time."""
        assert fr_bond.maturity == 5.0
    
    def test_n_coupons_property(self, fr_bond):
        """Number of coupons should match coupon_times length."""
        assert fr_bond.n_coupons == 5


# =============================================================================
# FIXED RATE BOND PRICING TESTS
# =============================================================================


class TestFixedRateBondPricing:
    """Tests for fixed rate bond pricing."""
    
    def test_fr_bond_price(self, fr_bond):
        """Fixed rate bond PV = Σ(C × DF) + Face × DF_maturity."""
        pricer = IrBondFixedRatePricerSimple()
        pv = pricer.price(fr_bond)
        
        # Manual calculation
        face = 100.0
        coupon = 5.0  # 5% of 100
        dfs = [0.97, 0.94, 0.91, 0.88, 0.85]
        
        expected = sum(coupon * df for df in dfs) + face * dfs[-1]
        
        assert abs(pv - expected) < 1e-10
    
    def test_clean_price(self, fr_bond_params):
        """Clean price should exclude accrued interest."""
        pricer = IrBondFixedRatePricerSimple()
        
        # Bond without accrued
        bond_no_accrued = IrBondFixedRateSimple(**fr_bond_params)
        dirty_no_accrued = pricer.price(bond_no_accrued)
        clean_no_accrued = pricer.clean_price(bond_no_accrued)
        
        assert abs(dirty_no_accrued - clean_no_accrued) < 1e-10
        
        # Bond with accrued
        params_accrued = {**fr_bond_params, "accrued_interest": 2.5}
        bond_accrued = IrBondFixedRateSimple(**params_accrued)
        
        dirty = pricer.price(bond_accrued)
        clean = pricer.clean_price(bond_accrued)
        
        assert abs(dirty - clean - 2.5) < 1e-10
    
    def test_fr_bond_greeks(self, fr_bond):
        """Fixed rate bond risk measures."""
        pricer = IrBondFixedRatePricerSimple()
        greeks = pricer.greeks(fr_bond)
        
        assert "dv01" in greeks
        assert "modified_duration" in greeks
        assert "macaulay_duration" in greeks
        assert "convexity" in greeks
        
        # Duration should be less than maturity for coupon bond
        assert greeks["macaulay_duration"] < 5.0
        assert greeks["macaulay_duration"] > 0
        
        # DV01 > 0
        assert greeks["dv01"] > 0
    
    def test_higher_coupon_lower_duration(self, fr_bond_params):
        """Higher coupon should result in lower duration."""
        pricer = IrBondFixedRatePricerSimple()
        
        low_coupon_params = {**fr_bond_params, "coupon_rate": 0.02}
        low_coupon_bond = IrBondFixedRateSimple(**low_coupon_params)
        
        high_coupon_params = {**fr_bond_params, "coupon_rate": 0.08}
        high_coupon_bond = IrBondFixedRateSimple(**high_coupon_params)
        
        dur_low = pricer.greeks(low_coupon_bond)["macaulay_duration"]
        dur_high = pricer.greeks(high_coupon_bond)["macaulay_duration"]
        
        assert dur_high < dur_low


# =============================================================================
# BOND OPTION VALIDATION TESTS
# =============================================================================


class TestBondOptionValidation:
    """Tests for bond option instrument validation."""
    
    def test_valid_bond_option(self, bond_option_params):
        """Valid bond option should be created without errors."""
        opt = IrBondEuropeanOptionSimple(**bond_option_params)
        assert opt.notional == bond_option_params["notional"]
        assert opt.strike == bond_option_params["strike"]
    
    def test_zero_notional_raises(self, bond_option_params):
        """Zero notional should raise."""
        bond_option_params["notional"] = 0.0
        with pytest.raises(ValueError, match="notional must be non-zero"):
            IrBondEuropeanOptionSimple(**bond_option_params)
    
    def test_negative_strike_raises(self, bond_option_params):
        """Negative strike should raise."""
        bond_option_params["strike"] = -10.0
        with pytest.raises(ValueError, match="strike must be > 0"):
            IrBondEuropeanOptionSimple(**bond_option_params)
    
    def test_negative_forward_raises(self, bond_option_params):
        """Negative forward bond price should raise."""
        bond_option_params["forward_bond_price"] = -100.0
        with pytest.raises(ValueError, match="forward_bond_price must be > 0"):
            IrBondEuropeanOptionSimple(**bond_option_params)
    
    def test_is_itm_call(self, call_bond_option):
        """Call is ITM when F > K."""
        assert call_bond_option.is_in_the_money  # 103.5 > 102.0
    
    def test_is_itm_put(self, bond_option_params):
        """Put is ITM when F < K."""
        params = {**bond_option_params, "option_type": "put", "forward_bond_price": 100.0}
        put = IrBondEuropeanOptionSimple(**params)
        assert put.is_in_the_money  # 100.0 < 102.0
    
    def test_moneyness(self, call_bond_option):
        """Moneyness = F/K."""
        expected = 103.5 / 102.0
        assert abs(call_bond_option.moneyness - expected) < 1e-10


# =============================================================================
# BOND OPTION PRICING TESTS
# =============================================================================


class TestBondOptionPricing:
    """Tests for bond option pricing using Black76."""
    
    def test_call_price_positive(self, call_bond_option):
        """Call option price should be positive."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        pv = pricer.price(call_bond_option)
        assert pv > 0
    
    def test_put_price_positive(self, put_bond_option):
        """Put option price should be positive."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        pv = pricer.price(put_bond_option)
        assert pv > 0
    
    def test_itm_higher_than_otm(self, bond_option_params):
        """ITM option should be more valuable than OTM."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        
        # ITM call: F > K
        itm_params = {**bond_option_params, "forward_bond_price": 110.0, "strike": 100.0}
        itm = IrBondEuropeanOptionSimple(**itm_params)
        
        # OTM call: F < K
        otm_params = {**bond_option_params, "forward_bond_price": 95.0, "strike": 100.0}
        otm = IrBondEuropeanOptionSimple(**otm_params)
        
        assert pricer.price(itm) > pricer.price(otm)
    
    def test_price_scales_with_notional(self, bond_option_params):
        """Price should scale linearly with notional."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        
        opt1 = IrBondEuropeanOptionSimple(**bond_option_params)
        
        params2 = {**bond_option_params, "notional": 2_000_000}
        opt2 = IrBondEuropeanOptionSimple(**params2)
        
        pv1 = pricer.price(opt1)
        pv2 = pricer.price(opt2)
        
        assert abs(pv2 - 2 * pv1) < 1e-6


# =============================================================================
# PUT-CALL PARITY TESTS
# =============================================================================


class TestBondOptionPutCallParity:
    """Tests for bond option put-call parity."""
    
    def test_put_call_parity(self, bond_option_params):
        """
        Put-call parity: Call - Put = DF × (F - K)
        """
        call = IrBondEuropeanOptionSimple(**bond_option_params)
        put_params = {**bond_option_params, "option_type": "put"}
        put = IrBondEuropeanOptionSimple(**put_params)
        
        pricer = IrBondEuropeanOptionB76PricerSimple()
        
        call_pv = pricer.price(call)
        put_pv = pricer.price(put)
        
        N = bond_option_params["notional"]
        df = bond_option_params["discount_factor"]
        F = bond_option_params["forward_bond_price"]
        K = bond_option_params["strike"]
        
        expected_diff = N * df * (F - K)
        actual_diff = call_pv - put_pv
        
        assert abs(actual_diff - expected_diff) < 1e-6, \
            f"Parity failed: {actual_diff} != {expected_diff}"


# =============================================================================
# BOND OPTION GREEKS TESTS
# =============================================================================


class TestBondOptionGreeks:
    """Tests for bond option Greeks."""
    
    def test_greeks_exist(self, call_bond_option):
        """Greeks should be computed."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(call_bond_option)
        
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks
    
    def test_call_delta_positive(self, call_bond_option):
        """Call delta should be positive."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(call_bond_option)
        assert greeks["delta"] > 0
    
    def test_put_delta_negative(self, put_bond_option):
        """Put delta should be negative."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(put_bond_option)
        assert greeks["delta"] < 0
    
    def test_gamma_positive(self, call_bond_option):
        """Gamma should be positive."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(call_bond_option)
        assert greeks["gamma"] > 0
    
    def test_vega_positive(self, call_bond_option):
        """Vega should be positive."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        greeks = pricer.greeks(call_bond_option)
        assert greeks["vega"] > 0


# =============================================================================
# FINITE DIFFERENCE VALIDATION TESTS
# =============================================================================


class TestFiniteDifferenceGreeks:
    """Finite difference validation for Greeks."""
    
    def test_delta_fd(self, bond_option_params):
        """Delta should match finite difference approximation."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        
        bump = 1e-4
        
        params_up = {**bond_option_params, "forward_bond_price": bond_option_params["forward_bond_price"] + bump}
        params_dn = {**bond_option_params, "forward_bond_price": bond_option_params["forward_bond_price"] - bump}
        
        opt_up = IrBondEuropeanOptionSimple(**params_up)
        opt_dn = IrBondEuropeanOptionSimple(**params_dn)
        opt_mid = IrBondEuropeanOptionSimple(**bond_option_params)
        
        pv_up = pricer.price(opt_up)
        pv_dn = pricer.price(opt_dn)
        
        fd_delta = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(opt_mid)
        analytic_delta = greeks["delta"]
        
        assert abs(fd_delta - analytic_delta) / abs(analytic_delta) < 0.01
    
    def test_vega_fd(self, bond_option_params):
        """Vega should match finite difference approximation."""
        pricer = IrBondEuropeanOptionB76PricerSimple()
        
        bump = 1e-5
        
        params_up = {**bond_option_params, "vol": bond_option_params["vol"] + bump}
        params_dn = {**bond_option_params, "vol": bond_option_params["vol"] - bump}
        
        opt_up = IrBondEuropeanOptionSimple(**params_up)
        opt_dn = IrBondEuropeanOptionSimple(**params_dn)
        opt_mid = IrBondEuropeanOptionSimple(**bond_option_params)
        
        pv_up = pricer.price(opt_up)
        pv_dn = pricer.price(opt_dn)
        
        fd_vega = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(opt_mid)
        analytic_vega = greeks["vega"]
        
        assert abs(fd_vega - analytic_vega) / abs(analytic_vega) < 0.01


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestBondOptionEdgeCases:
    """Tests for edge cases."""
    
    def test_atm_option(self, bond_option_params):
        """ATM option should have positive value."""
        params = {**bond_option_params, "forward_bond_price": bond_option_params["strike"]}
        opt = IrBondEuropeanOptionSimple(**params)
        
        pricer = IrBondEuropeanOptionB76PricerSimple()
        pv = pricer.price(opt)
        
        assert pv > 0
    
    def test_zero_vol_option(self, bond_option_params):
        """Zero vol option should equal intrinsic."""
        params = {**bond_option_params, "vol": 0.0}
        opt = IrBondEuropeanOptionSimple(**params)
        
        pricer = IrBondEuropeanOptionB76PricerSimple()
        pv = pricer.price(opt)
        
        # Intrinsic for call = N × DF × max(F - K, 0)
        N = params["notional"]
        df = params["discount_factor"]
        F = params["forward_bond_price"]
        K = params["strike"]
        intrinsic = N * df * max(F - K, 0)
        
        assert abs(pv - intrinsic) < 1e-6
    
    def test_expired_option(self, bond_option_params):
        """Expired option should return intrinsic value."""
        params = {**bond_option_params, "expiry": 0.0}
        opt = IrBondEuropeanOptionSimple(**params)
        
        pricer = IrBondEuropeanOptionB76PricerSimple()
        pv = pricer.price(opt)
        
        N = params["notional"]
        df = params["discount_factor"]
        F = params["forward_bond_price"]
        K = params["strike"]
        expected = N * df * max(F - K, 0)
        
        assert abs(pv - expected) < 1e-6
