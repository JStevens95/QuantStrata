# tests/unit/pricers/ir/test_ir_linear_pricer.py
"""
Unit tests for Interest Rate Linear Pricers (FRA and IRS).

Tests cover:
- FRA pricing and Greeks
- IRS pricing and Greeks
- Par rate calculations
- Payer vs receiver sign conventions
- Market data integration
- Edge cases

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import pytest

from src.instruments.ir.linear.fra import (
    ForwardRateAgreement,
    ForwardRateAgreementSimple,
)
from src.instruments.ir.linear.swap import (
    InterestRateSwap,
    InterestRateSwapSimple,
    FixedLeg,
    FloatingLeg,
    generate_swap_schedule,
)
from src.pricers.ir.linear import (
    FRAPricer,
    FRAPricerSimple,
    IRSwapPricer,
    IRSwapPricerSimple,
)
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def fra_params():
    """Base parameters for a FRA."""
    return {
        "notional": 10_000_000,
        "fixed_rate": 0.05,         # 5% contract rate
        "fixing_time": 0.25,        # 3 months to fixing
        "payment_time": 0.5,        # 6 months to payment
        "accrual_factor": 0.25,     # Quarterly
        "forward_rate": 0.052,      # 5.2% forward (slightly higher)
        "discount_factor": 0.975,
        "direction": "payer",
    }


@pytest.fixture
def payer_fra(fra_params):
    """Create a payer FRA (pay fixed, receive floating)."""
    return ForwardRateAgreementSimple(**fra_params)


@pytest.fixture
def receiver_fra(fra_params):
    """Create a receiver FRA (receive fixed, pay floating)."""
    params = {**fra_params, "direction": "receiver"}
    return ForwardRateAgreementSimple(**params)


@pytest.fixture
def swap_legs():
    """Create swap leg fixtures for a 2-year annual swap."""
    fixed_legs = (
        FixedLeg(
            start_time=0.0, end_time=1.0, accrual_factor=1.0,
            discount_factor=0.95, notional=1_000_000, fixed_rate=0.05,
        ),
        FixedLeg(
            start_time=1.0, end_time=2.0, accrual_factor=1.0,
            discount_factor=0.90, notional=1_000_000, fixed_rate=0.05,
        ),
    )
    
    floating_legs = (
        FloatingLeg(
            start_time=0.0, end_time=1.0, accrual_factor=1.0,
            discount_factor=0.95, notional=1_000_000, forward_rate=0.048,
        ),
        FloatingLeg(
            start_time=1.0, end_time=2.0, accrual_factor=1.0,
            discount_factor=0.90, notional=1_000_000, forward_rate=0.052,
        ),
    )
    
    return fixed_legs, floating_legs


@pytest.fixture
def receiver_swap(swap_legs):
    """Create a receiver swap (receive fixed, pay floating)."""
    fixed_legs, floating_legs = swap_legs
    return InterestRateSwapSimple(
        notional=1_000_000,
        fixed_rate=0.05,
        fixed_leg=fixed_legs,
        floating_leg=floating_legs,
        direction="receiver",
    )


@pytest.fixture
def payer_swap(swap_legs):
    """Create a payer swap (pay fixed, receive floating)."""
    fixed_legs, floating_legs = swap_legs
    return InterestRateSwapSimple(
        notional=1_000_000,
        fixed_rate=0.05,
        fixed_leg=fixed_legs,
        floating_leg=floating_legs,
        direction="payer",
    )


@pytest.fixture
def curve_id():
    """Standard curve ID."""
    return MarketId("IR", "CURVE", "USD.SOFR", (("ccy", "USD"),))


@pytest.fixture
def market(curve_id):
    """Create a simple market for testing."""
    return Market(
        asof="2026-01-15",
        quotes={},
        curves={
            curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05),
        },
        vols={},
    )


# =============================================================================
# FRA VALIDATION TESTS
# =============================================================================


class TestFRAValidation:
    """Tests for FRA instrument validation."""
    
    def test_valid_fra(self, fra_params):
        """Valid FRA should be created without errors."""
        fra = ForwardRateAgreementSimple(**fra_params)
        assert fra.notional == fra_params["notional"]
        assert fra.fixed_rate == fra_params["fixed_rate"]
    
    def test_zero_notional_raises(self, fra_params):
        """Zero notional should raise."""
        fra_params["notional"] = 0.0
        with pytest.raises(ValueError, match="notional must be non-zero"):
            ForwardRateAgreementSimple(**fra_params)
    
    def test_payment_before_fixing_raises(self, fra_params):
        """Payment time before fixing time should raise."""
        fra_params["payment_time"] = 0.1  # Before fixing at 0.25
        with pytest.raises(ValueError, match="payment_time must be > fixing_time"):
            ForwardRateAgreementSimple(**fra_params)
    
    def test_invalid_direction_raises(self, fra_params):
        """Invalid direction should raise."""
        fra_params["direction"] = "invalid"
        with pytest.raises(ValueError, match="direction must be"):
            ForwardRateAgreementSimple(**fra_params)
    
    def test_tenor_description(self, curve_id):
        """Test tenor description generation."""
        fra = ForwardRateAgreement(
            notional=10_000_000,
            fixed_rate=0.05,
            fixing_time=0.25,   # 3 months
            payment_time=0.5,   # 6 months
            curve_id=curve_id,
        )
        assert fra.tenor_description == "3x6"


# =============================================================================
# FRA PRICING TESTS
# =============================================================================


class TestFRAPricing:
    """Tests for FRA pricing."""
    
    def test_payer_fra_positive_when_rates_rise(self, payer_fra):
        """Payer FRA should have positive PV when forward > fixed."""
        # Forward (5.2%) > Fixed (5%), so payer benefits.
        pricer = FRAPricerSimple()
        pv = pricer.price(payer_fra)
        assert pv > 0
    
    def test_receiver_fra_negative_when_rates_rise(self, receiver_fra):
        """Receiver FRA should have negative PV when forward > fixed."""
        # Forward (5.2%) > Fixed (5%), so receiver loses.
        pricer = FRAPricerSimple()
        pv = pricer.price(receiver_fra)
        assert pv < 0
    
    def test_payer_receiver_opposite_sign(self, payer_fra, receiver_fra):
        """Payer and receiver PVs should be opposite."""
        pricer = FRAPricerSimple()
        
        payer_pv = pricer.price(payer_fra)
        receiver_pv = pricer.price(receiver_fra)
        
        assert abs(payer_pv + receiver_pv) < 1e-6
    
    def test_par_fra_zero_pv(self, fra_params):
        """FRA at par rate should have zero PV."""
        # Set fixed rate = forward rate.
        params = {**fra_params, "fixed_rate": fra_params["forward_rate"]}
        par_fra = ForwardRateAgreementSimple(**params)
        
        pricer = FRAPricerSimple()
        pv = pricer.price(par_fra)
        
        assert abs(pv) < 1e-6
    
    def test_fra_pv_formula(self, fra_params):
        """Verify FRA PV formula directly."""
        fra = ForwardRateAgreementSimple(**fra_params)
        pricer = FRAPricerSimple()
        
        pv = pricer.price(fra)
        
        # Expected: N × τ × DF × (F - K)
        expected = (
            fra_params["notional"] *
            fra_params["accrual_factor"] *
            fra_params["discount_factor"] *
            (fra_params["forward_rate"] - fra_params["fixed_rate"])
        )
        
        assert abs(pv - expected) < 1e-6
    
    def test_fra_scales_with_notional(self, fra_params):
        """FRA PV should scale linearly with notional."""
        pricer = FRAPricerSimple()
        
        fra_1 = ForwardRateAgreementSimple(**fra_params)
        
        params_2 = {**fra_params, "notional": 20_000_000}
        fra_2 = ForwardRateAgreementSimple(**params_2)
        
        pv_1 = pricer.price(fra_1)
        pv_2 = pricer.price(fra_2)
        
        assert abs(pv_2 - 2 * pv_1) < 1e-6


# =============================================================================
# FRA GREEKS TESTS
# =============================================================================


class TestFRAGreeks:
    """Tests for FRA Greeks."""
    
    def test_greeks_exist(self, payer_fra):
        """Greeks should be computed."""
        pricer = FRAPricerSimple()
        greeks = pricer.greeks(payer_fra)
        
        assert "delta" in greeks
        assert "dv01" in greeks
        assert "pv01" in greeks
    
    def test_payer_delta_positive(self, payer_fra):
        """Payer FRA should have positive delta (benefit from rate increase)."""
        pricer = FRAPricerSimple()
        greeks = pricer.greeks(payer_fra)
        assert greeks["delta"] > 0
    
    def test_receiver_delta_negative(self, receiver_fra):
        """Receiver FRA should have negative delta."""
        pricer = FRAPricerSimple()
        greeks = pricer.greeks(receiver_fra)
        assert greeks["delta"] < 0
    
    def test_dv01_positive(self, payer_fra):
        """DV01 should be positive (absolute sensitivity)."""
        pricer = FRAPricerSimple()
        greeks = pricer.greeks(payer_fra)
        assert greeks["dv01"] > 0
    
    def test_delta_fd(self, fra_params):
        """Delta should match finite difference approximation."""
        pricer = FRAPricerSimple()
        
        bump = 1e-4
        
        params_up = {**fra_params, "forward_rate": fra_params["forward_rate"] + bump}
        params_dn = {**fra_params, "forward_rate": fra_params["forward_rate"] - bump}
        
        fra_up = ForwardRateAgreementSimple(**params_up)
        fra_dn = ForwardRateAgreementSimple(**params_dn)
        fra_mid = ForwardRateAgreementSimple(**fra_params)
        
        pv_up = pricer.price(fra_up)
        pv_dn = pricer.price(fra_dn)
        
        fd_delta = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(fra_mid)
        analytic_delta = greeks["delta"]
        
        assert abs(fd_delta - analytic_delta) / abs(analytic_delta) < 0.01


# =============================================================================
# SWAP VALIDATION TESTS
# =============================================================================


class TestSwapValidation:
    """Tests for swap instrument validation."""
    
    def test_valid_swap(self, receiver_swap):
        """Valid swap should be created without errors."""
        assert receiver_swap.notional == 1_000_000
        assert receiver_swap.fixed_rate == 0.05
    
    def test_empty_fixed_leg_raises(self, swap_legs):
        """Empty fixed leg should raise."""
        _, floating_legs = swap_legs
        with pytest.raises(ValueError, match="fixed_leg must have at least one period"):
            InterestRateSwapSimple(
                notional=1_000_000,
                fixed_rate=0.05,
                fixed_leg=(),
                floating_leg=floating_legs,
            )
    
    def test_empty_floating_leg_raises(self, swap_legs):
        """Empty floating leg should raise."""
        fixed_legs, _ = swap_legs
        with pytest.raises(ValueError, match="floating_leg must have at least one period"):
            InterestRateSwapSimple(
                notional=1_000_000,
                fixed_rate=0.05,
                fixed_leg=fixed_legs,
                floating_leg=(),
            )


# =============================================================================
# SWAP PRICING TESTS
# =============================================================================


class TestSwapPricing:
    """Tests for swap pricing."""
    
    def test_receiver_payer_opposite_sign(self, receiver_swap, payer_swap):
        """Receiver and payer swap PVs should be opposite."""
        pricer = IRSwapPricerSimple()
        
        receiver_pv = pricer.price(receiver_swap)
        payer_pv = pricer.price(payer_swap)
        
        assert abs(receiver_pv + payer_pv) < 1e-6
    
    def test_swap_pv_decomposition(self, receiver_swap):
        """Swap PV should equal fixed leg PV minus floating leg PV."""
        pricer = IRSwapPricerSimple()
        
        total_pv = pricer.price(receiver_swap)
        fixed_pv = pricer.fixed_leg_pv(receiver_swap)
        floating_pv = pricer.floating_leg_pv(receiver_swap)
        
        assert abs(total_pv - (fixed_pv - floating_pv)) < 1e-6
    
    def test_par_swap_zero_pv(self, swap_legs):
        """Swap at par rate should have approximately zero PV."""
        fixed_legs, floating_legs = swap_legs
        
        # Create swap with arbitrary fixed rate.
        swap = InterestRateSwapSimple(
            notional=1_000_000,
            fixed_rate=0.05,
            fixed_leg=fixed_legs,
            floating_leg=floating_legs,
            direction="receiver",
        )
        
        pricer = IRSwapPricerSimple()
        par_rate = pricer.par_rate(swap)
        
        # Create swap at par rate.
        # Need to recreate fixed legs with par rate.
        par_fixed_legs = tuple(
            FixedLeg(
                start_time=leg.start_time,
                end_time=leg.end_time,
                accrual_factor=leg.accrual_factor,
                discount_factor=leg.discount_factor,
                notional=leg.notional,
                fixed_rate=par_rate,
            )
            for leg in fixed_legs
        )
        
        par_swap = InterestRateSwapSimple(
            notional=1_000_000,
            fixed_rate=par_rate,
            fixed_leg=par_fixed_legs,
            floating_leg=floating_legs,
            direction="receiver",
        )
        
        par_pv = pricer.price(par_swap)
        assert abs(par_pv) < 1e-6
    
    def test_annuity_calculation(self, receiver_swap):
        """Test annuity (PV01) calculation."""
        # Annuity = Σ[τ × DF] = 1.0 × 0.95 + 1.0 × 0.90 = 1.85
        expected_annuity = 0.95 + 0.90
        assert abs(receiver_swap.annuity - expected_annuity) < 1e-6


# =============================================================================
# SWAP GREEKS TESTS
# =============================================================================


class TestSwapGreeks:
    """Tests for swap Greeks."""
    
    def test_greeks_exist(self, receiver_swap):
        """Greeks should be computed."""
        pricer = IRSwapPricerSimple()
        greeks = pricer.greeks(receiver_swap)
        
        assert "delta" in greeks
        assert "dv01" in greeks
        assert "pv01" in greeks
    
    def test_receiver_delta_negative(self, receiver_swap):
        """Receiver swap should have negative delta (lose when rates rise)."""
        pricer = IRSwapPricerSimple()
        greeks = pricer.greeks(receiver_swap)
        assert greeks["delta"] < 0
    
    def test_payer_delta_positive(self, payer_swap):
        """Payer swap should have positive delta (gain when rates rise)."""
        pricer = IRSwapPricerSimple()
        greeks = pricer.greeks(payer_swap)
        assert greeks["delta"] > 0
    
    def test_dv01_formula(self, receiver_swap):
        """Test DV01 formula: N × A × 0.0001."""
        pricer = IRSwapPricerSimple()
        
        expected_dv01 = abs(receiver_swap.notional) * receiver_swap.annuity * 0.0001
        
        greeks = pricer.greeks(receiver_swap)
        
        # DV01 is signed for receiver/payer, but magnitude should match.
        assert abs(abs(greeks["dv01"]) - expected_dv01) < 1e-6


# =============================================================================
# MARKET DATA PRICER TESTS
# =============================================================================


class TestMarketDataPricers:
    """Tests for pricers with market data lookup."""
    
    def test_fra_market_pricer(self, market, curve_id):
        """FRA market pricer should produce reasonable price."""
        fra = ForwardRateAgreement(
            notional=10_000_000,
            fixed_rate=0.05,
            fixing_time=0.25,
            payment_time=0.5,
            day_count="ACT/360",
            direction="payer",
            curve_id=curve_id,
        )
        
        pricer = FRAPricer()
        pv = pricer.price(fra, market)
        
        # With flat 5% curve, forward ≈ 5%, so PV should be near zero.
        assert abs(pv) < 10_000  # Allow for day count effects
    
    def test_swap_market_pricer(self, market, curve_id):
        """Swap market pricer should produce reasonable price."""
        swap = InterestRateSwap(
            notional=10_000_000,
            fixed_rate=0.05,
            start_time=0.0,
            end_time=5.0,
            fixed_frequency=0.5,
            floating_frequency=0.25,
            curve_id=curve_id,
            direction="receiver",
        )
        
        pricer = IRSwapPricer()
        pv = pricer.price(swap, market)
        
        # With flat 5% curve and 5% fixed rate, PV should be near zero.
        assert abs(pv) < 100_000  # Allow for day count effects
    
    def test_swap_par_rate_near_flat_curve_rate(self, market, curve_id):
        """Par swap rate should be close to flat curve rate."""
        swap = InterestRateSwap(
            notional=10_000_000,
            fixed_rate=0.05,
            start_time=0.0,
            end_time=5.0,
            curve_id=curve_id,
        )
        
        pricer = IRSwapPricer()
        par_rate = pricer.par_rate(swap, market)
        
        # With flat 5% curve, par rate should be close to 5%.
        assert abs(par_rate - 0.05) < 0.01


# =============================================================================
# SCHEDULE GENERATION TESTS
# =============================================================================


class TestScheduleGeneration:
    """Tests for swap schedule generation."""
    
    def test_annual_schedule(self):
        """Test annual schedule generation."""
        schedule = generate_swap_schedule(0.0, 3.0, 1.0)
        
        assert len(schedule) == 3
        assert schedule[0] == (0.0, 1.0)
        assert schedule[1] == (1.0, 2.0)
        assert schedule[2] == (2.0, 3.0)
    
    def test_quarterly_schedule(self):
        """Test quarterly schedule generation."""
        schedule = generate_swap_schedule(0.0, 1.0, 0.25)
        
        assert len(schedule) == 4
        assert schedule[0] == (0.0, 0.25)
        assert schedule[3] == (0.75, 1.0)
    
    def test_partial_period(self):
        """Test schedule with partial final period."""
        schedule = generate_swap_schedule(0.0, 2.5, 1.0)
        
        assert len(schedule) == 3
        assert schedule[2] == (2.0, 2.5)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_negative_notional_fra(self, fra_params):
        """FRA with negative notional should work (reverses direction)."""
        params = {**fra_params, "notional": -10_000_000}
        fra = ForwardRateAgreementSimple(**params)
        
        pricer = FRAPricerSimple()
        pv = pricer.price(fra)
        
        # Negative notional payer should behave like positive notional receiver.
        positive_fra = ForwardRateAgreementSimple(**fra_params)
        positive_pv = pricer.price(positive_fra)
        
        assert abs(pv + positive_pv) < 1e-6
    
    def test_atm_fra(self, fra_params):
        """ATM FRA (forward = fixed) should have zero PV."""
        params = {**fra_params, "fixed_rate": fra_params["forward_rate"]}
        fra = ForwardRateAgreementSimple(**params)
        
        pricer = FRAPricerSimple()
        pv = pricer.price(fra)
        
        assert abs(pv) < 1e-10
    
    def test_fra_is_in_the_money(self, fra_params):
        """Test ITM/OTM detection."""
        # Payer ITM when F > K.
        payer_itm = ForwardRateAgreementSimple(**fra_params)  # F=5.2% > K=5%
        assert payer_itm.is_in_the_money
        
        # Receiver ITM when F < K.
        receiver_params = {**fra_params, "direction": "receiver"}
        receiver_otm = ForwardRateAgreementSimple(**receiver_params)
        assert not receiver_otm.is_in_the_money
