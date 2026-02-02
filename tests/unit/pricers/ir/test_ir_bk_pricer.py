"""
Unit tests for Black-Karasinski MC Pricers.

Tests cover:
1. ZC bond MC pricing
2. Bond option MC pricing
3. Caplet/Floorlet MC pricing
4. Greeks computation
5. Edge cases
"""

import numpy as np
import pytest

from src.models.short_rate.black_karasinski import (
    BlackKarasinskiParameters,
)
from src.pricers.ir.european_bk_mc import (
    BKMCConfig,
    DEFAULT_BK_MC_CONFIG,
    IrBondZeroCouponBKMCPricerSimple,
    IrBondEuropeanOptionBKMCPricerSimple,
    IrCapletEuropeanOptionBKMCPricerSimple,
    IrFloorletEuropeanOptionBKMCPricerSimple,
    MonteCarloEstimate,
)
from src.instruments.ir.linear.bond import IrBondZeroCouponSimple
from src.instruments.ir.options.bond import IrBondEuropeanOptionSimple
from src.instruments.ir.options.capfloor import (
    IrCapletEuropeanOptionSimple,
    IrFloorletEuropeanOptionSimple,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def bk_params() -> BlackKarasinskiParameters:
    """Standard Black-Karasinski parameters for testing."""
    return BlackKarasinskiParameters(
        a=0.1,       # Mean reversion speed
        sigma=0.15,  # 15% vol of log-rate
        r0=0.03,     # 3% initial rate
        theta=-3.5,  # Long-term log-rate (≈3% long-term rate)
    )


@pytest.fixture
def mc_config() -> BKMCConfig:
    """MC configuration for faster tests."""
    return BKMCConfig(
        n_paths=10_000,
        n_steps=100,
        seed=42,
        antithetic=True,
    )


@pytest.fixture
def zc_bond() -> IrBondZeroCouponSimple:
    """Zero coupon bond instrument."""
    return IrBondZeroCouponSimple(
        maturity=1.0,
        face_value=100.0,
        discount_factor=0.97,  # ~3% continuously compounded
    )


@pytest.fixture
def bond_option() -> IrBondEuropeanOptionSimple:
    """Bond option instrument."""
    return IrBondEuropeanOptionSimple(
        strike=95.0,
        expiry=0.5,
        notional=100.0,
        option_type="call",
        forward_bond_price=96.0,
        vol=0.08,  # 8% bond price vol
        discount_factor=0.985,
    )


@pytest.fixture
def caplet() -> IrCapletEuropeanOptionSimple:
    """Caplet instrument."""
    return IrCapletEuropeanOptionSimple(
        strike=0.03,
        fixing_time=0.5,
        payment_time=1.0,
        notional=1_000_000.0,
        accrual_factor=0.5,
        forward_rate=0.035,
        vol=0.20,  # 20% vol
        discount_factor=0.97,
    )


@pytest.fixture
def floorlet() -> IrFloorletEuropeanOptionSimple:
    """Floorlet instrument."""
    return IrFloorletEuropeanOptionSimple(
        strike=0.03,
        fixing_time=0.5,
        payment_time=1.0,
        notional=1_000_000.0,
        accrual_factor=0.5,
        forward_rate=0.025,
        vol=0.20,  # 20% vol
        discount_factor=0.97,
    )


# =============================================================================
# ZC Bond MC Pricer Tests
# =============================================================================


class TestIrBondZeroCouponBKMCPricerSimple:
    """Tests for ZC bond BK MC pricer."""

    def test_price_positive(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        zc_bond: IrBondZeroCouponSimple,
    ) -> None:
        """Test that bond price is positive."""
        pricer = IrBondZeroCouponBKMCPricerSimple(params=bk_params, config=mc_config)
        price = pricer.price(zc_bond)

        assert price > 0.0
        assert price < 100.0  # Should be less than face value

    def test_price_reasonable_range(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        zc_bond: IrBondZeroCouponSimple,
    ) -> None:
        """Test that bond price is in reasonable range."""
        pricer = IrBondZeroCouponBKMCPricerSimple(params=bk_params, config=mc_config)
        price = pricer.price(zc_bond)

        # 1Y bond at ~3% rate: expect price around 97
        assert 90 < price < 100

    def test_price_with_estimate(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        zc_bond: IrBondZeroCouponSimple,
    ) -> None:
        """Test price_with_estimate returns MonteCarloEstimate."""
        pricer = IrBondZeroCouponBKMCPricerSimple(params=bk_params, config=mc_config)
        estimate = pricer.price_with_estimate(zc_bond)

        assert isinstance(estimate, MonteCarloEstimate)
        assert estimate.mean > 0
        assert estimate.stderr > 0
        assert estimate.n_paths >= mc_config.n_paths

    def test_greeks(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        zc_bond: IrBondZeroCouponSimple,
    ) -> None:
        """Test Greeks computation."""
        pricer = IrBondZeroCouponBKMCPricerSimple(params=bk_params, config=mc_config)
        greeks = pricer.greeks(zc_bond)

        assert "delta" in greeks
        assert "dv01" in greeks
        assert "vega" in greeks
        # Delta should be negative (bond price decreases as rate increases)
        assert greeks["delta"] < 0
        # DV01 should be positive
        assert greeks["dv01"] > 0

    def test_zero_maturity(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
    ) -> None:
        """Test zero maturity bond returns face value."""
        bond = IrBondZeroCouponSimple(
            maturity=0.0,
            face_value=100.0,
            discount_factor=1.0,
        )
        pricer = IrBondZeroCouponBKMCPricerSimple(params=bk_params, config=mc_config)
        price = pricer.price(bond)

        assert price == 100.0


# =============================================================================
# Bond Option MC Pricer Tests
# =============================================================================


class TestIrBondEuropeanOptionBKMCPricerSimple:
    """Tests for bond option BK MC pricer."""

    def test_call_price_positive(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        bond_option: IrBondEuropeanOptionSimple,
    ) -> None:
        """Test that call option price is positive."""
        pricer = IrBondEuropeanOptionBKMCPricerSimple(params=bk_params, config=mc_config)
        price = pricer.price(bond_option)

        assert price >= 0.0

    def test_put_price_positive(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
    ) -> None:
        """Test that put option price is positive."""
        put_option = IrBondEuropeanOptionSimple(
            strike=98.0,
            expiry=0.5,
            notional=100.0,
            option_type="put",
            forward_bond_price=96.0,
            vol=0.08,
            discount_factor=0.985,
        )
        pricer = IrBondEuropeanOptionBKMCPricerSimple(params=bk_params, config=mc_config)
        price = pricer.price(put_option)

        assert price >= 0.0

    def test_price_with_estimate(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        bond_option: IrBondEuropeanOptionSimple,
    ) -> None:
        """Test price_with_estimate returns MonteCarloEstimate."""
        pricer = IrBondEuropeanOptionBKMCPricerSimple(params=bk_params, config=mc_config)
        estimate = pricer.price_with_estimate(bond_option)

        assert isinstance(estimate, MonteCarloEstimate)
        assert estimate.mean >= 0

    def test_greeks(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        bond_option: IrBondEuropeanOptionSimple,
    ) -> None:
        """Test Greeks computation."""
        pricer = IrBondEuropeanOptionBKMCPricerSimple(params=bk_params, config=mc_config)
        greeks = pricer.greeks(bond_option)

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks

    def test_expired_option_intrinsic(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
    ) -> None:
        """Test expired option returns intrinsic value."""
        itm_call = IrBondEuropeanOptionSimple(
            strike=90.0,
            expiry=0.0,  # Expired
            notional=100.0,
            option_type="call",
            forward_bond_price=95.0,
            vol=0.08,
            discount_factor=1.0,
        )
        pricer = IrBondEuropeanOptionBKMCPricerSimple(params=bk_params, config=mc_config)
        price = pricer.price(itm_call)

        # Intrinsic = 100 × max(95 - 90, 0) = 500
        assert price == pytest.approx(500.0, rel=0.01)


# =============================================================================
# Caplet MC Pricer Tests
# =============================================================================


class TestIrCapletEuropeanOptionBKMCPricerSimple:
    """Tests for caplet BK MC pricer."""

    def test_price_positive(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        caplet: IrCapletEuropeanOptionSimple,
    ) -> None:
        """Test that caplet price is positive."""
        pricer = IrCapletEuropeanOptionBKMCPricerSimple(params=bk_params, config=mc_config)
        price = pricer.price(caplet)

        assert price >= 0.0

    def test_greeks(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        caplet: IrCapletEuropeanOptionSimple,
    ) -> None:
        """Test Greeks computation."""
        pricer = IrCapletEuropeanOptionBKMCPricerSimple(params=bk_params, config=mc_config)
        greeks = pricer.greeks(caplet)

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks

    def test_expired_caplet(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
    ) -> None:
        """Test expired caplet returns intrinsic value."""
        expired_caplet = IrCapletEuropeanOptionSimple(
            strike=0.03,
            fixing_time=0.0,  # Expired
            payment_time=0.5,
            notional=1_000_000.0,
            accrual_factor=0.5,
            forward_rate=0.04,  # ITM
            vol=0.20,
            discount_factor=0.985,
        )
        pricer = IrCapletEuropeanOptionBKMCPricerSimple(params=bk_params, config=mc_config)
        price = pricer.price(expired_caplet)

        # Intrinsic = N × τ × df × max(F - K, 0)
        # = 1M × 0.5 × 0.985 × max(0.04 - 0.03, 0) = 1M × 0.5 × 0.985 × 0.01 = 4925
        expected = 1_000_000.0 * 0.5 * 0.985 * 0.01
        assert price == pytest.approx(expected, rel=0.01)


# =============================================================================
# Floorlet MC Pricer Tests
# =============================================================================


class TestIrFloorletEuropeanOptionBKMCPricerSimple:
    """Tests for floorlet BK MC pricer."""

    def test_price_positive(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        floorlet: IrFloorletEuropeanOptionSimple,
    ) -> None:
        """Test that floorlet price is positive."""
        pricer = IrFloorletEuropeanOptionBKMCPricerSimple(
            params=bk_params, config=mc_config
        )
        price = pricer.price(floorlet)

        assert price >= 0.0

    def test_greeks(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
        floorlet: IrFloorletEuropeanOptionSimple,
    ) -> None:
        """Test Greeks computation."""
        pricer = IrFloorletEuropeanOptionBKMCPricerSimple(
            params=bk_params, config=mc_config
        )
        greeks = pricer.greeks(floorlet)

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks

    def test_expired_floorlet(
        self,
        bk_params: BlackKarasinskiParameters,
        mc_config: BKMCConfig,
    ) -> None:
        """Test expired floorlet returns intrinsic value."""
        expired_floorlet = IrFloorletEuropeanOptionSimple(
            strike=0.03,
            fixing_time=0.0,  # Expired
            payment_time=0.5,
            notional=1_000_000.0,
            accrual_factor=0.5,
            forward_rate=0.02,  # ITM (K > F)
            vol=0.20,
            discount_factor=0.985,
        )
        pricer = IrFloorletEuropeanOptionBKMCPricerSimple(
            params=bk_params, config=mc_config
        )
        price = pricer.price(expired_floorlet)

        # Intrinsic = N × τ × df × max(K - F, 0)
        # = 1M × 0.5 × 0.985 × max(0.03 - 0.02, 0) = 1M × 0.5 × 0.985 × 0.01 = 4925
        expected = 1_000_000.0 * 0.5 * 0.985 * 0.01
        assert price == pytest.approx(expected, rel=0.01)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestBKMCConfig:
    """Tests for BK MC configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DEFAULT_BK_MC_CONFIG
        assert config.n_paths == 100_000
        assert config.n_steps == 252
        assert config.seed is None
        assert config.antithetic is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BKMCConfig(
            n_paths=50_000,
            n_steps=500,
            seed=123,
            antithetic=False,
        )
        assert config.n_paths == 50_000
        assert config.n_steps == 500
        assert config.seed == 123
        assert config.antithetic is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestBKPricerEdgeCases:
    """Tests for edge cases."""

    def test_high_mean_reversion(
        self,
        mc_config: BKMCConfig,
        zc_bond: IrBondZeroCouponSimple,
    ) -> None:
        """Test with high mean reversion."""
        params = BlackKarasinskiParameters(
            a=1.0,  # High mean reversion
            sigma=0.15,
            r0=0.03,
            theta=-3.5,
        )
        pricer = IrBondZeroCouponBKMCPricerSimple(params=params, config=mc_config)
        price = pricer.price(zc_bond)

        # Should still give reasonable price.
        assert 90 < price < 100

    def test_low_volatility(
        self,
        mc_config: BKMCConfig,
        zc_bond: IrBondZeroCouponSimple,
    ) -> None:
        """Test with low volatility."""
        params = BlackKarasinskiParameters(
            a=0.1,
            sigma=0.01,  # Low vol
            r0=0.03,
            theta=-3.5,
        )
        pricer = IrBondZeroCouponBKMCPricerSimple(params=params, config=mc_config)
        price = pricer.price(zc_bond)

        # Should be close to deterministic case (≈ exp(-r0 × T) × face)
        expected = 100.0 * np.exp(-0.03 * 1.0)
        assert price == pytest.approx(expected, rel=0.05)

    def test_reproducibility_with_seed(
        self,
        bk_params: BlackKarasinskiParameters,
        zc_bond: IrBondZeroCouponSimple,
    ) -> None:
        """Test reproducibility with fixed seed."""
        config1 = BKMCConfig(n_paths=5000, n_steps=50, seed=42)
        config2 = BKMCConfig(n_paths=5000, n_steps=50, seed=42)

        pricer1 = IrBondZeroCouponBKMCPricerSimple(params=bk_params, config=config1)
        pricer2 = IrBondZeroCouponBKMCPricerSimple(params=bk_params, config=config2)

        price1 = pricer1.price(zc_bond)
        price2 = pricer2.price(zc_bond)

        # Should be identical with same seed.
        assert price1 == price2
