"""
Unit tests for IR Bond instruments.

Tests cover:
1. Zero coupon bond construction
2. Validation of invalid parameters
3. Immutability
"""

import pytest
from dataclasses import FrozenInstanceError

from src.instruments.ir.linear.bond import IrBondZeroCouponSimple


class TestIrBondZeroCouponSimple:
    """Tests for IrBondZeroCouponSimple instrument."""

    @pytest.fixture
    def sample_bond(self) -> IrBondZeroCouponSimple:
        """Sample ZC bond for testing."""
        return IrBondZeroCouponSimple(
            face_value=100.0,
            maturity=5.0,
            discount_factor=0.85,
        )

    def test_construction_basic(self, sample_bond: IrBondZeroCouponSimple) -> None:
        """Test basic construction."""
        assert sample_bond.face_value == 100.0
        assert sample_bond.maturity == 5.0
        assert sample_bond.discount_factor == 0.85

    def test_zero_maturity_allowed(self) -> None:
        """Test that zero maturity is allowed (at maturity)."""
        bond = IrBondZeroCouponSimple(
            face_value=100.0,
            maturity=0.0,
            discount_factor=1.0,
        )
        assert bond.maturity == 0.0

    def test_long_maturity(self) -> None:
        """Test long maturity bond."""
        bond = IrBondZeroCouponSimple(
            face_value=1000.0,
            maturity=30.0,  # 30-year bond
            discount_factor=0.40,
        )
        assert bond.maturity == 30.0
        assert bond.face_value == 1000.0

    def test_negative_face_value_raises(self) -> None:
        """Test that negative face value raises error."""
        with pytest.raises(ValueError, match="face_value"):
            IrBondZeroCouponSimple(
                face_value=-100.0,  # Invalid
                maturity=5.0,
                discount_factor=0.85,
            )

    def test_negative_maturity_raises(self) -> None:
        """Test that negative maturity raises error."""
        with pytest.raises(ValueError, match="maturity"):
            IrBondZeroCouponSimple(
                face_value=100.0,
                maturity=-1.0,  # Invalid
                discount_factor=0.85,
            )

    def test_zero_discount_factor_raises(self) -> None:
        """Test that zero discount factor raises error."""
        with pytest.raises(ValueError, match="discount_factor"):
            IrBondZeroCouponSimple(
                face_value=100.0,
                maturity=5.0,
                discount_factor=0.0,  # Invalid
            )

    def test_negative_discount_factor_raises(self) -> None:
        """Test that negative discount factor raises error."""
        with pytest.raises(ValueError, match="discount_factor"):
            IrBondZeroCouponSimple(
                face_value=100.0,
                maturity=5.0,
                discount_factor=-0.5,  # Invalid
            )

    def test_frozen_dataclass(self, sample_bond: IrBondZeroCouponSimple) -> None:
        """Test that the dataclass is frozen (immutable)."""
        with pytest.raises(FrozenInstanceError):
            sample_bond.face_value = 200.0  # type: ignore

    def test_equality(self, sample_bond: IrBondZeroCouponSimple) -> None:
        """Test equality comparison."""
        bond2 = IrBondZeroCouponSimple(
            face_value=100.0,
            maturity=5.0,
            discount_factor=0.85,
        )
        assert sample_bond == bond2

    def test_inequality(self, sample_bond: IrBondZeroCouponSimple) -> None:
        """Test inequality comparison."""
        bond2 = IrBondZeroCouponSimple(
            face_value=100.0,
            maturity=5.0,
            discount_factor=0.90,  # Different DF
        )
        assert sample_bond != bond2
