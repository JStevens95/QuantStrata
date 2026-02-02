from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.surfaces.vol_surface import (
    FlatVolSurface,
    GridVolSurface,
    SwaptionVolCube,
    FlatSwaptionVolCube,
    CapFloorVolSurface,
    FlatCapFloorVolSurface,
    create_atm_swaption_vol_cube,
    create_cap_vol_surface_from_term_structure,
)


def test_flat_vol_surface_validates_and_returns_constant() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _ = FlatVolSurface(sigma=0.0)

    s = FlatVolSurface(sigma=0.123)
    assert abs(s.implied_vol(0.0, 100.0) - 0.123) < 1e-15
    assert abs(s.implied_vol(2.0, 50.0) - 0.123) < 1e-15
    assert abs(s.vol(1.0, 100.0) - 0.123) < 1e-15


def test_grid_vol_surface_validates_inputs() -> None:
    exp = np.array([0.5, 1.0], dtype=float)
    k = np.array([90.0, 100.0, 110.0], dtype=float)
    vols = np.full((2, 3), 0.2, dtype=float)

    _ = GridVolSurface(expiries=exp, strikes=k, implied_vols=vols)

    with pytest.raises(ValueError, match="strictly increasing"):
        _ = GridVolSurface(expiries=np.array([1.0, 1.0], float), strikes=k, implied_vols=vols)

    with pytest.raises(ValueError, match="strictly increasing"):
        _ = GridVolSurface(expiries=exp, strikes=np.array([100.0, 100.0], float), implied_vols=np.full((2, 2), 0.2))

    with pytest.raises(ValueError, match="must be a 2D"):
        _ = GridVolSurface(expiries=exp, strikes=k, implied_vols=np.array([0.2, 0.2], float))

    with pytest.raises(ValueError, match="must be strictly positive"):
        bad = vols.copy()
        bad[0, 0] = 0.0
        _ = GridVolSurface(expiries=exp, strikes=k, implied_vols=bad)


def test_grid_vol_surface_interp_and_flat_extrapolation() -> None:
    exp = np.array([0.5, 1.0], dtype=float)
    k = np.array([90.0, 110.0], dtype=float)

    # a simple plane so bilinear interpolation is predictable
    # z(T,K):
    # T=0.5: [0.10, 0.20]
    # T=1.0: [0.20, 0.30]
    vols = np.array([[0.10, 0.20], [0.20, 0.30]], dtype=float)

    s = GridVolSurface(expiries=exp, strikes=k, implied_vols=vols, extrapolation="flat")

    # midpoint in both dims should be average of corners = 0.20
    assert abs(s.implied_vol(0.75, 100.0) - 0.20) < 1e-12

    # flat extrapolation clamps
    assert abs(s.implied_vol(0.10, 100.0) - s.implied_vol(0.5, 100.0)) < 1e-15
    assert abs(s.implied_vol(2.00, 100.0) - s.implied_vol(1.0, 100.0)) < 1e-15
    assert abs(s.implied_vol(0.75, 50.0) - s.implied_vol(0.75, 90.0)) < 1e-15
    assert abs(s.implied_vol(0.75, 200.0) - s.implied_vol(0.75, 110.0)) < 1e-15


def test_grid_vol_surface_error_extrapolation_raises() -> None:
    exp = np.array([0.5, 1.0], dtype=float)
    k = np.array([90.0, 110.0], dtype=float)
    vols = np.full((2, 2), 0.2, dtype=float)

    s = GridVolSurface(expiries=exp, strikes=k, implied_vols=vols, extrapolation="error")

    with pytest.raises(ValueError, match="outside grid"):
        _ = s.implied_vol(0.1, 100.0)

    with pytest.raises(ValueError, match="outside grid"):
        _ = s.implied_vol(0.75, 50.0)


# =============================================================================
# SwaptionVolCube Tests
# =============================================================================


class TestSwaptionVolCube:
    """Tests for SwaptionVolCube."""

    @pytest.fixture
    def simple_cube(self) -> SwaptionVolCube:
        """Create a simple swaption vol cube for testing."""
        expiries = np.array([0.5, 1.0, 2.0, 5.0])
        tenors = np.array([1.0, 2.0, 5.0, 10.0])
        strikes = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        vols = np.zeros((4, 4, 5))
        for i, exp in enumerate(expiries):
            for j, ten in enumerate(tenors):
                for k, strike in enumerate(strikes):
                    base = 0.004 + 0.001 * np.log(exp + 1) + 0.0005 * np.log(ten + 1)
                    smile = 0.0001 * (strike - 0.03) ** 2
                    vols[i, j, k] = base + smile

        return SwaptionVolCube(
            expiries=expiries,
            tenors=tenors,
            strikes=strikes,
            vols=vols,
            vol_type="normal",
        )

    def test_creation_valid(self, simple_cube: SwaptionVolCube) -> None:
        """Test valid cube creation."""
        assert simple_cube.vol_type == "normal"
        assert len(simple_cube.expiries) == 4
        assert len(simple_cube.tenors) == 4
        assert len(simple_cube.strikes) == 5
        assert simple_cube.vols.shape == (4, 4, 5)

    def test_creation_invalid_shape(self) -> None:
        """Test that invalid vol shape raises error."""
        with pytest.raises(ValueError, match="vols shape"):
            SwaptionVolCube(
                expiries=np.array([1.0, 2.0]),
                tenors=np.array([1.0, 2.0]),
                strikes=np.array([0.01, 0.02, 0.03]),
                vols=np.zeros((2, 2, 2)),
            )

    def test_creation_non_increasing_expiries(self) -> None:
        """Test that non-increasing expiries raises error."""
        with pytest.raises(ValueError, match="strictly increasing"):
            SwaptionVolCube(
                expiries=np.array([1.0, 0.5, 2.0]),
                tenors=np.array([1.0, 2.0]),
                strikes=np.array([0.01, 0.02]),
                vols=np.zeros((3, 2, 2)),
            )

    def test_implied_vol_on_grid(self, simple_cube: SwaptionVolCube) -> None:
        """Test vol retrieval on grid points."""
        vol = simple_cube.implied_vol(1.0, 2.0, 0.03)
        assert isinstance(vol, float)
        assert vol > 0

    def test_implied_vol_interpolation(self, simple_cube: SwaptionVolCube) -> None:
        """Test vol interpolation between grid points."""
        vol = simple_cube.implied_vol(0.75, 1.5, 0.025)
        assert isinstance(vol, float)
        assert vol > 0

    def test_implied_vol_extrapolation_flat(self, simple_cube: SwaptionVolCube) -> None:
        """Test flat extrapolation outside grid."""
        vol_below = simple_cube.implied_vol(0.1, 0.5, 0.005)
        vol_edge = simple_cube.implied_vol(0.5, 1.0, 0.01)
        assert vol_below == vol_edge

    def test_implied_vol_extrapolation_error(self) -> None:
        """Test error extrapolation mode."""
        cube = SwaptionVolCube(
            expiries=np.array([1.0, 2.0]),
            tenors=np.array([1.0, 2.0]),
            strikes=np.array([0.01, 0.02]),
            vols=np.full((2, 2, 2), 0.005),
            extrapolation="error",
        )
        with pytest.raises(ValueError, match="below grid"):
            cube.implied_vol(0.5, 1.0, 0.01)

    def test_atm_vol(self, simple_cube: SwaptionVolCube) -> None:
        """Test ATM vol retrieval."""
        atm = simple_cube.atm_vol(1.0, 2.0)
        assert isinstance(atm, float)
        assert atm > 0

    def test_smile(self, simple_cube: SwaptionVolCube) -> None:
        """Test smile retrieval."""
        strikes, vols = simple_cube.smile(1.0, 2.0)
        assert len(strikes) == len(simple_cube.strikes)
        assert len(vols) == len(strikes)
        assert np.all(vols > 0)


class TestFlatSwaptionVolCube:
    """Tests for FlatSwaptionVolCube."""

    def test_creation(self) -> None:
        """Test flat cube creation."""
        cube = FlatSwaptionVolCube(vol=0.005, vol_type="normal")
        assert cube.vol == 0.005
        assert cube.vol_type == "normal"

    def test_implied_vol_constant(self) -> None:
        """Test that implied vol is constant."""
        cube = FlatSwaptionVolCube(vol=0.005)
        assert cube.implied_vol(1.0, 2.0, 0.03) == 0.005
        assert cube.implied_vol(5.0, 10.0, 0.05) == 0.005

    def test_atm_vol(self) -> None:
        """Test ATM vol."""
        cube = FlatSwaptionVolCube(vol=0.005)
        assert cube.atm_vol(1.0, 2.0) == 0.005

    def test_negative_vol_raises(self) -> None:
        """Test that negative vol raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            FlatSwaptionVolCube(vol=-0.001)


# =============================================================================
# CapFloorVolSurface Tests
# =============================================================================


class TestCapFloorVolSurface:
    """Tests for CapFloorVolSurface."""

    @pytest.fixture
    def simple_surface(self) -> CapFloorVolSurface:
        """Create a simple cap/floor vol surface."""
        expiries = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        strikes = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        vols = np.zeros((5, 5))
        base_vols = np.array([0.0045, 0.0050, 0.0048, 0.0045, 0.0042])
        for i in range(5):
            for j, k in enumerate(strikes):
                smile = 0.0002 * (k - 0.03) ** 2
                vols[i, j] = base_vols[i] + smile

        return CapFloorVolSurface(
            expiries=expiries,
            strikes=strikes,
            vols=vols,
            vol_type="normal",
        )

    def test_creation_valid(self, simple_surface: CapFloorVolSurface) -> None:
        """Test valid surface creation."""
        assert simple_surface.vol_type == "normal"
        assert len(simple_surface.expiries) == 5
        assert len(simple_surface.strikes) == 5

    def test_creation_invalid_shape(self) -> None:
        """Test that invalid vol shape raises error."""
        with pytest.raises(ValueError, match="vols shape"):
            CapFloorVolSurface(
                expiries=np.array([1.0, 2.0]),
                strikes=np.array([0.01, 0.02, 0.03]),
                vols=np.zeros((2, 2)),
            )

    def test_implied_vol_on_grid(self, simple_surface: CapFloorVolSurface) -> None:
        """Test vol retrieval on grid points."""
        vol = simple_surface.implied_vol(1.0, 0.03)
        assert isinstance(vol, float)
        assert vol > 0

    def test_implied_vol_interpolation(self, simple_surface: CapFloorVolSurface) -> None:
        """Test vol interpolation."""
        vol = simple_surface.implied_vol(0.75, 0.025)
        assert isinstance(vol, float)
        assert vol > 0

    def test_vol_alias(self, simple_surface: CapFloorVolSurface) -> None:
        """Test vol() alias."""
        assert simple_surface.vol(1.0, 0.03) == simple_surface.implied_vol(1.0, 0.03)

    def test_smile(self, simple_surface: CapFloorVolSurface) -> None:
        """Test smile retrieval."""
        strikes, vols = simple_surface.smile(1.0)
        assert len(strikes) == len(simple_surface.strikes)
        assert len(vols) == len(strikes)


class TestFlatCapFloorVolSurface:
    """Tests for FlatCapFloorVolSurface."""

    def test_creation(self) -> None:
        """Test flat surface creation."""
        surface = FlatCapFloorVolSurface(vol=0.005, vol_type="normal")
        assert surface.vol == 0.005

    def test_implied_vol_constant(self) -> None:
        """Test constant vol."""
        surface = FlatCapFloorVolSurface(vol=0.005)
        assert surface.implied_vol(1.0, 0.03) == 0.005
        assert surface.implied_vol(10.0, 0.05) == 0.005


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestVolSurfaceFactoryFunctions:
    """Tests for factory functions."""

    def test_create_atm_swaption_vol_cube(self) -> None:
        """Test ATM swaption vol cube creation."""
        expiries = np.array([1.0, 2.0, 5.0])
        tenors = np.array([2.0, 5.0, 10.0])
        atm_vols = np.array([
            [0.0045, 0.0050, 0.0055],
            [0.0048, 0.0052, 0.0057],
            [0.0050, 0.0054, 0.0058],
        ])

        cube = create_atm_swaption_vol_cube(
            expiries=expiries,
            tenors=tenors,
            atm_vols=atm_vols,
            smile_width=0.01,
            smile_curvature=0.5,
        )

        assert cube.strike_type == "relative_atm"
        assert len(cube.strikes) == 9
        assert cube.implied_vol(1.0, 2.0, 0.0) == pytest.approx(0.0045, rel=1e-6)

    def test_create_cap_vol_surface(self) -> None:
        """Test cap vol surface creation from term structure."""
        expiries = np.array([1.0, 2.0, 5.0, 10.0])
        atm_vols = np.array([0.0045, 0.0050, 0.0048, 0.0045])
        strikes = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        surface = create_cap_vol_surface_from_term_structure(
            expiries=expiries,
            atm_vols=atm_vols,
            strikes=strikes,
            skew=-0.05,
        )

        assert len(surface.expiries) == 4
        assert len(surface.strikes) == 5