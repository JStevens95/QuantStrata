"""
Unit tests for Equity synthetic market data generators.

Tests verify:
- Generator registration and requirements
- Spot panel generation (GBM paths)
- Vol surface generation (strike-based)
- Dividend adjustment utilities

Author: QuantStrata Team
"""
from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.specs import CurveZeroSpec, VolGridSmileSpec, SpotGbmSpec
from src.marketdata.providers.synthetic.context import SyntheticGenerationState
from src.marketdata.providers.synthetic.generators.equity import (
    _EquityGenerators,
    _default_ir_curve_id,
    adjust_spot_for_discrete_dividend,
    compute_forward_with_dividends,
)
from src.marketdata.providers.synthetic.generators.interest_rate import _IrGenerators


def _make_state(*, n_time: int, n_scenarios: int) -> SyntheticGenerationState:
    """Build a minimal state container for generator-level tests."""
    return SyntheticGenerationState(
        dates=["2026-01-01"] * n_time,
        n_time=n_time,
        n_scenarios=n_scenarios,
        quote_panels={},
        curve_param_panels={},
        curve_factories={},
        vol_param_panels={},
        vol_factories={},
        spot_cache={},
    )


# =============================================================================
# Tests: Requirements
# =============================================================================

def test_equity_vol_requirements_include_spot() -> None:
    """EQUITY VOL should require SPOT for the same ticker."""
    cfg = SyntheticProviderConfig()
    eq = _EquityGenerators(base_seed=7, config=cfg)

    vol_mid = MarketId(
        asset_class="EQUITY",
        mkt_type="VOL",
        name="AAPL",
        qualifiers=None,
    )

    reqs = eq.requirements_for_vol(vol_mid)
    keys = {r.key() for r in reqs}

    assert "EQUITY.SPOT.AAPL" in keys


def test_equity_vol_requirements_include_curve_when_ccy_present() -> None:
    """EQUITY VOL should require discount curve when currency qualifier is provided."""
    cfg = SyntheticProviderConfig()
    eq = _EquityGenerators(base_seed=7, config=cfg)

    vol_mid = MarketId(
        asset_class="EQUITY",
        mkt_type="VOL",
        name="AAPL",
        qualifiers=(("ccy", "USD"),),
    )

    reqs = eq.requirements_for_vol(vol_mid)
    keys = {r.key() for r in reqs}

    assert "EQUITY.SPOT.AAPL|ccy=USD" in keys
    assert _default_ir_curve_id("USD").key() in keys


# =============================================================================
# Tests: SPOT Generation
# =============================================================================

def test_equity_spot_produces_valid_panel() -> None:
    """EQUITY SPOT should produce a [T, S] panel with positive values."""
    n_time = 5
    n_scenarios = 10

    cfg = SyntheticProviderConfig(
        spot=SpotGbmSpec(
            initial_level=100.0,
            drift=0.05,
            vol=0.20,
            dt=1.0 / 252,
            initial_dispersion=0.0,
        ),
    )

    state = _make_state(n_time=n_time, n_scenarios=n_scenarios)
    eq = _EquityGenerators(base_seed=42, config=cfg)

    spot_mid = MarketId(asset_class="EQUITY", mkt_type="SPOT", name="AAPL")
    eq.generate_spot(spot_mid, state)

    spot = np.asarray(state.quote_panels[spot_mid].data, dtype=float)

    assert spot.shape == (n_time, n_scenarios)
    assert np.all(np.isfinite(spot))
    assert np.all(spot > 0)


def test_equity_spot_is_deterministic() -> None:
    """Same seed should produce identical SPOT panels."""
    n_time = 3
    n_scenarios = 5

    cfg = SyntheticProviderConfig()

    state1 = _make_state(n_time=n_time, n_scenarios=n_scenarios)
    state2 = _make_state(n_time=n_time, n_scenarios=n_scenarios)

    eq1 = _EquityGenerators(base_seed=42, config=cfg)
    eq2 = _EquityGenerators(base_seed=42, config=cfg)

    spot_mid = MarketId(asset_class="EQUITY", mkt_type="SPOT", name="SPX")

    eq1.generate_spot(spot_mid, state1)
    eq2.generate_spot(spot_mid, state2)

    spot1 = np.asarray(state1.quote_panels[spot_mid].data, dtype=float)
    spot2 = np.asarray(state2.quote_panels[spot_mid].data, dtype=float)

    np.testing.assert_array_equal(spot1, spot2)


def test_equity_spot_different_seeds_produce_different_panels() -> None:
    """Different seeds should produce different SPOT panels."""
    n_time = 3
    n_scenarios = 5

    cfg = SyntheticProviderConfig()

    state1 = _make_state(n_time=n_time, n_scenarios=n_scenarios)
    state2 = _make_state(n_time=n_time, n_scenarios=n_scenarios)

    eq1 = _EquityGenerators(base_seed=42, config=cfg)
    eq2 = _EquityGenerators(base_seed=99, config=cfg)

    spot_mid = MarketId(asset_class="EQUITY", mkt_type="SPOT", name="SPX")

    eq1.generate_spot(spot_mid, state1)
    eq2.generate_spot(spot_mid, state2)

    spot1 = np.asarray(state1.quote_panels[spot_mid].data, dtype=float)
    spot2 = np.asarray(state2.quote_panels[spot_mid].data, dtype=float)

    assert not np.allclose(spot1, spot2)


# =============================================================================
# Tests: FIXING Generation
# =============================================================================

def test_equity_fixing_reuses_spot_if_present() -> None:
    """EQUITY FIXING should return the same values as SPOT if already generated."""
    n_time = 3
    n_scenarios = 2

    cfg = SyntheticProviderConfig()
    state = _make_state(n_time=n_time, n_scenarios=n_scenarios)
    eq = _EquityGenerators(base_seed=7, config=cfg)

    spot_mid = MarketId(asset_class="EQUITY", mkt_type="SPOT", name="AAPL")
    fixing_mid = MarketId(asset_class="EQUITY", mkt_type="FIXING", name="AAPL")

    # Generate SPOT first.
    eq.generate_spot(spot_mid, state)
    # Generate FIXING.
    eq.generate_fixing(fixing_mid, state)

    spot = np.asarray(state.quote_panels[spot_mid].data, dtype=float)
    fixing = np.asarray(state.quote_panels[fixing_mid].data, dtype=float)

    np.testing.assert_array_equal(spot, fixing)


def test_equity_fixing_fallback_when_spot_missing() -> None:
    """EQUITY FIXING should produce constant fallback when SPOT is missing."""
    n_time = 3
    n_scenarios = 2

    cfg = SyntheticProviderConfig()
    state = _make_state(n_time=n_time, n_scenarios=n_scenarios)
    eq = _EquityGenerators(base_seed=7, config=cfg)

    fixing_mid = MarketId(asset_class="EQUITY", mkt_type="FIXING", name="UNKNOWN")

    # Generate FIXING without SPOT.
    eq.generate_fixing(fixing_mid, state)

    fixing = np.asarray(state.quote_panels[fixing_mid].data, dtype=float)

    assert fixing.shape == (n_time, n_scenarios)
    assert np.all(fixing == 100.0)  # Default fallback value


# =============================================================================
# Tests: VOL Generation
# =============================================================================

def test_equity_vol_produces_valid_cube() -> None:
    """EQUITY VOL should produce [T, S, n_exp, n_k] cube with positive values."""
    n_time = 3
    n_scenarios = 2

    cfg = SyntheticProviderConfig(
        vol=VolGridSmileSpec(
            expiries=np.array([0.25, 0.5, 1.0], dtype=float),
            strikes=np.array([80.0, 100.0, 120.0], dtype=float),
            atm_vol=0.20,
            skew=-0.10,
            smile=0.05,
        ),
    )

    state = _make_state(n_time=n_time, n_scenarios=n_scenarios)
    eq = _EquityGenerators(base_seed=7, config=cfg)

    spot_mid = MarketId(asset_class="EQUITY", mkt_type="SPOT", name="SPX")
    vol_mid = MarketId(asset_class="EQUITY", mkt_type="VOL", name="SPX")

    # Generate SPOT first (dependency).
    eq.generate_spot(spot_mid, state)

    # Generate VOL.
    eq.generate_vol_grid_strike_based(vol_mid, state)

    vol = np.asarray(state.vol_param_panels[vol_mid].data, dtype=float)

    assert vol.shape == (n_time, n_scenarios, 3, 3)  # [T, S, n_exp, n_k]
    assert np.all(np.isfinite(vol))
    assert np.min(vol) > 0.0


def test_equity_vol_exhibits_skew() -> None:
    """Equity vol surface should show negative skew (lower strikes have higher vol)."""
    n_time = 1
    n_scenarios = 1

    cfg = SyntheticProviderConfig(
        spot=SpotGbmSpec(initial_level=100.0),  # ATM at 100
        vol=VolGridSmileSpec(
            expiries=np.array([1.0], dtype=float),
            strikes=np.array([80.0, 100.0, 120.0], dtype=float),
            atm_vol=0.20,
            skew=-0.20,  # Negative skew
            smile=0.05,
            noise_scale=0.0,  # No noise for deterministic test
        ),
    )

    state = _make_state(n_time=n_time, n_scenarios=n_scenarios)
    eq = _EquityGenerators(base_seed=7, config=cfg)

    spot_mid = MarketId(asset_class="EQUITY", mkt_type="SPOT", name="SPX")
    vol_mid = MarketId(asset_class="EQUITY", mkt_type="VOL", name="SPX")

    eq.generate_spot(spot_mid, state)
    eq.generate_vol_grid_strike_based(vol_mid, state)

    vol = np.asarray(state.vol_param_panels[vol_mid].data, dtype=float)
    # Shape: [1, 1, 1, 3] -> squeeze to [3]
    vol_slice = vol[0, 0, 0, :]

    # With negative skew: vol(80) > vol(100) > vol(120)
    # (lower strikes have higher vol for equity)
    assert vol_slice[0] > vol_slice[1], f"Expected vol(80) > vol(100), got {vol_slice}"
    assert vol_slice[1] > vol_slice[2], f"Expected vol(100) > vol(120), got {vol_slice}"


def test_equity_vol_requires_spot() -> None:
    """EQUITY VOL generation should raise if SPOT is missing."""
    n_time = 2
    n_scenarios = 2

    cfg = SyntheticProviderConfig()
    state = _make_state(n_time=n_time, n_scenarios=n_scenarios)
    eq = _EquityGenerators(base_seed=7, config=cfg)

    vol_mid = MarketId(asset_class="EQUITY", mkt_type="VOL", name="SPX")

    with pytest.raises(ValueError, match="requires SPOT"):
        eq.generate_vol_grid_strike_based(vol_mid, state)


# =============================================================================
# Tests: Dividend Utilities
# =============================================================================

def test_adjust_spot_for_discrete_dividend_future() -> None:
    """Future dividend should subtract from spot."""
    spot = 100.0
    dividend = 2.0
    ex_date = 0.25  # 3 months from now
    current_time = 0.0

    adjusted = adjust_spot_for_discrete_dividend(
        spot=spot,
        dividend_amount=dividend,
        ex_date_fraction=ex_date,
        current_time=current_time,
    )

    assert adjusted == pytest.approx(98.0)


def test_adjust_spot_for_discrete_dividend_past() -> None:
    """Past dividend should not affect spot."""
    spot = 100.0
    dividend = 2.0
    ex_date = 0.1  # 1.2 months ago
    current_time = 0.25  # Current time is after ex-date

    adjusted = adjust_spot_for_discrete_dividend(
        spot=spot,
        dividend_amount=dividend,
        ex_date_fraction=ex_date,
        current_time=current_time,
    )

    assert adjusted == pytest.approx(100.0)


def test_compute_forward_with_continuous_dividend_yield() -> None:
    """Forward with continuous dividend yield: F = S × exp((r - q) × T)."""
    spot = 100.0
    r = 0.05
    q = 0.02
    T = 1.0

    forward = compute_forward_with_dividends(
        spot=spot,
        discount_rate=r,
        dividend_yield=q,
        expiry=T,
        discrete_dividends=None,
    )

    expected = spot * np.exp((r - q) * T)
    assert forward == pytest.approx(expected, rel=1e-10)


def test_compute_forward_with_discrete_dividends() -> None:
    """Forward with discrete dividends: F = (S - PV(divs)) × exp(r × T)."""
    spot = 100.0
    r = 0.05
    q = 0.0  # No continuous yield
    T = 1.0
    discrete_dividends = [(0.25, 1.0), (0.75, 1.0)]  # Two $1 dividends

    forward = compute_forward_with_dividends(
        spot=spot,
        discount_rate=r,
        dividend_yield=q,
        expiry=T,
        discrete_dividends=discrete_dividends,
    )

    # Manual calculation:
    # PV(div1) = 1.0 × exp(-0.05 × 0.25) ≈ 0.9876
    # PV(div2) = 1.0 × exp(-0.05 × 0.75) ≈ 0.9632
    # S_adj = 100 - 0.9876 - 0.9632 ≈ 98.049
    # F = 98.049 × exp(0.05 × 1.0) ≈ 103.07

    pv_div1 = 1.0 * np.exp(-r * 0.25)
    pv_div2 = 1.0 * np.exp(-r * 0.75)
    s_adj = spot - pv_div1 - pv_div2
    expected = s_adj * np.exp(r * T)

    assert forward == pytest.approx(expected, rel=1e-10)


def test_compute_forward_with_both_continuous_and_discrete() -> None:
    """Forward with both continuous yield and discrete dividends."""
    spot = 100.0
    r = 0.05
    q = 0.01  # 1% continuous yield
    T = 1.0
    discrete_dividends = [(0.5, 2.0)]  # $2 dividend at 6 months

    forward = compute_forward_with_dividends(
        spot=spot,
        discount_rate=r,
        dividend_yield=q,
        expiry=T,
        discrete_dividends=discrete_dividends,
    )

    # PV(div) = 2.0 × exp(-0.05 × 0.5) ≈ 1.9506
    # S_adj = 100 - 1.9506 ≈ 98.049
    # F = 98.049 × exp((0.05 - 0.01) × 1.0) ≈ 102.05

    pv_div = 2.0 * np.exp(-r * 0.5)
    s_adj = spot - pv_div
    expected = s_adj * np.exp((r - q) * T)

    assert forward == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Tests: Edge Cases
# =============================================================================

def test_equity_spot_single_time_point() -> None:
    """EQUITY SPOT with single time point should work."""
    cfg = SyntheticProviderConfig()
    state = _make_state(n_time=1, n_scenarios=5)
    eq = _EquityGenerators(base_seed=7, config=cfg)

    spot_mid = MarketId(asset_class="EQUITY", mkt_type="SPOT", name="AAPL")
    eq.generate_spot(spot_mid, state)

    spot = np.asarray(state.quote_panels[spot_mid].data, dtype=float)
    assert spot.shape == (1, 5)


def test_equity_spot_single_scenario() -> None:
    """EQUITY SPOT with single scenario should work."""
    cfg = SyntheticProviderConfig()
    state = _make_state(n_time=10, n_scenarios=1)
    eq = _EquityGenerators(base_seed=7, config=cfg)

    spot_mid = MarketId(asset_class="EQUITY", mkt_type="SPOT", name="AAPL")
    eq.generate_spot(spot_mid, state)

    spot = np.asarray(state.quote_panels[spot_mid].data, dtype=float)
    assert spot.shape == (10, 1)
