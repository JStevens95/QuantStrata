from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.surfaces.calibration.fx_smile_surface import (
    FxGridToSmileConfig,
    FxSmileToGridConfig,
    calibrate_fx_smile_to_grid_surface,
    extract_fx_smile_from_grid_surface,
)
from src.marketdata.surfaces.quotes.fx_smile import FxSmileQuotes, FxSmileSliceQuotes
from src.marketdata.surfaces.vol_surface import GridVolSurface


def _df_const(r: float):
    r = float(r)

    def _df(t: float) -> float:
        t = float(t)
        return float(np.exp(-r * max(t, 0.0)))

    return _df


def _make_simple_smile_quotes() -> FxSmileQuotes:
    # Two expiries, with mild skew/smile via RR/BF. Keep values realistic/stable.
    s1 = FxSmileSliceQuotes(
        expiry=0.5,
        atm_vol=0.12,
        rr_by_delta={0.25: -0.02, 0.10: -0.04},
        bf_by_delta={0.25: 0.005, 0.10: 0.010},
    )
    s2 = FxSmileSliceQuotes(
        expiry=1.0,
        atm_vol=0.11,
        rr_by_delta={0.25: -0.015, 0.10: -0.03},
        bf_by_delta={0.25: 0.004, 0.10: 0.008},
    )
    return FxSmileQuotes(slices=[s1, s2])


def test_calibrate_builds_grid_surface_basic_properties() -> None:
    smile = _make_simple_smile_quotes()
    spot = 1.10
    df_d = _df_const(0.02)
    df_f = _df_const(0.01)

    surf = calibrate_fx_smile_to_grid_surface(
        smile=smile,
        spot=spot,
        df_domestic=df_d,
        df_foreign=df_f,
        config=FxSmileToGridConfig(n_strikes=21, moneyness_width=0.25, extrapolation="flat"),
        validate=False,
        surface_id="TEST",
    )

    assert isinstance(surf, GridVolSurface)
    assert surf.strike_space == "absolute"
    assert surf.extrapolation == "flat"
    assert surf.surface_id == "TEST"

    # grid sanity
    assert surf.expiries.ndim == 1 and surf.expiries.size == 2
    assert np.all(np.diff(surf.expiries) > 0.0)
    assert surf.strikes.ndim == 1 and surf.strikes.size >= 5
    assert np.all(np.diff(surf.strikes) > 0.0)

    # vol grid positive finite
    assert surf.implied_vols.shape == (surf.expiries.size, surf.strikes.size)
    assert np.all(np.isfinite(surf.implied_vols))
    assert np.all(surf.implied_vols > 0.0)


def test_calibrate_validate_flag_controls_arbitrage_check_call(monkeypatch) -> None:
    # We don't care about the arbitrage logic here; just that validate=True calls it
    # and validate=False skips it.
    smile = _make_simple_smile_quotes()
    spot = 1.10
    df_d = _df_const(0.02)
    df_f = _df_const(0.01)

    called = {"n": 0}

    def _fake_check(**kwargs):
        called["n"] += 1

    monkeypatch.setattr(
        "src.marketdata.surfaces.calibration.fx_smile_surface.check_fx_grid_surface_no_static_arb",
        _fake_check,
    )

    calibrate_fx_smile_to_grid_surface(
        smile=smile,
        spot=spot,
        df_domestic=df_d,
        df_foreign=df_f,
        validate=False,
    )
    assert called["n"] == 0

    calibrate_fx_smile_to_grid_surface(
        smile=smile,
        spot=spot,
        df_domestic=df_d,
        df_foreign=df_f,
        validate=True,
    )
    assert called["n"] == 1


def test_extract_requires_absolute_strike_space() -> None:
    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([0.9, 1.0, 1.1], dtype=float)
    vols = np.full((2, 3), 0.12, dtype=float)

    surf = GridVolSurface(
        expiries=expiries,
        strikes=strikes,
        implied_vols=vols,
        extrapolation="flat",
        strike_space="spot_moneyness",  # should be rejected by extractor
    )

    with pytest.raises(ValueError, match="strike_space"):
        extract_fx_smile_from_grid_surface(
            surface=surf,
            spot=1.10,
            df_domestic=_df_const(0.02),
            df_foreign=_df_const(0.01),
        )


def test_calibrate_then_extract_roundtrip_recovers_quotes_reasonably() -> None:
    smile_in = _make_simple_smile_quotes()
    spot = 1.10
    df_d = _df_const(0.02)
    df_f = _df_const(0.01)

    # Calibrate to surface (no arb check to avoid coupling tests to arb thresholds)
    surf = calibrate_fx_smile_to_grid_surface(
        smile=smile_in,
        spot=spot,
        df_domestic=df_d,
        df_foreign=df_f,
        config=FxSmileToGridConfig(n_strikes=41, moneyness_width=0.35),
        validate=False,
    )

    # Extract back
    smile_out = extract_fx_smile_from_grid_surface(
        surface=surf,
        spot=spot,
        df_domestic=df_d,
        df_foreign=df_f,
        config=FxGridToSmileConfig(deltas=(0.25, 0.10), max_iter=80, tol=1e-12, damping=0.6),
    )

    # Same expiries
    assert smile_out.expiries() == pytest.approx(smile_in.expiries())

    # Compare per-expiry ATM/RR/BF approximately.
    # This is an *approximate* roundtrip because:
    # - calibration interpolates onto a common strike grid
    # - extraction uses fixed-point inversion on an interpolated surface
    tol_atm = 2e-3
    tol_rr = 5e-3
    tol_bf = 5e-3

    in_by_t = {float(s.expiry): s for s in smile_in}
    out_by_t = {float(s.expiry): s for s in smile_out}

    for t, s_in in in_by_t.items():
        s_out = out_by_t[t]
        assert float(s_out.atm_vol) == pytest.approx(float(s_in.atm_vol), abs=tol_atm)

        for d in (0.25, 0.10):
            assert float(s_out.rr_by_delta[d]) == pytest.approx(float(s_in.rr_by_delta[d]), abs=tol_rr)
            assert float(s_out.bf_by_delta[d]) == pytest.approx(float(s_in.bf_by_delta[d]), abs=tol_bf)


@pytest.mark.parametrize("bad_spot", [0.0, -1.0])
def test_calibrate_rejects_non_positive_spot(bad_spot: float) -> None:
    with pytest.raises(ValueError, match="spot must be > 0"):
        calibrate_fx_smile_to_grid_surface(
            smile=_make_simple_smile_quotes(),
            spot=bad_spot,
            df_domestic=_df_const(0.02),
            df_foreign=_df_const(0.01),
            validate=False,
        )


@pytest.mark.parametrize("bad_deltas", [(), (0.0,), (1.0,), (-0.1,), (1.2,)])
def test_extract_rejects_invalid_deltas(bad_deltas) -> None:
    expiries = np.array([0.5], dtype=float)
    strikes = np.array([0.9, 1.0, 1.1], dtype=float)
    vols = np.full((1, 3), 0.12, dtype=float)

    surf = GridVolSurface(
        expiries=expiries,
        strikes=strikes,
        implied_vols=vols,
        extrapolation="flat",
        strike_space="absolute",
    )

    with pytest.raises(ValueError):
        extract_fx_smile_from_grid_surface(
            surface=surf,
            spot=1.10,
            df_domestic=_df_const(0.02),
            df_foreign=_df_const(0.01),
            config=FxGridToSmileConfig(deltas=bad_deltas),
        )