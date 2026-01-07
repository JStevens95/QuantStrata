from __future__ import annotations

import math

from src.marketdata.surfaces.fx.calibration import (
    FxGridToSmileConfig,
    FxSmileToGridConfig,
    calibrate_fx_smile_to_grid_surface,
    extract_fx_smile_from_grid_surface,
)
from src.marketdata.surfaces.fx.quotes import FxSmileQuotes, FxSmileSliceQuotes


def test_fx_smile_to_grid_and_extract_roundtrip_flat_smile() -> None:
    # Flat smile: RR=0, BF=0 so all wings equal ATM.
    smile = FxSmileQuotes(
        slices=[
            FxSmileSliceQuotes(expiry=0.5, atm_vol=0.20, rr_by_delta={0.25: 0.0}, bf_by_delta={0.25: 0.0}),
            FxSmileSliceQuotes(expiry=1.0, atm_vol=0.20, rr_by_delta={0.25: 0.0}, bf_by_delta={0.25: 0.0}),
        ]
    )

    spot = 100.0
    r_d = 0.02
    r_f = 0.01
    df_dom = lambda t: math.exp(-r_d * float(t))
    df_for = lambda t: math.exp(-r_f * float(t))

    surf = calibrate_fx_smile_to_grid_surface(
        smile=smile,
        spot=spot,
        df_domestic=df_dom,
        df_foreign=df_for,
        config=FxSmileToGridConfig(n_strikes=21, moneyness_width=0.25, extrapolation="flat"),
        validate=True,
    )

    assert surf.implied_vols.shape[0] == 2
    assert surf.implied_vols.shape[1] >= 21
    assert surf.strike_space == "absolute"

    # Extract back
    out = extract_fx_smile_from_grid_surface(
        surface=surf,
        spot=spot,
        df_domestic=df_dom,
        df_foreign=df_for,
        config=FxGridToSmileConfig(deltas=(0.25,), max_iter=50, tol=1e-10, damping=0.5),
    )

    # Should recover RR≈0, BF≈0, ATM≈0.20
    for slc in out:
        assert abs(float(slc.atm_vol) - 0.20) < 5e-4
        assert abs(float(slc.rr_by_delta[0.25])) < 5e-4
        assert abs(float(slc.bf_by_delta[0.25])) < 5e-4