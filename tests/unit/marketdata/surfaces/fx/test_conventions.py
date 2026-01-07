from __future__ import annotations

import math

from src.marketdata.surfaces.conventions.fx_vol import FxDeltaConvention


def test_fx_delta_convention_strike_sanity_relative_to_forward() -> None:
    # With a standard smile convention, for the same |delta|:
    # - 25d put strike < forward
    # - 25d call strike > forward
    conv = FxDeltaConvention(delta_type="forward", premium_adjusted=False)

    spot = 100.0
    r_d = 0.02
    r_f = 0.01
    t = 1.0
    df_dom = math.exp(-r_d * t)
    df_for = math.exp(-r_f * t)
    fwd = spot * df_for / df_dom

    vol = 0.20
    d = 0.25

    k_put = conv.strike_from_abs_delta(
        option_type="put",
        abs_delta=d,
        spot=spot,
        df_dom=df_dom,
        df_for=df_for,
        vol=vol,
        expiry=t,
    )
    k_call = conv.strike_from_abs_delta(
        option_type="call",
        abs_delta=d,
        spot=spot,
        df_dom=df_dom,
        df_for=df_for,
        vol=vol,
        expiry=t,
    )

    assert k_put < fwd
    assert k_call > fwd