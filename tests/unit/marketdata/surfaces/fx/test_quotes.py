from __future__ import annotations

from src.marketdata.surfaces.fx.quotes import FxSmileQuotes, FxSmileSliceQuotes


def test_fx_smile_slice_quotes_call_put_reconstruction() -> None:
    slc = FxSmileSliceQuotes(
        expiry=1.0,
        atm_vol=0.20,
        rr_by_delta={0.25: 0.02},
        bf_by_delta={0.25: 0.01},
    )

    # sigma_call = atm + bf + 0.5*rr = 0.20 + 0.01 + 0.01 = 0.22
    # sigma_put  = atm + bf - 0.5*rr = 0.20 + 0.01 - 0.01 = 0.20
    assert abs(slc.vol_call(0.25) - 0.22) < 1e-15
    assert abs(slc.vol_put(0.25) - 0.20) < 1e-15


def test_fx_smile_quotes_sorts_by_expiry() -> None:
    a = FxSmileSliceQuotes(expiry=2.0, atm_vol=0.2, rr_by_delta={}, bf_by_delta={})
    b = FxSmileSliceQuotes(expiry=1.0, atm_vol=0.2, rr_by_delta={}, bf_by_delta={})

    q = FxSmileQuotes(slices=[a, b])
    assert q.expiries() == [1.0, 2.0]