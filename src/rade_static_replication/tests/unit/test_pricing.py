"""Unit tests for pricing kernels (sanity / parity)."""
import math

import pytest

from src.rade_static_replication.pricing.kernels.fx import price_fx_forward, price_fx_vanilla


def test_fx_put_call_parity():
    S, K, T, r_d, r_f, vol = 1.10, 1.12, 0.75, 0.045, 0.032, 0.10
    call = price_fx_vanilla(S, K, T, r_d, r_f, vol, True, 1.0)
    put = price_fx_vanilla(S, K, T, r_d, r_f, vol, False, 1.0)
    forward_pv = S * math.exp(-r_f * T) - K * math.exp(-r_d * T)
    assert call - put == pytest.approx(price_fx_forward(S, K, T, r_d, r_f, 1.0, 1.0), abs=1e-12)
    assert call - put == pytest.approx(forward_pv, abs=1e-12)


def test_fx_call_intrinsic_at_zero_vol():
    pv = price_fx_vanilla(1.20, 1.10, 0.0, 0.04, 0.03, 0.0, True, 1.0)
    assert pv == pytest.approx(0.10)
