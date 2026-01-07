from __future__ import annotations

import math
import numpy as np
import pytest

from src.marketdata.surfaces.arbitrage import (
    FxSurfaceArbitrageConfig,
    check_calendar_no_arb_total_variance,
    check_fx_grid_surface_no_static_arb,
)


def test_calendar_no_arb_total_variance_detects_decrease() -> None:
    expiries = np.array([1.0, 2.0], dtype=float)
    strikes = np.array([90.0, 100.0], dtype=float)

    # make total variance decrease at strike 90:
    # w1 = 0.30^2 * 1 = 0.09
    # w2 = 0.20^2 * 2 = 0.08  (decrease)
    vols = np.array([[0.30, 0.25], [0.20, 0.25]], dtype=float)

    with pytest.raises(ValueError, match="Calendar arbitrage sanity check failed"):
        check_calendar_no_arb_total_variance(expiries=expiries, strikes=strikes, vols=vols)


def test_fx_grid_surface_static_arb_checks_can_pass_on_flat_surface() -> None:
    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([90.0, 100.0, 110.0], dtype=float)
    vols = np.full((2, 3), 0.2, dtype=float)

    r_d = 0.02
    r_f = 0.01
    df_dom = lambda t: math.exp(-r_d * float(t))
    df_for = lambda t: math.exp(-r_f * float(t))

    # should not raise
    check_fx_grid_surface_no_static_arb(
        expiries=expiries,
        strikes=strikes,
        vols=vols,
        spot=100.0,
        df_domestic=df_dom,
        df_foreign=df_for,
        config=FxSurfaceArbitrageConfig(tol=1e-10, check_butterfly=True),
    )