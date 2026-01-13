from __future__ import annotations

import numpy as np

from src.marketdata.core.panel import Panel
from src.marketdata.builders.panels import (
    make_time_grid,
    make_quote_panel,
    make_zero_curve_panel,
    make_grid_vol_panel,
)


def test_make_time_grid_basic() -> None:
    dates = make_time_grid(start="2026-01-07", n_t=3, step="D")
    assert dates == ["2026-01-07", "2026-01-08", "2026-01-09"]


def test_make_quote_panel_from_callable_shape_and_values() -> None:
    p = make_quote_panel(
        n_t=3,
        n_s=2,
        values=lambda ti, si: 100.0 + ti + 10.0 * si,
    )
    assert isinstance(p, Panel)
    assert p.data.shape == (3, 2)
    assert p.axis_names == ("time", "scenario")
    assert p.scalar_at(0, 0) == 100.0
    assert p.scalar_at(2, 1) == 100.0 + 2 + 10.0


def test_make_zero_curve_panel_shape_and_blocks() -> None:
    dates = ["2026-01-07", "2026-01-08"]
    tenors = np.array([0.0, 0.5, 1.0], dtype=float)

    def z_fn(ti: int, si: int) -> np.ndarray:
        base = np.array([0.02, 0.021, 0.022], dtype=float)
        return base + 0.001 * ti + 0.01 * si

    p = make_zero_curve_panel(
        dates=dates,
        n_s=2,
        tenors=tenors,
        zero_rates=z_fn,
    )

    assert p.data.shape == (2, 2, 3, 2)
    # tenor column is constant
    assert np.allclose(p.data[0, 0, :, 0], tenors)
    assert np.allclose(p.data[1, 1, :, 0], tenors)
    # zero column matches callable
    assert np.allclose(p.data[0, 0, :, 1], z_fn(0, 0))
    assert np.allclose(p.data[1, 1, :, 1], z_fn(1, 1))


def test_make_grid_vol_panel_shape_and_positive() -> None:
    dates = ["2026-01-07", "2026-01-08"]
    spot = make_quote_panel(n_t=2, n_s=2, values=lambda ti, si: 1.10 + 0.01 * ti + 0.02 * si)

    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([1.0, 1.1, 1.2], dtype=float)

    p = make_grid_vol_panel(
        dates=dates,
        n_s=2,
        spot_panel=spot,
        expiries=expiries,
        strikes=strikes,
    )

    assert p.data.shape == (2, 2, 2, 3)
    assert np.all(np.isfinite(p.data))
    assert float(np.min(p.data)) > 0.0