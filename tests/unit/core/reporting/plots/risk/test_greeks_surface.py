"""Smoke tests for Greeks surface plotter (no display: use Agg backend)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")  # Non-interactive; unit tests must not open plot windows

import numpy as np

from src.core.reporting.plots.risk.greeks_surface import (
    greek_grid_from_sensitivities,
    plot_greeks_surface,
)
from src.risk.sensitivities.result import SensitivitiesReport, SensitivityKey, SensitivityRow


def test_plot_greeks_surface_smoke() -> None:
    """Call plotter with dummy grid; no crash."""
    expiries = np.array([0.25, 0.5, 1.0])
    strikes = np.array([0.95, 1.0, 1.05])
    grid = np.zeros((3, 3))
    grid[0, :] = [0.3, 0.5, 0.7]
    grid[1, :] = [0.35, 0.55, 0.75]
    grid[2, :] = [0.4, 0.6, 0.8]
    fig = plot_greeks_surface(expiries, strikes, grid, greek_name="delta")
    assert fig is not None
    # Main axes + colorbar axes
    assert len(fig.axes) >= 1


def test_greek_grid_from_sensitivities_empty() -> None:
    """Empty report returns empty arrays and 0x0 matrix."""
    report = SensitivitiesReport(rows=[])
    expiries, strikes, z = greek_grid_from_sensitivities(report, "delta")
    assert expiries.size == 0
    assert strikes.size == 0
    assert z.shape == (0, 0)


def test_greek_grid_from_sensitivities_with_expiry_strike() -> None:
    """Report with expiry/strike keys builds grid."""
    report = SensitivitiesReport(rows=[
        SensitivityRow(key=SensitivityKey(greek="delta", expiry=0.25, strike=1.0), value=0.5, method="analytic"),
        SensitivityRow(key=SensitivityKey(greek="delta", expiry=0.5, strike=1.0), value=0.55, method="analytic"),
    ])
    expiries, strikes, z = greek_grid_from_sensitivities(report, "delta")
    assert expiries.shape == (2,)
    assert strikes.shape == (1,)
    assert z.shape == (2, 1)
    assert z[0, 0] == pytest.approx(0.5)
    assert z[1, 0] == pytest.approx(0.55)
