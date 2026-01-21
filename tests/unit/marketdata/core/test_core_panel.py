from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.panel import Panel


def test_panel_axis_mismatch_raises() -> None:
    x = np.zeros((3, 2), dtype=float)
    with pytest.raises(ValueError, match="axis mismatch"):
        Panel(data=x, axis_names=("time",))  # wrong length


def test_panel_empty_axis_name_raises() -> None:
    x = np.zeros((3,), dtype=float)
    with pytest.raises(ValueError, match="non-empty"):
        Panel(data=x, axis_names=(" ",))


def test_panel_scalar_at_ndim1() -> None:
    x = np.array([1.0, 2.0, 3.0], dtype=float)
    p = Panel(data=x, axis_names=("time",))
    assert p.scalar_at(1) == 2.0


def test_panel_scalar_at_ndim2() -> None:
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float)  # [T,S]
    p = Panel(data=x, axis_names=("time", "scenario"))
    assert p.scalar_at(2, 1) == 30.0


def test_panel_scalar_at_wrong_ndim_raises() -> None:
    x = np.zeros((3, 2, 4), dtype=float)
    p = Panel(data=x, axis_names=("time", "scenario", "k"))
    with pytest.raises(ValueError, match="supports ndim"):
        p.scalar_at(0, 0)