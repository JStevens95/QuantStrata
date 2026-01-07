from __future__ import annotations

import numpy as np

from src.marketdata.providers.synthetic.specs import (
    SpotGbmSpec,
    CurveZeroSpec,
    CurveBootstrapSpec,
    VolGridSmileSpec,
)


def test_spot_gbm_spec_constructs() -> None:
    spec = SpotGbmSpec(initial_level=1.10)
    assert spec.initial_level == 1.10
    assert spec.vol >= 0.0


def test_curve_zero_spec_constructs() -> None:
    spec = CurveZeroSpec(tenors=np.array([0.25, 1.0], dtype=float))
    assert spec.tenors.shape == (2,)
    assert isinstance(spec.extrapolation, str)
    assert spec.extrapolation.strip() != ""


def test_curve_bootstrap_spec_has_engine_field() -> None:
    spec = CurveBootstrapSpec()
    # engine is a Literal alias, runtime it’s a string
    assert isinstance(spec.engine, str)
    assert spec.engine in {"native", "quantlib"}


def test_vol_grid_smile_spec_constructs() -> None:
    spec = VolGridSmileSpec(
        expiries=np.array([0.25, 0.5], dtype=float),
        strikes=np.array([0.9, 1.0, 1.1], dtype=float),
    )
    assert spec.expiries.shape == (2,)
    assert spec.strikes.shape == (3,)
    assert spec.atm_vol > 0.0