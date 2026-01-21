from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.synthetic.schemas import (
    SyntheticSchemaRuntime,
    curve_params_schema,
    quote_ts_schema,
    vol_params_schema_flattened,
)


def test_quote_ts_schema_accepts_exact_shape() -> None:
    """Quote schema is strictly [T,S]."""
    schema = quote_ts_schema(schema_id="QUOTE.TEST")
    rt = SyntheticSchemaRuntime(n_time=2, n_scenarios=3)

    ok = np.ones((2, 3), dtype=float)
    schema.panel_schema.shape_fn(ok, rt)  # should not raise

    bad = np.ones((2, 3, 1), dtype=float)
    with pytest.raises(ValueError):
        schema.panel_schema.shape_fn(bad, rt)


def test_curve_params_schema_requires_last_dim_2() -> None:
    """Curve params schema requires [T,S,K,2]."""
    schema = curve_params_schema(schema_id="CURVE.TEST")
    rt = SyntheticSchemaRuntime(n_time=2, n_scenarios=3)

    ok = np.ones((2, 3, 5, 2), dtype=float)
    schema.panel_schema.shape_fn(ok, rt)

    bad = np.ones((2, 3, 5, 3), dtype=float)
    with pytest.raises(ValueError):
        schema.panel_schema.shape_fn(bad, rt)


def test_vol_params_flattened_schema_requires_3d() -> None:
    """Flattened vol params schema requires [T,S,P]."""
    schema = vol_params_schema_flattened(schema_id="VOL.TEST")
    rt = SyntheticSchemaRuntime(n_time=2, n_scenarios=3)

    ok = np.ones((2, 3, 10), dtype=float)
    schema.panel_schema.shape_fn(ok, rt)

    bad = np.ones((2, 3, 10, 1), dtype=float)
    with pytest.raises(ValueError):
        schema.panel_schema.shape_fn(bad, rt)