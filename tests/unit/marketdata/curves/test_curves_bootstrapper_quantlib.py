from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.curves.bootstrapper import DepositQuote, ParSwapQuote, bootstrap_discount_curve
from src.marketdata.integration.quantlib.context import require_quantlib


def test_bootstrap_discount_curve_quantlib_engine_runs_or_skips() -> None:
    try:
        _ = require_quantlib()
    except ImportError:
        pytest.skip("QuantLib not installed; skipping quantlib bootstrap engine test.")

    instruments = [
        DepositQuote(maturity=0.25, rate=0.02, compounding="simple"),
        DepositQuote(maturity=0.50, rate=0.021, compounding="simple"),
        ParSwapQuote(maturity=1.0, fixed_rate=0.022, pay_freq=2),
        ParSwapQuote(maturity=2.0, fixed_rate=0.024, pay_freq=2),
    ]

    res = bootstrap_discount_curve(
        instruments=instruments,
        engine="quantlib",
        asof="2026-01-05",
        output_tenors=np.array([0.25, 0.5, 1.0, 2.0], dtype=float),
    )

    assert res.tenors.ndim == 1
    assert res.dfs.ndim == 1
    assert res.tenors.size == res.dfs.size == res.zero_rates.size

    assert np.all(np.isfinite(res.dfs))
    assert np.all(res.dfs > 0.0)

    df_2y = float(res.curve.df(2.0))
    assert np.isfinite(df_2y)
    assert df_2y > 0.0