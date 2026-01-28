from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.curves.bootstrapper import DepositQuote, ParSwapQuote, bootstrap_discount_curve


def test_bootstrap_native_deposits_only() -> None:
    res = bootstrap_discount_curve(
        instruments=[
            DepositQuote(label="DEP 6M", t=0.5, rate=0.02, compounding="simple"),
            DepositQuote(label="DEP 1Y", t=1.0, rate=0.025, compounding="continuous"),
        ],
        engine="native",
    )

    assert res.tenors.shape == (2,)
    assert res.dfs.shape == (2,)
    assert np.all(np.isfinite(res.dfs))
    assert np.all(res.dfs > 0.0)
    assert np.all(np.isfinite(res.zero_rates))

    # curve object should price df at the bootstrapped tenors
    df_1y = float(res.curve.df(1.0))
    assert np.isfinite(df_1y)
    assert df_1y > 0.0


def test_bootstrap_native_swap_requires_prior_coupon_dfs() -> None:
    # fixed_freq="6M" implies coupon at 0.5 before maturity=1.0
    with pytest.raises(ValueError, match="earlier payment DF is missing"):
        _ = bootstrap_discount_curve(
            instruments=[ParSwapQuote(label="SWAP 1Y", kind="IRS", maturity_t=1.0, par_rate=0.02, fixed_freq="6M")],
            engine="native",
        )

    # include a 0.5y deposit so the swap can be bootstrapped
    res = bootstrap_discount_curve(
        instruments=[
            DepositQuote(label="DEP 6M", t=0.5, rate=0.02, compounding="simple"),
            ParSwapQuote(label="SWAP 1Y", kind="IRS", maturity_t=1.0, par_rate=0.02, fixed_freq="6M"),
        ],
        engine="native",
    )

    assert res.tenors.size == 2
    assert np.all(res.dfs > 0.0)