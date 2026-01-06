from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pytest

from src.marketdata.curves.bootstrapper import (
    DepositQuote,
    ParSwapQuote,
    bootstrap_discount_curve,
)


def _par_rate_from_curve(*, maturity: float, pay_freq: int, df) -> float:
    """
    Par rate implied by a discount curve under the same single-curve identity used in bootstrap:

        R * sum_i alpha_i DF(t_i) = 1 - DF(T)
        => R = (1 - DF(T)) / sum_i alpha_i DF(t_i)

    where t_i are fixed-leg pay times (regular schedule) and alpha_i are accruals.
    """
    t = float(maturity)
    if t <= 0.0:
        raise ValueError("maturity must be > 0.")
    if pay_freq < 1:
        raise ValueError("pay_freq must be >= 1.")

    step = 1.0 / float(pay_freq)
    times = np.arange(step, t + 1e-12, step, dtype=float)
    if times.size == 0 or abs(times[-1] - t) > 1e-12:
        times = np.concatenate([times, np.array([t], dtype=float)])

    # accruals
    prev = 0.0
    denom = 0.0
    for ti in times.tolist():
        alpha = float(ti - prev)
        denom += alpha * float(df(ti))
        prev = float(ti)

    df_t = float(df(t))
    numer = 1.0 - df_t
    if denom <= 0.0:
        raise ValueError("Invalid denominator for par rate.")
    return float(numer / denom)


def test_bootstrap_deposits_only_produces_valid_curve() -> None:
    inst = [
        DepositQuote(maturity=0.25, rate=0.02, compounding="simple"),
        DepositQuote(maturity=0.50, rate=0.021, compounding="simple"),
        DepositQuote(maturity=1.00, rate=0.022, compounding="simple"),
    ]

    res = bootstrap_discount_curve(instruments=inst, extrapolation="flat")

    assert res.tenors.shape == (3,)
    assert res.dfs.shape == (3,)
    assert res.zero_rates.shape == (3,)

    # Basic sanity
    assert np.all(np.isfinite(res.dfs))
    assert np.all(res.dfs > 0.0)
    assert np.all(res.dfs <= 1.0 + 1e-12)

    # Curve works
    assert res.curve.df(0.0) == pytest.approx(1.0)
    assert res.curve.df(0.25) == pytest.approx(res.dfs[0])
    assert res.curve.df(1.0) == pytest.approx(res.dfs[-1])

    # For positive rates, DF should decrease with maturity
    assert np.all(np.diff(res.dfs) < 0.0)


def test_bootstrap_deposits_and_swaps_reprices_par_swaps() -> None:
    inst = [
        DepositQuote(maturity=0.25, rate=0.0200, compounding="simple"),
        DepositQuote(maturity=0.50, rate=0.0210, compounding="simple"),
        # Annual-pay swaps => coupon dates are integers.
        ParSwapQuote(maturity=1.0, fixed_rate=0.0220, pay_freq=1),
        ParSwapQuote(maturity=2.0, fixed_rate=0.0240, pay_freq=1),
        ParSwapQuote(maturity=3.0, fixed_rate=0.0250, pay_freq=1),
        ParSwapQuote(maturity=4.0, fixed_rate=0.0260, pay_freq=1),
        ParSwapQuote(maturity=5.0, fixed_rate=0.0270, pay_freq=1),
    ]

    res = bootstrap_discount_curve(instruments=inst, extrapolation="flat")
    curve = res.curve

    for q in [x for x in inst if isinstance(x, ParSwapQuote)]:
        r_impl = _par_rate_from_curve(maturity=float(q.maturity), pay_freq=int(q.pay_freq), df=curve.df)
        assert r_impl == pytest.approx(float(q.fixed_rate), rel=0.0, abs=5e-10)


def test_swap_bootstrap_missing_earlier_df_raises() -> None:
    # Pay freq=2 implies payment at 0.5 and 1.0. We *don't* provide a DF at 0.5 -> should fail.
    inst = [
        DepositQuote(maturity=0.25, rate=0.02, compounding="simple"),
        ParSwapQuote(maturity=1.0, fixed_rate=0.02, pay_freq=2),
    ]

    with pytest.raises(ValueError, match="missing.*DF|Missing DF|Cannot bootstrap swap"):
        bootstrap_discount_curve(instruments=inst)


def test_deposit_simple_compounding_guard_raises_when_denominator_nonpositive() -> None:
    # 1 + r*T <= 0 is invalid for simple compounding
    inst = [DepositQuote(maturity=1.0, rate=-1.0, compounding="simple")]
    with pytest.raises(ValueError, match="non-positive discount factor|1 \\+ r\\*T"):
        bootstrap_discount_curve(instruments=inst)