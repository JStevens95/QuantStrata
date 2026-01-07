from __future__ import annotations

import numpy as np
from typing import Union

from src.marketdata.curves.discount import FlatDiscountCurve, ZeroRateDiscountCurve
from src.marketdata.integration.quantlib.context import QlContext, require_quantlib, yearfrac_to_ql_date


CurveLike = Union[FlatDiscountCurve, ZeroRateDiscountCurve]


def curve_to_yts_handle(curve: CurveLike, *, ctx: QlContext):
    """
    Convert a QuantStrata discount-curve object into a QuantLib YieldTermStructureHandle.

    Supported (V2 milestone)
    ------------------------
    - FlatDiscountCurve       -> ql.FlatForward (continuous compounding)
    - ZeroRateDiscountCurve   -> ql.ZeroCurve (continuous compounding, linear interpolation)

    Notes
    -----
    - This adapter is intentionally minimal: it preserves your internal interfaces as canonical.
    - Later upgrades can build ql.RateHelper-based curves from instruments directly.
    """
    ql = require_quantlib()
    ctx2 = ctx.with_defaults()

    # Ensure QuantLib global evaluation date is aligned with the Market.asof.
    asof_date = ctx2.set_evaluation_date()

    # ---- Flat curve ----
    if isinstance(curve, FlatDiscountCurve):
        r = float(curve.continuously_compounded_rate)

        # FlatForward builds a term structure with a single constant rate.
        # We use continuous compounding to match your curve definitions.
        ts = ql.FlatForward(
            asof_date,                    # reference date
            ql.QuoteHandle(ql.SimpleQuote(r)),  # market quote handle
            ctx2.day_count,               # day count convention
            ql.Continuous,                # compounding
            ql.Annual,                    # frequency (irrelevant for continuous, but required)
        )
        return ql.YieldTermStructureHandle(ts)

    # ---- Zero rate curve on a tenor grid ----
    if isinstance(curve, ZeroRateDiscountCurve):
        tenors = np.asarray(curve.tenors, dtype=float).reshape(-1)
        zeros = np.asarray(curve.zero_rates, dtype=float).reshape(-1)

        if tenors.size == 0:
            raise ValueError("ZeroRateDiscountCurve.tenors must not be empty.")
        if tenors.size != zeros.size:
            raise ValueError("ZeroRateDiscountCurve.tenors and zero_rates must have same length.")

        # Convert each year-fraction node into a QuantLib Date.
        dates = [yearfrac_to_ql_date(asof=asof_date, yearfrac=float(t)) for t in tenors]

        # QuantLib expects strictly increasing dates.
        # If your tenors are strictly increasing, these dates should be increasing too.
        # (In rare cases with small tenors, rounding could collide — guard that.)
        for i in range(1, len(dates)):
            if dates[i] <= dates[i - 1]:
                raise ValueError(
                    "Tenor->date mapping produced non-increasing dates. "
                    "This can happen due to day-rounding. Increase tenor spacing or improve date mapping."
                )

        # Build a ZeroCurve using continuous compounding.
        ts = ql.ZeroCurve(
            dates,
            [float(r) for r in zeros],    # list of zero rates
            ctx2.day_count,
            ctx2.calendar,
            ql.Linear(),                  # interpolation
            ql.Continuous,                # compounding convention
            ql.Annual,                    # frequency (again irrelevant for continuous)
        )
        return ql.YieldTermStructureHandle(ts)

    raise TypeError(f"Unsupported curve type for QuantLib adapter: {type(curve).__name__}")