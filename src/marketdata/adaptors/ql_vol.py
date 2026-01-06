from __future__ import annotations

import numpy as np
from typing import Union

from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface
from src.marketdata.adaptors.ql_utils import QlContext, require_quantlib, yearfrac_to_ql_date


VolLike = Union[FlatVolSurface, GridVolSurface]


def vol_surface_to_black_vol_handle(surface: VolLike, *, ctx: QlContext):
    """
    Convert a QuantStrata vol surface into a QuantLib BlackVolTermStructureHandle.

    Supported (V2 milestone)
    ------------------------
    - FlatVolSurface  -> ql.BlackConstantVol
    - GridVolSurface  -> ql.BlackVarianceSurface

    Assumptions
    -----------
    - GridVolSurface.strike_space must be "absolute" (K in price units).
    - Expiries are year-fractions; we convert to dates by adding ~T*365 days.
    """
    ql = require_quantlib()
    ctx2 = ctx.with_defaults()

    # Align QuantLib global evaluation date with the Market.asof.
    asof_date = ctx2.set_evaluation_date()

    # ---- Flat vol ----
    if isinstance(surface, FlatVolSurface):
        sigma = float(surface.sigma)

        bv = ql.BlackConstantVol(
            asof_date,     # reference date
            ctx2.calendar,
            sigma,
            ctx2.day_count,
        )
        return ql.BlackVolTermStructureHandle(bv)

    # ---- Grid vol surface ----
    if isinstance(surface, GridVolSurface):
        if surface.strike_space != "absolute":
            raise ValueError(
                "QuantLib adapter expects GridVolSurface.strike_space == 'absolute' "
                f"but got {surface.strike_space!r}."
            )

        expiries = np.asarray(surface.expiries, dtype=float).reshape(-1)
        strikes = np.asarray(surface.strikes, dtype=float).reshape(-1)
        vols = np.asarray(surface.implied_vols, dtype=float)

        if vols.shape != (expiries.size, strikes.size):
            raise ValueError(
                "GridVolSurface implied_vols shape mismatch. "
                f"Expected {(expiries.size, strikes.size)}, got {vols.shape}."
            )

        # Convert expiry nodes into QuantLib Dates.
        dates = [yearfrac_to_ql_date(asof=asof_date, yearfrac=float(t)) for t in expiries]

        for j in range(1, len(dates)):
            if dates[j] <= dates[j - 1]:
                raise ValueError(
                    "Expiry->date mapping produced non-increasing dates. "
                    "This can happen due to day-rounding. Improve date mapping for very short expiries."
                )

        # QuantLib vol matrices are indexed [strike][date].
        mat = ql.Matrix(int(strikes.size), int(len(dates)))
        for j in range(len(dates)):
            for i in range(strikes.size):
                mat[i][j] = float(vols[j, i])

        bvs = ql.BlackVarianceSurface(
            asof_date,
            ctx2.calendar,
            dates,
            [float(k) for k in strikes],
            mat,
            ctx2.day_count,
        )
        return ql.BlackVolTermStructureHandle(bvs)

    raise TypeError(f"Unsupported vol surface type for QuantLib adapter: {type(surface).__name__}")