from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Tuple

from src.marketdata.core.conventions import DayCount


def yearfrac(*, start: date, end: date, dc: DayCount) -> float:
    """
    Compute year fraction between two dates under a given day-count convention.

    Parameters
    ----------
    start, end:
        Python `datetime.date` objects.
    dc:
        DayCount enum.

    Returns
    -------
    float
        Year fraction. If end < start, the result is negative.

    Implemented conventions (V2)
    ----------------------------
    - ACT365F: (end-start).days / 365
    - ACT360 : (end-start).days / 360
    - THIRTY360: 30/360 "basic" (implemented here as 30/360 US/NASD)

    Not implemented yet
    -------------------
    - ACTACT_ISDA (raise NotImplementedError)
    """
    if not isinstance(start, date) or not isinstance(end, date):
        raise TypeError("start and end must be datetime.date instances.")
    if not isinstance(dc, DayCount):
        # Defensive: ensure caller passed your canonical enum.
        raise TypeError(f"dc must be a DayCount. Got {type(dc).__name__}.")

    # Allow negative year fractions by swapping and reapplying sign.
    if end >= start:
        sign = 1.0
        d0, d1 = start, end
    else:
        sign = -1.0
        d0, d1 = end, start

    if dc == DayCount.ACT365F:
        return sign * ((d1 - d0).days / 365.0)

    if dc == DayCount.ACT360:
        return sign * ((d1 - d0).days / 360.0)

    if dc == DayCount.THIRTY360:
        return sign * _yearfrac_30_360_us(d0, d1)

    if dc == DayCount.ACTACT_ISDA:
        raise NotImplementedError("ACTACT_ISDA is not implemented yet (V2).")

    # If you add DayCount members later and forget to implement them, fail loudly.
    raise NotImplementedError(f"DayCount {dc} is not implemented.")


def _yearfrac_30_360_us(start: date, end: date) -> float:
    """
    30/360 US (NASD) day count.

    Rule summary (US/NASD)
    ----------------------
    Let d1 = start.day, d2 = end.day.

    1) If d1 == 31 -> d1 = 30
    2) If d2 == 31 and d1 in {30,31} -> d2 = 30
    3) Some variants treat end-of-Feb specially; we keep a pragmatic implementation:
       - If start is last day of Feb -> d1 = 30
       - If end is last day of Feb and start day >= 30 -> d2 = 30

    Year fraction:
        (360*(Y2-Y1) + 30*(M2-M1) + (d2-d1)) / 360
    """
    y1, m1, d1 = start.year, start.month, start.day
    y2, m2, d2 = end.year, end.month, end.day

    # End-of-Feb helpers (common practical adjustment)
    if _is_last_day_of_feb(start):
        d1 = 30
    if _is_last_day_of_feb(end) and d1 >= 30:
        d2 = 30

    # Standard US/NASD adjustments around month-end 31st
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 in (30, 31):
        d2 = 30

    days_360 = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return float(days_360 / 360.0)


def _is_last_day_of_feb(d: date) -> bool:
    """Return True if `d` is the last calendar day of February for that year."""
    if d.month != 2:
        return False
    # Feb has 29 days in leap years, else 28
    return (d.day == 29) or (d.day == 28 and not _is_leap_year(d.year))


def _is_leap_year(year: int) -> bool:
    """Gregorian leap year rule."""
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))