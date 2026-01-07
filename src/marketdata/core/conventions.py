# src/marketdata/core/conventions.py
from __future__ import annotations

from enum import Enum
from typing import Any


class _StrEnum(str, Enum):
    """
    Small helper base class: string-valued Enum.

    Benefits
    --------
    - JSON/cache-friendly: values are plain strings.
    - Stable repr/printing: str(MyEnum.X) == "X" (via .value).
    """

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def from_value(cls, value: Any) -> "_StrEnum":
        """
        Parse an enum from a string-like value.

        Accepts:
        - the Enum instance (returns as-is)
        - a string matching either name or value (case-insensitive)
        """
        if isinstance(value, cls):
            return value

        raw = str(value).strip()
        if not raw:
            raise ValueError(f"Cannot parse empty {cls.__name__}.")

        # Try name match (case-insensitive)
        for member in cls:
            if member.name.upper() == raw.upper():
                return member

        # Try value match (case-insensitive)
        for member in cls:
            if str(member.value).upper() == raw.upper():
                return member

        valid = ", ".join(m.name for m in cls)
        raise ValueError(f"Unknown {cls.__name__}={value!r}. Valid: {valid}.")


class DayCount(_StrEnum):
    """
    Day-count convention identifiers.

    Notes
    -----
    - ACTACT_ISDA is included for completeness, but is not implemented in `core/daycount.py` yet.
    """
    ACT365F = "ACT365F"         # Actual/365 Fixed
    ACT360 = "ACT360"           # Actual/360
    THIRTY360 = "THIRTY360"     # 30/360 basic (implemented as 30/360 US/NASD in daycount.py)
    ACTACT_ISDA = "ACTACT_ISDA" # Actual/Actual (ISDA) - not implemented yet


class BusinessDayConvention(_StrEnum):
    """
    Business-day adjustment convention identifiers.

    Notes
    -----
    These are *types*; actual calendar adjustment lives in calendar modules / QuantLib adapters.
    """
    FOLLOWING = "FOLLOWING"
    MODFOLLOWING = "MODFOLLOWING"
    PRECEDING = "PRECEDING"
    UNADJUSTED = "UNADJUSTED"


class Compounding(_StrEnum):
    """
    Interest-rate compounding convention identifiers.

    V2 scope
    --------
    We only need SIMPLE/CONTINUOUS for current bootstraps/pricing interfaces.
    """
    SIMPLE = "SIMPLE"
    CONTINUOUS = "CONTINUOUS"
    # Future (Vn): ANNUAL, SEMIANNUAL, QUARTERLY, etc.


class CalendarId(_StrEnum):
    """
    Calendar identifiers used by builders/adapters.

    V2 scope
    --------
    NULL is enough for deterministic year-fraction flows.
    Others are included as future extension points.
    """
    NULL = "NULL"
    TARGET = "TARGET"
    UK = "UK"
    USNY = "USNY"