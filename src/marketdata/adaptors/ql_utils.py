from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


def require_quantlib() -> Any:
    """
    Lazily import and return QuantLib.

    This makes QuantLib an optional dependency:
    - core library can import without QuantLib installed
    - only code paths that call this function require QuantLib
    """
    try:
        import QuantLib as ql  # type: ignore
    except Exception as e:
        raise ImportError(
            "QuantLib is not installed (or failed to import). "
            "Install it to use QuantLib adapters/pricers."
        ) from e
    return ql


def parse_iso_date(iso_yyyy_mm_dd: str) -> Tuple[int, int, int]:
    """
    Parse an ISO date string 'YYYY-MM-DD' into (year, month, day).
    """
    raw = str(iso_yyyy_mm_dd).strip()
    if not raw:
        raise ValueError("asof date must be a non-empty ISO string 'YYYY-MM-DD'.")

    parts = raw.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid asof date {iso_yyyy_mm_dd!r}; expected 'YYYY-MM-DD'.")

    y, m, d = (int(p) for p in parts)
    return y, m, d


def to_ql_date(iso_yyyy_mm_dd: str) -> Any:
    """
    Convert an ISO date 'YYYY-MM-DD' into a QuantLib Date.
    """
    ql = require_quantlib()
    y, m, d = parse_iso_date(iso_yyyy_mm_dd)
    return ql.Date(d, m, y)


def yearfrac_to_ql_date(*, asof: Any, yearfrac: float) -> Any:
    """
    Convert a year-fraction T into a QuantLib Date by adding ~T*365 days.

    Notes
    -----
    - Approximation (good enough for V2 demos).
    - For production bootstraps, map instruments to real dates via calendars/schedules.
    """
    ql = require_quantlib()

    t = float(yearfrac)
    if t < 0.0:
        raise ValueError(f"yearfrac must be >= 0. Got {t}.")

    days = int(round(t * 365.0))
    return asof + ql.Period(days, ql.Days)


@dataclass(frozen=True, slots=True)
class QlContext:
    """
    Shared QuantLib context for adapters/pricers.

    Keep this tiny and explicit; it’s a V2 integration layer, not the canonical source of truth.
    """
    asof: str
    day_count: Optional[Any] = None
    calendar: Optional[Any] = None

    def with_defaults(self) -> "QlContext":
        """
        Return a context with default day count + calendar filled in.
        """
        ql = require_quantlib()
        dc = self.day_count if self.day_count is not None else ql.Actual365Fixed()
        cal = self.calendar if self.calendar is not None else ql.NullCalendar()
        return QlContext(asof=self.asof, day_count=dc, calendar=cal)

    def set_evaluation_date(self) -> Any:
        """
        Set QuantLib global evaluationDate from ctx.asof and return the ql.Date.
        """
        ql = require_quantlib()
        asof_date = to_ql_date(self.asof)
        ql.Settings.instance().evaluationDate = asof_date
        return asof_date


def parse_iso_date(iso_yyyy_mm_dd: str) -> tuple[int, int, int]:
    """
    Parse an ISO date string 'YYYY-MM-DD' into (year, month, day).

    We keep this tiny and strict for speed + clarity.
    """
    raw = str(iso_yyyy_mm_dd).strip()
    if not raw:
        raise ValueError("asof date must be a non-empty ISO string 'YYYY-MM-DD'.")

    parts = raw.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid asof date {iso_yyyy_mm_dd!r}; expected 'YYYY-MM-DD'.")

    y, m, d = (int(p) for p in parts)
    return y, m, d


def to_ql_date(iso_yyyy_mm_dd: str) -> "ql.Date":  # type: ignore[name-defined]
    """
    Convert an ISO date 'YYYY-MM-DD' into a QuantLib Date.

    QuantLib uses (day, month, year) ordering.
    """
    ql_mod = require_quantlib()
    y, m, d = parse_iso_date(iso_yyyy_mm_dd)
    return ql_mod.Date(d, m, y)


def yearfrac_to_ql_date(*, asof: "ql.Date", yearfrac: float) -> "ql.Date":  # type: ignore[name-defined]
    """
    Convert a year-fraction T into a QuantLib Date by adding ~T*365 days.

    Notes
    -----
    - This is an approximation when you only have year-fractions (no Period schedule).
    - For interview-quality architecture + V2 milestone demos, this is fine.
    - For production OIS/LIBOR curve bootstraps, you'd map instruments to actual dates.
    """
    ql_mod = require_quantlib()

    t = float(yearfrac)
    if t < 0.0:
        raise ValueError(f"yearfrac must be >= 0. Got {t}.")

    # Convert year-fractions to an integer day offset (simple, deterministic).
    days = int(round(t * 365.0))
    return asof + ql_mod.Period(days, ql_mod.Days)


@dataclass(frozen=True, slots=True)
class QlContext:
    """
    Shared QuantLib context for adapters/pricers.

    You can extend this later with:
    - calendar conventions
    - settlement lag
    - daycount per asset class
    """
    asof: str
    day_count: Optional["ql.DayCounter"] = None  # type: ignore[name-defined]
    calendar: Optional["ql.Calendar"] = None     # type: ignore[name-defined]

    def with_defaults(self) -> "QlContext":
        """
        Return a context with default day count + calendar filled in.
        """
        ql_mod = require_quantlib()

        dc = self.day_count if self.day_count is not None else ql_mod.Actual365Fixed()
        cal = self.calendar if self.calendar is not None else ql_mod.NullCalendar()

        return QlContext(asof=self.asof, day_count=dc, calendar=cal)

    def set_evaluation_date(self) -> "ql.Date":  # type: ignore[name-defined]
        """
        Set QuantLib global evaluationDate from ctx.asof and return the ql.Date.
        """
        ql_mod = require_quantlib()
        asof_date = to_ql_date(self.asof)

        # QuantLib keeps a global evaluation date used by pricing engines.
        ql_mod.Settings.instance().evaluationDate = asof_date
        return asof_date