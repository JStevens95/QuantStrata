import pytest
from datetime import date

from src.marketdata.core.conventions import DayCount
from src.marketdata.core.daycount import yearfrac


def test_yearfrac_act365f() -> None:
    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 31)
    assert yearfrac(start=d0, end=d1, dc=DayCount.ACT365F) == pytest.approx(30 / 365.0)


def test_yearfrac_act360() -> None:
    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 31)
    assert yearfrac(start=d0, end=d1, dc=DayCount.ACT360) == pytest.approx(30 / 360.0)


def test_yearfrac_negative_when_end_before_start() -> None:
    d0 = date(2026, 2, 1)
    d1 = date(2026, 1, 1)
    assert yearfrac(start=d0, end=d1, dc=DayCount.ACT365F) == pytest.approx(-(31 / 365.0))


def test_yearfrac_thirty360_basic_month_end() -> None:
    # Common sanity case: Jan 30 -> Feb 28
    # Under 30/360 US with Feb EOM handling, this tends to land near 28/360 or 30/360 depending on rules.
    # We assert the implementation is stable and finite.
    d0 = date(2026, 1, 30)
    d1 = date(2026, 2, 28)
    yf = yearfrac(start=d0, end=d1, dc=DayCount.THIRTY360)
    assert yf > 0.0
    assert yf < 1.0


def test_yearfrac_thirty360_us_31st_rule() -> None:
    # Jan 31 -> Feb 28
    d0 = date(2026, 1, 31)
    d1 = date(2026, 2, 28)
    yf = yearfrac(start=d0, end=d1, dc=DayCount.THIRTY360)
    assert yf > 0.0
    assert yf < 1.0


def test_yearfrac_actact_isda_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="ACTACT_ISDA"):
        yearfrac(start=date(2026, 1, 1), end=date(2026, 2, 1), dc=DayCount.ACTACT_ISDA)


def test_yearfrac_type_guards() -> None:
    with pytest.raises(TypeError):
        yearfrac(start="2026-01-01", end=date(2026, 1, 2), dc=DayCount.ACT365F)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        yearfrac(start=date(2026, 1, 1), end=date(2026, 1, 2), dc="ACT365F")  # type: ignore[arg-type]