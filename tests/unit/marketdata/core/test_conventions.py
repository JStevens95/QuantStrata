import pytest

from src.marketdata.core.conventions import (
    DayCount,
    BusinessDayConvention,
    Compounding,
    CalendarId,
)


def test_enums_are_string_valued() -> None:
    assert isinstance(DayCount.ACT365F.value, str)
    assert str(DayCount.ACT365F) == "ACT365F"


@pytest.mark.parametrize(
    "enum_cls, value",
    [
        (DayCount, "act365f"),
        (DayCount, "ACT365F"),
        (BusinessDayConvention, "following"),
        (Compounding, "continuous"),
        (CalendarId, "NULL"),
    ],
)
def test_from_value_parses_case_insensitive(enum_cls, value) -> None:
    parsed = enum_cls.from_value(value)
    assert isinstance(parsed, enum_cls)


def test_from_value_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        DayCount.from_value("BAD_DC")