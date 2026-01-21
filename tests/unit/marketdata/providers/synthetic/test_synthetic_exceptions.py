from __future__ import annotations

from src.marketdata.providers.synthetic.exceptions import (
    InvalidPanelShapeError,
    MissingDependencyError,
    SyntheticMarketDataError,
    UnknownMarketSchemaError,
)


def test_synthetic_market_data_error_str_is_message() -> None:
    """Base error should stringify to its message (desk-friendly logs)."""
    err = SyntheticMarketDataError(message="boom")
    assert str(err) == "boom"


def test_exception_hierarchy() -> None:
    """All custom exceptions should inherit from the base synthetic error."""
    assert issubclass(MissingDependencyError, SyntheticMarketDataError)
    assert issubclass(UnknownMarketSchemaError, SyntheticMarketDataError)
    assert issubclass(InvalidPanelShapeError, SyntheticMarketDataError)