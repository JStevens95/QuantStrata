from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from src.portfolio.core import Portfolio, Position


@dataclass(frozen=True, slots=True)
class _DummyInstrument:
    name: str = "dummy"


def test_position_requires_non_empty_position_id() -> None:
    with pytest.raises(ValueError, match="position_id must be a non-empty string"):
        Position(position_id="", instrument=_DummyInstrument(), quantity=1.0)

    with pytest.raises(ValueError, match="position_id must be a non-empty string"):
        Position(position_id="   ", instrument=_DummyInstrument(), quantity=1.0)


def test_position_quantity_must_be_numeric() -> None:
    with pytest.raises(TypeError, match="quantity must be a number"):
        Position(position_id="p1", instrument=_DummyInstrument(), quantity="1.0")  # type: ignore[arg-type]


def test_position_quantity_must_not_be_nan() -> None:
    with pytest.raises(ValueError, match="quantity must be finite"):
        Position(position_id="p1", instrument=_DummyInstrument(), quantity=float("nan"))


def test_portfolio_rejects_duplicate_position_ids() -> None:
    p1 = Position(position_id="dup", instrument=_DummyInstrument("a"), quantity=1.0)
    p2 = Position(position_id="dup", instrument=_DummyInstrument("b"), quantity=2.0)

    with pytest.raises(ValueError, match=r"Duplicate position_id values found: dup"):
        Portfolio(positions=[p1, p2])


def test_portfolio_iter_and_len_work() -> None:
    p1 = Position(position_id="p1", instrument=_DummyInstrument("a"), quantity=1.0)
    p2 = Position(position_id="p2", instrument=_DummyInstrument("b"), quantity=-2.0)
    pf = Portfolio(positions=[p1, p2])

    assert len(pf) == 2
    assert [p.position_id for p in pf] == ["p1", "p2"]
    assert math.isfinite(pf.positions[0].quantity)