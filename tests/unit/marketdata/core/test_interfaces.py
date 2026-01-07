# tests/unit/marketdata/core/test_interfaces.py

import math
import pytest

from src.marketdata.core.interfaces import Quote


def test_quote_accepts_finite_float() -> None:
    q = Quote(1.234)
    assert q.value == 1.234


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_quote_rejects_nonfinite(bad: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        Quote(bad)