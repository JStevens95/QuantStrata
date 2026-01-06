from __future__ import annotations

import pytest

from src.marketdata.adaptors.ql_utils import (
    QlContext,
    parse_iso_date,
    require_quantlib,
    to_ql_date,
    yearfrac_to_ql_date,
)


def _has_quantlib() -> bool:
    try:
        require_quantlib()
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_quantlib(), reason="QuantLib not installed")


def test_parse_iso_date() -> None:
    assert parse_iso_date("2025-12-29") == (2025, 12, 29)

    with pytest.raises(ValueError):
        parse_iso_date("")

    with pytest.raises(ValueError):
        parse_iso_date("2025/12/29")


def test_to_ql_date_and_yearfrac_to_ql_date() -> None:
    ql = require_quantlib()
    d0 = to_ql_date("2025-12-29")
    assert isinstance(d0, ql.Date)

    d1 = yearfrac_to_ql_date(asof=d0, yearfrac=1.0)
    assert d1 > d0


def test_ql_context_sets_eval_date() -> None:
    ql = require_quantlib()
    ctx = QlContext(asof="2025-12-29").with_defaults()
    d = ctx.set_evaluation_date()
    assert ql.Settings.instance().evaluationDate == d