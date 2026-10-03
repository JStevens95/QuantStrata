"""
FX elementary-instrument definitions and shared helpers.

Keeps the *what to emit* (grid defaults, payoff set) and small parsing helpers in one
place so the generator and pricer agree on conventions.
"""
from __future__ import annotations

import datetime as _dt
from typing import Tuple

DEFAULT_GRID = {
    "moneyness": [0.90, 0.95, 1.00, 1.05, 1.10],
    "expiries": [0.25, 0.50, 1.00],
    "instruments": ["call", "put", "digital_call", "forward"],
}


def parse_fx_factor(factor_id: str) -> Tuple[str, str, str]:
    """``FX.SPOT.USD.EUR`` -> (foreign='EUR', domestic='USD', pair='EURUSD')."""
    parts = factor_id.split(".")
    foreign, domestic = parts[-1], parts[-2]
    return foreign, domestic, f"{foreign}{domestic}"


def cob_to_date(cob: str) -> _dt.date:
    return _dt.datetime.strptime(str(cob), "%Y%m%d").date()
