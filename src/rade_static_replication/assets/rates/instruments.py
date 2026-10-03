"""Rates elementary-instrument definitions and helpers."""
from __future__ import annotations

import datetime as _dt

DEFAULT_GRID = {
    "expiries": [1.0, 2.0, 5.0],
    "swap_tenors": [2.0, 5.0, 10.0],
    "strike_offsets_bp": [-50.0, 0.0, 50.0],
    "instruments": ["payer", "receiver"],
    "freq": 0.5,
}


def parse_rates_factor(factor_id: str) -> str:
    """Return the currency token, e.g. ``IR_CURVE_SWAP.GBP`` -> ``GBP``."""
    return factor_id.split(".")[-1]


def cob_to_date(cob: str) -> _dt.date:
    return _dt.datetime.strptime(str(cob), "%Y%m%d").date()
