"""
Attribute fixtures shared by the encoder and graph tests.

Deliberately hand-written rather than generated: the encoder's output
depends on which levels a categorical attribute takes and on how many
labels a multi-label attribute carries, so a test that asserts on column
layout needs those to be visible in the file rather than hidden behind a
random draw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest


@pytest.fixture
def attributes() -> Mapping[str, Sequence[Any]]:
    """
    Six instruments: four elementary, two target.

    Chosen so that every branch of the encoder is exercised at least once.
    ``product_subtype`` takes two levels and ``product_type`` takes two, so
    a one-hot block of width one -- which is a degenerate case the width
    normalisation would divide by -- is not the only thing under test. The
    risk-factor sets have different cardinalities, so the per-row norm has
    something to do.

    Returns
    -------
    Mapping
        Attribute name to one value per instrument.
    """
    return {
        "trade_id": ["E1", "E2", "E3", "E4", "T1", "T2"],
        "moneyness": [0.95, 1.00, 1.05, 1.10, 0.98, 1.02],
        "yrs_to_maturity": [0.25, 0.50, 1.00, 2.00, 0.75, 1.50],
        "delta": [0.30, 0.45, 0.55, 0.70, 0.40, 0.60],
        "vega": [10.0, 12.5, 15.0, 20.0, 11.0, 18.0],
        "product_type": [
            "vanilla_option",
            "vanilla_option",
            "forward",
            "forward",
            "vanilla_option",
            "forward",
        ],
        "product_subtype": [
            "european",
            "european",
            "outright",
            "outright",
            "european",
            "outright",
        ],
        "trade_type": [
            "elementary",
            "elementary",
            "elementary",
            "elementary",
            "target",
            "target",
        ],
        "underlying_risk_factors": [
            ["FX"],
            ["FX", "RATES"],
            ["RATES"],
            ["FX"],
            ["FX", "RATES"],
            ["RATES"],
        ],
    }
