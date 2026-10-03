"""
Market snapshot base.

A snapshot is the consistent COB market for one risk factor. The abstract base lives
here; asset-class-specific snapshots (``FXSnapshot``, ``RatesSnapshot``) live in
``marketdata/fx`` and ``marketdata/rates`` — mirroring the ``assets/`` layout.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSnapshot:
    """Base class for an asset-class-specific COB market snapshot."""
    factor_id: str
    asset_class: str
    as_of: _dt.date

    def summary(self) -> dict:
        return {"factor_id": self.factor_id, "asset_class": self.asset_class, "as_of": str(self.as_of)}
