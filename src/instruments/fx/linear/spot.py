from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.marketdata.ids import MarketId


@dataclass(frozen=True, slots=True)
class FxSpot:
    """
    FX spot exposure instrument (one-unit).

    Represents one unit of exposure to an FX spot quote identified by `spot_id`.
    Portfolio sizing is applied via Position.quantity.

    Parameters
    ----------
    spot_id:
        MarketId for the FX spot quote, e.g. MarketId("FX","SPOT","EURUSD").
    contract_multiplier:
        Multiplier applied to the quote for one instrument unit (default 1.0).
    description:
        Optional label for reporting/debugging.
    """
    spot_id: MarketId
    contract_multiplier: float = 1.0
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if self.spot_id is None:
            raise ValueError("spot_id must not be None.")

        if not isinstance(self.contract_multiplier, (int, float)):
            raise TypeError("contract_multiplier must be numeric.")
        if not np.isfinite(float(self.contract_multiplier)):
            raise ValueError("contract_multiplier must be finite.")