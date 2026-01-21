from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import MutableMapping, Sequence

from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel
from src.marketdata.curves.factory import ZeroRateCurveFactory
from src.marketdata.surfaces.factory import GridVolFactory


@dataclass(slots=True)
class SyntheticGenerationState:
    """
    Mutable state container passed across generators during dataset generation.

    Why this exists
    ---------------
    - Generators should not need to know how a MarketDataset is assembled.
    - Generators should only write into a consistent state object.
    - The engine owns orchestration and final MarketDataset construction.

    Notes
    -----
    - `spot_cache` exists so FIXING can reuse SPOT paths when possible.
    - All dicts are keyed by MarketId for consistent storage and snapshot lookup.
    """

    # --- time axis ---
    dates: Sequence[str]
    n_time: int
    n_scenarios: int

    # --- outputs: quote-like panels ---
    quote_panels: MutableMapping[MarketId, Panel]

    # --- outputs: curve params + factories ---
    curve_param_panels: MutableMapping[MarketId, Panel]
    curve_factories: MutableMapping[MarketId, ZeroRateCurveFactory]

    # --- outputs: vol params + factories ---
    vol_param_panels: MutableMapping[MarketId, Panel]
    vol_factories: MutableMapping[MarketId, GridVolFactory]

    # --- cached raw arrays for reuse ---
    spot_cache: MutableMapping[MarketId, np.ndarray]