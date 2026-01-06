from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, List, Dict

from src.marketdata.ids import MarketId
from src.marketdata.market import Market
from src.marketdata.interfaces import Quote, Curve, VolSurface, Panel


class CurveFactory(Protocol):
    """
    Factory that reconstructs a Curve snapshot from a parameter block.

    Typical pattern
    ---------------
    - Store curve parameters in a Panel as arrays across time/scenarios.
    - Build an object implementing the Curve protocol on demand.

    This keeps MarketDataset ML-friendly while still supporting pricing via snapshot().
    """

    def build(self, params: np.ndarray) -> Curve: ...


class VolSurfaceFactory(Protocol):
    """
    Factory that reconstructs a VolSurface snapshot from a parameter block.

    Typical pattern
    ---------------
    - Store surface parameters (e.g., flat vol, SVI params, grid vols) in a Panel.
    - Build a VolSurface object on demand.
    """

    def build(self, params: np.ndarray) -> VolSurface: ...


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """
    Time series / scenario panel container for ML and RL.

    Core capability
    ---------------
    `snapshot(time_idx, scenario_idx)` converts a panel slice into a pricing `Market`,
    guaranteeing your ML/RL pipelines and pricing share the same market contract.

    Storage layout
    --------------
    - dates: length T (ISO strings)
    - n_scenarios: number of scenarios S (>=1)

    - panels: scalar panels (spots, fixings, etc.) indexed by MarketId
    - curve_params + curve_factories: curve parameter panels + factories
    - vol_params   + vol_factories  : surface parameter panels + factories
    """
    # initiate required variables.
    dates: List[str]
    n_scenarios: int
    panels: Mapping[MarketId, Panel]

    curve_params: Mapping[MarketId, Panel]
    curve_factories: Mapping[MarketId, CurveFactory]

    vol_params: Mapping[MarketId, Panel]
    vol_factories: Mapping[MarketId, VolSurfaceFactory]

    meta: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Post init method"""
        if self.n_scenarios < 1:
            raise ValueError("MarketDataset.n_scenarios must be >= 1")
        if not self.dates:
            raise ValueError("MarketDataset.dates must not be empty")

        # ensure factories exist for each param panel (fail first)
        missing_curve_factories = set(self.curve_params.keys()) - set(self.curve_factories.keys())
        if missing_curve_factories:
            miss = ", ".join(mid.key() for mid in sorted(missing_curve_factories, key=lambda m: m.key()))
            raise ValueError(f"Missing curve factories for: {miss}")

        missing_vol_factories = set(self.vol_params.keys()) - set(self.vol_factories.keys())
        if missing_vol_factories:
            miss = ", ".join(mid.key() for mid in sorted(missing_vol_factories, key=lambda m: m.key()))
            raise ValueError(f"Missing vol factories for: {miss}")

    def snapshot(self, time_idx: int, scenario_idx: int = 0) -> Market:
        """
        Build a pricing Market snapshot from the dataset at a given time/scenario index.

        This is the most important bridge in the architecture:
        - Pricing consumes Market
        - ML/RL consumes MarketDataset
        - snapshot() links the two cleanly and deterministically.

        :param time_idx: index into `dates`
        :param scenario_idx: index into scenario axis for scenario-aware panels (default 0)
        """
        if not (0 <= time_idx < len(self.dates)):
            raise IndexError(f"time_idx out of range: {time_idx}")
        if not (0 <= scenario_idx < self.n_scenarios):
            raise IndexError(f"scenario_idx out of range: {scenario_idx}")
        asof = self.dates[time_idx]

        # --- quotes (scalar panels) ---
        quotes: Dict[MarketId, Quote] = {}
        for mkt_id, panel in self.panels.items():
            # scalar_at supports [T] or [T, S]; it fails fast otherwise.
            quotes[mkt_id] = Quote(panel.scalar_at(time_idx=time_idx, scenario_idx=scenario_idx))

        # --- curves (parameter panels --> curve objects via factories) ---
        curves: Dict[MarketId, Curve] = {}
        for mkt_id, panel in self.curve_params.items():
            params = _slice_params(panel=panel, time_idx=time_idx, scenario_idx=scenario_idx)
            curves[mkt_id] = self.curve_factories[mkt_id].build(params=params)

        # --- vol surfaces (parameter panels --> VolSurface objects via factories ---
        vols: Dict[MarketId, VolSurface] = {}
        for mkt_id, panel in self.vol_params.items():
            params = _slice_params(panel=panel, time_idx=time_idx, scenario_idx=scenario_idx)
            vols[mkt_id] = self.vol_factories[mkt_id].build(params=params)

        return Market(asof=asof, quotes=quotes, curves=curves, vols=vols)


def _slice_params(panel: Panel, time_idx: int, scenario_idx: int = 0) -> np.ndarray:
    """
    Slice parameter panels consistently.

    Supported shapes
    ----------------
    - [T]                     : scalar parameter per time
    - [T, S]                  : scalar parameter per time/scenario
    - [T, K...]               : block parameter per time
    - [T, S, K...]            : block parameter per time/scenario
      (e.g. curve grids [T,S,K,2], vol grids [T,S,n_exp,n_k], etc.)
    """
    x = panel.data

    if x.ndim == 1:
        return np.asarray(x[time_idx], dtype=float)

    if x.ndim == 2:
        # Ambiguous: [T,S] scalar-by-scenario OR [T,K] block params.
        if len(panel.axis_names) >= 2 and panel.axis_names[1] == "scenario":
            return np.asarray(x[time_idx, scenario_idx], dtype=float)
        return np.asarray(x[time_idx], dtype=float)

    # General case: ndim >= 3
    if len(panel.axis_names) >= 2 and panel.axis_names[1] == "scenario":
        slicer = (time_idx, scenario_idx) + (slice(None),) * (x.ndim - 2)
        return np.asarray(x[slicer], dtype=float)

    slicer = (time_idx,) + (slice(None),) * (x.ndim - 1)
    return np.asarray(x[slicer], dtype=float)
