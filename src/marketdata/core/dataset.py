from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Protocol

import numpy as np

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Curve, Quote, VolSurface
from src.marketdata.core.market import Market
from src.marketdata.core.panel import Panel


class CurveFactory(Protocol):
    """Factory that reconstructs a Curve snapshot from a parameter block."""
    def build(self, params: np.ndarray) -> Curve: ...


class VolSurfaceFactory(Protocol):
    """Factory that reconstructs a VolSurface snapshot from a parameter block."""
    def build(self, params: np.ndarray) -> VolSurface: ...


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """
    Time series / scenario container for ML/RL that can produce pricing snapshots.

    The key bridge:
        MarketDataset.snapshot(time_idx, scenario_idx) -> Market
    """
    dates: List[str]
    n_scenarios: int
    panels: Mapping[MarketId, Panel]

    curve_params: Mapping[MarketId, Panel]
    curve_factories: Mapping[MarketId, CurveFactory]

    vol_params: Mapping[MarketId, Panel]
    vol_factories: Mapping[MarketId, VolSurfaceFactory]

    meta: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # ---- Basic invariants ----
        if int(self.n_scenarios) < 1:
            raise ValueError("MarketDataset.n_scenarios must be >= 1.")
        if not self.dates:
            raise ValueError("MarketDataset.dates must not be empty.")

        n_t = len(self.dates)  # number of time points

        # ---- Ensure every param panel has a factory (fail fast) ----
        missing_curve = set(self.curve_params.keys()) - set(self.curve_factories.keys())
        if missing_curve:
            miss = ", ".join(mid.key() for mid in sorted(missing_curve, key=lambda m: m.key()))
            raise ValueError(f"Missing curve factories for: {miss}")

        missing_vol = set(self.vol_params.keys()) - set(self.vol_factories.keys())
        if missing_vol:
            miss = ", ".join(mid.key() for mid in sorted(missing_vol, key=lambda m: m.key()))
            raise ValueError(f"Missing vol factories for: {miss}")

        # ---- Validate panel shapes align with (time, scenario) expectations ----
        # Quotes panels must be scalar-like panels: [T] or [T,S]
        for mid, p in self.panels.items():
            _validate_panel_time_axis(panel=p, expected_t=n_t, mkt_id=mid)
            _validate_panel_scenario_axis(panel=p, expected_s=self.n_scenarios, mkt_id=mid)

            if p.data.ndim not in (1, 2):
                raise ValueError(
                    f"Quote panel must have ndim 1 or 2. Got ndim={p.data.ndim} for {mid.key()}."
                )

        # Curve and vol param panels can be block panels: [T,...] or [T,S,...]
        for mid, p in self.curve_params.items():
            _validate_panel_time_axis(panel=p, expected_t=n_t, mkt_id=mid)
            _validate_panel_scenario_axis(panel=p, expected_s=self.n_scenarios, mkt_id=mid)

        for mid, p in self.vol_params.items():
            _validate_panel_time_axis(panel=p, expected_t=n_t, mkt_id=mid)
            _validate_panel_scenario_axis(panel=p, expected_s=self.n_scenarios, mkt_id=mid)

    def snapshot(self, time_idx: int, scenario_idx: int = 0) -> Market:
        """
        Build an immutable pricing Market snapshot at (time_idx, scenario_idx).

        This method is intentionally deterministic:
        - All slicing is controlled via Panel + axis_names.
        - Factories reconstruct curves/surfaces from sliced parameter blocks.
        """
        # Validate indices up-front for clearer errors.
        if not (0 <= int(time_idx) < len(self.dates)):
            raise IndexError(f"time_idx out of range: {time_idx} for T={len(self.dates)}.")
        if not (0 <= int(scenario_idx) < int(self.n_scenarios)):
            raise IndexError(f"scenario_idx out of range: {scenario_idx} for S={self.n_scenarios}.")

        # Resolve as-of date string.
        asof = self.dates[int(time_idx)]

        # ---- Build quotes dict from scalar panels ----
        quotes: Dict[MarketId, Quote] = {}
        for mkt_id, panel in self.panels.items():
            # Pull scalar value at (t,s).
            v = panel.scalar_at(time_idx=int(time_idx), scenario_idx=int(scenario_idx))
            # Wrap in Quote (validated finite float).
            quotes[mkt_id] = Quote(value=v)

        # ---- Build curves from parameter panels + factories ----
        curves: Dict[MarketId, Curve] = {}
        for mkt_id, panel in self.curve_params.items():
            # Slice parameter block at (t,s) consistently for [T] / [T,S] / [T,...] / [T,S,...]
            params = _slice_params(panel=panel, time_idx=int(time_idx), scenario_idx=int(scenario_idx))
            # Use factory to reconstruct curve object.
            curves[mkt_id] = self.curve_factories[mkt_id].build(params=params)

        # ---- Build vol surfaces from parameter panels + factories ----
        vols: Dict[MarketId, VolSurface] = {}
        for mkt_id, panel in self.vol_params.items():
            params = _slice_params(panel=panel, time_idx=int(time_idx), scenario_idx=int(scenario_idx))
            vols[mkt_id] = self.vol_factories[mkt_id].build(params=params)

        # Assemble Market snapshot (meta is passed through for traceability/debugging).
        return Market(asof=asof, quotes=quotes, curves=curves, vols=vols, meta=self.meta)


def _validate_panel_time_axis(*, panel: Panel, expected_t: int, mkt_id: MarketId) -> None:
    """Validate that panel has time axis in dimension 0 and matches expected length."""
    x = panel.data
    if x.ndim < 1:
        raise ValueError(f"Panel must have at least 1 dimension for {mkt_id.key()}.")
    if x.shape[0] != int(expected_t):
        raise ValueError(
            f"Panel time axis mismatch for {mkt_id.key()}: shape[0]={x.shape[0]} vs expected T={expected_t}."
        )


def _validate_panel_scenario_axis(*, panel: Panel, expected_s: int, mkt_id: MarketId) -> None:
    """
    Validate scenario axis if present.

    Convention:
    - if axis_names[1] == "scenario", then panel.data.shape[1] must equal expected_s
    """
    if len(panel.axis_names) >= 2 and panel.axis_names[1] == "scenario":
        x = panel.data
        if x.ndim < 2:
            raise ValueError(f"Panel declares scenario axis but ndim < 2 for {mkt_id.key()}.")
        if x.shape[1] != int(expected_s):
            raise ValueError(
                f"Panel scenario axis mismatch for {mkt_id.key()}: shape[1]={x.shape[1]} vs expected S={expected_s}."
            )


def _slice_params(panel: Panel, time_idx: int, scenario_idx: int = 0) -> np.ndarray:
    """
    Slice parameter panels consistently.

    Supported shapes
    ----------------
    - [T]                     : scalar parameter per time
    - [T, S]                  : scalar parameter per time/scenario
    - [T, K...]               : block parameter per time
    - [T, S, K...]            : block parameter per time/scenario
    """
    x = panel.data

    # Case 1: [T]
    if x.ndim == 1:
        return np.asarray(x[time_idx], dtype=float)

    # Case 2: [T,S] (scalar-by-scenario) OR [T,K] (block-by-time)
    if x.ndim == 2:
        # If axis_names says second axis is scenario, interpret as [T,S].
        if len(panel.axis_names) >= 2 and panel.axis_names[1] == "scenario":
            return np.asarray(x[time_idx, scenario_idx], dtype=float)
        # Otherwise interpret as [T,K] (block parameters) and return the whole row.
        return np.asarray(x[time_idx], dtype=float)

    # Case 3: ndim >= 3
    if len(panel.axis_names) >= 2 and panel.axis_names[1] == "scenario":
        # [T,S,K...] -> slice first two axes, keep remaining block.
        slicer = (time_idx, scenario_idx) + (slice(None),) * (x.ndim - 2)
        return np.asarray(x[slicer], dtype=float)

    # [T,K...] -> slice time axis only, keep remaining block.
    slicer = (time_idx,) + (slice(None),) * (x.ndim - 1)
    return np.asarray(x[slicer], dtype=float)