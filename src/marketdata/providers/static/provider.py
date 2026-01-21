from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Mapping, Tuple

import numpy as np

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.panel import Panel
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest
from src.marketdata.providers.interfaces import MarketDataProvider


@dataclass(frozen=True, slots=True)
class StaticProvider(MarketDataProvider):
    """
    Replay/frozen dataset provider.

    Purpose
    -------
    - "Replays" an existing MarketDataset deterministically.
    - Allows the rest of the stack to treat stored data like a live provider.

    Behavior
    --------
    - get_timeseries slices the stored dataset to the requested (start,end,freq,scenarios).
    - get_market returns a single Market snapshot by building a 1-date sliced dataset and snapshotting it.

    Notes
    -----
    - This provider is intentionally strict:
        * If requested dates are not present in the store, raise ValueError.
        * If requested scenarios exceed stored scenarios, raise ValueError.
    """

    dataset: MarketDataset
    _name: str = "StaticProvider"

    @property
    def name(self) -> str:
        return str(self._name)

    # ---------------------------------------------------------------------
    # MarketDataProvider API
    # ---------------------------------------------------------------------

    def get_market(self, request: MarketRequest) -> Market:
        scenario_idx = 0 if request.scenario is None else int(request.scenario)
        if scenario_idx < 0:
            raise ValueError("MarketRequest.scenario must be >= 0 when provided.")

        # Ensure we can include scenario_idx in the sliced dataset.
        required_scenarios = max(1, scenario_idx + 1)

        # Slice to one as-of date.
        ts = self.get_timeseries(
            TimeseriesRequest(
                start=request.asof,
                end=request.asof,
                freq="D",
                universe=request.universe,
                scenarios=required_scenarios,
            )
        )
        return ts.snapshot(time_idx=0, scenario_idx=scenario_idx)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        # Validate scenario count.
        req_scenarios = int(request.scenarios)
        if req_scenarios < 1:
            raise ValueError("TimeseriesRequest.scenarios must be >= 1.")
        if req_scenarios > int(self.dataset.n_scenarios):
            raise ValueError(
                f"Requested scenarios={req_scenarios} exceeds stored scenarios={self.dataset.n_scenarios}."
            )

        # Build desired date grid and map to indices in stored dataset.
        desired_dates = _generate_dates(start=request.start, end=request.end, freq=request.freq)
        time_idx = _resolve_time_indices(stored_dates=self.dataset.dates, desired_dates=desired_dates)

        # Slice all dataset blocks to requested (time, scenarios) and requested universe.
        return _slice_dataset(
            dataset=self.dataset,
            time_idx=time_idx,
            scenario_count=req_scenarios,
            universe_ids=request.universe.ids,
            provider_name=self.name,
        )


# -------------------------------------------------------------------------
# Slicing helpers (consistent with MarketDataset conventions)
# -------------------------------------------------------------------------

def _slice_dataset(
    *,
    dataset: MarketDataset,
    time_idx: List[int],
    scenario_count: int,
    universe_ids: Tuple[MarketId, ...],
    provider_name: str,
) -> MarketDataset:
    """
    Slice MarketDataset to a subset of times and scenarios, restricted to a Universe.

    Universe policy
    ---------------
    - Only keep MarketIds present in the stored dataset.
    - Missing ids are considered a user/config error -> raise ValueError.
    """
    if not time_idx:
        raise ValueError("time_idx must not be empty.")
    if int(scenario_count) < 1:
        raise ValueError("scenario_count must be >= 1.")

    # Validate universe coverage across stored containers.
    missing = _find_missing_ids(dataset=dataset, universe_ids=universe_ids)
    if missing:
        miss = ", ".join(mid.key() for mid in sorted(missing, key=lambda m: m.key()))
        raise ValueError(f"StaticProvider store missing MarketIds: {miss}")

    # Slice dates.
    sliced_dates = [dataset.dates[i] for i in time_idx]

    # Slice quote panels.
    sliced_panels: Dict[MarketId, Panel] = {}
    for mid in universe_ids:
        if mid in dataset.panels:
            sliced_panels[mid] = _slice_panel(panel=dataset.panels[mid], time_idx=time_idx, scenario_count=scenario_count)

    # Slice curve params (+ keep factories).
    sliced_curve_params: Dict[MarketId, Panel] = {}
    sliced_curve_factories: Dict[MarketId, object] = {}
    for mid in universe_ids:
        if mid in dataset.curve_params:
            sliced_curve_params[mid] = _slice_panel(
                panel=dataset.curve_params[mid], time_idx=time_idx, scenario_count=scenario_count
            )
            sliced_curve_factories[mid] = dataset.curve_factories[mid]

    # Slice vol params (+ keep factories).
    sliced_vol_params: Dict[MarketId, Panel] = {}
    sliced_vol_factories: Dict[MarketId, object] = {}
    for mid in universe_ids:
        if mid in dataset.vol_params:
            sliced_vol_params[mid] = _slice_panel(
                panel=dataset.vol_params[mid], time_idx=time_idx, scenario_count=scenario_count
            )
            sliced_vol_factories[mid] = dataset.vol_factories[mid]

    # Meta: preserve but add a small provenance marker (non-breaking).
    meta: Mapping[str, object] = dict(dataset.meta or {})
    meta = {
        **meta,
        "provider": str(provider_name),
        "provider_mode": "static_replay",
    }

    return MarketDataset(
        dates=sliced_dates,
        n_scenarios=int(scenario_count),
        panels=sliced_panels,
        curve_params=sliced_curve_params,
        curve_factories=sliced_curve_factories,  # type: ignore[arg-type]
        vol_params=sliced_vol_params,
        vol_factories=sliced_vol_factories,      # type: ignore[arg-type]
        meta=meta,
    )


def _slice_panel(*, panel: Panel, time_idx: List[int], scenario_count: int) -> Panel:
    """
    Slice a Panel consistently with your MarketDataset conventions:

    - time axis is always axis 0 and is sliced by time_idx.
    - scenario axis is ONLY present when axis_names[1] == "scenario" and then is axis 1.
    """
    arr = np.asarray(panel.data)
    axis_names = tuple(panel.axis_names)

    # Slice time (axis 0 always).
    arr = np.take(arr, indices=time_idx, axis=0)

    # Slice scenarios ONLY if scenario axis is explicitly axis 1.
    if len(axis_names) >= 2 and axis_names[1] == "scenario":
        slicer = [slice(None)] * arr.ndim
        slicer[1] = slice(0, int(scenario_count))
        arr = arr[tuple(slicer)]

    return Panel(data=arr, axis_names=axis_names)


def _find_missing_ids(*, dataset: MarketDataset, universe_ids: Tuple[MarketId, ...]) -> List[MarketId]:
    """
    Return any ids requested by universe that are not present in any stored container.
    """
    missing: List[MarketId] = []
    for mid in universe_ids:
        if (mid not in dataset.panels) and (mid not in dataset.curve_params) and (mid not in dataset.vol_params):
            missing.append(mid)
    return missing


def _resolve_time_indices(*, stored_dates: List[str], desired_dates: List[str]) -> List[int]:
    """
    Map desired date strings to indices in stored_dates (strict).
    """
    index_by_date = {d: i for i, d in enumerate(stored_dates)}
    out: List[int] = []
    for d in desired_dates:
        if d not in index_by_date:
            raise ValueError(f"StaticProvider store missing requested date {d}.")
        out.append(int(index_by_date[d]))
    return out


def _generate_dates(*, start: str, end: str, freq: str) -> List[str]:
    """
    Deterministic date grid generator (same semantics as your synthetic engine).

    Supported:
      - D: daily
      - B: business days (Mon-Fri)
      - W: weekly (7-day step)
      - M: month approximated as 30D (deterministic; not calendar-accurate by design)
    """
    start_d = date.fromisoformat(str(start))
    end_d = date.fromisoformat(str(end))
    if end_d < start_d:
        raise ValueError(f"end < start: start={start}, end={end}")

    f = str(freq).strip().upper()
    if f not in {"D", "B", "W", "M"}:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: D, B, W, M.")

    dates: List[str] = []
    current = start_d

    if f == "D":
        step = timedelta(days=1)
        while current <= end_d:
            dates.append(current.isoformat())
            current += step
        return dates

    if f == "B":
        while current <= end_d:
            if current.weekday() < 5:
                dates.append(current.isoformat())
            current += timedelta(days=1)
        return dates

    if f == "W":
        step = timedelta(days=7)
        while current <= end_d:
            dates.append(current.isoformat())
            current += step
        return dates

    # M: deterministic 30-day step.
    step = timedelta(days=30)
    while current <= end_d:
        dates.append(current.isoformat())
        current += step
    return dates