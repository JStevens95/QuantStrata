from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.panel import Panel
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest
from src.marketdata.providers.interfaces import MarketDataProvider
from src.marketdata.providers.static.config import StaticProviderConfig


@dataclass(frozen=True, slots=True)
class StaticProvider(MarketDataProvider):
    """
    Replay/frozen dataset provider.

    - Replays an existing MarketDataset deterministically.
    - Enforces request policies via StaticProviderConfig.
    """

    dataset: MarketDataset
    config: StaticProviderConfig = field(default_factory=StaticProviderConfig)

    # IMPORTANT: concrete field (not @property)
    name: str = "StaticProvider"

    def get_market(self, request: MarketRequest) -> Market:
        scenario_idx = 0 if request.scenario is None else int(request.scenario)
        if scenario_idx < 0:
            raise ValueError("MarketRequest.scenario must be >= 0 when provided.")

        if scenario_idx >= int(self.dataset.n_scenarios):
            raise ValueError(
                f"Requested scenario_idx={scenario_idx} is not available in stored dataset "
                f"(n_scenarios={self.dataset.n_scenarios})."
            )

        ts = self.get_timeseries(
            TimeseriesRequest(
                start=request.asof,
                end=request.asof,
                freq="D",
                universe=request.universe,
                scenarios=scenario_idx + 1,
            )
        )
        return ts.snapshot(time_idx=0, scenario_idx=scenario_idx)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        # ---- freq policy ----
        req_freq = _norm_freq(request.freq)
        stored_freq = _norm_freq(str((self.dataset.meta or {}).get("freq", "") or ""))

        if self.config.strict_freq and stored_freq:
            if req_freq != stored_freq:
                raise ValueError(
                    "StaticProvider strict_freq violation.\n"
                    f"  requested freq={req_freq!r}\n"
                    f"  stored freq={stored_freq!r}"
                )

        # ---- scenarios policy ----
        req_scenarios = int(request.scenarios)
        if req_scenarios < 1:
            raise ValueError("TimeseriesRequest.scenarios must be >= 1.")

        stored_scenarios = int(self.dataset.n_scenarios)
        if self.config.strict_scenario_coverage:
            if req_scenarios > stored_scenarios:
                raise ValueError(
                    "StaticProvider strict_scenario_coverage violation.\n"
                    f"  requested scenarios={req_scenarios}\n"
                    f"  stored scenarios={stored_scenarios}"
                )
            scenario_count = req_scenarios
        else:
            scenario_count = min(req_scenarios, stored_scenarios)

        # ---- date policy ----
        desired_dates = _generate_dates(start=request.start, end=request.end, freq=req_freq)
        time_idx = _resolve_time_indices(
            stored_dates=list(self.dataset.dates),
            desired_dates=desired_dates,
            strict=self.config.strict_date_coverage,
        )

        # ---- universe filtering policy ----
        if self.config.include_only_requested_ids:
            keep_ids = tuple(request.universe.ids)
        else:
            keep_ids = _all_ids_in_dataset(self.dataset)

        return _slice_dataset(
            dataset=self.dataset,
            time_idx=time_idx,
            scenario_count=scenario_count,
            keep_ids=keep_ids,
            provider_name=str(self.name),
            provider_mode="static_replay",
            freq=req_freq,
        )


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _norm_freq(freq: str) -> str:
    return str(freq).strip().upper()


def _all_ids_in_dataset(ds: MarketDataset) -> Tuple[MarketId, ...]:
    """
    Return all MarketIds present in any dataset container, in stable key order.
    """
    keys = set(ds.panels.keys()) | set(ds.curve_params.keys()) | set(ds.vol_params.keys())
    return tuple(sorted(keys, key=lambda m: m.key()))


def _slice_dataset(
    *,
    dataset: MarketDataset,
    time_idx: List[int],
    scenario_count: int,
    keep_ids: Tuple[MarketId, ...],
    provider_name: str,
    provider_mode: str,
    freq: str,
) -> MarketDataset:
    """
    Slice MarketDataset to a subset of times and scenarios, restricted to keep_ids.

    Policy
    ------
    - If keep_ids includes an id not found anywhere in the dataset -> ValueError.
    """
    if not time_idx:
        raise ValueError("time_idx must not be empty.")
    if int(scenario_count) < 1:
        raise ValueError("scenario_count must be >= 1.")

    missing = _find_missing_ids(dataset=dataset, keep_ids=keep_ids)
    if missing:
        miss = ", ".join(mid.key() for mid in sorted(missing, key=lambda m: m.key()))
        raise ValueError(f"StaticProvider store missing MarketIds: {miss}")

    sliced_dates = [dataset.dates[i] for i in time_idx]

    sliced_panels: Dict[MarketId, Panel] = {}
    sliced_curve_params: Dict[MarketId, Panel] = {}
    sliced_curve_factories: Dict[MarketId, object] = {}
    sliced_vol_params: Dict[MarketId, Panel] = {}
    sliced_vol_factories: Dict[MarketId, object] = {}

    for mid in keep_ids:
        if mid in dataset.panels:
            sliced_panels[mid] = _slice_panel(panel=dataset.panels[mid], time_idx=time_idx, scenario_count=scenario_count)

        if mid in dataset.curve_params:
            sliced_curve_params[mid] = _slice_panel(
                panel=dataset.curve_params[mid], time_idx=time_idx, scenario_count=scenario_count
            )
            sliced_curve_factories[mid] = dataset.curve_factories[mid]

        if mid in dataset.vol_params:
            sliced_vol_params[mid] = _slice_panel(
                panel=dataset.vol_params[mid], time_idx=time_idx, scenario_count=scenario_count
            )
            sliced_vol_factories[mid] = dataset.vol_factories[mid]

    meta0: Mapping[str, Any] = dict(dataset.meta or {})
    meta = {
        **meta0,
        "provider": str(provider_name),
        "provider_mode": str(provider_mode),
        "freq": str(freq),
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
    Slice a Panel consistently with MarketDataset conventions:
    - time axis is axis 0
    - scenario axis exists iff axis_names[1] == "scenario" (then it's axis 1)
    """
    arr = np.asarray(panel.data)
    axis_names = tuple(panel.axis_names)

    # Slice time
    arr = np.take(arr, indices=time_idx, axis=0)

    # Slice scenarios only if explicitly present
    if len(axis_names) >= 2 and axis_names[1] == "scenario":
        slicer = [slice(None)] * arr.ndim
        slicer[1] = slice(0, int(scenario_count))
        arr = arr[tuple(slicer)]

    return Panel(data=arr, axis_names=axis_names)


def _find_missing_ids(*, dataset: MarketDataset, keep_ids: Tuple[MarketId, ...]) -> List[MarketId]:
    missing: List[MarketId] = []
    for mid in keep_ids:
        if (mid not in dataset.panels) and (mid not in dataset.curve_params) and (mid not in dataset.vol_params):
            missing.append(mid)
    return missing


def _resolve_time_indices(*, stored_dates: List[str], desired_dates: List[str], strict: bool) -> List[int]:
    """
    Map desired date strings to indices in stored_dates.

    - strict=True  : every desired date must exist
    - strict=False : return intersection in desired order (must be non-empty)
    """
    index_by_date = {d: i for i, d in enumerate(stored_dates)}
    if strict:
        out: List[int] = []
        for d in desired_dates:
            if d not in index_by_date:
                raise ValueError(f"StaticProvider store missing requested date {d}.")
            out.append(int(index_by_date[d]))
        return out

    # Non-strict: clip to intersection
    out = [int(index_by_date[d]) for d in desired_dates if d in index_by_date]
    if not out:
        raise ValueError(
            "StaticProvider strict_date_coverage=False but requested range has no overlap with stored dataset.\n"
            f"  requested={desired_dates[0]}..{desired_dates[-1]} ({len(desired_dates)} dates)\n"
            f"  stored={stored_dates[0]}..{stored_dates[-1]} ({len(stored_dates)} dates)"
        )
    return out


def _generate_dates(*, start: str, end: str, freq: str) -> List[str]:
    """
    Deterministic date grid generator.

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

    f = _norm_freq(freq)
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

    # M: deterministic 30-day step
    step = timedelta(days=30)
    while current <= end_d:
        dates.append(current.isoformat())
        current += step
    return dates