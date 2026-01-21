from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Tuple

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

    Purpose
    -------
    - Replays an existing MarketDataset deterministically.
    - Allows the rest of the stack (pricers/orchestrators/examples) to treat stored data
      like a provider without any change in calling code.

    Behavior
    --------
    - get_timeseries slices the stored dataset to the requested (start, end, freq, scenarios).
    - get_market returns a Market snapshot by slicing to a 1-date dataset and snapshotting it.

    Determinism & safety
    --------------------
    - No randomness.
    - Strict validation (configurable) to prevent silent mismatches.
    """

    dataset: MarketDataset
    config: StaticProviderConfig = StaticProviderConfig()
    _name: str = "StaticProvider"

    @property
    def name(self) -> str:
        # Read-only property (Protocol-safe).
        return str(self._name)

    # ---------------------------------------------------------------------
    # MarketDataProvider API
    # ---------------------------------------------------------------------

    def get_market(self, request: MarketRequest) -> Market:
        scenario_idx = 0 if request.scenario is None else int(request.scenario)
        if scenario_idx < 0:
            raise ValueError("MarketRequest.scenario must be >= 0 when provided.")

        # Ensure the sliced dataset has enough scenarios to include scenario_idx.
        required_scenarios = max(1, scenario_idx + 1)

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
        req_scenarios = int(request.scenarios)
        if req_scenarios < 1:
            raise ValueError("TimeseriesRequest.scenarios must be >= 1.")

        # --- Frequency policy (if meta has freq) ---
        _enforce_freq_policy(config=self.config, dataset=self.dataset, requested_freq=request.freq)

        # --- Scenario policy ---
        scenario_count = _resolve_scenario_count(
            config=self.config,
            stored_scenarios=int(self.dataset.n_scenarios),
            requested_scenarios=req_scenarios,
        )

        # --- Date policy ---
        desired_dates = _generate_dates(start=request.start, end=request.end, freq=request.freq)
        time_idx, sliced_dates = _resolve_time_indices_with_policy(
            config=self.config,
            stored_dates=self.dataset.dates,
            desired_dates=desired_dates,
        )

        # --- Universe policy ---
        universe_ids = tuple(request.universe.ids)
        if self.config.include_only_requested_ids:
            keep_ids = universe_ids
        else:
            keep_ids = _all_dataset_ids(dataset=self.dataset)

        # Slice all dataset blocks to requested (time, scenarios) and selected ids.
        return _slice_dataset(
            dataset=self.dataset,
            sliced_dates=sliced_dates,
            time_idx=time_idx,
            scenario_count=scenario_count,
            keep_ids=keep_ids,
            provider_name=self.name,
        )


# =============================================================================
# Policy helpers
# =============================================================================

def _enforce_freq_policy(*, config: StaticProviderConfig, dataset: MarketDataset, requested_freq: str) -> None:
    """
    If strict_freq=True and dataset.meta contains a freq, enforce exact match.

    Rationale
    ---------
    Old stored datasets may not have meta["freq"]. In that case we do not block.
    """
    if not config.strict_freq:
        return

    meta = dataset.meta or {}
    stored_freq = meta.get("freq", None)
    if stored_freq is None:
        return

    req = str(requested_freq).strip().upper()
    stored = str(stored_freq).strip().upper()
    if req != stored:
        raise ValueError(f"StaticProvider strict_freq mismatch: requested freq={req!r} vs stored freq={stored!r}.")


def _resolve_scenario_count(
    *,
    config: StaticProviderConfig,
    stored_scenarios: int,
    requested_scenarios: int,
) -> int:
    """
    Resolve how many scenarios to return under strict/non-strict policy.
    """
    if stored_scenarios < 1:
        raise ValueError("Stored dataset has invalid n_scenarios < 1.")

    if requested_scenarios <= stored_scenarios:
        return int(requested_scenarios)

    if config.strict_scenario_coverage:
        raise ValueError(
            f"Requested scenarios={requested_scenarios} exceeds stored scenarios={stored_scenarios}."
        )

    # Clip.
    return int(stored_scenarios)


def _resolve_time_indices_with_policy(
    *,
    config: StaticProviderConfig,
    stored_dates: List[str],
    desired_dates: List[str],
) -> Tuple[List[int], List[str]]:
    """
    Resolve time indices with strict vs clipped coverage policy.

    Returns
    -------
    (time_idx, sliced_dates)
    """
    if not stored_dates:
        raise ValueError("Stored dataset dates are empty.")

    index_by_date = {d: i for i, d in enumerate(stored_dates)}

    if config.strict_date_coverage:
        out_idx: List[int] = []
        for d in desired_dates:
            if d not in index_by_date:
                raise ValueError(f"StaticProvider store missing requested date {d}.")
            out_idx.append(int(index_by_date[d]))
        return out_idx, list(desired_dates)

    # Non-strict: intersect, preserve desired order.
    intersect_dates = [d for d in desired_dates if d in index_by_date]
    if not intersect_dates:
        raise ValueError(
            "StaticProvider strict_date_coverage=False but requested range has no overlap with stored dataset."
        )

    out_idx = [int(index_by_date[d]) for d in intersect_dates]
    return out_idx, intersect_dates


def _all_dataset_ids(*, dataset: MarketDataset) -> Tuple[MarketId, ...]:
    """
    Return all MarketIds contained in any dataset container, in a stable order.
    """
    keys = {}
    for mid in dataset.panels.keys():
        keys[mid.key()] = mid
    for mid in dataset.curve_params.keys():
        keys[mid.key()] = mid
    for mid in dataset.vol_params.keys():
        keys[mid.key()] = mid

    # Stable ordering by canonical key.
    return tuple(keys[k] for k in sorted(keys.keys()))


# =============================================================================
# Slicing helpers (consistent with MarketDataset conventions)
# =============================================================================

def _slice_dataset(
    *,
    dataset: MarketDataset,
    sliced_dates: List[str],
    time_idx: List[int],
    scenario_count: int,
    keep_ids: Tuple[MarketId, ...],
    provider_name: str,
) -> MarketDataset:
    """
    Slice MarketDataset to subset of time indices and scenarios.

    Notes
    -----
    - We include only ids present in dataset.
    - If config.include_only_requested_ids=True, the caller already passed keep_ids=request.universe.ids.
      Missing ids are treated as a user/config error.
    """
    if not time_idx:
        raise ValueError("time_idx must not be empty.")
    if int(scenario_count) < 1:
        raise ValueError("scenario_count must be >= 1.")

    # Validate coverage if user requested a restricted universe set (detect missing ids).
    _ensure_ids_exist_if_requested(dataset=dataset, requested_ids=keep_ids)

    # Slice quote panels.
    sliced_panels: Dict[MarketId, Panel] = {}
    for mid in keep_ids:
        if mid in dataset.panels:
            sliced_panels[mid] = _slice_panel(panel=dataset.panels[mid], time_idx=time_idx, scenario_count=scenario_count)

    # Slice curve params (+ keep factories).
    sliced_curve_params: Dict[MarketId, Panel] = {}
    sliced_curve_factories: Dict[MarketId, Any] = {}
    for mid in keep_ids:
        if mid in dataset.curve_params:
            sliced_curve_params[mid] = _slice_panel(
                panel=dataset.curve_params[mid],
                time_idx=time_idx,
                scenario_count=scenario_count,
            )
            sliced_curve_factories[mid] = dataset.curve_factories[mid]

    # Slice vol params (+ keep factories).
    sliced_vol_params: Dict[MarketId, Panel] = {}
    sliced_vol_factories: Dict[MarketId, Any] = {}
    for mid in keep_ids:
        if mid in dataset.vol_params:
            sliced_vol_params[mid] = _slice_panel(
                panel=dataset.vol_params[mid],
                time_idx=time_idx,
                scenario_count=scenario_count,
            )
            sliced_vol_factories[mid] = dataset.vol_factories[mid]

    # Meta: preserve, plus provenance. Do not delete upstream keys.
    meta_in = dict(dataset.meta or {})
    meta_out: Mapping[str, object] = {
        **meta_in,
        "provider": str(provider_name),
        "provider_mode": "static_replay",
    }

    return MarketDataset(
        dates=list(sliced_dates),
        n_scenarios=int(scenario_count),
        panels=sliced_panels,
        curve_params=sliced_curve_params,
        curve_factories=sliced_curve_factories,  # type: ignore[arg-type]
        vol_params=sliced_vol_params,
        vol_factories=sliced_vol_factories,      # type: ignore[arg-type]
        meta=meta_out,
    )


def _ensure_ids_exist_if_requested(*, dataset: MarketDataset, requested_ids: Tuple[MarketId, ...]) -> None:
    """
    If user explicitly requested ids (include_only_requested_ids=True), ensure they exist.

    We treat missing ids as a configuration/user error (fail fast).
    """
    missing: List[MarketId] = []
    for mid in requested_ids:
        if (mid not in dataset.panels) and (mid not in dataset.curve_params) and (mid not in dataset.vol_params):
            missing.append(mid)

    if missing:
        miss = ", ".join(m.key() for m in sorted(missing, key=lambda x: x.key()))
        raise ValueError(f"StaticProvider store missing MarketIds: {miss}")


def _slice_panel(*, panel: Panel, time_idx: List[int], scenario_count: int) -> Panel:
    """
    Slice a Panel consistently with MarketDataset conventions.

    - time axis is always axis 0 (sliced by time_idx)
    - scenario axis is present only when axis_names[1] == "scenario" (then slice axis 1)
    """
    arr = np.asarray(panel.data)
    axis_names = tuple(panel.axis_names)

    # Slice time (axis 0).
    arr = np.take(arr, indices=time_idx, axis=0)

    # Slice scenario axis only if explicitly declared.
    if len(axis_names) >= 2 and axis_names[1] == "scenario":
        slicer = [slice(None)] * arr.ndim
        slicer[1] = slice(0, int(scenario_count))
        arr = arr[tuple(slicer)]

    return Panel(data=arr, axis_names=axis_names)


def _generate_dates(*, start: str, end: str, freq: str) -> List[str]:
    """
    Deterministic date grid generator.

    Supported:
      - D: daily
      - B: business days (Mon-Fri)
      - W: weekly (7-day step)
      - M: month approximated as 30D (deterministic; intentionally not calendar-accurate)
    """
    start_d = date.fromisoformat(str(start))
    end_d = date.fromisoformat(str(end))
    if end_d < start_d:
        raise ValueError(f"end < start: start={start}, end={end}")

    f = str(freq).strip().upper()
    if f not in {"D", "B", "W", "M"}:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: D, B, W, M.")

    out: List[str] = []
    current = start_d

    if f == "D":
        step = timedelta(days=1)
        while current <= end_d:
            out.append(current.isoformat())
            current += step
        return out

    if f == "B":
        while current <= end_d:
            if current.weekday() < 5:
                out.append(current.isoformat())
            current += timedelta(days=1)
        return out

    if f == "W":
        step = timedelta(days=7)
        while current <= end_d:
            out.append(current.isoformat())
            current += step
        return out

    # f == "M": deterministic 30-day step
    step = timedelta(days=30)
    while current <= end_d:
        out.append(current.isoformat())
        current += step
    return out