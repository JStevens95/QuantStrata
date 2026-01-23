"""
Generic Marketdata Timeseries Builder Pipeline (V1)

This module provides a reusable, built-in pipeline that:
  1) Builds a MarketDataProvider (V1: SyntheticProvider)
  2) Parses MarketIds from config
  3) Builds Universe
  4) Builds TimeseriesRequest
  5) Builds an in-memory MarketDataset via provider.get_timeseries(...)
  6) Optionally takes a Market snapshot (marketdata.snapshot)

Design goals
------------
- Modular: each Step has one responsibility.
- Deterministic: synthetic data can be seeded (important for tests).
- Vn-proof: stable ctx.state keys + clear extension points for future providers.
"""

from __future__ import annotations  # allow forward references in type hints (py<3.11 friendly)

# --- Standard library imports ---
from dataclasses import dataclass  # used to define immutable/structured Step classes
from typing import Any, Dict, List, Mapping, Optional, Union  # typing primitives for clarity

# --- Orchestrator imports (execution framework) ---
from src.orchestrator.config.schemas import RunConfig  # validated run config passed into pipelines
from src.orchestrator.core.context import Context  # execution context passed step-to-step
from src.orchestrator.core.pipeline import Pipeline  # pipeline container with ordered steps
from src.orchestrator.core.step import Step  # base Step interface (run(ctx)->ctx)

# --- Marketdata imports (marketdata framework) ---
from src.marketdata.core.ids import MarketId  # canonical identifiers for quotes/curves/vols
from src.marketdata.core.requests import TimeseriesRequest, Universe  # request objects
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig  # synthetic provider config
from src.marketdata.providers.synthetic.provider import SyntheticProvider  # synthetic provider implementation

# --- State Keys imports ---
from src.orchestrator.core.state_keys import StateKeys as Keys


# =============================================================================
# Config helpers (pure functions, easy to reuse/test)
# =============================================================================

def _require_dict(parent: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Fetch parent[key] and assert it is a dict (with a helpful error message)."""
    if key not in parent:  # check required key exists
        raise KeyError(f"Missing required config key: {key}")  # fail fast with exact missing key
    value = parent[key]  # pull the value out of the mapping
    if not isinstance(value, dict):  # validate type
        raise TypeError(f"Config key '{key}' must be a dict, got: {type(value).__name__}")  # clear typing error
    return value  # return the validated dict


def _require_str(value: Any, *, key_name: str) -> str:
    """Coerce/validate a required non-empty string config field."""
    if not isinstance(value, str):  # validate type is str
        raise TypeError(f"'{key_name}' must be a str, got: {type(value).__name__}")  # explicit error
    out = value.strip()  # normalize whitespace
    if not out:  # reject empty strings after stripping
        raise ValueError(f"'{key_name}' must be a non-empty string")  # explicit error
    return out  # return normalized string


def _require_int(value: Any, *, key_name: str, min_value: Optional[int] = None) -> int:
    """Coerce/validate an int-like config field, optionally enforcing a minimum."""
    try:
        out = int(value)  # attempt to coerce to int (supports "1", numpy ints, etc.)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"'{key_name}' must be int-like, got: {type(value).__name__}") from exc  # preserve cause
    if min_value is not None and out < min_value:  # enforce min constraint if provided
        raise ValueError(f"'{key_name}' must be >= {min_value}, got: {out}")  # explicit min-value error
    return out  # return validated int


def _require_str_list(value: Any, *, key_name: str) -> List[str]:
    """Validate that value is a non-empty list of non-empty strings."""
    if not isinstance(value, list):  # must be a list
        raise TypeError(f"'{key_name}' must be a list[str], got: {type(value).__name__}")  # explicit error

    out: List[str] = []  # create an output list we will populate with normalized values

    for i, item in enumerate(value):  # iterate with index for better error messages
        if not isinstance(item, str):  # each element must be a string
            raise TypeError(f"'{key_name}[{i}]' must be str, got: {type(item).__name__}")  # explicit error
        s = item.strip()  # normalize whitespace
        if not s:  # reject empty strings
            raise ValueError(f"'{key_name}[{i}]' must be a non-empty string")  # explicit error
        out.append(s)  # store normalized string

    if not out:  # reject empty lists
        raise ValueError(f"'{key_name}' must not be empty")  # explicit error

    return out  # return validated list


def _marketdata_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """
    Extract the `marketdata` block from RunConfig.params and validate structure.

    Expected:
      cfg.params["marketdata"] is a dict
    """
    if not isinstance(cfg.params, dict):  # ensure params is a mapping
        raise TypeError("RunConfig.params must be a dict")  # explicit error (config should already validate)
    return _require_dict(cfg.params, "marketdata")  # required config block


def _build_provider(md_cfg: Mapping[str, Any]) -> Any:
    """
    Provider factory.

    V1 supports:
      - synthetic

    Extension points (V2/Vn):
      - static provider (from a dataset or from disk)
      - hybrid provider (primary + fallback with provenance)
    """
    provider_cfg = md_cfg.get("provider", {}) or {}  # fetch provider sub-config (default empty dict)
    if not isinstance(provider_cfg, dict):  # validate config type
        raise TypeError("'marketdata.provider' must be a dict if provided")  # explicit error

    provider_type = str(provider_cfg.get("type", "synthetic")).strip().lower()  # normalized provider type

    if provider_type in {"synthetic", "syn"}:  # synthetic provider path
        seed = _require_int(provider_cfg.get("seed", 123), key_name="marketdata.provider.seed")  # deterministic seed
        name = str(provider_cfg.get("name", "SyntheticProvider")).strip() or "SyntheticProvider"  # readable name
        return SyntheticProvider(  # construct provider instance
            seed=seed,  # deterministic seed used by synthetic generator
            config=SyntheticProviderConfig(),  # default synthetic config (customizable later)
            name=name,  # debug-friendly provider name
        )

    # if we reach here: provider type is not supported in V1
    raise ValueError(  # raise explicit error with guidance
        f"Unsupported provider type '{provider_type}'. V1 supports only 'synthetic'."
    )


def _resolve_snapshot_time_index(dataset: Any, time_spec: Union[str, int]) -> int:
    """
    Convert a snapshot time spec into an explicit 0..T-1 integer index.

    Supported:
      - "last"
      - an integer index
    """
    num_times = len(dataset.dates)  # dataset.dates is the canonical time axis (list-like)
    if num_times <= 0:  # cannot snapshot a dataset with no dates
        raise ValueError("MarketDataset has no dates; cannot snapshot.")  # explicit error

    if isinstance(time_spec, int):  # integer index path
        if time_spec < 0 or time_spec >= num_times:  # enforce explicit bounds (no negative indexing)
            raise IndexError(f"snapshot.time out of range: {time_spec} for T={num_times}")  # explicit error
        return int(time_spec)  # return validated index

    t = str(time_spec).strip().lower()  # normalize string time spec

    if t == "last":  # special keyword for last index
        return num_times - 1  # compute explicit last index

    # unsupported string spec
    raise ValueError("marketdata.snapshot.time must be 'last' or an int index")  # explicit error


# =============================================================================
# Steps (single responsibility; stable ctx.state keys)
# =============================================================================

@dataclass(frozen=True, slots=True)
class BuildProviderStep(Step):
    """
    Build the MarketDataProvider and attach it to Context.provider.

    Output:
      ctx.provider is set (SyntheticProvider in V1).
    """

    def run(self, ctx: Context) -> Context:
        md_cfg = _marketdata_cfg(ctx.cfg)  # read/validate marketdata config block
        ctx.provider = _build_provider(md_cfg)  # build provider and attach to Context
        return ctx  # return Context for downstream steps


@dataclass(frozen=True, slots=True)
class BuildMarketIdsStep(Step):
    """
    Parse configured MarketId strings and store MarketId objects in ctx.state.

    Outputs (stable keys)
    ---------------------
    ctx.state["market_id_strings"] : List[str]
    ctx.state["market_ids"]        : List[MarketId]
    ctx.state["market_ids_pretty"] : List[str]
    """

    def run(self, ctx: Context) -> Context:
        md_cfg = _marketdata_cfg(ctx.cfg)  # read/validate marketdata config block

        id_strings = _require_str_list(md_cfg.get("ids"), key_name="marketdata.ids")  # validate list[str]
        market_ids = [MarketId.parse(s) for s in id_strings]  # parse each id string into a MarketId

        ctx.put(Keys.MARKET_ID_STRINGS, id_strings)  # store normalized id strings (useful for debugging)
        ctx.put(Keys.MARKET_IDS, market_ids)  # store parsed MarketId objects (used by Universe)
        ctx.put(Keys.MARKET_IDS_PRETTY, [str(mid) for mid in market_ids])  # store string representations for printing

        return ctx  # return Context for downstream steps


@dataclass(frozen=True, slots=True)
class BuildUniverseStep(Step):
    """
    Build a Universe from MarketIds.

    Input:
      ctx.state["market_ids"] : List[MarketId]

    Output:
      ctx.state["universe"] : Universe
    """

    def run(self, ctx: Context) -> Context:
        market_ids: List[MarketId] = ctx.get(Keys.MARKET_IDS)  # read MarketIds produced by previous step
        universe = Universe(market_ids)  # Universe validates duplicates/empties internally
        ctx.put(Keys.UNIVERSE, universe)  # store Universe for request-building step
        return ctx  # return Context for downstream steps


@dataclass(frozen=True, slots=True)
class BuildTimeseriesRequestStep(Step):
    """
    Build a TimeseriesRequest from config + ctx.state["universe"].

    Outputs (stable keys)
    ---------------------
    ctx.state["request"]         : TimeseriesRequest
    ctx.state["request_summary"] : Dict[str, Any]
    """

    def run(self, ctx: Context) -> Context:
        md_cfg = _marketdata_cfg(ctx.cfg)  # read/validate marketdata config block

        start = _require_str(md_cfg.get("start", ""), key_name="marketdata.start")  # required ISO date string
        end = _require_str(md_cfg.get("end", ""), key_name="marketdata.end")  # required ISO date string
        freq = str(md_cfg.get("freq", "D")).strip() or "D"  # default daily frequency if missing/blank
        scenarios = _require_int(md_cfg.get("scenarios", 1), key_name="marketdata.scenarios", min_value=1)  # >=1

        universe: Universe = ctx.get(Keys.UNIVERSE)  # read Universe built by previous step

        request = TimeseriesRequest(  # create provider request object
            universe=universe,  # market ids to include
            start=start,  # ISO start date string (provider expects str)
            end=end,  # ISO end date string (provider expects str)
            freq=freq,  # frequency string (e.g. "D")
            scenarios=scenarios,  # number of scenarios requested
        )

        ctx.put(Keys.REQUEST, request)  # store request for dataset-building step
        ctx.put(  # store a lightweight summary for console output and tests
            Keys.REQUEST_SUMMARY,
            {
                "start": start,  # echo back normalized start
                "end": end,  # echo back normalized end
                "freq": freq,  # echo back frequency
                "scenarios": scenarios,  # echo back scenario count
                "n_ids": len(universe.ids),  # number of requested market ids
            },
        )

        return ctx  # return Context for downstream steps


@dataclass(frozen=True, slots=True)
class BuildDatasetStep(Step):
    """
    Build an in-memory MarketDataset via provider.get_timeseries(request).

    Inputs:
      ctx.provider
      ctx.state["request"]

    Output:
      ctx.state["dataset"]
    """

    def run(self, ctx: Context) -> Context:
        if ctx.provider is None:  # provider must exist
            raise RuntimeError("Context.provider is None (BuildProviderStep did not run).")  # fail fast

        request: TimeseriesRequest = ctx.get(Keys.REQUEST)  # read request built earlier
        dataset = ctx.provider.get_timeseries(request)  # ask provider to generate/fetch dataset

        ctx.put(Keys.DATASET, dataset)  # store dataset for snapshot/pricing steps downstream
        return ctx  # return Context for downstream steps


@dataclass(frozen=True, slots=True)
class BuildSnapshotStep(Step):
    """
    Optional: take a Market snapshot from the dataset.

    Controlled by:
      cfg.params["marketdata"]["snapshot"] (dict)

    If snapshot config is absent:
      this step is a no-op and the pipeline still succeeds.

    Outputs (if enabled):
      ctx.state["market"]
      ctx.state["market_snapshot_summary"]
    """

    def run(self, ctx: Context) -> Context:
        md_cfg = _marketdata_cfg(ctx.cfg)  # read/validate marketdata config block
        snapshot_cfg = md_cfg.get("snapshot")  # fetch snapshot sub-config (optional)

        if snapshot_cfg is None:  # if not requested
            return ctx  # do nothing (no-op step)

        if not isinstance(snapshot_cfg, dict):  # validate snapshot config is a dict
            raise TypeError("marketdata.snapshot must be a dict if provided")  # explicit error

        dataset = ctx.get(Keys.DATASET)  # read dataset built earlier
        time_spec: Union[str, int] = snapshot_cfg.get("time", "last")  # default snapshot time is last date
        scenario_idx = _require_int(  # parse scenario index
            snapshot_cfg.get("scenario_idx", 0),  # default base scenario
            key_name="marketdata.snapshot.scenario_idx",  # error context
            min_value=0,  # scenario idx cannot be negative
        )

        time_idx = _resolve_snapshot_time_index(dataset, time_spec)  # convert spec -> explicit index
        market = dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)  # build Market snapshot

        ctx.put(Keys.MARKET, market)  # store Market for downstream pricing steps
        ctx.put(  # store a tiny summary for debugging/tests
            Keys.MARKET_SNAPSHOT_SUMMARY,
            {
                "time_idx": time_idx,  # explicit resolved time index
                "scenario_idx": scenario_idx,  # scenario chosen
                "T": len(dataset.dates),  # total time points in dataset
            },
        )

        return ctx  # return Context for downstream steps


# =============================================================================
# Pipeline builder (registered via orchestrator runtime discovery)
# =============================================================================

def build_pipeline(cfg: RunConfig) -> Pipeline:
    """
    Build the built-in pipeline: "marketdata.timeseries.v1".

    Notes:
    - `cfg` is provided for builder consistency (some pipelines may use cfg at build time).
    - V1 does not need cfg at build time, but we accept it for uniform builder signature.
    """
    _ = cfg  # keep signature consistent; avoids unused parameter lint warnings

    steps: List[Step] = [  # create ordered steps for the pipeline
        BuildProviderStep(name="build_provider"),  # 1) provider
        BuildMarketIdsStep(name="build_market_ids"),  # 2) ids
        BuildUniverseStep(name="build_universe"),  # 3) universe
        BuildTimeseriesRequestStep(name="build_request"),  # 4) request
        BuildDatasetStep(name="build_dataset"),  # 5) dataset
        BuildSnapshotStep(name="build_snapshot"),  # 6) optional snapshot
    ]

    return Pipeline(  # return pipeline instance
        name="marketdata.timeseries",  # registry name / identity
        steps=steps,  # ordered steps
    )