from __future__ import annotations

# NumPy is used for deterministic RNG creation and fallback constant arrays.
import numpy as np

# dataclass provides a clean immutable engine definition (frozen).
from dataclasses import dataclass

# date/timedelta used for deterministic time grid generation.
from datetime import date, timedelta

# Typing utilities for container structures and dependency closure.
from typing import Dict, Iterable, List, Set

# Core dataset + identifiers + market snapshot object.
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market

# Panel is the internal multi-axis container for quotes/curve params/vol params.
from src.marketdata.core.panel import Panel

# Requests define provider interface contracts.
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest

# Factories reconstruct pricing objects from dataset blocks at snapshot time.
from src.marketdata.curves.factories import ZeroRateCurveFactory
from src.marketdata.surfaces.factories import GridVolFactory

# Shared mutable state passed to generators.
from src.marketdata.synthetic.context import SyntheticGenerationState

# Registry routes MarketId -> generator + prerequisites.
from src.marketdata.synthetic.registry import SyntheticRegistry


@dataclass(frozen=True, slots=True)
class SyntheticMarketEngine:
    """
    Synthetic market-data engine (Vn core).

    Production-grade behaviors implemented here
    -------------------------------------------
    1) Dependency closure:
       - If a requested MarketId has prerequisites (e.g., FX VOL requires SPOT + CURVES),
         the engine auto-adds them into the generation set.

    2) Dependency-safe ordering:
       - VOL is generated after SPOT/CURVE/FIXING so forward-moneyness vols can use spot+curves.
       - Ordering does not affect determinism because RNG is per MarketId.

    3) Order-independent determinism:
       - Each MarketId uses a stable RNG substream derived from (seed, MarketId.key()).
    """

    # Global seed for this engine instance (used to derive per-MarketId sub-streams).
    seed: int
    # Registry used to route MarketIds to generator functions.
    registry: SyntheticRegistry

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get_market(self, request: MarketRequest) -> Market:
        """
        Generate a single Market snapshot for a given as-of date.

        Implementation detail
        ---------------------
        We generate a one-date MarketDataset and slice time_idx=0.
        """
        # If request.scenario is None, default to scenario 0.
        scenario_idx = 0 if request.scenario is None else int(request.scenario)
        # Enforce scenario_idx is non-negative.
        if scenario_idx < 0:
            raise ValueError("MarketRequest.scenario must be >= 0 when provided.")

        # Build a one-date dataset (start=end=asof) with enough scenarios to include scenario_idx.
        dataset = self.get_timeseries(
            TimeseriesRequest(
                start=request.asof,                    # Start date = asof.
                end=request.asof,                      # End date = asof (single point).
                freq="D",                              # Daily frequency (one point).
                universe=request.universe,             # Universe of requested MarketIds.
                scenarios=max(1, scenario_idx + 1),    # Ensure we have scenario_idx available.
            )
        )

        # Slice out the single snapshot and return it as a Market object.
        return dataset.snapshot(time_idx=0, scenario_idx=scenario_idx)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        """
        Generate a MarketDataset for the request's date range and scenario count.

        Determinism guarantee
        ---------------------
        - RNG is stable per MarketId key (seed + MarketId.key()).
        - Universe ordering does not affect generated values.
        """
        # Build the deterministic date grid for this timeseries request.
        dates = _generate_dates(start=request.start, end=request.end, freq=request.freq)
        # Count number of time points (T).
        n_time = len(dates)

        # Parse and validate scenario count (S).
        n_scenarios = int(request.scenarios)
        if n_scenarios < 1:
            raise ValueError("TimeseriesRequest.scenarios must be >= 1.")

        # ------------------------------------------------------------------
        # Expand Universe -> closure(required market ids)
        # ------------------------------------------------------------------
        # Copy requested MarketIds (the user-specified universe).
        requested_ids = list(request.universe.ids)
        # Expand to include prerequisites declared by generators.
        all_ids = self._dependency_closure(requested_ids)

        # ------------------------------------------------------------------
        # Allocate outputs owned by the engine (mutated by generators)
        # ------------------------------------------------------------------
        # Quote panels keyed by MarketId (e.g., SPOT, FIXING, scalar quotes).
        quote_panels: Dict[MarketId, Panel] = {}
        # Curve parameter panels keyed by MarketId (e.g., tenors + zero rates).
        curve_param_panels: Dict[MarketId, Panel] = {}
        # Vol parameter panels keyed by MarketId (e.g., expiry x strike grids).
        vol_param_panels: Dict[MarketId, Panel] = {}

        # Factories that reconstruct curve objects from curve_param_panels at snapshot time.
        curve_factories: Dict[MarketId, ZeroRateCurveFactory] = {}
        # Factories that reconstruct vol objects from vol_param_panels at snapshot time.
        vol_factories: Dict[MarketId, GridVolFactory] = {}

        # Cache for spot panels so FIXING can reuse SPOT exactly (by MarketId).
        spot_cache: Dict[MarketId, np.ndarray] = {}

        # Build the shared mutable generation state that generators will write into.
        state = SyntheticGenerationState(
            dates=dates,                         # The date grid.
            n_time=n_time,                       # T.
            n_scenarios=n_scenarios,             # S.
            quote_panels=quote_panels,           # Output quote panels.
            curve_param_panels=curve_param_panels,  # Output curve params.
            curve_factories=curve_factories,     # Output curve factories.
            vol_param_panels=vol_param_panels,   # Output vol params.
            vol_factories=vol_factories,         # Output vol factories.
            spot_cache=spot_cache,               # Spot cache for FIXING reuse.
        )

        # ------------------------------------------------------------------
        # Two-phase generation ordering:
        #   - Generate non-VOL first
        #   - Generate VOL last (so it can read SPOT/CURVE panels)
        # ------------------------------------------------------------------
        # Separate IDs into non-vol and vol groups by mkt_type.
        non_vol = [mid for mid in all_ids if mid.mkt_type.strip().upper() != "VOL"]
        vols = [mid for mid in all_ids if mid.mkt_type.strip().upper() == "VOL"]

        # Optional: keep FIXING after SPOT but before VOL by stable sort key.
        # We keep this deterministic and readable; it doesn’t affect RNG determinism.
        def _phase_rank(mid: MarketId) -> int:
            # Normalize mkt_type once for routing.
            t = mid.mkt_type.strip().upper()
            # Spots first so everything downstream can reuse them.
            if t == "SPOT":
                return 0
            # Curves next since vol surfaces often need forwards/carry.
            if t == "CURVE":
                return 1
            # Fixings after spot (may reuse spot) and before vol.
            if t == "FIXING":
                return 2
            # Everything else last (unknown or other quote types).
            return 3

        # Sort non-vol IDs deterministically by (phase, MarketId.key()).
        non_vol_sorted = sorted(non_vol, key=lambda m: (_phase_rank(m), m.key()))

        # Generate all non-vol items first.
        for market_id in non_vol_sorted:
            self._generate_one(market_id=market_id, state=state)

        # Generate vol surfaces last, sorted deterministically by MarketId.key().
        for market_id in sorted(vols, key=lambda m: m.key()):
            self._generate_one(market_id=market_id, state=state)

        # Build and return the final MarketDataset (snapshot() will use factories).
        return MarketDataset(
            dates=list(dates),                   # Persist dates as list[str].
            n_scenarios=n_scenarios,             # Persist scenario count.
            panels=quote_panels,                 # Quote panels produced by generation.
            curve_params=curve_param_panels,     # Curve param panels produced by generation.
            curve_factories=curve_factories,     # Curve factories produced by generation.
            vol_params=vol_param_panels,         # Vol param panels produced by generation.
            vol_factories=vol_factories,         # Vol factories produced by generation.
            meta={                               # Helpful metadata for debugging/traceability.
                "provider": "SyntheticMarketEngine",
                "seed": int(self.seed),
                "freq": request.freq,
            },
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _generate_one(self, *, market_id: MarketId, state: SyntheticGenerationState) -> None:
        """
        Generate one MarketId using the registry if possible, else fall back to a safe scalar quote.

        This keeps dataset construction robust while clearly signaling missing generator coverage.
        """
        # Look up generator function based on (asset_class, mkt_type).
        generator = self.registry.get(market_id=market_id)

        # If no generator registered, produce a conservative constant quote panel.
        if generator is None:
            _generate_default_scalar_quote(market_id=market_id, state=state)
            return

        # Otherwise call the registered generator (mutates state in-place).
        generator(market_id, state)

        # ------------------------------------------------------------------
        # Desk-hardening: ensure generator wrote the expected output container.
        # This catches silent generator bugs early (especially as you add EQ/CR).
        # ------------------------------------------------------------------
        t = market_id.mkt_type.strip().upper()

        if t == "CURVE":
            if market_id not in state.curve_param_panels or market_id not in state.curve_factories:
                raise ValueError(
                    "IR CURVE generator did not populate curve_param_panels and curve_factories.\n"
                    f"  market_id={market_id.key()}"
                )

        elif t == "VOL":
            if market_id not in state.vol_param_panels or market_id not in state.vol_factories:
                raise ValueError(
                    "VOL generator did not populate vol_param_panels and vol_factories.\n"
                    f"  market_id={market_id.key()}"
                )

        else:
            if market_id not in state.quote_panels:
                raise ValueError(
                    "Quote-like generator did not populate quote_panels.\n"
                    f"  market_id={market_id.key()}"
                )

    def _dependency_closure(self, requested: Iterable[MarketId]) -> List[MarketId]:
        """
        Expand requested MarketIds to include prerequisites via registry.requirements().

        Design
        ------
        - This mimics desk systems: if you ask for VOL, you implicitly need SPOT (and often curves).
        - We keep it deterministic by using MarketId.key() for de-duplication + ordering.

        Notes
        -----
        - We do *not* attempt to detect cycles beyond a simple visited set; requirements should be acyclic.
        """
        # Output list of MarketIds after dependency expansion.
        out: List[MarketId] = []
        # Track seen MarketId keys to avoid duplicates and infinite loops.
        seen: Set[str] = set()

        def _add(mid: MarketId) -> None:
            # Convert MarketId to its canonical string key for stable deduplication.
            k = mid.key()
            # Skip if already added.
            if k in seen:
                return
            # Mark key as seen.
            seen.add(k)
            # Append MarketId to output list.
            out.append(mid)

        # Start BFS-like traversal from requested IDs.
        queue: List[MarketId] = list(requested)

        # Process queue until no new prerequisites remain.
        while queue:
            # Pop from front to preserve a predictable traversal order.
            mid = queue.pop(0)
            # Add this MarketId if not already seen.
            _add(mid)

            # Ask registry for prerequisites for this MarketId.
            reqs = self.registry.requirements(market_id=mid)
            # Push unseen prerequisites into the queue.
            for r in reqs:
                if r.key() not in seen:
                    queue.append(r)

        # Sort by key for deterministic ordering (generation phase ordering happens later).
        return sorted(out, key=lambda m: m.key())


# -------------------------------------------------------------------------
# Deterministic helpers + engine-level fallback
# -------------------------------------------------------------------------

def stable_seed(base_seed: int, *parts: str) -> int:
    """
    Compute a deterministic uint32 seed from a base seed and stable string parts.

    Notes
    -----
    - Intentionally simple and deterministic across Python versions.
    - Suitable for np.random.default_rng(seed).
    """
    # Initialize FNV-1a hash seed.
    h = 2166136261
    # Incorporate each string part into the hash.
    for p in parts:
        # Hash over the UTF-8 bytes for deterministic behavior.
        for b in str(p).encode("utf-8"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
    # Mix in the base_seed and return as uint32 int.
    return int((int(base_seed) ^ h) & 0xFFFFFFFF)


def rng_for_market_id(*, base_seed: int, market_id: MarketId) -> np.random.Generator:
    """
    Create a deterministic RNG substream for a MarketId.

    Why
    ---
    - Prevent Universe ordering from changing generated data.
    - Make debugging reproducible: each MarketId has its own RNG stream.
    """
    # Derive a stable per-market seed from base_seed + MarketId.key().
    sub_seed = stable_seed(int(base_seed), market_id.key())
    # Create and return a NumPy Generator with that seed.
    return np.random.default_rng(sub_seed)


def _generate_default_scalar_quote(*, market_id: MarketId, state: SyntheticGenerationState) -> None:
    """
    Conservative fallback generator for unknown MarketIds.

    Policy
    ------
    - Generate a constant 1.0 quote across time/scenarios.
    - This keeps dataset construction robust while highlighting missing generator coverage.
    """
    # Create a [T,S] array filled with 1.0.
    constant = np.full((int(state.n_time), int(state.n_scenarios)), 1.0, dtype=float)
    # Store as a quote panel with axes ("time","scenario").
    state.quote_panels[market_id] = Panel(data=constant, axis_names=("time", "scenario"))


def _generate_dates(*, start: str, end: str, freq: str) -> list[str]:
    """
    Deterministic date grid generator.

    Supported frequencies
    ---------------------
    - D: daily
    - B: business days (Mon-Fri)
    - W: weekly (7-day step)
    - M: month approximated as 30D (deterministic; not calendar-accurate by design)
    """
    # Parse start date from ISO string to datetime.date.
    start_d = date.fromisoformat(str(start))
    # Parse end date from ISO string to datetime.date.
    end_d = date.fromisoformat(str(end))

    # Reject invalid ranges early.
    if end_d < start_d:
        raise ValueError(f"end < start: start={start}, end={end}")

    # Normalize frequency string.
    f = str(freq).strip().upper()
    # Ensure frequency is in the supported set.
    if f not in {"D", "B", "W", "M"}:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: D, B, W, M.")

    # Accumulate ISO date strings here.
    dates: list[str] = []
    # Current cursor date (will step forward).
    current = start_d

    # Daily frequency: add every calendar day.
    if f == "D":
        step = timedelta(days=1)
        while current <= end_d:
            dates.append(current.isoformat())
            current += step
        return dates

    # Business day frequency: add only Mon-Fri.
    if f == "B":
        while current <= end_d:
            if current.weekday() < 5:
                dates.append(current.isoformat())
            current += timedelta(days=1)
        return dates

    # Weekly frequency: add every 7 days.
    if f == "W":
        step = timedelta(days=7)
        while current <= end_d:
            dates.append(current.isoformat())
            current += step
        return dates

    # Monthly frequency: deterministic 30-day approximation (not calendar-accurate).
    step = timedelta(days=30)
    while current <= end_d:
        dates.append(current.isoformat())
        current += step
    return dates