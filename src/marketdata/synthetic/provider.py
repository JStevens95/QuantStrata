from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, MutableMapping

from src.marketdata.curves.factories import FlatCurveFactory
from src.marketdata.dataset import MarketDataset
from src.marketdata.ids import MarketId
from src.marketdata.interfaces import Panel
from src.marketdata.market import Market
from src.marketdata.requests import MarketRequest, TimeseriesRequest
from src.marketdata.surfaces.factories import FlatVolFactory

# define logging at module level.
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _GenerationState:
    """
    Mutable state passed to each generator handler.

    We keep this as a small container so:
    - handlers can write output panels/params without returning large objects,
    - the provider remains easy to extend with new kinds or asset classes.
    """
    rng: np.random.Generator
    n_time: int
    n_scenarios: int

    # Scalar quote panels (SPOT, FIXING, etc.)
    quote_panels: MutableMapping[MarketId, Panel]

    # Parameter panels + factories (CURVE, VOL)
    curve_param_panels: MutableMapping[MarketId, Panel]
    curve_factories: MutableMapping[MarketId, FlatCurveFactory]

    vol_param_panels: MutableMapping[MarketId, Panel]
    vol_factories: MutableMapping[MarketId, FlatVolFactory]

    # Cache of generated SPOT arrays so FIXING can reuse them if needed.
    spot_cache: MutableMapping[MarketId, np.ndarray]


# A handler generates data for a single MarketId and writes into the shared state.
_MarketIdHandler = Callable[[MarketId, _GenerationState], None]


@dataclass(frozen=True, slots=True)
class SyntheticProvider:
    """
    Generic synthetic market-data provider (V1).

    What it does
    ------------
    - Produces deterministic synthetic data for any MarketId (by routing on MarketId.mkt_type).
    - Returns:
        * MarketDataset for ML/RL pipelines (arrays in Panels)
        * Market snapshots for pricing (via MarketDataset.snapshot())

    Why this design works long-term
    -------------------------------
    - Curves and vol surfaces are stored as *parameter panels* in the dataset.
    - Factories reconstruct Curve/VolSurface objects on demand in snapshot().
    - Later, you can replace these simple generators with:
        * term-structured curves
        * grid/smile vol surfaces
        * correlated multi-asset simulations
        * path-based scenarios
      without changing the consumer-facing API.

    Supported 'kind' routing (V1)
    -----------------------------
    - SPOT  : GBM-like synthetic path (positive)
    - CURVE : flat continuously-compounded rate (scalar)
    - VOL   : flat implied vol (positive scalar)
    - FIXING: copies SPOT if present, otherwise constant
    - other : constant fallback scalar panel

    Determinism
    -----------
    For the same (seed, request.start, request.end, request.freq) you get the same panels.
    """

    # random seed, default to 7.
    seed: int = 7

    # Default levels used as sensible initial values.
    default_fx_spot: float = 1.10
    default_eq_spot: float = 100.0
    default_rate: float = 0.02
    default_fx_vol: float = 0.12
    default_eq_vol: float = 0.25

    # SPOT path parameters (V1: simple GBM).
    spot_drift: float = 0.00
    spot_vol_fx: float = 0.10
    spot_vol_eq: float = 0.20

    # Small noise for CURVE/VOL so series are not perfectly constant (optional).
    curve_noise: float = 0.0005
    vol_noise: float = 0.005

    # Factories used by MarketDataset.snapshot() to reconstruct objects.
    curve_factory: FlatCurveFactory = FlatCurveFactory()
    vol_factory: FlatVolFactory = FlatVolFactory()

    # Registry mapping MarketId.kind -> handler function.
    _handlers_by_kind: Dict[str, _MarketIdHandler] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Build the dispatch table once.
        handlers: Dict[str, _MarketIdHandler] = {
            "SPOT": self._generate_spot,
            "CURVE": self._generate_flat_curve_params,
            "VOL": self._generate_flat_vol_params,
            "FIXING": self._generate_fixing,
        }
        object.__setattr__(self, "_handlers_by_kind", handlers)

    # -------------------------------------------------------------------------
    # Public provider API
    # -------------------------------------------------------------------------

    def get_market(self, request: MarketRequest) -> Market:
        """
        Return a Market snapshot for one as-of date.

        Implementation strategy
        -----------------------
        To keep one consistent code-path, we:
        1) generate a 1-date MarketDataset via get_timeseries()
        2) convert it into a Market using snapshot(0, scenario)

        This prevents divergence between "market" and "timeseries" logic over time.
        """
        scenario_idx = 0 if request.scenario is None else int(request.scenario)
        if scenario_idx < 0:
            raise ValueError("MarketRequest.scenario must be >= 0 when provided.")

        logger.debug(
            "SyntheticProvider.get_market(asof=%s, scenario=%s, universe=%d)",
            request.asof,
            scenario_idx,
            len(request.universe.ids),
        )

        # Create a dataset covering exactly one date. We generate enough scenarios
        # so that scenario_idx is a valid index.
        dataset = self.get_timeseries(
            TimeseriesRequest(
                start=request.asof,
                end=request.asof,
                freq="D",
                universe=request.universe,
                scenarios=max(1, scenario_idx + 1),
            )
        )

        # Convert the dataset slice into an object-based Market snapshot.
        return dataset.snapshot(time_idx=0, scenario_idx=scenario_idx)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        """
        Return a MarketDataset with panels shaped [time, scenario] for all IDs.

        Notes
        -----
        - We always materialize scalar data as [T, S] with axis_names ("time","scenario")
          to keep snapshot slicing unambiguous.
        - CURVE/VOL are returned as parameter panels + factories (not as quotes),
          so snapshot() can reconstruct Curve/VolSurface objects.
        """
        dates = _generate_dates(start=request.start, end=request.end, freq=request.freq)
        n_time = len(dates)
        n_scenarios = int(request.scenarios)

        if n_scenarios < 1:
            raise ValueError("TimeseriesRequest.scenarios must be >= 1.")

        # Create a deterministic RNG stream derived from request metadata.
        rng = np.random.default_rng(_stable_seed(self.seed, request.start, request.end, request.freq))

        # Prepare the containers we will populate.
        quote_panels: Dict[MarketId, Panel] = {}
        curve_param_panels: Dict[MarketId, Panel] = {}
        vol_param_panels: Dict[MarketId, Panel] = {}
        curve_factories: Dict[MarketId, FlatCurveFactory] = {}
        vol_factories: Dict[MarketId, FlatVolFactory] = {}
        spot_cache: Dict[MarketId, np.ndarray] = {}

        generation_state = _GenerationState(
            rng=rng,
            n_time=n_time,
            n_scenarios=n_scenarios,
            quote_panels=quote_panels,
            curve_param_panels=curve_param_panels,
            curve_factories=curve_factories,
            vol_param_panels=vol_param_panels,
            vol_factories=vol_factories,
            spot_cache=spot_cache,
        )

        # Generate data for each MarketId using dispatch-by-kind.
        for market_id in request.universe.ids:
            mkt_type = market_id.mkt_type.upper()

            # Select a handler; if unknown kind, fall back to a safe constant panel.
            handler = self._handlers_by_kind.get(mkt_type, self._generate_default_scalar_quote)

            # Handler writes into generation_state dictionaries.
            handler(market_id, generation_state)

        # Assemble the MarketDataset used for ML/RL and for snapshot()->Market.
        return MarketDataset(
            dates=dates,
            n_scenarios=n_scenarios,
            panels=quote_panels,
            curve_params=curve_param_panels,
            curve_factories=curve_factories,
            vol_params=vol_param_panels,
            vol_factories=vol_factories,
            meta={"provider": "SyntheticProvider", "seed": self.seed, "freq": request.freq},
        )

    # -------------------------------------------------------------------------
    # Handlers (generate data per MarketId.mkt_type)
    # -------------------------------------------------------------------------

    def _generate_spot(self, market_id: MarketId, state: _GenerationState) -> None:
        """
        Generate a SPOT panel.

        Output
        ------
        - state.quote_panels[market_id] = Panel([T,S], ("time","scenario"))
        - state.spot_cache[...] is populated so FIXING can reuse SPOT if appropriate.
        """
        initial_level = _default_spot_level(
            market_id,
            default_fx=self.default_fx_spot,
            default_eq=self.default_eq_spot,
        )
        spot_vol = _default_spot_vol(
            market_id,
            fx_vol=self.spot_vol_fx,
            eq_vol=self.spot_vol_eq,
        )

        # Generate a GBM-like positive panel: shape [T, S].
        spot_panel_array = _generate_gbm_spot_panel(
            rng=state.rng,
            n_time=state.n_time,
            n_scenarios=state.n_scenarios,
            initial_level=initial_level,
            drift=self.spot_drift,
            vol=spot_vol,
            dt=1.0 / 252.0,  # business-day-ish time step
        )

        # Wrap the raw ndarray in a Panel with clear axis naming.
        state.quote_panels[market_id] = Panel(data=spot_panel_array, axis_names=("time", "scenario"))

        # Cache the array for FIXING generation.
        state.spot_cache[market_id] = spot_panel_array

    def _generate_flat_curve_params(self, market_id: MarketId, state: _GenerationState) -> None:
        """
        Generate CURVE params (flat continuously-compounded rate series).

        Output
        ------
        - state.curve_param_panels[market_id] = Panel([T,S], ("time","scenario"))
        - state.curve_factories[market_id] = FlatCurveFactory
        """
        # Flat rate with tiny noise so the time series isn't perfectly constant (optional).
        rate_array = _generate_constant_panel(
            rng=state.rng,
            n_time=state.n_time,
            n_scenarios=state.n_scenarios,
            level=self.default_rate,
            noise_scale=self.curve_noise,
        )

        state.curve_param_panels[market_id] = Panel(data=rate_array, axis_names=("time", "scenario"))
        state.curve_factories[market_id] = self.curve_factory

    def _generate_flat_vol_params(self, market_id: MarketId, state: _GenerationState) -> None:
        """
        Generate VOL params (flat implied vol series).

        Output
        ------
        - state.vol_param_panels[market_id] = Panel([T,S], ("time","scenario"))
        - state.vol_factories[market_id] = FlatVolFactory
        """
        base_vol = _default_implied_vol(
            market_id,
            default_fx=self.default_fx_vol,
            default_eq=self.default_eq_vol,
        )

        vol_array = _generate_positive_constant_panel(
            rng=state.rng,
            n_time=state.n_time,
            n_scenarios=state.n_scenarios,
            level=base_vol,
            noise_scale=self.vol_noise,
            floor=1e-4,
        )

        state.vol_param_panels[market_id] = Panel(data=vol_array, axis_names=("time", "scenario"))
        state.vol_factories[market_id] = self.vol_factory

    def _generate_fixing(self, market_id: MarketId, state: _GenerationState) -> None:
        """
        Generate FIXING panel.

        Default behavior:
        - If there is a matching SPOT in the universe (same asset_class/name/qualifiers),
          reuse it (fixings look like observed spots).
        - Otherwise, fall back to a constant scalar panel.
        """
        # Construct the "spot-equivalent" ID.
        spot_id = MarketId(
            asset_class=market_id.asset_class,
            kind="SPOT",
            name=market_id.name,
            qualifiers=market_id.qualifiers,
        )

        if spot_id in state.spot_cache:
            # Reuse the already-generated spot panel array.
            state.quote_panels[market_id] = Panel(data=state.spot_cache[spot_id], axis_names=("time", "scenario"))
            return

        # If we didn't generate a spot, just make a constant fixing series.
        self._generate_default_scalar_quote(market_id, state)

    def _generate_default_scalar_quote(self, market_id: MarketId, state: _GenerationState) -> None:
        """
        Default handler for unknown MarketId kinds.

        We generate a constant scalar panel [T,S]. This is a useful placeholder for:
        - dividends
        - spreads
        - repo rates
        - borrow costs
        - anything you haven't implemented yet
        """
        constant_array = _generate_constant_panel(
            rng=state.rng,
            n_time=state.n_time,
            n_scenarios=state.n_scenarios,
            level=1.0,
            noise_scale=0.0,
        )
        state.quote_panels[market_id] = Panel(data=constant_array, axis_names=("time", "scenario"))


# -----------------------------------------------------------------------------
# Helper functions (small, focused, easy to unit test)
# -----------------------------------------------------------------------------

def _stable_seed(base_seed: int, *parts: str) -> int:
    """
    Build a deterministic seed from (base_seed + request metadata).

    This guarantees that two identical requests create identical synthetic data,
    which is important for tests and reproducible demos.
    """
    h = 2166136261  # FNV-1a 32-bit offset basis
    for p in parts:
        for b in p.encode("utf-8"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
    return int((base_seed ^ h) & 0xFFFFFFFF)


def _generate_dates(start: str, end: str, freq: str) -> list[str]:
    """
    Generate ISO date strings from start to end inclusive.

    Supported in V1:
    - D: daily
    - B: business day (Mon-Fri)
    - W: weekly (7-day step)
    - M: monthly (30-day step placeholder)
    """
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    if end_d < start_d:
        raise ValueError(f"end < start: start={start}, end={end}")

    f = freq.strip().upper()
    if f not in {"D", "B", "W", "M"}:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: D, B, W, M.")

    dates: list[str] = []
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

    # f == "M": simple placeholder for V1 (30-day increments)
    step = timedelta(days=30)
    while current <= end_d:
        dates.append(current.isoformat())
        current += step
    return dates


def _generate_constant_panel(
    rng: np.random.Generator,
    n_time: int,
    n_scenarios: int,
    level: float,
    noise_scale: float,
) -> np.ndarray:
    """Generate scalar data shaped [time, scenario] with optional small noise."""
    array = np.full((n_time, n_scenarios), float(level), dtype=float)
    if noise_scale > 0.0:
        array += rng.normal(loc=0.0, scale=float(noise_scale), size=(n_time, n_scenarios))
    return array


def _generate_positive_constant_panel(
    rng: np.random.Generator,
    n_time: int,
    n_scenarios: int,
    level: float,
    noise_scale: float,
    floor: float,
) -> np.ndarray:
    """Same as constant panel but floored positive (useful for implied vols)."""
    array = _generate_constant_panel(rng, n_time, n_scenarios, level, noise_scale)
    return np.maximum(array, float(floor))


def _generate_gbm_spot_panel(
    rng: np.random.Generator,
    n_time: int,
    n_scenarios: int,
    initial_level: float,
    drift: float,
    vol: float,
    dt: float,
) -> np.ndarray:
    """
    Generate a simple GBM-like spot panel with shape [time, scenario].

    log(S_{t+1}/S_t) = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    """
    if initial_level <= 0.0:
        raise ValueError("initial_level must be > 0.")
    if vol < 0.0:
        raise ValueError("vol must be >= 0.")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")

    spot = np.empty((n_time, n_scenarios), dtype=float)
    spot[0, :] = float(initial_level)

    # If only one time point, we are done.
    if n_time == 1:
        return spot

    # Draw standard normal innovations.
    z = rng.normal(loc=0.0, scale=1.0, size=(n_time - 1, n_scenarios))

    # Compute log-returns.
    mu = float(drift)
    sigma = float(vol)
    log_returns = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z

    # Build the path forward.
    for t in range(1, n_time):
        spot[t, :] = spot[t - 1, :] * np.exp(log_returns[t - 1, :])

    return spot


def _default_spot_level(market_id: MarketId, default_fx: float, default_eq: float) -> float:
    """Heuristic: EQ starts around 100, FX starts around ~1.10 (V1)."""
    return float(default_eq) if market_id.asset_class.upper() == "EQ" else float(default_fx)


def _default_spot_vol(market_id: MarketId, fx_vol: float, eq_vol: float) -> float:
    """Heuristic: equities tend to have higher vol than FX (V1)."""
    return float(eq_vol) if market_id.asset_class.upper() == "EQ" else float(fx_vol)


def _default_implied_vol(market_id: MarketId, default_fx: float, default_eq: float) -> float:
    """Heuristic: equity implied vols often higher than FX (V1)."""
    return float(default_eq) if market_id.asset_class.upper() == "EQ" else float(default_fx)