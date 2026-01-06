from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, MutableMapping

import numpy as np

from src.marketdata.curves.factories import ZeroCurveFactory
from src.marketdata.dataset import MarketDataset
from src.marketdata.ids import MarketId
from src.marketdata.interfaces import Panel
from src.marketdata.market import Market
from src.marketdata.requests import MarketRequest, TimeseriesRequest
from src.marketdata.surfaces.factories import GridVolFactory

from src.marketdata.providers.synthetic.config import SyntheticProviderConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _GenerationState:
    rng: np.random.Generator
    n_time: int
    n_scenarios: int

    quote_panels: MutableMapping[MarketId, Panel]
    curve_param_panels: MutableMapping[MarketId, Panel]
    curve_factories: MutableMapping[MarketId, ZeroCurveFactory]
    vol_param_panels: MutableMapping[MarketId, Panel]
    vol_factories: MutableMapping[MarketId, GridVolFactory]

    spot_cache: MutableMapping[MarketId, np.ndarray]


_MarketIdHandler = Callable[[MarketId, _GenerationState], None]


@dataclass(frozen=True, slots=True)
class SyntheticProvider:
    """
    Synthetic market-data provider (V1→Vn ready).

    Key upgrades vs previous version
    --------------------------------
    1) Config object holds all “defaults” (no giant provider dataclass).
    2) Deterministic per MarketId: outputs do not depend on universe ordering.
    3) CURVE generates term-structured zero curves (params [T,S,K,2]).
    4) VOL generates grid smiles (params [T,S,n_exp*n_strikes]) -> GridVolSurface.
    """
    seed: int = 7
    config: SyntheticProviderConfig = SyntheticProviderConfig()

    _handlers_by_kind: Dict[str, _MarketIdHandler] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        handlers: Dict[str, _MarketIdHandler] = {
            "SPOT": self._generate_spot_gbm,
            "CURVE": self._generate_zero_curve_params,
            "VOL": self._generate_grid_vol_params,
            "FIXING": self._generate_fixing,
        }
        object.__setattr__(self, "_handlers_by_kind", handlers)

    def get_market(self, request: MarketRequest) -> Market:
        scenario_idx = 0 if request.scenario is None else int(request.scenario)
        if scenario_idx < 0:
            raise ValueError("MarketRequest.scenario must be >= 0 when provided.")

        dataset = self.get_timeseries(
            TimeseriesRequest(
                start=request.asof,
                end=request.asof,
                freq="D",
                universe=request.universe,
                scenarios=max(1, scenario_idx + 1),
            )
        )
        return dataset.snapshot(time_idx=0, scenario_idx=scenario_idx)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        dates = _generate_dates(start=request.start, end=request.end, freq=request.freq)
        n_time = len(dates)
        n_scenarios = int(request.scenarios)
        if n_scenarios < 1:
            raise ValueError("TimeseriesRequest.scenarios must be >= 1.")

        base_rng = np.random.default_rng(_stable_seed(self.seed, request.start, request.end, request.freq))

        quote_panels: Dict[MarketId, Panel] = {}
        curve_param_panels: Dict[MarketId, Panel] = {}
        vol_param_panels: Dict[MarketId, Panel] = {}
        curve_factories: Dict[MarketId, ZeroCurveFactory] = {}
        vol_factories: Dict[MarketId, GridVolFactory] = {}
        spot_cache: Dict[MarketId, np.ndarray] = {}

        state = _GenerationState(
            rng=base_rng,
            n_time=n_time,
            n_scenarios=n_scenarios,
            quote_panels=quote_panels,
            curve_param_panels=curve_param_panels,
            curve_factories=curve_factories,
            vol_param_panels=vol_param_panels,
            vol_factories=vol_factories,
            spot_cache=spot_cache,
        )

        for market_id in request.universe.ids:
            handler = self._handlers_by_kind.get(market_id.mkt_type.upper(), self._generate_default_scalar_quote)
            handler(market_id, state)

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

    # ---------------------------------------------------------------------
    # RNG helper: deterministic per MarketId (order-independent)
    # ---------------------------------------------------------------------

    def _rng_for_id(self, base_rng: np.random.Generator, mid: MarketId) -> np.random.Generator:
        # Derive a stable sub-seed from base seed + MarketId key.
        # This makes generation independent of request.universe ordering.
        sub_seed = _stable_seed(int(self.seed), mid.key())
        return np.random.default_rng(sub_seed)

    # ---------------------------------------------------------------------
    # Handlers
    # ---------------------------------------------------------------------

    def _generate_spot_gbm(self, market_id: MarketId, state: _GenerationState) -> None:
        rng = self._rng_for_id(state.rng, market_id)
        spec = self.config.spot_spec(market_id)

        arr = _generate_gbm_spot_panel(
            rng=rng,
            n_time=state.n_time,
            n_scenarios=state.n_scenarios,
            initial_level=float(spec.initial_level),
            drift=float(spec.drift),
            vol=float(spec.vol),
            dt=float(spec.dt),
            initial_dispersion=float(spec.initial_dispersion),
        )

        state.quote_panels[market_id] = Panel(data=arr, axis_names=("time", "scenario"))
        state.spot_cache[market_id] = arr

    def _generate_zero_curve_params(self, market_id: MarketId, state: _GenerationState) -> None:
        rng = self._rng_for_id(state.rng, market_id)
        spec = self.config.curve_spec(market_id)

        tenors = np.asarray(spec.tenors, dtype=float).reshape(-1)
        params = _generate_zero_curve_param_panel(
            rng=rng,
            n_time=state.n_time,
            n_scenarios=state.n_scenarios,
            tenors=tenors,
            base_rate=float(spec.base_rate),
            slope=float(spec.slope),
            curvature=float(spec.curvature),
            noise_scale=float(spec.noise_scale),
        )

        state.curve_param_panels[market_id] = Panel(data=params, axis_names=("time", "scenario", "tenor", "cols"))
        state.curve_factories[market_id] = ZeroCurveFactory(extrapolation=str(spec.extrapolation))

    def _generate_grid_vol_params(self, market_id: MarketId, state: _GenerationState) -> None:
        rng = self._rng_for_id(state.rng, market_id)
        spec = self.config.vol_spec(market_id)

        expiries = np.asarray(spec.expiries, dtype=float).reshape(-1)
        strikes = np.asarray(spec.strikes, dtype=float).reshape(-1)

        vol_params = _generate_grid_vol_param_panel(
            rng=rng,
            n_time=state.n_time,
            n_scenarios=state.n_scenarios,
            expiries=expiries,
            strikes=strikes,
            atm_vol=float(spec.atm_vol),
            skew=float(spec.skew),
            smile=float(spec.smile),
            term=float(spec.term),
            noise_scale=float(spec.noise_scale),
        )

        state.vol_param_panels[market_id] = Panel(data=vol_params, axis_names=("time", "scenario", "params"))
        state.vol_factories[market_id] = GridVolFactory(expiries=expiries, strikes=strikes, extrapolation=str(spec.extrapolation))

    def _generate_fixing(self, market_id: MarketId, state: _GenerationState) -> None:
        spot_id = MarketId(
            asset_class=market_id.asset_class,
            mkt_type="SPOT",
            name=market_id.name,
            qualifiers=market_id.qualifiers,
        )
        if spot_id in state.spot_cache:
            state.quote_panels[market_id] = Panel(data=state.spot_cache[spot_id], axis_names=("time", "scenario"))
            return
        self._generate_default_scalar_quote(market_id, state)

    def _generate_default_scalar_quote(self, market_id: MarketId, state: _GenerationState) -> None:
        arr = np.full((state.n_time, state.n_scenarios), 1.0, dtype=float)
        state.quote_panels[market_id] = Panel(data=arr, axis_names=("time", "scenario"))


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _stable_seed(base_seed: int, *parts: str) -> int:
    h = 2166136261
    for p in parts:
        for b in p.encode("utf-8"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
    return int((base_seed ^ h) & 0xFFFFFFFF)


def _generate_dates(start: str, end: str, freq: str) -> list[str]:
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

    step = timedelta(days=30)
    while current <= end_d:
        dates.append(current.isoformat())
        current += step
    return dates


def _generate_gbm_spot_panel(
    rng: np.random.Generator,
    n_time: int,
    n_scenarios: int,
    initial_level: float,
    drift: float,
    vol: float,
    dt: float,
    initial_dispersion: float,
) -> np.ndarray:
    if initial_level <= 0.0:
        raise ValueError("initial_level must be > 0.")
    if vol < 0.0:
        raise ValueError("vol must be >= 0.")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")

    spot = np.empty((n_time, n_scenarios), dtype=float)

    # t=0: deterministic by default, optional cross-scenario dispersion if requested
    if initial_dispersion > 0.0:
        z0 = rng.normal(size=(n_scenarios,))
        spot0 = float(initial_level) * np.exp(float(initial_dispersion) * z0)
        spot[0, :] = spot0
    else:
        spot[0, :] = float(initial_level)

    if n_time == 1:
        return spot

    z = rng.normal(size=(n_time - 1, n_scenarios))
    mu = float(drift)
    sigma = float(vol)
    log_returns = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z

    for t in range(1, n_time):
        spot[t, :] = spot[t - 1, :] * np.exp(log_returns[t - 1, :])

    return spot


def _generate_zero_curve_param_panel(
    rng: np.random.Generator,
    n_time: int,
    n_scenarios: int,
    tenors: np.ndarray,
    base_rate: float,
    slope: float,
    curvature: float,
    noise_scale: float,
) -> np.ndarray:
    """
    Return params shaped [T, S, K, 2] with columns [tenor, zero_rate].
    """
    tenors = np.asarray(tenors, dtype=float).reshape(-1)
    k = tenors.size
    if k == 0:
        raise ValueError("tenors must be non-empty.")

    out = np.empty((n_time, n_scenarios, k, 2), dtype=float)
    out[..., 0] = tenors[None, None, :, None].squeeze(-1)

    # Smooth base curve: r(tau) = base + slope*tau + curvature*exp(-tau)
    base_curve = base_rate + slope * tenors + curvature * np.exp(-tenors)

    noise = 0.0
    if noise_scale > 0.0:
        noise = rng.normal(loc=0.0, scale=float(noise_scale), size=(n_time, n_scenarios, k))

    rates = base_curve[None, None, :] + noise
    out[..., 1] = rates
    return out


def _generate_grid_vol_param_panel(
    rng: np.random.Generator,
    n_time: int,
    n_scenarios: int,
    expiries: np.ndarray,
    strikes: np.ndarray,
    atm_vol: float,
    skew: float,
    smile: float,
    term: float,
    noise_scale: float,
) -> np.ndarray:
    """
    Generate flattened vol grid params [T,S,n_exp*n_strikes].

    We use a simple “smile in log-strike” proxy around the middle strike
    plus a mild term structure in sqrt(T).
    """
    expiries = np.asarray(expiries, dtype=float).reshape(-1)
    strikes = np.asarray(strikes, dtype=float).reshape(-1)
    n_exp = expiries.size
    n_k = strikes.size
    if n_exp == 0 or n_k == 0:
        raise ValueError("expiries/strikes must be non-empty.")

    k_mid = float(strikes[n_k // 2])
    logm = np.log(strikes / k_mid)  # proxy

    base_smile = atm_vol * (1.0 + skew * logm + smile * (logm ** 2))

    # term structure multiplier: 1 + term*(sqrt(T) - sqrt(T_mid))
    t_mid = float(expiries[n_exp // 2])
    term_mult = 1.0 + term * (np.sqrt(expiries) - np.sqrt(max(t_mid, 1e-12)))

    grid = term_mult[:, None] * base_smile[None, :]
    grid = np.maximum(grid, 1e-4)

    if noise_scale > 0.0:
        grid = grid[None, None, :, :] + rng.normal(loc=0.0, scale=float(noise_scale), size=(n_time, n_scenarios, n_exp, n_k))
    else:
        grid = grid[None, None, :, :].repeat(n_time, axis=0).repeat(n_scenarios, axis=1)

    grid = np.maximum(grid, 1e-4)
    flat = grid.reshape(n_time, n_scenarios, n_exp * n_k)
    return flat