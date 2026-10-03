"""
Market-data client seam.

``MarketDataClient`` is the contract every asset ``_fetch_*`` hook calls. To run
against your firm's systems, implement this protocol (one class, ten methods) and
inject it via the orchestrator — no other code changes.

The return shapes are the dicts the asset ``_build_*`` methods consume; they are
documented per method below and mirror ``PIPELINE.md`` §13.

``MockMarketDataClient`` produces synthetic-but-consistent FX + IR data so the
whole pipeline runs offline. It is careful to satisfy the asset ``validate()``
axis-alignment contracts (shock vol axes == surface axes, shock curve tenors ==
curve tenors), and exposes the same scenario count for every factor.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from src.rade_sr.mock.market import (
    CCY_RATES,
    FX_UNIVERSE,
    generate_scenarios,
)

# Shared grids (labels) — used consistently for build + shock alignment.
FX_VOL_TENORS: List[str] = ["1W", "1M", "3M", "6M", "1Y", "2Y"]
FX_SMILE_DELTAS: List[str] = ["10DP", "25DP", "ATM", "25DC", "10DC"]
FX_FWD_TENORS: List[str] = ["1W", "1M", "3M", "6M", "1Y"]
IR_TENORS: List[str] = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
IR_VOL_EXPIRIES: List[str] = ["3M", "6M", "1Y", "2Y", "5Y"]
IR_VOL_SWAP_TENORS: List[str] = ["1Y", "2Y", "5Y", "10Y"]
IR_VOL_STRIKES: List[float] = [-100.0, -50.0, 0.0, 50.0, 100.0]


@runtime_checkable
class MarketDataClient(Protocol):
    """Transport contract for market data + scenario shocks.

    Implement this against your internal API to replace the mock. Every method
    returns plain dicts/lists (no rade_sr types), so it is a clean adapter layer.
    """

    # ── FX ────────────────────────────────────────────────────────────
    def get_fx_spot(self, pair: str, cob: Any = None) -> Dict[str, Any]: ...
    def get_fx_forward_points(self, pair: str, cob: Any = None) -> Dict[str, Any]: ...
    def get_fx_atm_vol(self, pair: str, cob: Any = None) -> Dict[str, float]: ...
    def get_fx_smile_vol(self, pair: str, cob: Any = None) -> Dict[str, Dict[str, float]]: ...
    def get_fx_shocks(self, pair: str, cob: Any = None) -> Dict[str, Any]: ...

    # ── IR ────────────────────────────────────────────────────────────
    def get_ir_curve(self, name: str, cob: Any = None) -> Dict[str, Any]: ...
    def get_ir_history(self, name: str, cob: Any = None) -> Dict[str, Any]: ...
    def get_ir_atm_vol(self, name: str, cob: Any = None) -> Dict[str, Any]: ...
    def get_ir_smile_vol(self, name: str, cob: Any = None) -> Dict[str, Any]: ...
    def get_ir_shocks(self, name: str, cob: Any = None) -> Dict[str, Any]: ...


_TENOR_MAP = {"D": 1 / 365, "W": 7 / 365, "M": 1 / 12, "Y": 1.0}


def _yf(label: str) -> float:
    for suffix, factor in _TENOR_MAP.items():
        if label.upper().endswith(suffix):
            return float(label[:-len(suffix)]) * factor
    return float(label)


class MockMarketDataClient:
    """Synthetic :class:`MarketDataClient` for offline pipeline runs.

    Parameters
    ----------
    n_scenarios : int
        Shock scenario count (kept identical across all factors).
    cob_date : str
        Close-of-business date for the spot/curve snapshot.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, n_scenarios: int = 250, cob_date: str = "2026-05-29", seed: int = 99) -> None:
        import pandas as pd

        self.n_scenarios = n_scenarios
        self._cob = pd.Timestamp(cob_date)
        self._rng = np.random.default_rng(seed)
        self._scenarios = generate_scenarios(self._cob, n_scenarios=n_scenarios, seed=seed)
        # business-day history window for spot / curve time series
        self._hist_dates = (
            pd.bdate_range(end=self._cob, periods=260).strftime("%Y-%m-%d").tolist()
        )

    # ── FX ────────────────────────────────────────────────────────────

    def get_fx_spot(self, pair: str, cob: Any = None) -> Dict[str, Any]:
        spot = FX_UNIVERSE[pair]["spot"]
        vol = FX_UNIVERSE[pair]["vol"]
        rets = self._rng.standard_normal(len(self._hist_dates)) * vol / np.sqrt(252.0)
        series = spot * np.exp(np.cumsum(rets) - np.cumsum(rets)[-1])
        return {"spot": float(spot), "dates": self._hist_dates, "values": series.tolist()}

    def get_fx_forward_points(self, pair: str, cob: Any = None) -> Dict[str, Any]:
        info = FX_UNIVERSE[pair]
        spot = info["spot"]
        carry = CCY_RATES[info["quote"]] - CCY_RATES[info["base"]]
        points = [spot * (np.exp(carry * _yf(t)) - 1.0) for t in FX_FWD_TENORS]
        return {"tenors": FX_FWD_TENORS, "points": points}

    def get_fx_atm_vol(self, pair: str, cob: Any = None) -> Dict[str, float]:
        base = FX_UNIVERSE[pair]["vol"]
        # mild upward term structure
        return {t: float(base * (1.0 + 0.05 * i)) for i, t in enumerate(FX_VOL_TENORS)}

    def get_fx_smile_vol(self, pair: str, cob: Any = None) -> Dict[str, Dict[str, float]]:
        atm = self.get_fx_atm_vol(pair)
        # symmetric smile: wings above ATM
        wing = {"10DP": 1.12, "25DP": 1.04, "25DC": 1.05, "10DC": 1.14}
        return {t: {d: float(atm[t] * mult) for d, mult in wing.items()} for t in FX_VOL_TENORS}

    def get_fx_shocks(self, pair: str, cob: Any = None) -> Dict[str, Any]:
        n = self.n_scenarios
        info = FX_UNIVERSE[pair]
        spot = info["spot"]

        spot_shocks = (spot * self._scenarios.spot_mult[pair]).tolist()

        atm = self.get_fx_atm_vol(pair)
        base_vols = np.array([atm[t] for t in FX_VOL_TENORS])           # (n_exp,)
        smile = np.array([1.12, 1.04, 1.00, 1.05, 1.14])                # (n_strikes,)
        vol_grid = base_vols[:, None] * smile[None, :]                  # (n_exp, n_strikes)
        vol_noise = 1.0 + self._rng.standard_normal((n, 1, 1)) * 0.05
        vol_shocks = (vol_grid[None, :, :] * vol_noise)                 # (n, n_exp, n_strikes)

        dom_tenors = IR_TENORS
        for_tenors = IR_TENORS
        rate_scale = 25e-4  # ~25bp
        dom_shocks = self._rng.standard_normal((n, len(dom_tenors))) * rate_scale
        for_shocks = self._rng.standard_normal((n, len(for_tenors))) * rate_scale

        return {
            "spot_shocks": spot_shocks,
            "vol_shocks": vol_shocks.tolist(),
            "vol_expiries": FX_VOL_TENORS,
            "vol_strikes": FX_SMILE_DELTAS,
            "domestic_rate_shocks": dom_shocks.tolist(),
            "domestic_tenors": dom_tenors,
            "foreign_rate_shocks": for_shocks.tolist(),
            "foreign_tenors": for_tenors,
            "forward_point_shocks": None,
            "forward_tenors": None,
        }

    # ── IR ────────────────────────────────────────────────────────────

    @staticmethod
    def _ccy(name: str) -> str:
        return name.split("_")[0].split(".")[0].upper()

    def get_ir_curve(self, name: str, cob: Any = None) -> Dict[str, Any]:
        base = CCY_RATES.get(self._ccy(name), 0.03)
        rates = [base + 0.002 * i for i in range(len(IR_TENORS))]  # gentle upward slope
        return {"tenors": IR_TENORS, "rates": rates}

    def get_ir_history(self, name: str, cob: Any = None) -> Dict[str, Any]:
        base = CCY_RATES.get(self._ccy(name), 0.03)
        n_d, n_t = len(self._hist_dates), len(IR_TENORS)
        slope = np.array([0.002 * i for i in range(n_t)])
        noise = self._rng.standard_normal((n_d, n_t)) * 5e-4
        values = base + slope[None, :] + noise
        return {"dates": self._hist_dates, "tenors": IR_TENORS, "values": values.tolist()}

    def get_ir_atm_vol(self, name: str, cob: Any = None) -> Dict[str, Any]:
        n_e, n_t = len(IR_VOL_EXPIRIES), len(IR_VOL_SWAP_TENORS)
        values = np.full((n_e, n_t), 0.006)  # 60bp normal vol
        return {"expiries": IR_VOL_EXPIRIES, "swap_tenors": IR_VOL_SWAP_TENORS, "values": values.tolist()}

    def get_ir_smile_vol(self, name: str, cob: Any = None) -> Dict[str, Any]:
        n_e, n_t, n_k = len(IR_VOL_EXPIRIES), len(IR_VOL_SWAP_TENORS), len(IR_VOL_STRIKES)
        atm = 0.006
        skew = 1.0 + 0.0008 * np.abs(np.array(IR_VOL_STRIKES)) / 50.0
        values = atm * np.ones((n_e, n_t, 1)) * skew[None, None, :]
        return {
            "expiries": IR_VOL_EXPIRIES,
            "swap_tenors": IR_VOL_SWAP_TENORS,
            "strikes": IR_VOL_STRIKES,
            "values": values.tolist(),
        }

    def get_ir_shocks(self, name: str, cob: Any = None) -> Dict[str, Any]:
        n = self.n_scenarios
        curve_shocks = self._rng.standard_normal((n, len(IR_TENORS))) * 25e-4
        return {
            "curve_shocks": curve_shocks.tolist(),
            "curve_tenors": IR_TENORS,
            "vol_shocks": None,
            "vol_expiries": None,
            "vol_swap_tenors": None,
            "vol_strikes": None,
        }
