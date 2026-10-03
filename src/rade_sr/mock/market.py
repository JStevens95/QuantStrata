"""
Mock FX market + scenario engine for synthetic portfolio generation.

Provides:
  - A small, realistic FX universe (spot, ATM vol, currency short rates).
  - A reproducible ScenarioSet: per-pair instantaneous spot/vol shocks across a
    set of historical-style scenario dates (YYYYMMDD).
  - Vectorised Garman-Kohlhagen pricing used to produce *semi-accurate* PnL and
    risk values for the mock portfolio.

Everything here is deliberately self-contained (numpy + scipy only) so mock data
generation has no dependency on the pricing kernels under test.

Pricing convention (Garman-Kohlhagen) for a pair XXXYYY quoted as YYY per XXX:
    domestic (numeraire) = YYY (quote)  -> r_d
    foreign  (asset)     = XXX (base)   -> r_f
    S = units of YYY per 1 XXX
All PnL/risk values are reported in USD via a currency->USD conversion table
derived from the universe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.special import ndtr

# ─────────────────────────────────────────────────────────────────────────
# Static market universe
# ─────────────────────────────────────────────────────────────────────────

# pair -> (spot = quote per base, ATM lognormal vol, base ccy, quote ccy)
FX_UNIVERSE: Dict[str, dict] = {
    "EURUSD": {"spot": 1.0850, "vol": 0.078, "base": "EUR", "quote": "USD"},
    "GBPUSD": {"spot": 1.2700, "vol": 0.085, "base": "GBP", "quote": "USD"},
    "USDJPY": {"spot": 156.00, "vol": 0.095, "base": "USD", "quote": "JPY"},
    "AUDUSD": {"spot": 0.6600, "vol": 0.105, "base": "AUD", "quote": "USD"},
    "USDCHF": {"spot": 0.9000, "vol": 0.075, "base": "USD", "quote": "CHF"},
    "USDCAD": {"spot": 1.3650, "vol": 0.065, "base": "USD", "quote": "CAD"},
    "NZDUSD": {"spot": 0.6100, "vol": 0.110, "base": "NZD", "quote": "USD"},
    "EURGBP": {"spot": 0.8550, "vol": 0.060, "base": "EUR", "quote": "GBP"},
    "EURJPY": {"spot": 169.00, "vol": 0.090, "base": "EUR", "quote": "JPY"},
    "GBPJPY": {"spot": 198.00, "vol": 0.100, "base": "GBP", "quote": "JPY"},
    "USDMXN": {"spot": 18.500, "vol": 0.130, "base": "USD", "quote": "MXN"},
    "USDZAR": {"spot": 18.700, "vol": 0.155, "base": "USD", "quote": "ZAR"},
    "USDTRY": {"spot": 32.500, "vol": 0.210, "base": "USD", "quote": "TRY"},
}

# continuously-compounded flat short rates per currency
CCY_RATES: Dict[str, float] = {
    "USD": 0.0450, "EUR": 0.0250, "GBP": 0.0420, "JPY": 0.0010, "AUD": 0.0380,
    "CHF": 0.0120, "CAD": 0.0400, "NZD": 0.0400, "MXN": 0.1050, "ZAR": 0.0820,
    "TRY": 0.4200,
}

_DAYS_PER_YEAR = 365.0


def _build_usd_per(universe: Dict[str, dict]) -> Dict[str, float]:
    """USD value of one unit of each currency, derived from USD-quoted pairs."""
    usd_per: Dict[str, float] = {"USD": 1.0}
    for info in universe.values():
        base, quote, spot = info["base"], info["quote"], info["spot"]
        if quote == "USD":          # 1 base = spot USD
            usd_per[base] = spot
        elif base == "USD":         # 1 quote = 1/spot USD
            usd_per[quote] = 1.0 / spot
    return usd_per


USD_PER_CCY: Dict[str, float] = _build_usd_per(FX_UNIVERSE)


def usd_value(currency: str) -> float:
    """USD value of one unit of ``currency`` (1.0 for USD)."""
    return USD_PER_CCY.get(currency, 1.0)


# ─────────────────────────────────────────────────────────────────────────
# Vectorised Garman-Kohlhagen pricing (S and vol may be scenario arrays)
# ─────────────────────────────────────────────────────────────────────────

def _disc(rate: float, T: float) -> float:
    return float(np.exp(-rate * T))


def gk_vanilla_pv(S, K, T, r_d, r_f, vol, is_call, notional):
    """European FX option PV (per ``notional`` units of base ccy), in quote ccy."""
    S = np.asarray(S, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    sqrt_T = np.sqrt(T)
    df_d, df_f = _disc(r_d, T), _disc(r_f, T)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * vol * vol) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    if is_call:
        pv = S * df_f * ndtr(d1) - K * df_d * ndtr(d2)
    else:
        pv = K * df_d * ndtr(-d2) - S * df_f * ndtr(-d1)
    return notional * pv


def gk_digital_pv(S, K, T, r_d, r_f, vol, is_call, payout):
    """Cash-or-nothing digital PV (pays ``payout`` quote ccy if in the money)."""
    S = np.asarray(S, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    df_d = _disc(r_d, T)
    d2 = (np.log(S / K) + (r_d - r_f - 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    return payout * df_d * (ndtr(d2) if is_call else ndtr(-d2))


def fx_forward_pv(S, K, T, r_d, r_f, notional):
    """FX forward PV (long base), in quote ccy."""
    S = np.asarray(S, dtype=np.float64)
    return notional * (S * _disc(r_f, T) - K * _disc(r_d, T))


def ko_up_out_call_pv(S, K, H, T, r_d, r_f, vol, notional):
    """Up-and-out call, statically replicated as C(K) - C(H) - (H-K)*Digital(H).

    A deliberate nod to the static-replication theme; floored at zero to keep the
    approximation well-behaved near the barrier.
    """
    c_k = gk_vanilla_pv(S, K, T, r_d, r_f, vol, True, 1.0)
    c_h = gk_vanilla_pv(S, H, T, r_d, r_f, vol, True, 1.0)
    dig_h = gk_digital_pv(S, H, T, r_d, r_f, vol, True, 1.0)
    pv = c_k - c_h - (H - K) * dig_h
    return notional * np.maximum(pv, 0.0)


def quanto_vanilla_pv(S, K, T, r_d, r_f, vol, is_call, notional, corr=0.3, q_vol=0.10):
    """Quanto vanilla, approximated as a GK vanilla with a quanto drift factor."""
    factor = np.exp(-corr * vol * q_vol * T)
    return gk_vanilla_pv(S, K, T, r_d, r_f, vol, is_call, notional) * factor


# ─────────────────────────────────────────────────────────────────────────
# Scenario set
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioSet:
    """Instantaneous market shocks across a set of scenario dates.

    Parameters
    ----------
    dates : list[str]
        Scenario labels as ``YYYYMMDD`` strings (the PnL column headers).
    spot_mult : dict[str, np.ndarray]
        ``pair -> (n_scenarios,)`` multiplicative spot shock (1.0 = unchanged).
    vol_level : dict[str, np.ndarray]
        ``pair -> (n_scenarios,)`` shocked ATM vol level.
    """
    dates: List[str]
    spot_mult: Dict[str, np.ndarray]
    vol_level: Dict[str, np.ndarray]

    @property
    def n_scenarios(self) -> int:
        return len(self.dates)


def generate_scenarios(
    cob_date: pd.Timestamp,
    n_scenarios: int = 250,
    seed: int = 7,
    universe: Dict[str, dict] = FX_UNIVERSE,
) -> ScenarioSet:
    """Build a reproducible historical-style scenario set.

    Each scenario is an instantaneous joint shock to spot and ATM vol for every
    pair. Spot shocks use daily-scaled lognormal moves with mild fat tails; vol
    shocks are small and negatively correlated with the spot move.
    """
    rng = np.random.default_rng(seed)
    dates = (
        pd.bdate_range(end=cob_date, periods=n_scenarios)
        .strftime("%Y%m%d")
        .tolist()
    )

    spot_mult: Dict[str, np.ndarray] = {}
    vol_level: Dict[str, np.ndarray] = {}
    for pair, info in universe.items():
        ann_vol = info["vol"]
        daily = ann_vol / np.sqrt(252.0)
        # Student-t-ish tails via mixing a wider normal occasionally.
        normal = rng.standard_normal(n_scenarios)
        jump = rng.standard_normal(n_scenarios) * (rng.random(n_scenarios) < 0.05)
        ret = daily * normal + 3.0 * daily * jump
        spot_mult[pair] = np.exp(ret)
        dvol = -0.4 * (ret / daily) * (0.08 * ann_vol) + rng.standard_normal(n_scenarios) * 0.05 * ann_vol
        vol_level[pair] = np.clip(ann_vol + dvol, 0.2 * ann_vol, 3.0 * ann_vol)

    return ScenarioSet(dates=dates, spot_mult=spot_mult, vol_level=vol_level)
