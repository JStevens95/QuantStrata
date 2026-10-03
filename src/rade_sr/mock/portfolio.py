"""
Synthetic FX desk portfolio generator.

Emits two raw-shaped artifacts that mirror the external risk-system exports:

1. **Attribute frame** — raw, un-aggregated: one row per
   ``(trade, risk_type, curve)``. ``AssetClass`` is ``"ALL"`` on ``Notional``
   rows and the real class ("FX") otherwise. Columns exactly match the export:

       AssetClass, BuySellInd, CallPut, Date, DeskName, MaturityDate,
       PTSCurveCode, PTSDealNumber, Product, ProductGroup, RiskType,
       SourceSystemGroup, StandardCurveCode, StrikePrice, SubDeskName,
       Sum_RiskValuesUSD_Net, TRBookKey

2. **PnL frame** — index = trade id (``PTSDealNumber``), columns = scenario dates
   (``YYYYMMDD``) with PnL values, plus an ``AssetClass`` column.

Risk types are a **configurable contract** (``RISK_TYPE_SETS``) so the set can
change per asset class as the universe grows. PnL and risk values are
*semi-accurate*: produced by full Garman-Kohlhagen revaluation under each
scenario's spot/vol shock, with light idiosyncratic noise so the target PnL is
not a perfect linear combination of any elementary basis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .market import (
    CCY_RATES,
    FX_UNIVERSE,
    ScenarioSet,
    fx_forward_pv,
    generate_scenarios,
    gk_digital_pv,
    gk_vanilla_pv,
    ko_up_out_call_pv,
    quanto_vanilla_pv,
    usd_value,
)

# ─────────────────────────────────────────────────────────────────────────
# Configurable contract: which risk types appear per asset class
# ─────────────────────────────────────────────────────────────────────────

# The notional risk type is special (drives NotionalSign and carries AssetClass="ALL").
NOTIONAL_RISK_TYPE = "Notional"

RISK_TYPE_SETS: Dict[str, List[str]] = {
    "FX": ["FXPV", "FXPOS", NOTIONAL_RISK_TYPE],
    # Extend as new asset classes are added, e.g.:
    # "IR": ["IRPV", "IRDelta", "IRVega", NOTIONAL_RISK_TYPE],
}

RAW_ATTRIBUTE_COLUMNS = [
    "AssetClass", "BuySellInd", "CallPut", "Date", "DeskName", "MaturityDate",
    "PTSCurveCode", "PTSDealNumber", "Product", "ProductGroup", "RiskType",
    "SourceSystemGroup", "StandardCurveCode", "StrikePrice", "SubDeskName",
    "Sum_RiskValuesUSD_Net", "TRBookKey",
]

# desk -> (sub-desks, product groups available, candidate pairs)
_G10 = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
_CROSSES = ["EURGBP", "EURJPY", "GBPJPY"]
_EM = ["USDMXN", "USDZAR", "USDTRY"]

_DESKS: Dict[str, dict] = {
    "FX VANILLA OPTIONS": {
        "subdesks": ["G10", "CROSSES"],
        "products": ["VanillaOption"],
        "pairs": _G10 + _CROSSES,
        "source": "MUREX",
    },
    "FX FORWARDS": {
        "subdesks": ["G10", "EM"],
        "products": ["Forward"],
        "pairs": _G10 + _EM,
        "source": "CALYPSO",
    },
    "FX EXOTICS": {
        "subdesks": ["G10", "CROSSES"],
        "products": ["Digital", "BarrierOption", "QuantoOption"],
        "pairs": _G10 + _CROSSES,
        "source": "MUREX",
    },
    "FX EMERGING MARKETS": {
        "subdesks": ["EM"],
        "products": ["VanillaOption", "Forward", "Digital"],
        "pairs": _EM,
        "source": "SUMMIT",
    },
}

_PRODUCT_GROUP = {
    "VanillaOption": "VANILLA",
    "Forward": "LINEAR",
    "Digital": "EXOTIC",
    "BarrierOption": "EXOTIC",
    "QuantoOption": "EXOTIC",
}

_OPTION_PRODUCTS = {"VanillaOption", "Digital", "BarrierOption", "QuantoOption"}


@dataclass
class MockTrade:
    """Internal representation of one synthetic trade."""
    trade_id: str
    pair: str
    product: str
    desk: str
    subdesk: str
    book: str
    source_system: str
    buy_sell: str            # "Buy" / "Sell"
    call_put: str            # "C" / "P" / "" (linear)
    strike: float            # absolute strike / forward rate
    barrier: float           # for barriers (else 0.0)
    notional_base: float     # notional in base currency units
    cob_date: pd.Timestamp
    maturity_date: pd.Timestamp

    @property
    def sign(self) -> float:
        return 1.0 if self.buy_sell == "Buy" else -1.0

    @property
    def expiry_years(self) -> float:
        days = (self.maturity_date - self.cob_date).days
        return max(days / 365.0, 1.0 / 365.0)


# ─────────────────────────────────────────────────────────────────────────
# Trade universe sampling
# ─────────────────────────────────────────────────────────────────────────

def _sample_trades(n_trades: int, cob_date: pd.Timestamp, seed: int) -> List[MockTrade]:
    rng = np.random.default_rng(seed)
    desks = list(_DESKS)
    trades: List[MockTrade] = []

    for i in range(n_trades):
        desk = desks[rng.integers(len(desks))]
        spec = _DESKS[desk]
        subdesk = spec["subdesks"][rng.integers(len(spec["subdesks"]))]
        product = spec["products"][rng.integers(len(spec["products"]))]
        pair = spec["pairs"][rng.integers(len(spec["pairs"]))]
        info = FX_UNIVERSE[pair]
        spot = info["spot"]

        # notional: lognormal, USD-equivalent 1mm .. ~1bn, expressed in base ccy
        notional_usd = float(np.exp(rng.normal(16.5, 1.1)))   # ~1.5e7 median
        notional_base = notional_usd / usd_value(info["base"])

        tenor_days = int(rng.choice([7, 30, 91, 182, 365, 730, 1095, 1825]))
        maturity = cob_date + pd.Timedelta(days=tenor_days)

        if product in _OPTION_PRODUCTS:
            call_put = "C" if rng.random() < 0.5 else "P"
            strike = spot * float(np.clip(rng.normal(1.0, 0.06), 0.80, 1.25))
            barrier = (
                strike * (1.12 if call_put == "C" else 0.88)
                if product == "BarrierOption" else 0.0
            )
        else:
            call_put = ""
            # at-market forward rate with small offset
            strike = spot * float(np.clip(rng.normal(1.0, 0.01), 0.95, 1.05))
            barrier = 0.0

        trades.append(MockTrade(
            trade_id=str(70_000_000 + i),
            pair=pair,
            product=product,
            desk=desk,
            subdesk=subdesk,
            book=f"BK{1000 + int(rng.integers(0, 250))}",
            source_system=spec["source"],
            buy_sell="Buy" if rng.random() < 0.5 else "Sell",
            call_put=call_put,
            strike=round(strike, 6),
            barrier=round(barrier, 6),
            notional_base=notional_base,
            cob_date=cob_date,
            maturity_date=maturity,
        ))
    return trades


# ─────────────────────────────────────────────────────────────────────────
# Pricing helpers (per trade, unsigned, in quote ccy)
# ─────────────────────────────────────────────────────────────────────────

def _pv(trade: MockTrade, S, vol, notional: float):
    """Unsigned PV for ``notional`` units, in quote ccy. S/vol may be arrays."""
    info = FX_UNIVERSE[trade.pair]
    T = trade.expiry_years
    r_d = CCY_RATES[info["quote"]]
    r_f = CCY_RATES[info["base"]]
    K = trade.strike
    is_call = trade.call_put == "C"

    if trade.product == "VanillaOption":
        return gk_vanilla_pv(S, K, T, r_d, r_f, vol, is_call, notional)
    if trade.product == "Forward":
        return fx_forward_pv(S, K, T, r_d, r_f, notional)
    if trade.product == "Digital":
        return gk_digital_pv(S, K, T, r_d, r_f, vol, is_call, notional)
    if trade.product == "BarrierOption":
        return ko_up_out_call_pv(S, K, trade.barrier, T, r_d, r_f, vol, notional)
    if trade.product == "QuantoOption":
        return quanto_vanilla_pv(S, K, T, r_d, r_f, vol, is_call, notional)
    raise ValueError(f"Unknown product {trade.product!r}")


def _risk_values_usd(trade: MockTrade) -> Dict[str, float]:
    """Semi-accurate base-case risk values (USD), keyed by risk type."""
    info = FX_UNIVERSE[trade.pair]
    spot, base_vol = info["spot"], info["vol"]
    quote_usd = usd_value(info["quote"])
    base_usd = usd_value(info["base"])
    notl_usd_base = trade.notional_base * base_usd

    pv_quote = float(_pv(trade, spot, base_vol, trade.notional_base))

    # unit-notional spot delta via central finite difference
    h = 1e-4 * spot
    up = float(_pv(trade, spot + h, base_vol, 1.0))
    dn = float(_pv(trade, spot - h, base_vol, 1.0))
    delta_unit = (up - dn) / (2.0 * h)         # dPV/dS for one base unit

    return {
        "FXPV": trade.sign * pv_quote * quote_usd,
        # delta-equivalent position in USD (delta_unit ~ option delta, ~1 for linear)
        "FXPOS": trade.sign * notl_usd_base * delta_unit,
        NOTIONAL_RISK_TYPE: notl_usd_base,     # magnitude; sign carried separately
    }


def _scenario_pnl_usd(trade: MockTrade, scenarios: ScenarioSet, rng: np.random.Generator) -> np.ndarray:
    """Full-reval PnL (USD) across all scenarios for one trade."""
    info = FX_UNIVERSE[trade.pair]
    spot, base_vol = info["spot"], info["vol"]
    quote_usd = usd_value(info["quote"])

    S_arr = spot * scenarios.spot_mult[trade.pair]
    vol_arr = scenarios.vol_level[trade.pair]

    pv_base = float(_pv(trade, spot, base_vol, trade.notional_base))
    pv_scen = np.asarray(_pv(trade, S_arr, vol_arr, trade.notional_base), dtype=np.float64)

    pnl_quote = pv_scen - pv_base
    noise = 1.0 + rng.standard_normal(scenarios.n_scenarios) * 0.015
    return trade.sign * pnl_quote * quote_usd * noise


# ─────────────────────────────────────────────────────────────────────────
# Raw frame builders
# ─────────────────────────────────────────────────────────────────────────

def _curve_rows(trade: MockTrade, risk_type: str, value_usd: float,
                rng: np.random.Generator) -> List[Tuple[str, str, float, str]]:
    """Return (asset_class, standard_curve, value, pts_curve) rows for a risk type.

    Risk values are split across the relevant curves so the raw file is exploded
    by ``(trade, risk_type, curve)`` exactly like the export. PTSCurveCode is
    mostly "N/A" (defensive, unused in logic).
    """
    info = FX_UNIVERSE[trade.pair]
    base, quote = info["base"], info["quote"]
    pts = "N/A" if rng.random() < 0.85 else f"PTS{int(rng.integers(100000, 999999))}"

    if risk_type == NOTIONAL_RISK_TYPE:
        return [("ALL", f"{trade.pair}.FX", value_usd, pts)]

    if risk_type == "FXPOS":
        return [("FX", f"{trade.pair}.FX", value_usd, pts)]

    # FXPV: split across the FX curve and the two IR legs (rates sensitivity)
    return [
        ("FX", f"{trade.pair}.FX", value_usd * 0.96, pts),
        ("FX", f"{base}.IR.OIS", value_usd * 0.025, "N/A"),
        ("FX", f"{quote}.IR.OIS", value_usd * 0.015, "N/A"),
    ]


def build_attribute_frame(trades: List[MockTrade], seed: int = 11) -> pd.DataFrame:
    """Build the raw, exploded attribute DataFrame (one row per trade×risk×curve)."""
    rng = np.random.default_rng(seed)
    rows: List[dict] = []

    for trade in trades:
        risk_values = _risk_values_usd(trade)
        for risk_type in RISK_TYPE_SETS["FX"]:
            for asset_class, std_curve, value, pts_curve in _curve_rows(
                trade, risk_type, risk_values[risk_type], rng,
            ):
                rows.append({
                    "AssetClass": asset_class,
                    "BuySellInd": trade.buy_sell,
                    "CallPut": trade.call_put,
                    "Date": trade.cob_date.strftime("%Y%m%d"),
                    "DeskName": trade.desk,
                    "MaturityDate": trade.maturity_date.strftime("%Y%m%d"),
                    "PTSCurveCode": pts_curve,
                    "PTSDealNumber": trade.trade_id,
                    "Product": f"{trade.pair}_{trade.product}",
                    "ProductGroup": _PRODUCT_GROUP[trade.product],
                    "RiskType": risk_type,
                    "SourceSystemGroup": trade.source_system,
                    "StandardCurveCode": std_curve,
                    "StrikePrice": trade.strike if trade.product in _OPTION_PRODUCTS else np.nan,
                    "SubDeskName": trade.subdesk,
                    "Sum_RiskValuesUSD_Net": round(value, 2),
                    "TRBookKey": trade.book,
                })

    df = pd.DataFrame(rows, columns=RAW_ATTRIBUTE_COLUMNS)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle rows


def build_pnl_frame(trades: List[MockTrade], scenarios: ScenarioSet, seed: int = 13) -> pd.DataFrame:
    """Build the raw PnL DataFrame: trade-id index, scenario-date columns, + AssetClass."""
    rng = np.random.default_rng(seed)
    pnl = np.empty((len(trades), scenarios.n_scenarios), dtype=np.float64)
    for i, trade in enumerate(trades):
        pnl[i] = _scenario_pnl_usd(trade, scenarios, rng)

    df = pd.DataFrame(pnl, index=[t.trade_id for t in trades], columns=scenarios.dates)
    df.index.name = "PTSDealNumber"
    df.insert(0, "AssetClass", "FX")
    return df


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────

def generate_mock_portfolio(
    n_trades: int = 2500,
    n_scenarios: int = 250,
    cob_date: str = "2026-05-29",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic FX desk portfolio.

    Returns
    -------
    (attributes, pnl) : tuple[pd.DataFrame, pd.DataFrame]
        ``attributes`` is the raw exploded attribute frame; ``pnl`` is the raw
        scenario-PnL frame. Both mirror the external risk-system export schema.
    """
    cob = pd.Timestamp(cob_date)
    trades = _sample_trades(n_trades, cob, seed=seed)
    scenarios = generate_scenarios(cob, n_scenarios=n_scenarios, seed=seed + 1)
    attributes = build_attribute_frame(trades, seed=seed + 2)
    pnl = build_pnl_frame(trades, scenarios, seed=seed + 3)
    return attributes, pnl
