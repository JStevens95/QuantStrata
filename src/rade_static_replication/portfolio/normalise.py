"""
Portfolio normalisation (stage 2).

Collapses the exploded raw export (one row per trade × risk-type × curve, notional
row tagged ``AssetClass="ALL"``) into one validated row per trade:

1. pivot ``RiskType`` into columns;
2. resolve ``AssetClass`` ignoring the ``"ALL"`` notional rows;
3. carry through every column constant within a trade (discovered dynamically);
4. derive ``NotionalSign``, ``SignedNotional``, ``yrs_to_maturity``.

Raw column names live in one place (below) — point them at your export if it differs.
"""
from __future__ import annotations

import datetime as _dt
import logging

import numpy as np
import pandas as pd

from src.rade_static_replication.domain.contracts import Portfolio, RawPortfolio
from src.rade_static_replication.domain.errors import PortfolioError

logger = logging.getLogger(__name__)

TRADE_ID_COL = "PTSDealNumber"
RISK_TYPE_COL = "RiskType"
RISK_VALUE_COL = "Sum_RiskValuesUSD_Net"
ASSET_CLASS_COL = "AssetClass"
BUY_SELL_COL = "BuySellInd"
MATURITY_COL = "MaturityDate"
NOTIONAL_RISK_TYPE = "Notional"
ALL_SENTINEL = "ALL"

# Per-(trade, curve) noise that should not be carried as attributes.
_DROP_COLS = {"PTSCurveCode", "StandardCurveCode"}


def _years_to_maturity(maturity: str, cob_date: str) -> float:
    try:
        m = _dt.datetime.strptime(str(maturity), "%Y%m%d").date()
        c = _dt.datetime.strptime(str(cob_date), "%Y%m%d").date()
        return max((m - c).days / 365.0, 0.0)
    except (ValueError, TypeError):
        return np.nan


def _first_non_null(s: pd.Series):
    for v in s:
        if pd.notna(v):
            return v
    return np.nan


def _first_non_all(s: pd.Series):
    for v in s:
        if pd.notna(v) and str(v) != ALL_SENTINEL:
            return v
    return np.nan


def normalise(raw: RawPortfolio) -> Portfolio:
    """Collapse the exploded raw export into one row per trade."""
    attributes_raw = raw.attributes.copy()
    if TRADE_ID_COL not in attributes_raw.columns:
        raise PortfolioError(f"raw attributes missing trade-id column {TRADE_ID_COL!r}")
    attributes_raw[TRADE_ID_COL] = attributes_raw[TRADE_ID_COL].astype(str)

    # 1) Pivot the risk-type rows into one column per risk type (Notional, FXPV, ...).
    risk_values = attributes_raw.pivot_table(
        index=TRADE_ID_COL, columns=RISK_TYPE_COL, values=RISK_VALUE_COL, aggfunc="first",
    )
    risk_values.columns = [str(c) for c in risk_values.columns]

    # 2) Carry through every other column that is constant within a trade. AssetClass uses
    #    a special reducer that ignores the "ALL" notional-row sentinel.
    exclude = {RISK_TYPE_COL, RISK_VALUE_COL, TRADE_ID_COL} | _DROP_COLS
    carry_columns = [c for c in attributes_raw.columns if c not in exclude]
    by_trade = attributes_raw.groupby(TRADE_ID_COL, sort=False)

    constant_attributes = pd.DataFrame(index=risk_values.index)
    for column in carry_columns:
        reducer = _first_non_all if column == ASSET_CLASS_COL else _first_non_null
        constant_attributes[column] = by_trade[column].apply(reducer)

    attributes = constant_attributes.join(risk_values)

    # 3) Derive direction and signed notional from the buy/sell indicator.
    notional_sign = attributes[BUY_SELL_COL].map(
        lambda flag: 1.0 if str(flag).upper().startswith("B") else -1.0
    )
    attributes["NotionalSign"] = notional_sign
    if NOTIONAL_RISK_TYPE in attributes.columns:
        attributes["SignedNotional"] = notional_sign * attributes[NOTIONAL_RISK_TYPE].astype(float)
    if MATURITY_COL in attributes.columns:
        attributes["yrs_to_maturity"] = attributes[MATURITY_COL].map(
            lambda maturity: _years_to_maturity(maturity, raw.cob_date)
        )

    attributes.index = attributes.index.astype(str)
    attributes.index.name = "trade_id"

    # 4) Align the target PnL frame to a clean trade_id index and keep only scenario columns.
    target = raw.target_pnl.copy()
    if target.index.name != "trade_id" and "trade_id" in target.columns:
        target = target.set_index("trade_id")
    target.index = target.index.astype(str)
    scenario_cols = [c for c in target.columns if c != ASSET_CLASS_COL]
    target = target[scenario_cols]

    portfolio = Portfolio(
        attributes=attributes, target_pnl=target,
        scenario_ids=np.array(scenario_cols), cob_date=raw.cob_date,
    )
    logger.info(
        "Normalised portfolio: %d trades, %d asset classes, %d scenarios",
        len(attributes), len(portfolio.asset_classes), len(scenario_cols),
    )
    return portfolio
