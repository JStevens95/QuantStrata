"""
Step 0 — portfolio ingestion.

Turns the raw external risk-system exports into the inputs the pipeline needs:

  - ``normalize_attributes``  : raw exploded attribute frame  → one row per trade,
    with risk types pivoted to columns and ``NotionalSign`` derived.
  - ``extract_asset_config``  : normalized trades            → ``asset_config``
    (``{risk_factor: {asset_class, ...}}``) + a dependency graph, via a pluggable
    :class:`RiskFactorResolver`.
  - ``load_pnl``              : raw scenario-PnL frame        → trade-indexed PnL.

The raw attribute frame has **more rows than trades** because each row is one
``(trade, risk_type, curve)`` combination, and ``AssetClass`` is ``"ALL"`` on the
``Notional`` rows. Aggregation rules:

  - static (trade-level) columns  → first value per trade
  - ``AssetClass``                → first **non-"ALL"** value per trade
  - ``RiskType``                  → pivot to columns, summing the risk value
                                    across curve rows
  - ``NotionalSign``              → from ``BuySellInd``

Risk-factor identification is asset-class specific and pluggable; the default FX
resolver infers the pair (and its IR dependencies) from the ``Product`` field.
Curve codes are intentionally **not** used here (they are mostly ``N/A``).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from src.rade_sr.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Ingest configuration (the schema contract — change here, not in code)
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IngestConfig:
    """Column names and conventions for the raw attribute / PnL files.

    Defaults match the external risk-system export. Override fields to retarget
    a different source without touching the ingestion logic.
    """
    trade_id_col: str = "PTSDealNumber"
    asset_class_col: str = "AssetClass"
    risk_type_col: str = "RiskType"
    value_col: str = "Sum_RiskValuesUSD_Net"
    buy_sell_col: str = "BuySellInd"

    notional_risk_type: str = "Notional"
    all_asset_class_label: str = "ALL"

    # static (trade-level) columns carried through as first-per-trade
    static_cols: Tuple[str, ...] = (
        "BuySellInd", "CallPut", "Date", "DeskName", "MaturityDate",
        "Product", "ProductGroup", "SourceSystemGroup", "StrikePrice",
        "SubDeskName", "TRBookKey",
    )
    curve_cols: Tuple[str, ...] = ("StandardCurveCode", "PTSCurveCode")

    # BuySellInd -> direction sign (case-insensitive prefix match on Buy/Sell)
    buy_values: Tuple[str, ...] = ("BUY", "B", "+1", "1")
    sell_values: Tuple[str, ...] = ("SELL", "S", "-1")


# ─────────────────────────────────────────────────────────────────────────
# Risk-factor resolution (pluggable per asset class)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class RiskFactorInfo:
    """Resolved risk-factor identity + dependency graph for one trade."""
    risk_factor: str
    asset_class: str
    dependencies: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class RiskFactorResolver(Protocol):
    """Maps a normalized trade row to its primary risk factor + dependencies."""

    asset_class: str

    def resolve(self, trade: pd.Series) -> RiskFactorInfo: ...


_PAIR_RE = re.compile(r"([A-Z]{6})")


class FXRiskFactorResolver:
    """Infer the FX risk factor (currency pair) and its IR dependencies.

    The pair is parsed from the ``Product`` field (e.g. ``"EURUSD_VanillaOption"``
    → ``"EURUSD"``). FX depends on the two IR curves for its base and quote
    currencies. This is the one place that encodes the "use currency to infer
    risk factors" rule — swap it out when the real convention is decided.
    """

    asset_class = "fx"

    def __init__(self, product_col: str = "Product") -> None:
        self._product_col = product_col

    def resolve(self, trade: pd.Series) -> RiskFactorInfo:
        product = str(trade.get(self._product_col, ""))
        match = _PAIR_RE.search(product.upper())
        if not match:
            raise ConfigurationError(
                f"Could not infer FX pair from Product={product!r}"
            )
        pair = match.group(1)
        base, quote = pair[:3], pair[3:6]
        return RiskFactorInfo(
            risk_factor=pair,
            asset_class="fx",
            dependencies=[f"{base}_IR", f"{quote}_IR"],
            extra={"pair": pair, "base_ccy": base, "quote_ccy": quote},
        )


def default_resolvers() -> Dict[str, RiskFactorResolver]:
    """Default resolver registry keyed by lower-cased asset class."""
    return {"fx": FXRiskFactorResolver()}


# ─────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────

def _direction_sign(value: Any, config: IngestConfig) -> float:
    token = str(value).strip().upper()
    if token in config.buy_values:
        return 1.0
    if token in config.sell_values:
        return -1.0
    if token.startswith("B"):
        return 1.0
    if token.startswith("S"):
        return -1.0
    logger.warning("Unrecognised BuySellInd %r; defaulting sign +1", value)
    return 1.0


def _resolve_asset_class(series: pd.Series, all_label: str) -> str:
    """First non-'ALL' asset class for a trade (the Notional rows carry 'ALL')."""
    non_all = series[series != all_label]
    if len(non_all) == 0:
        return all_label
    return non_all.mode().iloc[0]


def normalize_attributes(
    raw: pd.DataFrame,
    config: Optional[IngestConfig] = None,
) -> pd.DataFrame:
    """Collapse the raw exploded attribute frame to one row per trade.

    Returns a DataFrame indexed by trade id with: the static trade-level columns,
    one column per risk type (summed across curves), the resolved ``AssetClass``,
    ``NotionalSign``, and ``SignedNotional`` (if a notional risk type is present).
    """
    config = config or IngestConfig()
    _validate_columns(raw, config)

    grouped = raw.groupby(config.trade_id_col, sort=False)

    # static columns: first per trade
    static = grouped[list(config.static_cols)].first()

    # asset class: first non-ALL per trade
    asset_class = grouped[config.asset_class_col].apply(
        lambda s: _resolve_asset_class(s, config.all_asset_class_label)
    ).rename(config.asset_class_col)

    # risk types → columns, summed across (trade, risk_type, curve) rows
    risk_pivot = raw.pivot_table(
        index=config.trade_id_col,
        columns=config.risk_type_col,
        values=config.value_col,
        aggfunc="sum",
    )
    risk_pivot.columns.name = None

    # curve inventory (informational; not used for logic)
    curves = grouped[list(config.curve_cols)].agg(
        lambda s: sorted({str(v) for v in s if str(v) not in ("N/A", "nan")})
    )

    out = static.join(asset_class).join(risk_pivot).join(curves)

    # direction sign + signed notional
    out["NotionalSign"] = out[config.buy_sell_col].map(
        lambda v: _direction_sign(v, config)
    )
    if config.notional_risk_type in out.columns:
        out["SignedNotional"] = out["NotionalSign"] * out[config.notional_risk_type]

    # derived: time to maturity in years (best-effort from YYYYMMDD strings)
    out = _add_expiry_years(out)

    out.index.name = "trade_id"
    logger.info(
        "Normalized %d raw rows → %d trades, %d risk-type columns",
        len(raw), len(out), risk_pivot.shape[1],
    )
    return out


def _add_expiry_years(df: pd.DataFrame) -> pd.DataFrame:
    if "MaturityDate" not in df.columns or "Date" not in df.columns:
        return df
    cob = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    mat = pd.to_datetime(df["MaturityDate"], format="%Y%m%d", errors="coerce")
    df["expiry_years"] = ((mat - cob).dt.days / 365.0).clip(lower=1.0 / 365.0)
    return df


def _validate_columns(raw: pd.DataFrame, config: IngestConfig) -> None:
    required = [
        config.trade_id_col, config.asset_class_col, config.risk_type_col,
        config.value_col, config.buy_sell_col,
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ConfigurationError(f"Raw attribute frame missing columns: {missing}")


# ─────────────────────────────────────────────────────────────────────────
# Asset config + dependency graph extraction
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioInputs:
    """Bundle of everything Step 0 produces, ready for the pipeline."""
    attributes: pd.DataFrame                       # one row per trade
    asset_config: Dict[str, Dict[str, Any]]        # {risk_factor: {asset_class, ...}}
    dependency_graph: Dict[str, List[str]]         # {risk_factor: [dependency_ids]}
    risk_factor_by_trade: pd.Series                # trade_id → risk_factor
    target_pnl: Optional[pd.DataFrame] = None      # trades × scenarios
    target_asset_class: Optional[pd.Series] = None # trade_id → asset class


def extract_asset_config(
    attributes: pd.DataFrame,
    resolvers: Optional[Dict[str, RiskFactorResolver]] = None,
    asset_class_col: str = "AssetClass",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], pd.Series]:
    """Resolve risk factors + dependencies for every trade.

    Returns ``(asset_config, dependency_graph, risk_factor_by_trade)``.
    """
    resolvers = resolvers or default_resolvers()

    asset_config: Dict[str, Dict[str, Any]] = {}
    dependency_graph: Dict[str, List[str]] = {}
    rf_by_trade: Dict[str, str] = {}

    for trade_id, row in attributes.iterrows():
        ac_key = str(row[asset_class_col]).lower()
        resolver = resolvers.get(ac_key)
        if resolver is None:
            logger.warning(
                "No resolver for asset_class=%r (trade %s); skipping",
                ac_key, trade_id,
            )
            continue

        info = resolver.resolve(row)
        rf_by_trade[trade_id] = info.risk_factor
        if info.risk_factor not in asset_config:
            asset_config[info.risk_factor] = {
                "asset_class": info.asset_class,
                **info.extra,
            }
            dependency_graph[info.risk_factor] = list(info.dependencies)

    logger.info(
        "Extracted %d unique risk factors from %d trades",
        len(asset_config), len(attributes),
    )
    return asset_config, dependency_graph, pd.Series(rf_by_trade, name="risk_factor")


# ─────────────────────────────────────────────────────────────────────────
# PnL loading
# ─────────────────────────────────────────────────────────────────────────

def load_pnl(
    raw_pnl: pd.DataFrame,
    config: Optional[IngestConfig] = None,
    asset_class_col: str = "AssetClass",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the raw PnL frame into a (trades × scenarios) matrix + asset class.

    The raw frame is indexed by trade id with scenario-date columns plus an
    ``AssetClass`` column. Returns ``(pnl, asset_class_by_trade)``.
    """
    config = config or IngestConfig()
    df = raw_pnl.copy()

    if df.index.name != config.trade_id_col and config.trade_id_col in df.columns:
        df = df.set_index(config.trade_id_col)
    df.index = df.index.astype(str)

    asset_class = (
        df[asset_class_col].astype(str)
        if asset_class_col in df.columns
        else pd.Series("UNKNOWN", index=df.index)
    )
    asset_class.name = "asset_class"

    pnl = df.drop(columns=[asset_class_col], errors="ignore").astype(np.float64)
    pnl.index.name = "trade_id"
    return pnl, asset_class


# ─────────────────────────────────────────────────────────────────────────
# Convenience: full Step 0
# ─────────────────────────────────────────────────────────────────────────

def ingest_portfolio(
    raw_attributes: pd.DataFrame,
    raw_pnl: pd.DataFrame,
    config: Optional[IngestConfig] = None,
    resolvers: Optional[Dict[str, RiskFactorResolver]] = None,
) -> PortfolioInputs:
    """Run the full Step 0 ingestion: normalize, resolve risk factors, load PnL."""
    config = config or IngestConfig()
    attributes = normalize_attributes(raw_attributes, config)
    asset_config, dep_graph, rf_by_trade = extract_asset_config(
        attributes, resolvers=resolvers, asset_class_col=config.asset_class_col,
    )
    target_pnl, target_ac = load_pnl(raw_pnl, config)

    return PortfolioInputs(
        attributes=attributes,
        asset_config=asset_config,
        dependency_graph=dep_graph,
        risk_factor_by_trade=rf_by_trade,
        target_pnl=target_pnl,
        target_asset_class=target_ac,
    )
