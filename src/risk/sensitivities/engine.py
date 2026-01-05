from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from src.marketdata.ids import MarketId
from src.marketdata.scenarios.base import MarketView
from src.portfolio.core import Portfolio

from src.marketdata.scenarios.shocks import SpotShock
from src.marketdata.scenarios.shocks import ParallelRateShock
from src.marketdata.scenarios.shocks import VolShock

from src.risk.scenarios.runner import run_portfolio_scenarios
from src.risk.sensitivities.config import SensitivitiesConfig
from src.risk.sensitivities.result import SensitivityKey, SensitivityRow, SensitivitiesReport

@dataclass(frozen=True, slots=True)
class _InferredRiskFactors:
    spot_ids: Tuple[MarketId, ...]
    vol_ids: Tuple[MarketId, ...]
    curve_ids: Tuple[MarketId, ...]


def compute_sensitivities(
    portfolio: Portfolio,
    market: MarketView,
    portfolio_pricer,  # noqa: ANN001 (duck-typed)
    *,
    config: SensitivitiesConfig = SensitivitiesConfig(),
    requested_greeks: Sequence[str] = ("delta", "gamma", "vega", "rho_domestic", "rho_foreign"),
) -> SensitivitiesReport:
    """
    Compute portfolio sensitivities either analytically or via bump-and-reprice.

    Parameters
    ----------
    portfolio:
        Portfolio object.
    market:
        Base market snapshot (MarketView).
    portfolio_pricer:
        PortfolioPricer-like object exposing price(portfolio, market)-> result with totals.pv and totals.greeks
    config:
        SensitivitiesConfig.
    requested_greeks:
        List of greek names to compute (portfolio-level keys for V1).

    Returns
    -------
    SensitivitiesReport
    """
    inferred = _infer_risk_factors_from_portfolio(portfolio)

    # Price base once (we’ll reuse PV + totals greeks for analytic and FD).
    base_result = portfolio_pricer.price(portfolio, market)
    pv_base = float(getattr(getattr(base_result, "totals"), "pv"))
    totals_greeks = _extract_totals_greeks(base_result)

    method = str(config.method).lower().strip()
    if method not in {"analytic", "fd_central"}:
        raise ValueError("SensitivitiesConfig.method must be 'analytic' or 'fd_central'.")

    rows: List[SensitivityRow] = []

    # Decide if analytic is ambiguous (multiple ids but greeks are not keyed per-id).
    # In V1: allow analytic if we only have one id per risk type.
    ambiguous = _is_analytic_ambiguous(inferred)

    for greek in requested_greeks:
        greek = str(greek)

        use_method = method
        if use_method == "analytic" and ambiguous and config.fallback_to_fd_when_ambiguous:
            use_method = "fd_central"

        if use_method == "analytic":
            rows.extend(_analytic_rows_for_greek(
                greek=greek,
                totals_greeks=totals_greeks,
                inferred=inferred,
                config=config,
            ))
        else:
            rows.extend(_fd_rows_for_greek(
                greek=greek,
                portfolio=portfolio,
                market=market,
                portfolio_pricer=portfolio_pricer,
                pv_base=pv_base,
                inferred=inferred,
                config=config,
            ))

    return SensitivitiesReport(rows=rows)


# -----------------------------------------------------------------------------
# Analytic path
# -----------------------------------------------------------------------------

def _analytic_rows_for_greek(
    *,
    greek: str,
    totals_greeks: Mapping[str, float],
    inferred: _InferredRiskFactors,
    config: SensitivitiesConfig,
) -> List[SensitivityRow]:
    """
    Convert portfolio totals greeks into SensitivityRows.

    Notes
    -----
    In V1 your totals greeks are portfolio-level scalars (not per MarketId).
    We attach a MarketId only when it’s unambiguous (exactly one inferred id).
    """
    value = float(totals_greeks.get(greek, 0.0))

    market_id: Optional[MarketId] = None
    units = _units_for_greek(greek)

    if greek in {"delta", "gamma"} and len(inferred.spot_ids) == 1:
        market_id = inferred.spot_ids[0]
    elif greek == "vega" and len(inferred.vol_ids) == 1:
        market_id = inferred.vol_ids[0]
    elif greek.startswith("rho"):
        # Prefer explicit mapping if provided.
        if config.rho_key_by_curve_id is not None and len(inferred.curve_ids) >= 1:
            # For analytic V1, rho_domestic / rho_foreign are already split keys.
            # We attach the curve_id that maps to this rho key if possible.
            for cid, key in config.rho_key_by_curve_id.items():
                if str(key) == greek:
                    market_id = cid
                    break

    return [
        SensitivityRow(
            key=SensitivityKey(greek=greek, market_id=market_id),
            value=value,
            method="analytic",
            bump=None,
            units=units,
        )
    ]


def _extract_totals_greeks(price_result) -> Dict[str, float]:  # noqa: ANN001
    totals = getattr(price_result, "totals", None)
    if totals is None:
        raise AttributeError("price_result must have attribute 'totals'.")
    greeks = getattr(totals, "greeks", None)
    if greeks is None:
        return {}
    return {str(k): float(v) for k, v in dict(greeks).items()}


def _is_analytic_ambiguous(inferred: _InferredRiskFactors) -> bool:
    # If you have >1 spot_id (or vol_id/curve_id), V1 portfolio-level keys cannot be attributed per-id.
    return (len(inferred.spot_ids) > 1) or (len(inferred.vol_ids) > 1) or (len(inferred.curve_ids) > 2)


# -----------------------------------------------------------------------------
# FD path
# -----------------------------------------------------------------------------

def _fd_rows_for_greek(
    *,
    greek: str,
    portfolio: Portfolio,
    market: MarketView,
    portfolio_pricer,  # noqa: ANN001
    pv_base: float,
    inferred: _InferredRiskFactors,
    config: SensitivitiesConfig,
) -> List[SensitivityRow]:
    """
    Compute a greek via bump-and-reprice.

    V1 supported:
    - delta, gamma: bump spot (per spot_id)
    - vega        : bump vol  (per vol_id)
    - rho_domestic/rho_foreign: bump curve (per curve_id, mapped by config when available)
    """
    if greek in {"delta", "gamma"}:
        return _fd_spot_rows(
            greek=greek,
            portfolio=portfolio,
            market=market,
            portfolio_pricer=portfolio_pricer,
            pv_base=pv_base,
            spot_ids=inferred.spot_ids,
            rel_bump=float(config.bumps.spot_rel),
        )

    if greek == "vega":
        return _fd_vol_rows(
            portfolio=portfolio,
            market=market,
            portfolio_pricer=portfolio_pricer,
            pv_base=pv_base,
            vol_ids=inferred.vol_ids,
            abs_bump=float(config.bumps.vol_abs),
        )

    if greek.startswith("rho"):
        return _fd_rho_rows(
            greek=greek,
            portfolio=portfolio,
            market=market,
            portfolio_pricer=portfolio_pricer,
            pv_base=pv_base,
            curve_ids=inferred.curve_ids,
            abs_bump=float(config.bumps.rate_abs),
            rho_key_by_curve_id=config.rho_key_by_curve_id,
        )

    # Unknown greek requested -> return empty row list (safe default).
    return []


def _fd_spot_rows(
    *,
    greek: str,
    portfolio: Portfolio,
    market: MarketView,
    portfolio_pricer,  # noqa: ANN001
    pv_base: float,
    spot_ids: Tuple[MarketId, ...],
    rel_bump: float,
) -> List[SensitivityRow]:\

    rows: List[SensitivityRow] = []
    for sid in spot_ids:
        s0 = float(market.quote(sid))
        h = float(s0 * rel_bump)

        shock_up = SpotShock(name=f"spot_up_{sid.name}", spot_id=sid, bump=rel_bump, bump_mode="relative")
        shock_dn = SpotShock(name=f"spot_dn_{sid.name}", spot_id=sid, bump=-rel_bump, bump_mode="relative")

        res = run_portfolio_scenarios(portfolio, market, portfolio_pricer, [shock_up, shock_dn])
        pv_up = float(res.pv_by_scenario[shock_up.name])
        pv_dn = float(res.pv_by_scenario[shock_dn.name])

        if greek == "delta":
            value = (pv_up - pv_dn) / (2.0 * h)
            rows.append(SensitivityRow(
                key=SensitivityKey(greek="delta", market_id=sid),
                value=float(value),
                method="fd_central",
                bump=float(rel_bump),
                units="per 1 spot",
            ))
        else:
            # gamma = d²PV/dS² ≈ (PV_up - 2*PV0 + PV_dn) / h²
            value = (pv_up - 2.0 * float(pv_base) + pv_dn) / (h * h)
            rows.append(SensitivityRow(
                key=SensitivityKey(greek="gamma", market_id=sid),
                value=float(value),
                method="fd_central",
                bump=float(rel_bump),
                units="per 1 spot^2",
            ))

    return rows


def _fd_vol_rows(
    *,
    portfolio: Portfolio,
    market: MarketView,
    portfolio_pricer,  # noqa: ANN001
    pv_base: float,
    vol_ids: Tuple[MarketId, ...],
    abs_bump: float,
) -> List[SensitivityRow]:

    rows: List[SensitivityRow] = []
    for vid in vol_ids:
        shock_up = VolShock(name=f"vol_up_{vid.name}", vol_id=vid, bump=abs_bump, bump_mode="absolute")
        shock_dn = VolShock(name=f"vol_dn_{vid.name}", vol_id=vid, bump=-abs_bump, bump_mode="absolute")

        res = run_portfolio_scenarios(portfolio, market, portfolio_pricer, [shock_up, shock_dn])
        pv_up = float(res.pv_by_scenario[shock_up.name])
        pv_dn = float(res.pv_by_scenario[shock_dn.name])

        vega = (pv_up - pv_dn) / (2.0 * abs_bump)
        rows.append(SensitivityRow(
            key=SensitivityKey(greek="vega", market_id=vid),
            value=float(vega),
            method="fd_central",
            bump=float(abs_bump),
            units="per 1 vol",
        ))

    return rows


def _fd_rho_rows(
    *,
    greek: str,
    portfolio: Portfolio,
    market: MarketView,
    portfolio_pricer,  # noqa: ANN001
    pv_base: float,
    curve_ids: Tuple[MarketId, ...],
    abs_bump: float,
    rho_key_by_curve_id: Optional[Mapping[MarketId, str]],
) -> List[SensitivityRow]:

    # If you provided a mapping curve_id -> rho_key, use that to decide which curves to bump for rho_domestic/rho_foreign.
    bump_these: List[MarketId] = []
    if rho_key_by_curve_id is None:
        bump_these = list(curve_ids)
    else:
        for cid, key in rho_key_by_curve_id.items():
            if str(key) == greek:
                bump_these.append(cid)

    rows: List[SensitivityRow] = []
    for cid in bump_these:
        shock_up = ParallelRateShock(name=f"r_up_{cid.name}", curve_id=cid, rate_shift=abs_bump)
        shock_dn = ParallelRateShock(name=f"r_dn_{cid.name}", curve_id=cid, rate_shift=-abs_bump)

        res = run_portfolio_scenarios(portfolio, market, portfolio_pricer, [shock_up, shock_dn])
        pv_up = float(res.pv_by_scenario[shock_up.name])
        pv_dn = float(res.pv_by_scenario[shock_dn.name])

        rho = (pv_up - pv_dn) / (2.0 * abs_bump)
        rows.append(SensitivityRow(
            key=SensitivityKey(greek=greek, market_id=cid),
            value=float(rho),
            method="fd_central",
            bump=float(abs_bump),
            units="per 1 rate",
        ))

    return rows


# -----------------------------------------------------------------------------
# Risk factor inference (V1)
# -----------------------------------------------------------------------------

def _infer_risk_factors_from_portfolio(portfolio: Portfolio) -> _InferredRiskFactors:
    spot_ids: Set[MarketId] = set()
    vol_ids: Set[MarketId] = set()
    curve_ids: Set[MarketId] = set()

    positions = getattr(portfolio, "positions", None)
    if positions is None:
        return _InferredRiskFactors(spot_ids=(), vol_ids=(), curve_ids=())

    for pos in positions:
        inst = getattr(pos, "instrument", None)
        if inst is None:
            continue

        sid = getattr(inst, "spot_id", None)
        if isinstance(sid, MarketId):
            spot_ids.add(sid)

        vid = getattr(inst, "vol_id", None)
        if isinstance(vid, MarketId):
            vol_ids.add(vid)

        cd = getattr(inst, "domestic_curve_id", None)
        if isinstance(cd, MarketId):
            curve_ids.add(cd)

        cf = getattr(inst, "foreign_curve_id", None)
        if isinstance(cf, MarketId):
            curve_ids.add(cf)

        c = getattr(inst, "curve_id", None)
        if isinstance(c, MarketId):
            curve_ids.add(c)

    return _InferredRiskFactors(
        spot_ids=tuple(sorted(spot_ids, key=str)),
        vol_ids=tuple(sorted(vol_ids, key=str)),
        curve_ids=tuple(sorted(curve_ids, key=str)),
    )


def _units_for_greek(greek: str) -> str:
    if greek == "delta":
        return "per 1 spot"
    if greek == "gamma":
        return "per 1 spot^2"
    if greek == "vega":
        return "per 1 vol"
    if greek.startswith("rho"):
        return "per 1 rate"
    return ""