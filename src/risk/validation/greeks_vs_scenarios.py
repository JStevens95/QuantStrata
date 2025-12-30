from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.marketdata.ids import MarketId
from src.marketdata.scenarios.base import MarketView, ScenarioPack, ScenarioShock


# =============================================================================
# Result objects
# =============================================================================

@dataclass(frozen=True, slots=True)
class GreeksVsScenarioRow:
    """
    One scenario row comparing realized scenario PnL vs greek-predicted PnL.
    """
    scenario: str
    pv_base: float
    pv_scenario: float
    pnl: float
    predicted_pnl: float
    residual: float
    rel_error: float


@dataclass(frozen=True, slots=True)
class GreeksVsScenarioReport:
    """
    Collection of Greeks-vs-scenario validation rows.
    """
    rows: List[GreeksVsScenarioRow]
    base_scenario_name: str = "BASE"

    def max_rel_error(self) -> float:
        errors = [r.rel_error for r in self.rows if r.scenario != self.base_scenario_name]
        return float(max(errors)) if errors else 0.0

    def to_console(self, *, decimals: int = 6) -> str:
        if not self.rows:
            return "GreeksVsScenarioReport(empty)"

        def fmt(x: float, signed: bool = False) -> str:
            x_f = float(x)
            if not math.isfinite(x_f):
                if math.isnan(x_f):
                    return "nan"
                return "+inf" if x_f > 0 else "-inf"
            return f"{x_f:{'+' if signed else ''}.{int(decimals)}f}"

        w_s = max(len("Scenario"), max(len(r.scenario) for r in self.rows))
        w_pv = max(len("PV"), max(len(fmt(r.pv_scenario)) for r in self.rows))
        w_pnl = max(len("PnL"), max(len(fmt(r.pnl, signed=True)) for r in self.rows))
        w_pred = max(len("Pred"), max(len(fmt(r.predicted_pnl, signed=True)) for r in self.rows))
        w_res = max(len("Residual"), max(len(fmt(r.residual, signed=True)) for r in self.rows))
        w_re = max(len("RelErr"), max(len(f"{r.rel_error:.3g}") for r in self.rows))

        header = (
            f"{'Scenario':<{w_s}} | {'PV':>{w_pv}} | {'PnL':>{w_pnl}} | "
            f"{'Pred':>{w_pred}} | {'Residual':>{w_res}} | {'RelErr':>{w_re}}"
        )
        sep = "-" * len(header)

        lines = [header, sep]
        for r in self.rows:
            lines.append(
                f"{r.scenario:<{w_s}} | {fmt(r.pv_scenario):>{w_pv}} | {fmt(r.pnl, signed=True):>{w_pnl}} | "
                f"{fmt(r.predicted_pnl, signed=True):>{w_pred}} | {fmt(r.residual, signed=True):>{w_res}} | "
                f"{r.rel_error:>{w_re}.3g}"
            )
        return "\n".join(lines)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class GreeksVsScenarioConfig:
    """
    Configure how we map scenario shocks to greek keys.

    Notes
    -----
    - Spot shocks typically map to ('delta', optionally 'gamma').
    - Vol shocks map to 'vega' (per 1.00 absolute vol).
    - Rate shocks map to a rho key. For multi-curve setups (e.g. FX domestic vs foreign),
      you *should* provide rho_key_by_curve_id.
    """
    spot_delta_keys: Tuple[str, ...] = ("delta", "delta_spot")
    spot_gamma_key: str = "gamma"
    include_gamma_for_spot: bool = True

    vega_key: str = "vega"

    # Map curve_id -> rho key in totals greeks.
    # Example: {rd_id: "rho_domestic", rf_id: "rho_foreign"}
    rho_key_by_curve_id: Optional[Mapping[MarketId, str]] = None


# =============================================================================
# Public API
# =============================================================================

def validate_greeks_vs_scenarios(
    portfolio: Any,
    base_market: MarketView,
    portfolio_pricer: Any,
    scenarios: Union[ScenarioPack, Sequence[ScenarioShock]],
    *,
    config: GreeksVsScenarioConfig = GreeksVsScenarioConfig(),
    base_name: str = "BASE",
    vol_reference_by_id: Optional[Mapping[MarketId, Tuple[float, float]]] = None,
) -> GreeksVsScenarioReport:
    """
    Validate first-order greek predictions against realized scenario PnL.

    Important
    ---------
    This is a *local* (small bump) validation harness.
    For larger stresses, residuals naturally include higher-order terms
    (gamma, volga, vanna, cross terms, and non-linear DF effects).

    Parameters
    ----------
    portfolio:
        Portfolio object (duck-typed; must work with portfolio_pricer.price()).
    base_market:
        Base MarketView (concrete Market or a wrapper).
    portfolio_pricer:
        Must expose .price(portfolio, market_view) -> object with .totals.pv and .totals.greeks.
    scenarios:
        ScenarioPack or ordered list of ScenarioShock.
    config:
        Mapping config for greek keys.
    base_name:
        Base scenario label.
    vol_reference_by_id:
        Optional explicit mapping vol_id -> (expiry_ref, strike_ref) used only for relative vol shocks.

    Returns
    -------
    GreeksVsScenarioReport
    """
    base_price = portfolio_pricer.price(portfolio, base_market)
    pv_base = _extract_total_pv(base_price)
    totals_greeks = _extract_total_greeks(base_price)

    scenario_names, market_views, shock_by_name = _build_scenario_markets(
        base_market=base_market,
        scenarios=scenarios,
        base_name=base_name,
    )

    vol_ref_points = dict(vol_reference_by_id or {})
    _ensure_vol_reference_points(
        vol_reference_points=vol_ref_points,
        portfolio=portfolio,
        base_market=base_market,
        scenarios=scenarios,
    )

    rows: List[GreeksVsScenarioRow] = []
    for scenario_name, market_view in zip(scenario_names, market_views):
        shocked_price = portfolio_pricer.price(portfolio, market_view)
        pv_scn = _extract_total_pv(shocked_price)
        pnl = float(pv_scn - pv_base)

        predicted = 0.0
        if scenario_name != base_name:
            shock = shock_by_name[scenario_name]
            predicted = float(_predict_first_order_pnl(
                shock=shock,
                base_market=base_market,
                totals_greeks=totals_greeks,
                config=config,
                vol_reference_points=vol_ref_points,
            ))

        residual = pnl - predicted

        # Relative error: scale by a robust magnitude so tiny PnLs don't explode.
        scale = max(1.0, abs(pnl), abs(pv_base))
        rel_error = abs(residual) / scale

        rows.append(GreeksVsScenarioRow(
            scenario=str(scenario_name),
            pv_base=float(pv_base),
            pv_scenario=float(pv_scn),
            pnl=float(pnl),
            predicted_pnl=float(predicted),
            residual=float(residual),
            rel_error=float(rel_error),
        ))

    return GreeksVsScenarioReport(rows=rows, base_scenario_name=str(base_name))


# =============================================================================
# Extraction helpers (duck-typed)
# =============================================================================

def _extract_total_pv(price_result: Any) -> float:
    totals = getattr(price_result, "totals", None)
    if totals is None:
        raise AttributeError("price_result must have attribute 'totals'.")
    pv = getattr(totals, "pv", None)
    if pv is None:
        raise AttributeError("price_result.totals must have attribute 'pv'.")
    return float(pv)


def _extract_total_greeks(price_result: Any) -> Dict[str, float]:
    totals = getattr(price_result, "totals", None)
    if totals is None:
        raise AttributeError("price_result must have attribute 'totals'.")
    greeks = getattr(totals, "greeks", None)
    if greeks is None:
        raise AttributeError(
            "price_result.totals must have attribute 'greeks' (dict-like). "
            "If totals greeks aggregation is not implemented yet, add it in PortfolioPricer."
        )
    return {str(k): float(v) for k, v in dict(greeks).items()}


# =============================================================================
# Scenario market construction
# =============================================================================

def _build_scenario_markets(
    base_market: MarketView,
    scenarios: Union[ScenarioPack, Sequence[ScenarioShock]],
    *,
    base_name: str,
) -> Tuple[List[str], List[MarketView], Dict[str, ScenarioShock]]:
    scenario_names: List[str] = [str(base_name)]
    scenario_views: List[MarketView] = [base_market]
    shock_by_name: Dict[str, ScenarioShock] = {}

    if isinstance(scenarios, ScenarioPack):
        for scenario_name, shock in scenarios.scenarios.items():
            name = str(scenario_name)
            scenario_names.append(name)
            scenario_views.append(shock.apply(base_market))
            shock_by_name[name] = shock
        return scenario_names, scenario_views, shock_by_name

    for shock in scenarios:
        name = str(getattr(shock, "name", shock.__class__.__name__))
        scenario_names.append(name)
        scenario_views.append(shock.apply(base_market))
        shock_by_name[name] = shock

    return scenario_names, scenario_views, shock_by_name


# =============================================================================
# First-order prediction logic (duck-typed shocks)
# =============================================================================

def _predict_first_order_pnl(
    *,
    shock: ScenarioShock,
    base_market: MarketView,
    totals_greeks: Mapping[str, float],
    config: GreeksVsScenarioConfig,
    vol_reference_points: Mapping[MarketId, Tuple[float, float]],
) -> float:
    # ---- Spot shock (delta * dS [+ 0.5*gamma*dS^2]) ----
    if _looks_like_spot_shock(shock):
        spot_id = getattr(shock, "spot_id")
        bump = float(getattr(shock, "bump"))
        bump_mode = str(getattr(shock, "bump_mode"))

        base_spot = float(base_market.quote(spot_id))
        dS = base_spot * bump if bump_mode == "relative" else bump

        delta_key = _first_matching_key(totals_greeks, config.spot_delta_keys)
        if delta_key is None:
            raise KeyError(f"No spot delta key found in totals greeks. Tried: {config.spot_delta_keys}")

        predicted = float(totals_greeks[delta_key]) * dS

        if config.include_gamma_for_spot and config.spot_gamma_key in totals_greeks:
            gamma = float(totals_greeks[config.spot_gamma_key])
            predicted += 0.5 * gamma * dS * dS

        return float(predicted)

    # ---- Vol shock (vega * dσ) ----
    if _looks_like_vol_shock(shock):
        vol_id = getattr(shock, "vol_id")
        d_sigma = _infer_vol_bump(shock=shock, base_market=base_market, vol_reference_points=vol_reference_points)

        if config.vega_key not in totals_greeks:
            raise KeyError(f"Totals greeks missing vega key '{config.vega_key}'.")
        vega = float(totals_greeks[config.vega_key])

        return float(vega * d_sigma)

    # ---- Parallel rate shock (rho * dr) ----
    if _looks_like_rate_shock(shock):
        curve_id = getattr(shock, "curve_id")
        dr = float(getattr(shock, "rate_shift"))

        rho_key = _resolve_rho_key(curve_id=curve_id, totals_greeks=totals_greeks, config=config)
        rho = float(totals_greeks[rho_key])

        return float(rho * dr)

    raise TypeError(
        "Unsupported ScenarioShock type for first-order PnL prediction. "
        "Supported: SpotShock-like, VolShock-like, ParallelRateShock-like."
    )


def _looks_like_spot_shock(shock: Any) -> bool:
    return hasattr(shock, "spot_id") and hasattr(shock, "bump") and hasattr(shock, "bump_mode")


def _looks_like_vol_shock(shock: Any) -> bool:
    return hasattr(shock, "vol_id") and (hasattr(shock, "vol_bump") or (hasattr(shock, "bump") and hasattr(shock, "bump_mode")))


def _looks_like_rate_shock(shock: Any) -> bool:
    return hasattr(shock, "curve_id") and hasattr(shock, "rate_shift")


def _resolve_rho_key(
    *,
    curve_id: MarketId,
    totals_greeks: Mapping[str, float],
    config: GreeksVsScenarioConfig,
) -> str:
    # 1) Explicit mapping by curve id (recommended for multi-curve books)
    if config.rho_key_by_curve_id is not None and curve_id in config.rho_key_by_curve_id:
        key = str(config.rho_key_by_curve_id[curve_id])
        if key not in totals_greeks:
            raise KeyError(f"rho_key_by_curve_id maps {curve_id} -> '{key}', but totals greeks lacks '{key}'.")
        return key

    # 2) Single generic key fallback
    if "rho" in totals_greeks:
        return "rho"

    raise KeyError(
        "Cannot map ParallelRateShock to a rho key. "
        "Provide GreeksVsScenarioConfig.rho_key_by_curve_id or include 'rho' in totals greeks."
    )


def _infer_vol_bump(
    *,
    shock: Any,
    base_market: MarketView,
    vol_reference_points: Mapping[MarketId, Tuple[float, float]],
) -> float:
    # FlatVolShock-style: explicit absolute bump
    if hasattr(shock, "vol_bump"):
        return float(getattr(shock, "vol_bump"))

    # Generic vol shock: bump + bump_mode
    bump = float(getattr(shock, "bump"))
    bump_mode = str(getattr(shock, "bump_mode"))
    vol_id = getattr(shock, "vol_id")

    if bump_mode == "absolute":
        return bump

    if bump_mode == "relative":
        expiry_ref, strike_ref = vol_reference_points.get(vol_id, (1.0, 1.0))
        sigma0 = float(base_market.vol_surface(vol_id).vol(float(expiry_ref), float(strike_ref)))
        return float(sigma0 * bump)

    raise ValueError("Vol shock bump_mode must be 'relative' or 'absolute'.")


def _first_matching_key(greeks: Mapping[str, float], candidate_keys: Sequence[str]) -> Optional[str]:
    for key in candidate_keys:
        if key in greeks:
            return str(key)
    return None


# =============================================================================
# Vol reference-point handling (only needed for *relative* vol shocks)
# =============================================================================

def _ensure_vol_reference_points(
    *,
    vol_reference_points: Dict[MarketId, Tuple[float, float]],
    portfolio: Any,
    base_market: MarketView,
    scenarios: Union[ScenarioPack, Sequence[ScenarioShock]],
) -> None:
    vol_ids = _collect_vol_ids_from_scenarios(scenarios)
    for vol_id in vol_ids:
        if vol_id in vol_reference_points:
            continue
        vol_reference_points[vol_id] = _infer_vol_reference_from_portfolio(
            vol_id=vol_id,
            portfolio=portfolio,
            base_market=base_market,
        )


def _collect_vol_ids_from_scenarios(scenarios: Union[ScenarioPack, Sequence[ScenarioShock]]) -> List[MarketId]:
    if isinstance(scenarios, ScenarioPack):
        iterable: Sequence[Any] = list(scenarios.scenarios.values())
    else:
        iterable = list(scenarios)

    vol_ids: List[MarketId] = []
    for shock in iterable:
        if hasattr(shock, "vol_id"):
            vol_ids.append(getattr(shock, "vol_id"))
    return vol_ids


def _infer_vol_reference_from_portfolio(
    *,
    vol_id: MarketId,
    portfolio: Any,
    base_market: MarketView,
) -> Tuple[float, float]:
    """
    Infer (expiry, strike) for vol_id by scanning portfolio instruments.

    Priority
    --------
    1) First instrument with matching vol_id that has (expiry, strike)
    2) If strike missing but has spot_id, use (expiry, spot)
    3) Fallback (1.0, 1.0)
    """
    positions = getattr(portfolio, "positions", None)
    if positions is not None:
        for pos in positions:
            inst = getattr(pos, "instrument", None)
            if inst is None:
                continue
            if getattr(inst, "vol_id", None) != vol_id:
                continue

            expiry = getattr(inst, "expiry", None)
            strike = getattr(inst, "strike", None)
            if expiry is not None and strike is not None:
                return float(expiry), float(strike)

            spot_id = getattr(inst, "spot_id", None)
            if expiry is not None and spot_id is not None:
                spot = float(base_market.quote(spot_id))
                return float(expiry), float(spot)

    return 1.0, 1.0


__all__ = [
    "GreeksVsScenarioRow",
    "GreeksVsScenarioReport",
    "GreeksVsScenarioConfig",
    "validate_greeks_vs_scenarios",
]