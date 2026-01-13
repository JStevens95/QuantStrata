from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.interfaces import MarketView, ScenarioPack, ScenarioShock
from src.portfolio.core import Portfolio
from src.risk.attribution.report import AttributionReport, AttributionRow
from src.risk.scenarios.runner import run_portfolio_scenarios


@dataclass(frozen=True, slots=True)
class AttributionConfig:
    """
    First-order (and optional gamma for spot) attribution configuration.
    """
    base_name: str = "BASE"
    include_gamma_for_spot: bool = True
    spot_delta_keys: Tuple[str, ...] = ("delta", "delta_spot")
    spot_gamma_key: str = "gamma"
    vega_key: str = "vega"
    rho_key_by_curve_id: Optional[Mapping[MarketId, str]] = None


def attribute_portfolio_scenarios(
    *,
    portfolio: Portfolio,
    base_market: MarketView,
    portfolio_pricer,  # noqa: ANN001 (duck-typed)
    scenarios: Union[ScenarioPack, Sequence[ScenarioShock]],
    config: AttributionConfig = AttributionConfig(),
) -> AttributionReport:
    """
    Attribute scenario PnL into factor contributions using totals greeks (V1: first-order + optional gamma).
    """
    base_price = portfolio_pricer.price(portfolio, base_market)
    pv_base = float(getattr(getattr(base_price, "totals"), "pv"))
    totals_greeks = _extract_totals_greeks(base_price)

    stress = run_portfolio_scenarios(
        portfolio=portfolio,
        base_market=base_market,
        portfolio_pricer=portfolio_pricer,
        scenarios=scenarios,
        base_name=config.base_name,
    )

    shock_by_name = _build_shock_lookup(scenarios)

    rows: list[AttributionRow] = []
    for scenario_name, pv_scn, pnl in zip(stress.scenario_names, stress.pv, stress.pnl):
        scenario_name = str(scenario_name)
        pv_scn_f = float(pv_scn)
        pnl_f = float(pnl)

        contributions = {} if scenario_name == config.base_name else _contributions_for_shock(
            shock=shock_by_name[scenario_name],
            base_market=base_market,
            totals_greeks=totals_greeks,
            config=config,
        )

        predicted = float(sum(contributions.values()))
        residual = pnl_f - predicted
        scale = max(1.0, abs(pnl_f), abs(pv_base))
        rel_error = abs(residual) / scale

        rows.append(
            AttributionRow(
                scenario=scenario_name,
                pv_base=float(pv_base),
                pv_scn=float(pv_scn_f),
                pnl=float(pnl_f),
                contributions=dict(contributions),
                predicted_pnl=float(predicted),
                residual=float(residual),
                rel_error=float(rel_error),
            )
        )

    return AttributionReport(rows=rows, base_scenario_name=config.base_name)


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

def _extract_totals_greeks(price_result) -> Dict[str, float]:  # noqa: ANN001
    totals = getattr(price_result, "totals", None)
    greeks = getattr(totals, "greeks", None) if totals is not None else None
    if greeks is None:
        return {}
    return {str(k): float(v) for k, v in dict(greeks).items()}


def _build_shock_lookup(
    scenarios: Union[ScenarioPack, Sequence[ScenarioShock]],
) -> Dict[str, ScenarioShock]:
    if isinstance(scenarios, ScenarioPack):
        return {str(name): shock for name, shock in scenarios.scenarios.items()}
    return {str(getattr(shock, "name", shock.__class__.__name__)): shock for shock in scenarios}


def _first_matching_key(greeks: Mapping[str, float], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in greeks:
            return str(k)
    return None


def _k(greek: str, market_id: object) -> str:
    return f"{greek}:{market_id}"


def _contributions_for_shock(
    *,
    shock: ScenarioShock,
    base_market: MarketView,
    totals_greeks: Mapping[str, float],
    config: AttributionConfig,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # SpotShock-like
    if hasattr(shock, "spot_id") and hasattr(shock, "bump") and hasattr(shock, "bump_mode"):
        spot_id = getattr(shock, "spot_id")
        bump = float(getattr(shock, "bump"))
        bump_mode = str(getattr(shock, "bump_mode"))

        s0 = float(base_market.quote(spot_id))
        dS = s0 * bump if bump_mode == "relative" else bump

        delta_key = _first_matching_key(totals_greeks, config.spot_delta_keys)
        if delta_key is None:
            raise KeyError(f"No delta key found. Tried: {config.spot_delta_keys}")
        delta = float(totals_greeks[delta_key])
        out[_k("delta", spot_id)] = delta * dS

        if config.include_gamma_for_spot and (config.spot_gamma_key in totals_greeks):
            gamma = float(totals_greeks[config.spot_gamma_key])
            out[_k("gamma", spot_id)] = 0.5 * gamma * dS * dS

        return out

    # VolShock-like
    if hasattr(shock, "vol_id"):
        vol_id = getattr(shock, "vol_id")
        d_sigma = float(getattr(shock, "vol_bump", getattr(shock, "bump", 0.0)))

        if config.vega_key not in totals_greeks:
            raise KeyError(f"Totals greeks missing '{config.vega_key}'.")
        vega = float(totals_greeks[config.vega_key])
        out[_k("vega", vol_id)] = vega * d_sigma
        return out

    # ParallelRateShock-like
    if hasattr(shock, "curve_id") and hasattr(shock, "rate_shift"):
        curve_id = getattr(shock, "curve_id")
        dr = float(getattr(shock, "rate_shift"))

        rho_key = None
        if config.rho_key_by_curve_id and curve_id in config.rho_key_by_curve_id:
            rho_key = str(config.rho_key_by_curve_id[curve_id])

        if rho_key is None:
            raise KeyError("Provide rho_key_by_curve_id for rate shocks (e.g., rho_domestic / rho_foreign).")

        if rho_key not in totals_greeks:
            raise KeyError(f"Totals greeks missing '{rho_key}'.")
        out[_k(rho_key, curve_id)] = float(totals_greeks[rho_key]) * dr
        return out

    raise TypeError(f"Unsupported shock type: {type(shock).__name__}")