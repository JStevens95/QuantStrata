from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.options.european import EuropeanFxOption
from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.scenarios.base import ScenarioPack
from src.marketdata.scenarios.shocks import FlatVolShock, ParallelRateShock, SpotShock
from src.marketdata.synthetic.provider import SyntheticProvider
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry_old import DefaultPricerRegistry
from src.risk.scenarios.runner import run_portfolio_scenarios


# -----------------------------------------------------------------------------
# IDs + Market builders
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FxVanillaMarketIds:
    """Canonical ids for the FX spot/forward/vanilla option example."""
    spot_id: MarketId
    vol_id: MarketId
    domestic_curve_id: MarketId
    foreign_curve_id: MarketId


def build_fx_vanilla_ids() -> FxVanillaMarketIds:
    """Create the MarketIds used in this example."""
    return FxVanillaMarketIds(
        spot_id=MarketId("FX", "SPOT", "EURUSD"),
        vol_id=MarketId("FX", "VOL", "EURUSD"),
        domestic_curve_id=MarketId("IR", "CURVE", "USD.OIS"),
        foreign_curve_id=MarketId("IR", "CURVE", "EUR.OIS"),
    )


def build_market(*, ids: FxVanillaMarketIds, asof: str = "2025-12-29", seed: int = 123):
    """Build a deterministic synthetic Market snapshot for the requested ids."""
    provider = SyntheticProvider(seed=seed)
    return provider.get_market(
        MarketRequest(
            asof=asof,
            universe=Universe([ids.spot_id, ids.vol_id, ids.domestic_curve_id, ids.foreign_curve_id]),
        )
    )


def fair_forward_strike(*, market, ids: FxVanillaMarketIds, expiry: float) -> float:
    """Fair forward strike: F0 = S0 * df_f(T) / df_d(T)."""
    spot = float(market.quote(ids.spot_id))
    df_d = float(market.curve(ids.domestic_curve_id).df(expiry))
    df_f = float(market.curve(ids.foreign_curve_id).df(expiry))
    return float(spot * df_f / df_d)


# -----------------------------------------------------------------------------
# Scenario packs
# -----------------------------------------------------------------------------

def scenario_pack_spot_only(*, ids: FxVanillaMarketIds) -> ScenarioPack:
    """Spot-only shocks (quick delta sanity)."""
    return ScenarioPack(
        scenarios={
            "spot_up_1pct": SpotShock("spot_up_1pct", ids.spot_id, 0.01, "relative"),
            "spot_down_1pct": SpotShock("spot_down_1pct", ids.spot_id, -0.01, "relative"),
        }
    )


def scenario_pack_full_risk(*, ids: FxVanillaMarketIds) -> ScenarioPack:
    """Spot + vol + domestic/foreign rate shocks."""
    return ScenarioPack(
        scenarios={
            "spot_up_1pct": SpotShock("spot_up_1pct", ids.spot_id, 0.01, "relative"),
            "vol_up_1vol": FlatVolShock("vol_up_1vol", ids.vol_id, 0.01),
            "rd_up_25bp": ParallelRateShock("rd_up_25bp", ids.domestic_curve_id, 0.0025),
            "rf_up_25bp": ParallelRateShock("rf_up_25bp", ids.foreign_curve_id, 0.0025),
        }
    )


# -----------------------------------------------------------------------------
# Reporting helpers (kept consistent with linear-only script)
# -----------------------------------------------------------------------------

def _scenario_key(name: object) -> str:
    """Normalize scenario name values coming from different runners/types."""
    return str(name).strip()


def _safe_float(greeks: Mapping[str, float], key: str) -> float:
    """Return greeks[key] as float if present, else 0.0."""
    return float(greeks.get(key, 0.0))


def print_scenario_table(*, title: str, result) -> None:
    """Print PV and PnL for every row returned by run_portfolio_scenarios()."""
    print(f"\n{title}")
    print("-" * len(title))
    for name, pv, pnl in zip(result.scenario_names, result.pv, result.pnl):
        print(f"{_scenario_key(name):>14s} | PV: {float(pv):>14.6f} | PnL: {float(pnl):>+14.6f}")
    print()


def predict_pnl_from_greeks(
    *,
    scenario: object,
    base_market: object,
    totals_greeks: Mapping[str, float],
    ids: FxVanillaMarketIds,
) -> float:
    """
    Predict scenario PnL using aggregated portfolio greeks.

    Supported predictions
    ---------------------
    SpotShock:
        dPV ≈ delta*dS + 0.5*gamma*dS^2     (gamma optional; 0 if missing)

    FlatVolShock:
        dPV ≈ vega*dSigma                    (vega per 1.00 vol)

    ParallelRateShock:
        dPV ≈ rho_domestic*dr  if domestic curve shocked
        dPV ≈ rho_foreign*dr   if foreign curve shocked
    """
    from src.marketdata.scenarios.shocks import FlatVolShock, ParallelRateShock, SpotShock  # noqa: WPS433

    delta = _safe_float(totals_greeks, "delta")
    gamma = _safe_float(totals_greeks, "gamma")
    vega = _safe_float(totals_greeks, "vega")
    rho_d = _safe_float(totals_greeks, "rho_domestic")
    rho_f = _safe_float(totals_greeks, "rho_foreign")

    if isinstance(scenario, SpotShock):
        base_spot = float(base_market.quote(ids.spot_id))
        dS = base_spot * float(scenario.bump) if scenario.bump_mode == "relative" else float(scenario.bump)
        return float(delta * dS + 0.5 * gamma * dS * dS)

    if isinstance(scenario, FlatVolShock):
        d_sigma = float(scenario.vol_bump)
        return float(vega * d_sigma)

    if isinstance(scenario, ParallelRateShock):
        dr = float(scenario.rate_shift)
        if scenario.curve_id == ids.domestic_curve_id:
            return float(rho_d * dr)
        if scenario.curve_id == ids.foreign_curve_id:
            return float(rho_f * dr)
        return 0.0

    raise TypeError(f"Unsupported scenario type for greek predictor: {type(scenario).__name__}")


def print_greeks_vs_scenarios(
    *,
    title: str,
    base_pv: float,
    base_greeks: Mapping[str, float],
    base_market: object,
    scenario_pack: ScenarioPack,
    scenario_result,
    ids: FxVanillaMarketIds,
) -> None:
    """Compare realized scenario PnL vs greek-predicted PnL (skips non-pack rows like BASE)."""
    header = f"Greeks vs Scenario PnL: {title}"
    print(f"\n{header}")
    print("-" * len(header))

    print(
        "Base totals: "
        f"PV={float(base_pv):,.6f} | "
        f"delta={_safe_float(base_greeks, 'delta'):,.6f} | "
        f"gamma={_safe_float(base_greeks, 'gamma'):,.6f} | "
        f"vega={_safe_float(base_greeks, 'vega'):,.6f} | "
        f"rho_d={_safe_float(base_greeks, 'rho_domestic'):,.6f} | "
        f"rho_f={_safe_float(base_greeks, 'rho_foreign'):,.6f}"
    )

    headers = ["Scenario", "PnL", "Predicted", "Residual", "RelError"]
    rows: List[List[str]] = []

    for raw_name, pnl in zip(scenario_result.scenario_names, scenario_result.pnl):
        name = _scenario_key(raw_name)

        scenario_obj = scenario_pack.scenarios.get(name)
        if scenario_obj is None:
            continue

        predicted = predict_pnl_from_greeks(
            scenario=scenario_obj,
            base_market=base_market,
            totals_greeks=base_greeks,
            ids=ids,
        )

        residual = float(pnl) - float(predicted)
        denom = max(1e-12, abs(float(pnl)))
        rel_error = abs(residual) / denom

        rows.append(
            [
                name,
                f"{float(pnl):>+,.6f}",
                f"{float(predicted):>+,.6f}",
                f"{float(residual):>+,.6f}",
                f"{100.0 * rel_error:>9.4f}%",
            ]
        )

    col_widths = (
        [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        if rows
        else [len(h) for h in headers]
    )
    print("\n" + " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))))
    print("-+-".join("-" * col_widths[i] for i in range(len(headers))))
    for r in rows:
        print(" | ".join(f"{r[i]:<{col_widths[i]}}" for i in range(len(headers))))


# -----------------------------------------------------------------------------
# Portfolio builders
# -----------------------------------------------------------------------------

def portfolio_spot_only(*, ids: FxVanillaMarketIds) -> Portfolio:
    return Portfolio(positions=[Position(position_id="SPOT", instrument=FxSpot(spot_id=ids.spot_id), quantity=1.0)])


def portfolio_forward_only(*, market, ids: FxVanillaMarketIds, expiry: float, notional: float) -> Portfolio:
    strike = fair_forward_strike(market=market, ids=ids, expiry=expiry)
    return Portfolio(
        positions=[
            Position(
                position_id="FWD",
                instrument=FxForward(
                    notional=notional,
                    strike=strike,
                    expiry=expiry,
                    spot_id=ids.spot_id,
                    domestic_curve_id=ids.domestic_curve_id,
                    foreign_curve_id=ids.foreign_curve_id,
                ),
                quantity=1.0,
            )
        ]
    )


def portfolio_spot_and_forward(*, market, ids: FxVanillaMarketIds, expiry: float) -> Portfolio:
    strike = fair_forward_strike(market=market, ids=ids, expiry=expiry)
    return Portfolio(
        positions=[
            Position(position_id="SPOT", instrument=FxSpot(spot_id=ids.spot_id), quantity=1.0),
            Position(
                position_id="FWD",
                instrument=FxForward(
                    notional=1_000_000.0,
                    strike=strike,
                    expiry=expiry,
                    spot_id=ids.spot_id,
                    domestic_curve_id=ids.domestic_curve_id,
                    foreign_curve_id=ids.foreign_curve_id,
                ),
                quantity=1.0,
            ),
        ]
    )


def portfolio_spot_forward_option(*, market, ids: FxVanillaMarketIds, expiry: float) -> Portfolio:
    spot = float(market.quote(ids.spot_id))
    strike_fwd = fair_forward_strike(market=market, ids=ids, expiry=expiry)

    return Portfolio(
        positions=[
            Position(position_id="SPOT", instrument=FxSpot(spot_id=ids.spot_id), quantity=1.0),
            Position(
                position_id="FWD",
                instrument=FxForward(
                    notional=1_000_000.0,
                    strike=strike_fwd,
                    expiry=expiry,
                    spot_id=ids.spot_id,
                    domestic_curve_id=ids.domestic_curve_id,
                    foreign_curve_id=ids.foreign_curve_id,
                ),
                quantity=1.0,
            ),
            Position(
                position_id="CALL",
                instrument=EuropeanFxOption(
                    option_type="call",
                    notional=1_000_000.0,
                    strike=spot,
                    expiry=expiry,
                    spot_id=ids.spot_id,
                    vol_id=ids.vol_id,
                    domestic_curve_id=ids.domestic_curve_id,
                    foreign_curve_id=ids.foreign_curve_id,
                ),
                quantity=1.0,
            ),
        ]
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ids = build_fx_vanilla_ids()
    market = build_market(ids=ids)

    portfolio_pricer = PortfolioPricer(pricer_registry=DefaultPricerRegistry().build())
    expiry = 1.0

    books: List[Tuple[str, Portfolio, ScenarioPack]] = [
        ("FX Spot only (spot shocks)", portfolio_spot_only(ids=ids), scenario_pack_spot_only(ids=ids)),
        ("FX Forward only (full risk)", portfolio_forward_only(market=market, ids=ids, expiry=expiry, notional=1_000_000.0), scenario_pack_full_risk(ids=ids)),
        ("FX Spot + Forward (full risk)", portfolio_spot_and_forward(market=market, ids=ids, expiry=expiry), scenario_pack_full_risk(ids=ids)),
        ("FX Spot + Forward + Call (full risk)", portfolio_spot_forward_option(market=market, ids=ids, expiry=expiry), scenario_pack_full_risk(ids=ids)),
    ]

    for title, portfolio, scenario_pack in books:
        base_result = portfolio_pricer.price(portfolio, market)
        base_pv = float(base_result.totals.pv)
        base_greeks = dict(base_result.totals.greeks)

        scenario_result = run_portfolio_scenarios(portfolio, market, portfolio_pricer, scenario_pack)

        print_scenario_table(title=title, result=scenario_result)

        print_greeks_vs_scenarios(
            title=title,
            base_pv=base_pv,
            base_greeks=base_greeks,
            base_market=market,
            scenario_pack=scenario_pack,
            scenario_result=scenario_result,
            ids=ids,
        )


if __name__ == "__main__":
    main()