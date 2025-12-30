from __future__ import annotations

from typing import Dict, List

from src.marketdata.ids import MarketId
from src.instruments.fx.options.european import EuropeanFxOption
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider
from src.pricers.analytic.black_scholes import BlackScholesPricer
from src.marketdata.scenarios.shocks import FlatVolShock, ParallelRateShock, SpotShock



def main() -> None:
    """
    End-to-end demo (console output only):
    - Build a Market snapshot via SyntheticProvider
    - Price an FX European call/put via BlackScholesPricer
    - Apply stress scenarios and report PV + PnL in a clean table

    Run
    ---
    python examples/pricers/fx/intermediate/pricing_and_stress.py
    """
    # -------------------------------------------------------------------------
    # 1) Build base market snapshot
    # -------------------------------------------------------------------------
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
    domestic_curve_id = MarketId("IR", "CURVE", "USD.OIS")
    foreign_curve_id = MarketId("IR", "CURVE", "EUR.OIS")

    provider = SyntheticProvider(seed=123)

    base_market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([spot_id, vol_id, domestic_curve_id, foreign_curve_id]),
        )
    )

    # -------------------------------------------------------------------------
    # 2) Define trades (ATM call + put for clean stress behavior)
    # -------------------------------------------------------------------------
    pricer = BlackScholesPricer()

    spot = float(base_market.quote(spot_id))
    expiry_years = 1.0
    strike = spot
    notional = 1_000_000.0

    call = EuropeanFxOption("call", notional, strike, expiry_years, spot_id, vol_id, domestic_curve_id, foreign_curve_id)
    put = EuropeanFxOption("put", notional, strike, expiry_years, spot_id, vol_id, domestic_curve_id, foreign_curve_id)

    # -------------------------------------------------------------------------
    # 3) Print base market inputs (so you can inspect what drove the PV)
    # -------------------------------------------------------------------------
    sigma = float(base_market.vol_surface(vol_id).vol(expiry=expiry_years, strike=strike))
    df_d = float(base_market.curve(domestic_curve_id).df(expiry_years))
    df_f = float(base_market.curve(foreign_curve_id).df(expiry_years))

    print("\nBase Inputs")
    print("----------")
    print(f"AsOf: 2025-12-29")
    print(f"Spot: {spot:.6f}")
    print(f"Strike: {strike:.6f}  Expiry: {expiry_years:.2f}y  Notional: {notional:,.0f}")
    print(f"Vol(sigma): {sigma:.6f}")
    print(f"DF_domestic(T): {df_d:.8f}")
    print(f"DF_foreign(T):  {df_f:.8f}")

    # -------------------------------------------------------------------------
    # 4) Compute base PV + base Greeks (printed once)
    # -------------------------------------------------------------------------
    base_call_pv = float(pricer.price(call, base_market))
    base_put_pv = float(pricer.price(put, base_market))

    base_call_greeks = pricer.greeks(call, base_market)
    base_put_greeks = pricer.greeks(put, base_market)

    print("\nBase PV + Greeks (for inspection)")
    print("--------------------------------")
    base_rows = [
        {
            "trade": "CALL",
            "pv": f"{base_call_pv:.2f}",
            "delta": f"{base_call_greeks['delta']:.2f}",
            "gamma": f"{base_call_greeks['gamma']:.6f}",
            "vega": f"{base_call_greeks['vega']:.2f}",
        },
        {
            "trade": "PUT",
            "pv": f"{base_put_pv:.2f}",
            "delta": f"{base_put_greeks['delta']:.2f}",
            "gamma": f"{base_put_greeks['gamma']:.6f}",
            "vega": f"{base_put_greeks['vega']:.2f}",
        },
    ]
    _print_table(base_rows, headers=["trade", "pv", "delta", "gamma", "vega"])

    # -------------------------------------------------------------------------
    # 5) Define scenarios (each returns a shocked market view)
    # -------------------------------------------------------------------------
    scenario_shocks = {
        "spot_up_1pct": SpotShock(name="spot_up_1pct", spot_id=spot_id, bump=0.01, bump_mode="relative"),
        "vol_up_1volpt": FlatVolShock(name="vol_up_1volpt", vol_id=vol_id, vol_bump=0.01),
        "domestic_rate_up_100bp": ParallelRateShock(name="domestic_rate_up_100bp", curve_id=domestic_curve_id, rate_shift=0.01),
    }

    # -------------------------------------------------------------------------
    # 6) Stress pricing report: PV + PnL under each scenario
    # -------------------------------------------------------------------------
    stress_rows: List[Dict[str, str]] = []

    for scenario_name, shock in scenario_shocks.items():
        shocked_market = shock.apply(base_market)

        call_pv = float(pricer.price(call, shocked_market))
        put_pv = float(pricer.price(put, shocked_market))

        stress_rows.append(
            {
                "scenario": scenario_name,
                "call_pv": f"{call_pv:.2f}",
                "call_pnl": f"{(call_pv - base_call_pv):.2f}",
                "put_pv": f"{put_pv:.2f}",
                "put_pnl": f"{(put_pv - base_put_pv):.2f}",
            }
        )

    print("\nStress Report (PV and PnL vs base)")
    print("---------------------------------")
    _print_table(stress_rows, headers=["scenario", "call_pv", "call_pnl", "put_pv", "put_pnl"])


def _print_table(rows: List[Dict[str, str]], headers: List[str]) -> None:
    """
    Print a clean aligned table with no third-party dependencies.
    """
    col_width = {h: max(len(h), *(len(r.get(h, "")) for r in rows)) for h in headers}

    header_line = " | ".join(h.ljust(col_width[h]) for h in headers)
    separator = "-+-".join("-" * col_width[h] for h in headers)

    print(header_line)
    print(separator)

    for r in rows:
        print(" | ".join(r.get(h, "").ljust(col_width[h]) for h in headers))


if __name__ == "__main__":
    main()