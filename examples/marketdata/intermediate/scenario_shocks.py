from __future__ import annotations

from typing import Dict, List

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.scenarios.shocks import FlatVolShock, ParallelRateShock, SpotShock
from src.marketdata.synthetic.provider import SyntheticProvider


def main() -> None:
    """
    Demonstrate scenario shocks on a Market snapshot (console output only).

    This example focuses on market-data inspection:
    - build a base Market
    - apply a few shocks (spot / vol / curve)
    - print base vs shocked values in a clean table

    Run
    ---
    python examples/marketdata/intermediate/scenario_shocks.py
    """
    # -------------------------------------------------------------------------
    # 1) Build a deterministic base Market snapshot (offline)
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

    # Choose a single (T, K) pair purely for demonstration of vol lookup.
    T = 1.0
    K = float(base_market.quote(spot_id))

    # -------------------------------------------------------------------------
    # 2) Define shocks
    # -------------------------------------------------------------------------
    scenario_shocks = {
        "spot_up_1pct": SpotShock(name="spot_up_1pct", spot_id=spot_id, bump=0.01, bump_mode="relative"),
        "vol_up_1volpt": FlatVolShock(name="vol_up_1volpt", vol_id=vol_id, vol_bump=0.01),
        "domestic_rate_up_100bp": ParallelRateShock(name="domestic_rate_up_100bp", curve_id=domestic_curve_id, rate_shift=0.01),
    }

    # -------------------------------------------------------------------------
    # 3) Print a quick "base inputs" snapshot
    # -------------------------------------------------------------------------
    base_spot = float(base_market.quote(spot_id))
    base_vol = float(base_market.vol_surface(vol_id).vol(expiry=T, strike=K))
    base_df_domestic = float(base_market.curve(domestic_curve_id).df(T))
    base_df_foreign = float(base_market.curve(foreign_curve_id).df(T))

    print("\nBase Market Snapshot")
    print("--------------------")
    print(f"AsOf: 2025-12-29")
    print(f"Spot  ({spot_id}): {base_spot:.6f}")
    print(f"Vol   ({vol_id}) at (T={T:.2f}, K={K:.6f}): {base_vol:.6f}")
    print(f"DF_d  ({domestic_curve_id}) at T={T:.2f}: {base_df_domestic:.8f}")
    print(f"DF_f  ({foreign_curve_id}) at T={T:.2f}: {base_df_foreign:.8f}")

    # -------------------------------------------------------------------------
    # 4) Build a report table: base vs each shocked market
    # -------------------------------------------------------------------------
    rows: List[Dict[str, str]] = []

    # Base row
    rows.append(
        {
            "scenario": "base",
            "spot": f"{base_spot:.6f}",
            "vol(T,K)": f"{base_vol:.6f}",
            "df_dom(T)": f"{base_df_domestic:.8f}",
            "df_for(T)": f"{base_df_foreign:.8f}",
        }
    )

    # Shocked rows
    for scenario_name, shock in scenario_shocks.items():
        shocked_market = shock.apply(base_market)

        shocked_spot = float(shocked_market.quote(spot_id))
        shocked_vol = float(shocked_market.vol_surface(vol_id).vol(expiry=T, strike=K))
        shocked_df_domestic = float(shocked_market.curve(domestic_curve_id).df(T))
        shocked_df_foreign = float(shocked_market.curve(foreign_curve_id).df(T))

        rows.append(
            {
                "scenario": scenario_name,
                "spot": f"{shocked_spot:.6f}",
                "vol(T,K)": f"{shocked_vol:.6f}",
                "df_dom(T)": f"{shocked_df_domestic:.8f}",
                "df_for(T)": f"{shocked_df_foreign:.8f}",
            }
        )

    print("\nScenario Shock Report (market inputs)")
    print("-------------------------------------")
    _print_table(rows, headers=["scenario", "spot", "vol(T,K)", "df_dom(T)", "df_for(T)"])


def _print_table(rows: List[Dict[str, str]], headers: List[str]) -> None:
    """
    Print a small aligned table with no external dependencies (no pandas).
    """
    # Compute per-column widths so the table aligns nicely.
    col_width = {h: max(len(h), *(len(r.get(h, "")) for r in rows)) for h in headers}

    # Header row + separator row.
    header_line = " | ".join(h.ljust(col_width[h]) for h in headers)
    separator = "-+-".join("-" * col_width[h] for h in headers)

    print(header_line)
    print(separator)

    # Data rows.
    for r in rows:
        print(" | ".join(r.get(h, "").ljust(col_width[h]) for h in headers))


if __name__ == "__main__":
    main()