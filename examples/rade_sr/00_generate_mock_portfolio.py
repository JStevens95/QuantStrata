"""
Generate a mock FX desk portfolio in the raw risk-system export schema, write the
two CSVs to disk, and demonstrate the Step 0 ingestion round-trip.

Run from the repository root:

    python examples/rade_sr/00_generate_mock_portfolio.py

Outputs (under data/mock/rade_sr/ by default):
    portfolio_attributes.csv   raw, exploded: one row per (trade, risk_type, curve)
    portfolio_pnl.csv          trade-id index, scenario-date columns, + AssetClass
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.rade_sr.mock.portfolio import generate_mock_portfolio
from src.rade_sr.replication.ingest import ingest_portfolio


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock rade_sr portfolio files")
    parser.add_argument("--n-trades", type=int, default=2500)
    parser.add_argument("--n-scenarios", type=int, default=250)
    parser.add_argument("--cob-date", type=str, default="2026-05-29")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data/mock/rade_sr")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating mock portfolio: {args.n_trades} trades, {args.n_scenarios} scenarios...")
    attributes, pnl = generate_mock_portfolio(
        n_trades=args.n_trades,
        n_scenarios=args.n_scenarios,
        cob_date=args.cob_date,
        seed=args.seed,
    )

    attr_path = out_dir / "portfolio_attributes.csv"
    pnl_path = out_dir / "portfolio_pnl.csv"
    attributes.to_csv(attr_path, index=False)
    pnl.to_csv(pnl_path)
    print(f"  wrote {attr_path}  ({len(attributes):,} rows)")
    print(f"  wrote {pnl_path}  ({len(pnl):,} trades x {args.n_scenarios} scenarios)")

    # ── Raw attribute preview ──────────────────────────────────────────
    print("\n── Raw attribute frame (head) ─────────────────────────────")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(attributes.head(8).to_string(index=False))
    print(f"\n  unique trades       : {attributes['PTSDealNumber'].nunique():,}")
    print(f"  total raw rows      : {len(attributes):,}")
    print(f"  RiskType values     : {sorted(attributes['RiskType'].unique())}")
    print(f"  AssetClass values   : {sorted(attributes['AssetClass'].unique())}")
    print(f"  rows on Notional    : {(attributes['RiskType'] == 'Notional').sum():,}"
          f" (all AssetClass='ALL')")

    # ── Step 0 ingestion round-trip ────────────────────────────────────
    print("\n── Step 0 ingestion ───────────────────────────────────────")
    inputs = ingest_portfolio(attributes, pnl)
    norm = inputs.attributes
    print(f"  normalized trades   : {len(norm):,} (one row per trade)")
    print(f"  risk-type columns   : "
          f"{[c for c in ('FXPV', 'FXPOS', 'Notional') if c in norm.columns]}")
    print(f"  NotionalSign values : {sorted(norm['NotionalSign'].unique())}")
    print(f"  unique risk factors : {len(inputs.asset_config)}")
    print("\n  asset_config sample:")
    for rf, cfg in list(inputs.asset_config.items())[:5]:
        deps = inputs.dependency_graph[rf]
        print(f"    {rf:8s} -> {cfg}   deps={deps}")

    print("\n  normalized attributes (head):")
    cols = [c for c in ("AssetClass", "DeskName", "Product", "ProductGroup",
                        "NotionalSign", "FXPV", "FXPOS", "Notional",
                        "SignedNotional", "expiry_years") if c in norm.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(norm[cols].head(8).to_string())

    print(f"\n  target PnL matrix    : {inputs.target_pnl.shape} (trades x scenarios)")
    print(f"  total portfolio PnL  : {inputs.target_pnl.values.sum():,.0f} USD (summed)")
    print("\nDone.")


if __name__ == "__main__":
    main()
