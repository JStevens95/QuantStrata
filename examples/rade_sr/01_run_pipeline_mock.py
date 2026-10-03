"""
End-to-end rade_sr run through the orchestrator, using injected mock providers.

Demonstrates the two swap seams:
  - PortfolioSource   (raw attribute + PnL frames)
  - MarketDataClient  (spot / curves / vol / shocks)

To run against your real environment, replace the two mock objects with your
implementations of `PortfolioSource` and `MarketDataClient` — nothing else changes:

    portfolio_source   = ApiPortfolioSource(my_client, portfolio_id="FX_BOOK")
    market_data_client = InternalAPIClient(base_url=..., auth_token=...)

Run from the repository root:

    python examples/rade_sr/01_run_pipeline_mock.py
"""
from __future__ import annotations

from src.rade_sr.contracts import run_all
from src.rade_sr.replication.orchestrator import run_from_source
from src.rade_sr.sources import MockMarketDataClient, MockPortfolioSource

OUTPUT_DIR = "data/rade_sr/mock_run"

N_TRADES = 600
N_SCENARIOS = 120
COB = "2026-05-29"

TRADE_CONFIG = {
    "notional": 1.0,
    "maturities": [0.25, 0.5, 1.0, 2.0],
    "fx": {
        "strike_pcts": [0.90, 0.95, 1.00, 1.05, 1.10],
        "include_forwards": True,
        "include_vanillas": True,
        "include_digitals": True,
    },
}


def main() -> None:
    # ── Inject providers (swap these two for your real implementations) ──
    portfolio_source = MockPortfolioSource(
        n_trades=N_TRADES, n_scenarios=N_SCENARIOS, cob_date=COB, seed=42,
    )
    market_data_client = MockMarketDataClient(
        n_scenarios=N_SCENARIOS, cob_date=COB, seed=99,
    )

    # ── One orchestrated call: ingest → load → trades → PnL → clusters → artifacts ──
    result = run_from_source(
        portfolio_source=portfolio_source,
        market_data_client=market_data_client,
        trade_config=TRADE_CONFIG,
        cluster_config={"method": "one_to_one"},
        max_workers=1,
        output_dir=OUTPUT_DIR,
    )

    inputs, jobs = result.inputs, result.jobs

    print("── Ingestion ──────────────────────────────────────────────")
    print(f"  trades              : {len(inputs.attributes):,}")
    print(f"  risk factors        : {sorted(inputs.asset_config)}")
    print(f"  target PnL matrix   : {inputs.target_pnl.shape}")

    print("\n── Pipeline output (one job per cluster) ──────────────────")
    print(f"  clusters            : {len(jobs)}")
    total_elem = sum(j.n_trades for j in jobs)
    print(f"  elementary trades   : {total_elem:,}")
    for job in jobs[:5]:
        print(f"    {job.cluster_id:8s}  factors={job.risk_factor_ids}  "
              f"elem_trades={job.n_trades}  scenarios={job.n_scenarios}")

    # sanity: elementary PnL matrix for the first cluster
    j0 = jobs[0]
    pnl = j0.pnl_matrix()
    print(f"\n  {j0.cluster_id} elementary PnL matrix: {pnl.shape} "
          f"(trades x scenarios)")
    print(f"  mean |PnL| = {abs(pnl).mean():,.4f}")

    print(f"\n── Artifacts (rade_ml_pt handoff) ─────────────────────────")
    print(f"  manifest : {result.jobs_manifest_path}")

    # ── Conformance: prove the seams + the written artifacts hold the contract ──
    run_all(
        market_data_client=market_data_client,
        portfolio_source=portfolio_source,
        pricer=None,  # uses the default OptionPricer inside the run already
        artifacts_dir=OUTPUT_DIR,
    )

    print("Done.")


if __name__ == "__main__":
    main()
