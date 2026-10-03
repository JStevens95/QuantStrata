"""
End-to-end example: run the full pipeline against the mock clients.

    python -m src.rade_static_replication.examples.run_pipeline_mock

Swap ``MockPortfolioClient`` / ``MockMarketDataClient`` for the Sage / STAR adapters
(or the file clients) and everything else stays the same.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.rade_static_replication import load_config, run
from src.rade_static_replication.clients.mock import MockMarketDataClient, MockPortfolioClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "orchestrator.yaml"


def main() -> None:
    config = load_config(CONFIG_PATH)
    ctx = run(config, MockPortfolioClient(), MockMarketDataClient())

    print("\n=== run summary ===")
    print(f"trades             : {len(ctx.portfolio.trade_ids)}")
    print(f"risk factors       : {len(ctx.universe.specs)} "
          f"({len(ctx.universe.primary_ids)} primary)")
    print(f"elementary trades  : {ctx.elementary.n_trades}")
    print(f"clusters           : {len(ctx.clusters.clusters)}")
    for c in ctx.clusters.clusters:
        print(f"  - {c.cluster_id}: {len(c.target_trade_ids)} targets, "
              f"{len(c.risk_factor_ids)} factors")
    print(f"artifacts          : {ctx.config.output.root}/{ctx.config.output.run_id}")
    print(f"timings (ms)       : { {k: round(v, 1) for k, v in ctx.timings_ms.items()} }")


if __name__ == "__main__":
    main()
