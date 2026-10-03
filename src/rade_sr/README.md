# rade_sr — Static Replication Preprocessing

`rade_sr` turns a target trade portfolio and its market data into the inputs a
downstream model (`rade_ml_pt` → hybrid GNN/RNN) needs to **predict exotic and
linear trade PnL via static replication**.

The core idea of static replication: any target payoff can be approximated by a
weighted basket of simple, liquid *elementary* instruments. `rade_sr` builds that
elementary basket per risk factor, prices it across the same scenario set as the
target trades, and hands the **elementary PnL matrix** to the model. The model
learns the replicating weights; `rade_sr` does **not** solve them.

```
target trades + target PnL  ──►  rade_sr  ──►  per-cluster artifacts (parquet) + jobs.pkl  ──►  rade_ml_pt
                                   │
                                   └─ elementary basis + elementary PnL across scenarios
```

---

## Scope (what this module does and does not do)

**Does:**

1. Ingest a portfolio CSV (trade-level attributes) and a scenario-PnL CSV.
2. Identify the unique **risk factors** and their **dependencies**
   (e.g. FX `GBPUSD` depends on IR `GBP` and IR `USD`).
3. Build a **self-contained market-data object** per risk factor: spot, ATM, smile
   → vol surface/cube, curves, and their **scenario shocks** — every component
   under a typed dataclass contract.
4. Generate a configurable **elementary trade universe** per risk factor
   (FX: forward / vanilla / digital / quanto; IR: swap / swaption / cap-floor).
5. Price the elementary trades as of COB and compute **elementary PnL** across all
   scenarios via compiled (`njit`) batch kernels.
6. Resolve **clusters** (each cluster is single-asset-class) and slice the
   portfolio-level data into per-cluster `ReplicationJob`s.
7. Save four parquet artifacts per cluster + a `jobs.pkl` handoff.

**Does not:**

- Solve the static-replication weights — that is the downstream model's job.
- Fetch market data itself — every `_fetch_*` method is a wiring point to your
  internal market-data / scenario API.

---

## Architecture at a glance

Two phases, **compute once → slice many**:

```
Phase 1 (portfolio-wide, computed once)
  Step 0  Ingest CSVs → identify risk factors + dependency graph → asset_config
  Step 1  Load one self-contained Asset per risk factor (market data + shocks)
  Step 2  Generate the elementary trade universe per risk factor
  Step 3  Compute elementary PnL across all scenarios (batch njit kernels)
  Step 4  Assemble PortfolioData (single source of truth)

Phase 2 (per cluster)
  Step 5  Resolve clusters → {cluster_id: [risk_factors]} (asset-class homogeneous)
  Step 6  Slice PortfolioData + target data; build attribute / PnL DataFrames
  Step 7  Save 4 parquet files per cluster + jobs.pkl
```

Three dictionaries share one **risk-factor key space** the whole way through:

```
assets:            Dict[risk_factor, Asset]
elementary_trades: Dict[risk_factor, List[ElementaryTrade]]
trade_pnls:        Dict[risk_factor, Dict[trade_id, ElementaryTradePnL]]
```

The authoritative, detailed contract lives in [`PIPELINE.md`](./PIPELINE.md).
This README is the orientation; `PIPELINE.md` is the reference.

### Design principles

1. **Compute once, slice many.** All pricing is portfolio-level; clusters get
   pre-computed slices, never re-pricing.
2. **Protocol-driven stages.** Every stage depends on a `Protocol`
   (`core/protocols.py`), not a concrete class — any stage is swappable at
   construction time.
3. **Registry dispatch.** New asset classes register via a decorator
   (`@RiskFactorRegistry.register`); the pipeline never changes.
4. **Typed contracts everywhere.** Market data and shocks are dataclasses that
   carry their own axis labels (tenors/strikes) so arrays are self-documenting
   and shape-validated (`validate_shapes()`).
5. **Self-contained assets.** An `FXAsset` loads its own IR dependencies; the
   pipeline never couples to asset-class-specific loading.

---

## Directory layout

```
src/rade_sr/
├── assets/         Market-data layer: Asset ABC, FXAsset, IRAsset, typed containers + shocks
├── instruments/    Elementary trade specs: InstrumentSpec ABC, FX/IR instruments, grids
├── pricing/        OptionPricer dispatcher + @njit kernels (pricing/kernels/)
├── core/           Domain types (ElementaryTrade, PortfolioData, ReplicationJob), protocols, exceptions
├── replication/    Pipeline engine, orchestrator, registry, builders, trade generator, PnL engine
├── config/         YAML → PipelineConfig loader
├── api/            Internal market-data API client (wiring point)
├── README.md       This file
├── GUIDE.md        How to extend and operate the module
├── PLAN.md         Phased build-out roadmap (status of each milestone)
└── PIPELINE.md     Authoritative data/stage contract
```

---

## Quickstart

> Status: the end-to-end path is being completed in Phase 0 (see `PLAN.md`).
> Until then this is the intended entry point.

```python
from src.rade_sr.replication.orchestrator import build_pipeline, run_preprocessing
from src.rade_sr.replication.pipeline import PipelineConfig
from src.rade_sr.pricing.option_pricer import OptionPricer

pipeline = build_pipeline(
    cluster_manager=my_cluster_manager,
    pricer=OptionPricer(),
    max_workers=8,
)

config = PipelineConfig(
    asset_config=asset_config,    # from Step 0 ingestion of the portfolio CSV
    trade_config=trade_config,    # elementary grids (hybrid: config + observed)
    cluster_config=cluster_config,
    path_config=path_config,
    target_data=target_data,      # target PnL + attributes from the risk system
)

jobs = run_preprocessing(pipeline, config)  # saves parquet artifacts + jobs.pkl
```

Output per run:

```
{root}/
├── {cluster_id}/
│   ├── target_pnl.parquet
│   ├── target_attributes.parquet
│   ├── elem_pnl.parquet
│   └── elem_attributes.parquet
└── jobs.pkl          # List[ReplicationJob] — the handoff to rade_ml_pt
```

---

## Key design decisions

| Decision | Choice |
|----------|--------|
| Replication weight-solving | **Downstream in `rade_ml_pt`.** `rade_sr` emits the elementary basis + PnL only. |
| Elementary grid selection | **Hybrid.** Static config defaults, optionally extended by the target portfolio's observed strikes/tenors so the basis spans the target payoffs. |
| Pricing concurrency | **`njit(nogil=True)` batch kernels + `ThreadPool`** — no Asset pickling. Asset loading (I/O-bound) parallelised separately with threads. |
| Cluster constraint | Each cluster holds exactly **one asset class** (validated; mixed-class clusters raise `PipelineError`). |

See `PLAN.md` for the phased roadmap and `GUIDE.md` for how to extend the module.
