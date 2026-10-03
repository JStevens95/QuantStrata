# rade_sr — Shippability Guide

What you can drop into your work environment now, and what needs wiring/testing
against your real systems. Read alongside `GUIDE.md` (how to wire) and `README.md`
(architecture).

## The two swap seams

Everything external goes through **two protocols**. Swap mock → production by
passing different objects to `run_from_source(...)` — no pipeline code changes.

| Seam | Protocol | Mock (offline) | Production (you implement) |
|------|----------|----------------|----------------------------|
| Raw portfolio (attributes + scenario PnL) | `sources.PortfolioSource` | `MockPortfolioSource` | `FilePortfolioSource` (CSV/Parquet) or `ApiPortfolioSource` |
| Market data (spot / curves / vol / shocks) | `sources.MarketDataClient` | `MockMarketDataClient` | `api.InternalAPIClient` (10 methods to wire) |

```python
from src.rade_sr.replication.orchestrator import run_from_source
result = run_from_source(
    portfolio_source   = my_portfolio_source,    # PortfolioSource
    market_data_client = my_market_data_client,  # MarketDataClient
    trade_config       = {...},
    output_dir         = "data/rade_sr/run_001", # writes the rade_ml_pt handoff
)
# result.jobs_manifest_path  ->  jobs.pkl consumed by rade_ml_pt.build_dataset()
```

## Validate before you run (conformance harness)

Wiring is a pass/fail checklist, not a debugging session. Run this in your env
after wiring each piece:

```python
from src.rade_sr.contracts import run_all
run_all(
    portfolio_source   = my_portfolio_source,
    market_data_client = my_market_data_client,  # loads + validates an FXAsset
    pricer             = my_pricer,               # checks (n_trades, n_scen) finite
    artifacts_dir      = "data/rade_sr/run_001",  # checks the on-disk contract
)
```

It exercises every `get_*` method, the asset axis-alignment `validate()`, the
pricer shape, and reads the written artifacts back to confirm the
`rade_ml_pt.build_dataset()` contract holds.

## Artifact contract (rade_sr → rade_ml_pt)

`run_from_source(output_dir=...)` writes, per cluster, the four files
`build_dataset()` reads, plus a `jobs.pkl` manifest. Verified against
`rade_ml_pt/data/hybrid_gnn_rnn/build.py` + `data/io.py`:

| File | Format | Shape / structure |
|------|--------|-------------------|
| `elementary_pnl.parquet` / `target_pnl.parquet` | Parquet | `[scenarios × trade-ids]`; index = scenarios. Elementary column ids keep the `UND\|TYPE\|...` shape. |
| `elementary_attributes.pkl` / `target_attributes.pkl` | Pickled `dict[str, list]` | column-oriented; one list per attribute; includes `trade_id` + `yrs_to_maturity`. |
| `jobs.pkl` | Pickled `list[dict]` | each has `cluster_info` (the 4 paths) — the `job` dicts `build_dataset` takes. |

**You must check:** the attribute keys cover your `AttributeEncoderConfig`
(`numeric_keys` / `categorical_keys` / `multi_label_keys`), and the elementary
shock scenarios and target PnL **share the same scenario (time) axis** — the
harness warns if the row counts differ.

## Shippability matrix

### 🟢 Ship now — pure, environment-agnostic, tested

| Component | Notes |
|-----------|-------|
| `core/` (types, protocols, exceptions) | Pure dataclasses + Protocols. Zero external deps. |
| `replication/ingest.py` (Step 0) | Normalises your raw schema. Covered by `tests/rade_sr/test_mock_ingest.py`. The only env-specific bit is the `FXRiskFactorResolver` (Product→pair rule) — swap when your convention is final. |
| `sources/portfolio_source.py` | `FilePortfolioSource` reads your exported CSV/Parquet today. |
| `replication/orchestrator.py` (`run_from_source`) | Clean DI entry point; proven end-to-end with mocks. |
| `replication/pipeline.py`, `cluster_resolver.py` (KeyGrouping), `path_resolver.py`, `portfolio.py` | Orchestration is protocol-only; no I/O assumptions. |
| `instruments/`, `trade_generator.py` | Elementary-trade construction is deterministic config. |
| `pricing/kernels/_numba.py` | Graceful fallback if `numba` is absent. |
| `replication/artifacts.py` | Writes the 4-file + `jobs.pkl` handoff in the exact `build_dataset()` format; tested by `tests/rade_sr/test_pipeline_artifacts.py`. |
| `contracts/` (conformance harness) | Validates your client/source/pricer/artifacts in your env. |

### 🟡 Light wiring — implement a documented contract, then runs

| Component | Work required |
|-----------|---------------|
| `api/client.py` (`InternalAPIClient`) | Implement the 10 `get_*` methods against your market-data APIs. Each docstring states the exact return dict (mirrors `MockMarketDataClient`). This is the **main** integration task. |
| `sources/portfolio_source.py` (`ApiPortfolioSource`) | Implement `load_attributes()` / `load_pnl()` if you pull the portfolio from an API rather than files. |
| `replication/cluster_resolver.py` (`ClusterResolver`) | Wrap your real `ClusterManager`, OR keep `KeyGroupingClusterResolver` (config-driven, no dependency). |
| Asset `validate()` axis contracts | Your real shocks must align axes (shock vol expiries == surface tenors; shock curve tenors == curve tenors). `validate()` will tell you if they don't. |

### 🔴 Needs review/testing before production numbers are trusted

These run today (the plumbing is correct) but have **known finance-correctness
gaps** tracked in `PLAN.md` Phase 1. Treat current PnL as structurally-correct,
not value-correct.

| Component | Issue (Phase 1) |
|-----------|-----------------|
| `pricing/option_pricer.py` FX path | Vol-surface strike axis is delta-index but queried with absolute strike → extrapolation. Vol & rate scenario shocks (3-D / 2-D) are currently dropped; only spot shocks feed FX PnL. |
| FX shock application | Domestic/foreign rate and vol shocks need to be threaded into the batch pricer. |
| Scenario-axis alignment | Elementary shocks and target PnL must share the scenario/time axis in your real data (the harness warns on mismatch). |

## Recommended adoption path (deadline order)

1. **Ingestion.** Point `FilePortfolioSource` at a real export; run
   `check_portfolio_source(...)`. Finalise the FX risk-factor resolver
   (Product→pair rule).
2. **Wire `InternalAPIClient`** (10 methods). Run `check_market_data_client(...)`
   until `FXAsset load + validate` is green.
3. **Plug your pricer** (similar to rade_sr's) and run `check_pricer(...)`, or use
   rade_sr's kernels as-is for v1.
4. **`run_from_source(output_dir=...)`** with `max_workers=1` → produces
   `jobs.pkl` + per-cluster artifacts. Run `check_artifacts_roundtrip(...)`.
5. **Map attribute keys** to your `AttributeEncoderConfig`; feed `jobs.pkl` to
   `rade_ml_pt.build_dataset()`.
6. *(Post-deadline)* Phase 1 pricing-correctness fixes before relying on PnL values.
