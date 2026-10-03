# rade_sr — Build-Out Plan

Phased roadmap to take the skeleton to a correct, fast, extensible preprocessing
module. Work top-down: each phase leaves the module in a better, runnable state.

**Locked decisions**

- Replication weight-solving lives **downstream in `rade_ml_pt`**. `rade_sr` emits
  the elementary basis + PnL only.
- Elementary grids are **hybrid**: static config defaults, optionally extended by
  the target portfolio's observed strikes/tenors.
- Pricing concurrency: **`njit(nogil=True)` batch kernels + `ThreadPool`** (no
  Asset pickling); asset loading parallelised separately (I/O-bound).

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Phase 0 — Make it run end-to-end (highest priority)

Goal: a synthetic portfolio flows through the whole pipeline to `jobs.pkl` + the
four parquet files per cluster, with **no real market-data API**.

### Raw input schema (locked, from the external risk-system export)

**Attribute file** — raw grain is one row per **(trade, risk_type, curve)**:

```
AssetClass, BuySellInd, CallPut, Date, DeskName, MaturityDate, PTSCurveCode,
PTSDealNumber (trade id), Product, ProductGroup, RiskType, SourceSystemGroup,
StandardCurveCode, StrikePrice, SubDeskName, Sum_RiskValuesUSD_Net, TRBookKey
```

Normalization rules (one row per `PTSDealNumber`):
- static columns → first per trade; **`AssetClass` → first non-`"ALL"`** (the
  `Notional` rows carry `AssetClass="ALL"`).
- `RiskType` → columns, `values=Sum_RiskValuesUSD_Net`, `aggfunc=sum` (across the
  per-curve rows, since the value is per `(trade, risk_type, curve)`).
- `NotionalSign` from `BuySellInd`; `SignedNotional = NotionalSign × Notional`.
- Risk-type set is a **configurable contract** (`FXPV`, `FXPOS`, `Notional` for FX).
- Curve codes are mostly `N/A` → **not** used in logic (kept defensively).
- FX risk factor + IR dependencies inferred via a **pluggable resolver** (currency).

**PnL file** — `PTSDealNumber` index, scenario columns as `YYYYMMDD` dates, plus
an `AssetClass` column.

### Tasks

- [x] **Mock data generator** (`mock/market.py`, `mock/portfolio.py`) — emits both
      raw files in the exact schema; semi-accurate FX PnL via vectorised GK
      (incl. digital + barrier-as-vanilla+digital). FX first; IR-ready structure.
- [x] **Step 0 ingestion** (`replication/ingest.py`) — `normalize_attributes`,
      `extract_asset_config` (risk factors + dependency graph), `load_pnl`,
      `ingest_portfolio`; pluggable `FXRiskFactorResolver`.
- [x] **Example script** — `examples/rade_sr/00_generate_mock_portfolio.py`.
- [x] **Tests** — `tests/rade_sr/test_mock_ingest.py` (10 passing): raw schema,
      `AssetClass="ALL"` handling, risk-type pivot/sum, `NotionalSign`,
      asset_config + dependencies, PnL shape/alignment.
- [x] **Swap seams** — `sources/` package: `MarketDataClient` protocol +
      `MockMarketDataClient`; `PortfolioSource` protocol + Mock/File/Api sources.
      Assets' `_fetch_*` delegate to the client; `InternalAPIClient` aligned to the
      same 10-method surface. Client injected via builders → pipeline (no inline
      construction).
- [x] **Config-driven clustering** — `KeyGroupingClusterResolver` (one_to_one /
      explicit) so the module runs without an external `ClusterManager`.
- [x] **Orchestrated entry** — `orchestrator.run_from_source(portfolio_source,
      market_data_client, ...)`: ingestion → load → trades → PnL → clusters → jobs.
- [x] **End-to-end** — `examples/rade_sr/01_run_pipeline_mock.py` runs green
      (13 FX factors, 1,144 elementary trades, PnL matrices populated).
- [ ] **Target-data flow** — group target PnL + attributes by risk factor; slice
      per cluster in Phase 2 (`PIPELINE.md` §5).
- [ ] **`ReplicationJob` builders** — implement `build_elementary_attributes()`,
      `build_elementary_pnl_df()`, and populate `target_pnl` / `target_attributes`
      in `_populate_clusters`.
- [ ] **Artifact save** — `save_cluster_artifacts(jobs)` (4 parquet/cluster) and
      `save_jobs(jobs, jobs.pkl)`; call them from the orchestrator.
- [ ] **`to_job_dict()`** — finalise the `cluster_info` paths contract expected by
      `rade_ml_pt.build_dataset()`.

Exit: `pytest src/rade_sr` green; mock run produces artifacts + `jobs.pkl`.

---

## Phase 1 — Correctness

Goal: PnL reflects all shocked risk factors, and vol lookups are right.

- [ ] **FX rate shocks applied** — `OptionPricer._scenario_rate` currently drops
      2-D `(n_scenarios, n_pillars)` curve shocks. Interpolate the shocked curve at
      tenor `T` per scenario for domestic + foreign rates.
- [ ] **FX vol shocks applied** — pull shocked vol at `(T, K)` per scenario from
      the 3-D vol-surface shock, not the base vol.
- [ ] **FX vol-surface strike axis** — surface is built on delta-pillar indices
      (`np.arange(n_k)`) but queried with absolute strike `K`. Add a delta↔strike
      mapping (or moneyness axis) so smile lookups are correct.
- [ ] **At-market forward default** — `FXForward.forward_rate` should default to
      `F = S·exp((r_d − r_f)·T)` so the elementary forward's base PV ≈ 0.
- [ ] **IR vol resolution** — replace nearest-neighbour `argmin` cube lookup with
      interpolation across (expiry, swap_tenor, strike).
- [ ] **Cross-check** — batch PnL must match the single-trade `price()` path within
      tolerance (regression test).

Exit: per-trade PnL responds correctly to spot, rate, and vol scenarios; golden
tests pass.

---

## Phase 2 — Performance & concurrency

Goal: replace Python scenario loops with compiled batch kernels and thread-level
parallelism.

- [ ] **Batch kernels** — `price_fx_vanilla_batch`, `price_fx_forward_batch`,
      `price_fx_digital_batch`, `price_ir_swap_vec`, `price_ir_swaption_vec`,
      `price_ir_capfloor_vec`; each loops scenarios (and trades) inside the JIT
      boundary → `(n_trades, n_scenarios)`.
- [ ] **`nogil=True`** on batch kernels.
- [ ] **PnL engine → ThreadPool** — parallelise over factor groups with threads
      (no Asset pickling); drop `ProcessPoolExecutor` from the pricing path.
- [ ] **Asset loading parallelism** — `ThreadPool` over risk factors for the
      I/O-bound `_fetch_*` calls (with a cap to respect API rate limits).
- [ ] **Benchmark** — record before/after on a representative portfolio; document
      in this file.

Exit: order-of-magnitude speedup on the pricing step; deterministic results
unchanged.

---

## Phase 3 — Configurability

Goal: drive everything from config; make instruments as additive as asset classes.

- [ ] **YAML `ConfigLoader`** — fully wire `config/loader.py` → `PipelineConfig`.
- [ ] **Instrument registry** — decorator-based instrument registration so adding
      an instrument is a single registration (no edits to dispatcher / generator).
- [ ] **Hybrid grid derivation** — extend static grids with strikes/tenors observed
      in the target portfolio (`augment_from_target` flag) so the basis spans target
      payoffs; de-duplicate and sort.
- [ ] **Config validation** — clear errors for missing/contradictory settings.

Exit: a run is fully specified by a YAML file + the two input CSVs.

---

## Phase 4 — Extensibility

Goal: broaden coverage now that the patterns are proven.

- [ ] **FX quanto option** — instrument + kernel (needs correlation / quanto drift
      input on `FXAsset`).
- [ ] **FX barrier option** — complete the `FXBarrierOption` stub
      (Reiner–Rubinstein closed form).
- [ ] **Equity asset class** — `EQAsset` + `EQShocks` + instruments + builder.
- [ ] **Credit asset class** — `CRAsset` (credit-spread curve) + instruments.
- [ ] **Richer IR set** — e.g. CMS, basis instruments as needed.

---

## Phase 5 — Hardening & ops

- [ ] **Logging/observability** — per-stage timings, trade/scenario counts,
      replication-error summary (diagnostic only — weights stay downstream).
- [ ] **Artifact versioning** — write a run manifest (config hash, COB date,
      git sha, package versions) alongside `jobs.pkl`.
- [ ] **CI** — run `pytest src/rade_sr` on PRs.
- [ ] **Docs** — keep `README.md` / `GUIDE.md` / `PIPELINE.md` in sync; update this
      plan's checkboxes as phases land.

---

## Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| 0 — End-to-end (mock) | runs green | ingestion + mock generator + tests + swap seams + orchestrated run done; artifact save + target-data attach remain |
| 1 — Correctness | not started | shock application + vol axis |
| 2 — Performance | not started | batch njit + threads |
| 3 — Configurability | not started | YAML + instrument registry + hybrid grids |
| 4 — Extensibility | not started | quanto/barrier/EQ/CR |
| 5 — Hardening | not started | obs/CI/manifest |
