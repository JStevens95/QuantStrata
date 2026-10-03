# rade_static_replication

Preprocessing library that turns a raw derivatives portfolio into the
**elementary-basis PnL tensors** consumed by the `rade_ml` hybrid GNN-RNN model,
using the theory of **static replication**.

> A portfolio's exotic/linear trades are replicated by a basis of elementary
> instruments (FX vanillas/digitals/forwards, swaptions, …). If the model learns the
> mapping from *elementary* scenario PnL to *target* scenario PnL on historical
> shocks, it can predict target PnL under new shocks. This library produces, per
> cluster: the elementary basis PnL, the target PnL, and aligned attributes.

---

## 1. What it produces

For every **cluster** (a single-asset-class slice of the portfolio) it writes the four
files `rade_ml` expects, plus a fully auditable run directory:

```
<output.root>/<output.run_id>/
├── run_manifest.json          # version, git sha, cob date, counts, timings, status
├── config.snapshot.yaml       # the exact config used (reproducibility)
├── logs/run.log
├── portfolio/
│   ├── attributes.parquet     # normalised portfolio (audit copy)
│   └── universe.json          # resolved risk factors + dependency graph
├── clusters/<cluster_id>/
│   ├── elementary_pnl.parquet     # [scenarios × elementary-trade-ids]
│   ├── elementary_attributes.pkl  # column-oriented dict
│   ├── target_pnl.parquet         # [scenarios × target-trade-ids]
│   └── target_attributes.pkl
└── jobs.pkl                   # manifest of cluster paths (rade_ml entry point)
```

---

## 2. Architecture (separation of concerns)

Dependencies point **inward**. The pure core knows nothing about I/O or asset classes;
infrastructure and asset plugins depend on the core, never the reverse.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         api.py  (public facade)                        │
├──────────────────────────────────────────────────────────────────────┤
│  pipeline/      orchestrator · context · stages/* · engine/            │  application
├───────────────┬──────────────────────────┬───────────────────────────┤
│  assets/      │     portfolio/           │       artifacts/           │
│  fx/ rates/   │   normalise·validate·    │  layout·store·writers·     │  domain services
│  (plugins)    │   resolution/            │  manifest                  │
├───────────────┴──────────────────────────┴───────────────────────────┤
│  domain/        contracts · instruments · enums · errors               │
│  marketdata/    base·snapshot·scenarios · common/ · fx/ · rates/        │  pure core
│  pricing/       kernels/ (fx, rates, math)                             │
├──────────────────────────────────────────────────────────────────────┤
│  clients/       base (ports) · payloads · mock/ · file/ · api/         │  infrastructure
└──────────────────────────────────────────────────────────────────────┘
```

### Folder map

| Path | Responsibility | You touch it to… |
|------|----------------|------------------|
| `domain/` | Stage contracts, `ElementaryTrade`, enums, errors. Pure. | Add a new contract field |
| `marketdata/base·snapshot·scenarios` | Shared spine (interpolation, conventions, validation) + abstract `MarketSnapshot`/`ScenarioSet`. | Strengthen market modelling |
| `marketdata/common/` | The one cross-asset instrument: `DiscountCurve`. | Refine shared discounting |
| `marketdata/<class>/` | **Per-asset instruments + snapshot + scenarios** (FX: `Spot`/`VolSurface`; rates: `VolCube`). | **Add an asset class's market data** |
| `pricing/kernels/` | `@njit` numerics (FX GK, Bachelier). Primitive in/out. | Add a closed-form/kernel |
| `assets/<class>/` | **Per-asset plugin**: `builder · instruments · generator · pricer`. | **Add/extend an asset class** |
| `portfolio/` | Normalise, validate, resolve risk factors. | Change ingestion/resolution |
| `config/` | `OrchestratorConfig` schema + YAML loader. | Add a config knob |
| `pipeline/` | Orchestrator, `RunContext`, one file per `stages/`, PnL `engine/`. | Re-order/inspect stages |
| `clients/` | **Ports** (`base.py`) + adapters: `mock/`, `file/`, `api/`. | **Wire Sage/STAR** |
| `artifacts/` | Auditable run store (layout, manifest, writers). | Change output layout |

---

## 3. The pipeline (stage contracts)

Each stage is a pure function consuming one contract and emitting the next; the
`Orchestrator` records each on the `RunContext` with timings.

```
client ─▶ RawPortfolio ─▶ Portfolio ─▶ RiskFactorUniverse ─▶ FactorDataSet
   load      normalise+validate   resolve        build_market (dependency order)
        ─▶ ElementaryUniverse ─▶ BasePriceSet ─▶ PnLResult ─▶ ClusterSet ─▶ artifacts
            generate              price          pnl(engine)   cluster        write
```

| Stage | Function | In → Out |
|------|----------|----------|
| 1 load | `stages.load_portfolio` | client → `RawPortfolio` |
| 2 normalise | `stages.normalise_portfolio` | `RawPortfolio` → `Portfolio` |
| 3 resolve | `stages.resolve_universe` | `Portfolio` → `RiskFactorUniverse` |
| 4 build | `stages.build_factor_data` | universe → `FactorDataSet` |
| 5 generate | `stages.generate_elementary` | → `ElementaryUniverse` |
| 6 price | `stages.price_base` | → `BasePriceSet` |
| 7 pnl | `stages.compute_pnl` | → `PnLResult` (threaded) |
| 8 cluster | `stages.resolve_clusters` | → `ClusterSet` |
| write | `Orchestrator.write` | → `ArtifactManifest` |

---

## 4. Quickstart

```bash
python -m src.rade_static_replication.examples.run_pipeline_mock
```

```python
from src.rade_static_replication import load_config, run
from src.rade_static_replication.clients.mock import MockPortfolioClient, MockMarketDataClient

config = load_config("src/rade_static_replication/configs/orchestrator.yaml")
ctx = run(config, MockPortfolioClient(), MockMarketDataClient())
# ctx.clusters, ctx.elementary, ctx.timings_ms ; artifacts under config.output
```

Step-by-step (debugging / partial runs):

```python
from src.rade_static_replication import Orchestrator
orch = Orchestrator(config, MockPortfolioClient(), MockMarketDataClient())
orch.load(); orch.normalise(); orch.resolve()
print(orch.ctx.universe.specs)        # inspect before building market data
orch.build(); orch.generate(); orch.price(); orch.pnl(); orch.cluster(); orch.write()
```

---

## 5. Market-data layer (the "strong base")

The layout mirrors `assets/`: a shared spine + shared primitives, then asset-class
subpackages for the objects that genuinely differ per asset class.

```
marketdata/
├── base.py            # Interpolator strategy · validators · conventions · year_fraction · MarketObject
├── snapshot.py        # MarketSnapshot   (abstract base)
├── scenarios.py       # ScenarioSet      (abstract base)
├── shocks.py          # ShockMode / ShockConvention — relative↔absolute resolution
├── common/            # the ONE cross-asset instrument
│   └── curves.py      #   DiscountCurve  (df/zero/forward; LINEAR_ZERO | LOG_LINEAR_DF)
├── fx/                # ASSET-SPECIFIC — instruments + snapshot + scenarios
│   ├── instruments.py #   Spot · VolSurface (2-D, lognormal, moneyness/delta/absolute)
│   ├── snapshot.py    #   FXSnapshot     (spot + dom/for curves + surface; forward() via CIP)
│   └── scenarios.py   #   FXScenarioSet  (validate_against the snapshot grids)
└── rates/
    ├── instruments.py #   VolCube        (3-D expiry×tenor×strike, normal/bp)
    ├── snapshot.py    #   RatesSnapshot  (curve + optional cube)
    └── scenarios.py   #   RatesScenarioSet
```

`base.py` carries the spine so the objects stay small and consistent:

- **`Interpolator`** strategy (`LinearInterpolator` today; `total_variance_interp` for
  surfaces) — change interpolation policy in one place.
- **Validation helpers** (`require_increasing`, `require_shape`, …) — every object is
  self-checking in `__post_init__`.
- **`year_fraction`** — single day-count implementation.
- **`MarketObject`** protocol (`label` / `validate` / `summary`) for typing + audit.

**Why split by asset class:** market instruments differ in *shape and convention* across
asset classes — the FX `VolSurface` is 2-D (lognormal, strike-by-moneyness) while the
rates `VolCube` is 3-D (normal/bp, expiry×tenor×strike). Each therefore lives with its
asset class in `marketdata/<class>/instruments.py`, alongside that class's `snapshot` and
`scenarios`. The **only** genuinely cross-asset instrument is the `DiscountCurve` (an FX
factor needs two; a rates factor needs one) — it is the same object everywhere, so it sits
in `marketdata/common/` rather than being duplicated. `ScenarioSet`s hold the **resolved
absolute** shocked states and `validate_against` the snapshot grids; relative-vs-absolute
conventions are first-class in `marketdata/shocks.py`.

---

## 6. Extending: add an asset class

1. Add the asset's market objects in `marketdata/<class>/`: its `instruments.py` (the
   asset-specific quote/surface/cube), `snapshot.py`, and `scenarios.py`. Reuse
   `marketdata/common/` for discounting; add to `common/` only if a new instrument is
   genuinely cross-asset.
2. Create `assets/<class>/` with four files implementing the protocols in
   `assets/base.py`:
   - `builder.py` → `RiskFactorBuilder` (client payloads → snapshot + scenarios)
   - `instruments.py` + `generator.py` → `ElementaryGenerator` (the replicating grid)
   - `pricer.py` → `Pricer` (`base_prices` + vectorised `scenario_pnl`)
3. Bundle in `assets/<class>/__init__.py` as `<CLASS>_PLUGIN = AssetClassPlugin(...)`.
4. Register one line in `assets/registry.py`.
5. Add a resolution rule + elementary grid block to the config, and the client methods
   that supply the new asset's payloads.

Because each plugin is isolated, FX changes can never break Rates, and vice versa.

---

## 7. Wiring your environment (Sage / STAR)

The pipeline depends only on two **ports** in `clients/base.py`. Implement these and
everything else is unchanged:

- `clients/api/sage.py :: SagePortfolioClient.load(cob_date) -> RawPortfolio`
  — return the exploded attributes frame + scenario-PnL frame.
- `clients/api/star.py :: StarMarketDataClient` — implement the six methods returning
  the payloads in `clients/payloads.py`.

The **mock client is the reference implementation**: its array shapes are exactly the
contract your adapters must satisfy. For offline runs from exported files, use
`clients/file/` (`FilePortfolioClient` is ready; `FileMarketDataClient` is a stub).

Raw column names are centralised in `portfolio/normalise.py` — point them at your
export if it differs.

---

## 8. Configuration

See `configs/orchestrator.yaml` (block order mirrors the stages) and
`configs/fx_mapping.csv`. Risk-factor resolution is config-driven per asset class under
`factor_config` (each sub-key like `fx_config` yields one class, trailing `_config`
stripped):

- `source: file` (or `mapping`) — CSV lookup. `attrs_key` is the **portfolio** column
  holding the key; `mapping_key` is the **CSV** key column (they may differ, e.g.
  `CurrencyCode` vs `currency`); `mapping_file` is the CSV path.
- `source: attribute` — factor ids already on the trade row.
- `factor_cols` are the primary factors; `dependency_cols` are built-but-not-priced
  factors (e.g. the IR curves an FX factor needs).
- A dependency's asset class is inferred from its factor-id prefix (`IR_* → rates`,
  `FX_* → fx`); pin it explicitly with `asset_class_of: {col: class}` when your plugin
  class names differ.

---

## 9. Testing & dependencies

```bash
python -m pytest src/rade_static_replication/tests -q
```

Runtime deps: `numpy`, `pandas`, `pyyaml`, `pyarrow`. `numba` is **optional** — the shim
in `pricing/kernels/_numba.py` falls back to pure Python when it is absent. See
`pyproject.toml` and `docs/DESIGN.md` for the deeper rationale.
