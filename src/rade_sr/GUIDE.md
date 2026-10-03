# rade_sr — Developer & Operator Guide

Practical how-to for extending and running `rade_sr`. For the *what/why* see
[`README.md`](./README.md); for the authoritative contract see
[`PIPELINE.md`](./PIPELINE.md); for the roadmap see [`PLAN.md`](./PLAN.md).

---

## 1. Mental model

Everything flows through one **risk-factor key space**. If you keep these three
dicts aligned by risk-factor name, the rest of the pipeline is mechanical:

```
assets[rf]            → a loaded, self-contained Asset
elementary_trades[rf] → the elementary instrument universe for rf
trade_pnls[rf][tid]   → PnL vector (n_scenarios,) for each elementary trade
```

A stage is just a class implementing a `Protocol` from `core/protocols.py`. The
orchestrator wires concrete implementations; the pipeline only ever sees the
protocols. To change behaviour, swap an implementation at construction time —
don't edit the pipeline.

---

## 2. Wiring the market-data API (the main integration work)

Every `_fetch_*` method on an Asset raises `NotImplementedError` and documents
its exact return shape. These are the only places that touch your internal
systems. The `_build_*` methods (already implemented) turn those raw dicts into
the typed market-data objects.

### FX (`assets/fx.py`)

| Method | Returns |
|--------|---------|
| `_fetch_spot` | `{"spot": float, "dates": [str], "values": [float]}` |
| `_fetch_forward_points` | `{"tenors": [str], "points": [float]}` |
| `_fetch_atm_vol` | `{"1W": 0.082, "1M": 0.091, ...}` |
| `_fetch_smile_vol` | `{"1W": {"10DP": .., "25DP": .., ...}, ...}` |
| `_fetch_shocks` | spot/vol/dom-rate/for-rate shock arrays + axis labels (see docstring) |

### IR (`assets/rates.py`)

| Method | Returns |
|--------|---------|
| `_fetch_curve` | `{"tenors": [str], "rates": [float]}` |
| `_fetch_history` | `{"dates": [str], "tenors": [str], "values": [[float]]}` |
| `_fetch_atm_vol` | `{"expiries": [str], "swap_tenors": [str], "values": [[float]]}` |
| `_fetch_smile_vol` | `{..., "strikes": [float], "values": [[[float]]]}` |
| `_fetch_shocks` | curve (+ optional vol) shock arrays + axis labels |

> **Tip:** Before wiring the real API, implement a `MockAPIClient` returning
> synthetic-but-shape-correct data. The whole pipeline then runs offline, which is
> how the tests and Phase 0 end-to-end run work.

### Wiring the client

The builders in `replication/builders.py` construct the client and call
`asset.load(client)`. Replace the inline `InternalAPIClient(...)` with your
authenticated client (injected, not constructed inline, ideally).

---

## 3. Adding a new asset class (e.g. Equity)

Three additive steps, **zero pipeline changes**:

**a. The asset** (`assets/equity.py`) — subclass `Asset`, implement the four
hooks (`_load_dependencies`, `_load`, `_load_shocks`, `validate`) and a typed
shock dataclass in `assets/types.py` (e.g. `EQShocks`) with `validate_shapes()`.

**b. The builder** (`replication/builders.py`):

```python
@RiskFactorRegistry.register
class EqRiskFactorBuilder:
    asset_class = "eq"
    def build(self, factor_id, factor_config):
        config = AssetConfig(asset_class="eq", asset_name=factor_id, extra=factor_config)
        asset = EQAsset(config)
        asset.load(api_client)
        return asset
```

**c. The instruments** (`instruments/equity.py`) — subclass `InstrumentSpec`
plus a `generate_eq_elementary_trades(...)` helper.

Then add the `eq` branch in `RiskFactorAwareTradeGenerator._generate_for_factor`
and in `OptionPricer._dispatch`. The registry discovers the builder via the
decorator automatically.

---

## 4. Adding an instrument to an existing asset class

1. Add the `@njit` pricing kernel(s) in `pricing/kernels/<asset>.py`
   (`price_*` scalar + a `price_*_batch` vectorised-over-scenarios variant).
2. Add the `InstrumentSpec` subclass in `instruments/<asset>.py`
   (`to_pricer_params`, `to_trade_id`, `price`, `sensitivities`).
3. Emit it from the asset's `generate_*_elementary_trades(...)` helper, gated by
   an `include_<type>` flag.
4. Route its `payoff_type` in `OptionPricer._dispatch`.

> Phase 3 (see `PLAN.md`) introduces a decorator-based **instrument registry** so
> steps 3–4 become a single registration, mirroring the asset-class pattern.

---

## 5. Pricing kernels — conventions

- Kernels live in `pricing/kernels/`, are `@njit(cache=True)`, and take **only**
  primitive floats / NumPy arrays. No Python objects cross the JIT boundary.
- Naming:
  - `price_*` — single-trade PV
  - `greeks_*` — first-order sensitivities tuple
  - `price_*_vec` — vectorised over scenarios → `(n_scenarios,)`
  - `price_*_batch` — vectorised over trades **and** scenarios → `(n_trades, n_scenarios)`
- FX uses Garman–Kohlhagen. **Watch the rate naming**: `FXAsset.domestic_ir` is
  the *base* ccy (GK `r_f`) and `FXAsset.foreign_ir` is the *quote* ccy
  (GK `r_d`). The mapping is documented in `pricing/kernels/fx.py`.
- For Phase 2, batch kernels should be `@njit(nogil=True)` so the `ThreadPool`
  in the PnL engine gets real parallelism without pickling Assets.

---

## 6. Configuring a run

`PipelineConfig` (see `replication/pipeline.py`) carries:

| Field | Purpose |
|-------|---------|
| `asset_config` | `{risk_factor: {asset_class, ...}}` — from Step 0 ingestion |
| `trade_config` | elementary grids per asset class (hybrid: defaults + observed) |
| `cluster_config` | passed to `ClusterResolver.resolve()` |
| `path_config` | output root + artifact filenames |
| `target_data` | target PnL + attributes from the risk system |

`trade_config` example:

```python
trade_config = {
    "fx": {
        "strike_pcts": [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20],
        "maturities":  [0.25, 0.5, 1.0, 2.0, 5.0],
        "include_forwards": True, "include_vanillas": True, "include_digitals": True,
        "augment_from_target": True,   # hybrid: add strikes/tenors observed in target trades
    },
    "rates": {
        "strike_offsets_bp": [-200, -100, -50, 0, 50, 100, 200],
        "maturities":        [1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
        "include_swaps": True, "include_swaptions": True, "include_caps_floors": True,
        "vol_type": "normal",
    },
}
```

---

## 7. Clustering rules

- A cluster maps `cluster_id → [risk_factors]`.
- **Every cluster must be single-asset-class.** The pipeline validates this and
  raises `PipelineError` on a mixed FX+IR cluster.
- Wire `ClusterResolver.resolve()` to your existing clustering (key grouping,
  k-means, hierarchical). Key grouping by ccy-pair gives 1:1 clusters; desk-level
  grouping gives many-RF clusters.

---

## 8. Running the tests (Phase 0 onward)

```bash
pytest src/rade_sr -q
```

Test layers to maintain:

- **Kernel golden tests** — put-call parity, forward parity, GK vs known values,
  digital ≤ vanilla, monotonicity in strike.
- **Shape/contract tests** — every shock dataclass `validate_shapes()` clean;
  pipeline produces aligned dicts.
- **End-to-end (mock API)** — synthetic portfolio → `jobs.pkl` + 4 parquet files
  per cluster, with a non-trivial replication-error sanity check.

---

## 9. Common pitfalls

- **Vol lookup axis.** The FX vol surface must be queried on the same axis it was
  built on (strike vs delta vs moneyness). Mismatches silently return corner vols
  — see Phase 1 fixes in `PLAN.md`.
- **Shock dimensionality.** Curve/rate shocks are 2-D `(n_scenarios, n_pillars)`
  and vol shocks are 3-D/4-D. Scenario extraction must interpolate per pillar/axis,
  not assume 1-D.
- **Forward base PV.** Use at-market forwards for elementary forwards so base PV ≈ 0.
- **Determinism.** Flatten trades by `sorted(risk_factor)` then trade order so the
  PnL matrix row order is stable and reproducible across runs.
