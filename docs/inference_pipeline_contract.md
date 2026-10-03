# Inference Pipeline Contract (Stages 1-6)

> **Purpose**: Authoritative reference for the public surface of the
> `EnsembleInferencePipeline` and its per-cluster helpers in
> `HybridGnnRnnInferencePipeline`. Both this repo **and** your work
> environment sync to this document — anything that doesn't match
> what's specified here is drift and gets fixed.
>
> **Out of scope**: API scaffolding (Stage 7+), UI callbacks
> (Stage 11+). Those are downstream contracts and have their own
> docs.
>
> **How to use this doc**:
> 1. Skim each stage's "Public surface" — these are the function
>    signatures and dataclass shapes that **must not change** without
>    updating this doc first.
> 2. The "On-disk layout" section at the bottom pins the exact
>    parquet / JSON paths Stage 6 writes.
> 3. The "Current state in this repo" notes at the end of each stage
>    flag any spot where the repo currently diverges (or is verified
>    aligned).
>
> **Last verified against this repo**: 2026-05-15 (full forensic audit — see Drift log; all 4 prior gaps closed).

---

## Pipeline waterfall

```
EnsembleInferencePipeline.run()
├── load()                                    [Stage 1]
├── load_scenarios()    or load_new_trades()  [Stage 2]   (mode-specific)
├── validate_scenarios() or validate_new_trades()  [Stage 3]   (mode-specific)
└── run_inference()                           [Stages 4-6 inclusive]
    ├── _build_member_inputs(input_mode)              [Stage 4]
    │   └── per cluster: _build_affected_inputs()     (affected path)
    │       or          _build_unaffected_inputs()    (cheap path)
    ├── per cluster: predict_member_chunked()         [Stage 5]
    │   └── _post_infer_cluster() → ClusterArtifact   [Stage 6, per-cluster]
    ├── _ensemble._combine(member_preds)              [Stage 5]
    ├── _build_result(...)                            [Stage 5]
    └── post_infer(result, cluster_artifacts=...)     [Stage 6, run-level]
        └── _write_run_manifest(...)
```

---

## Stage 1 — `load()`

**File**: `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Signature**:
```python
def load(self) -> None: ...
```

**Behaviour**:
- Idempotent (returns early if `self._ensemble is not None`).
- If `self._session is not None`: warm-load from session (`session.ensemble`, `session.inference_contexts`).
- If `self._session is None`: cold-load from `config.registry_dir` via `EnsembleBuilder.from_registry(version)`. Then for each cluster, parallel-load its `InferenceContext` from `<registry>/<cid>/<version>/` via `load_inference_context_from_dir(...)` and convert through `_dict_to_inference_context(...)`.
- Populates: `self._ensemble`, `self._inference_contexts: Dict[str, InferenceContext]`.
- Emits: `event(STAGE_INFERENCE, ...)` per loading layer — `"Loading ensemble from registry"` → `"Ensemble assembled"` → one `"Cluster context loading"` / `"Cluster context loaded"` pair per cluster. There is no `STAGE_LOAD` constant; everything in the inference pipeline (`load → load_scenarios → validate → run`) shares `STAGE_INFERENCE` so the UI's activity-log filter chip toggles the entire stream as one unit (see `infer_events.py` for the three valid stages: `ingest` / `validate` / `inference`).
- **Parallel cluster loading**: cold-load uses a `ThreadPoolExecutor` with worker count = `config.metadata['inference']['loader_max_workers']` → `config.max_workers` → `min(8, n_clusters)`. Per-cluster failures are aggregated and raised once after the executor shuts down.

**Errors raised**:
- `RuntimeError` if a session is provided whose `ensemble_version` mismatches, or whose `all_inference_ready` is False.
- `ValueError` (aggregated) listing every cluster whose context failed to load.

**Constructor signature (must not change)**:
```python
def __init__(
    self,
    ensemble_config:  EnsembleConfig,
    ensemble_version: str = "latest",
    session:          Optional[EnsembleSession] = None,
    *,
    on_event:         Optional[EmitFn] = None,
) -> None: ...
```

**Current state**: ✅ Verified aligned.

---

## Stage 2a — `load_scenarios()`

**File**: `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Signature**:
```python
def load_scenarios(
    self,
    new_scenario_dir: Optional[Union[str, Path]] = None,
) -> LoadedScenariosReport: ...
```

**Behaviour**:
1. Resolve scenario directory: argument override, else `config.metadata["inference"]["new_scenario_dir"]`.
2. Parse every `<risk_factor>.csv` under that directory into a nested dict `{rf_name: {scenario_label: {knot: value}}}`.
3. Cross-check that all shock files share the same scenario index (raises `ValueError` if not).
4. Populate `self._new_scenario_shocks` (private, heavy) and `self._loaded_scenarios` (public, light report).

**Returns**:
```python
@dataclass(frozen=True)
class LoadedScenariosReport:
    new_scenario_dir:    str
    risk_factor_names:   List[str]
    n_risk_factors:      int
    n_scenarios:         int
    scenario_labels:     List[str]
    def to_dict(self) -> Dict[str, Any]: ...
```

**Errors raised**:
- `RuntimeError` if `load()` not called yet.
- `ValueError` if scenario directory missing/empty, or scenario indices inconsistent across files.

**Stage 2b stub** — `load_new_trades(new_trades_path: str) -> LoadedNewTradesReport` exists but raises `NotImplementedError`.

**Current state**: ✅ Verified aligned.

---

## Stage 3a — `validate_scenarios()`

**File**: `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Signature**:
```python
def validate_scenarios(self) -> ValidationReport: ...
```

**Behaviour**:
For each cluster `cid` in `self._ensemble.router.cluster_ids`:
1. Get `ctx = self._inference_contexts[cid]`.
2. Call `HybridGnnRnnInferencePipeline.intersecting_risk_factors(ctx, shock_rfs)`.
3. If intersection non-empty: `is_affected = True`, skip cheap-path eligibility.
4. Else: `is_affected = False`, call `missing_scenario_labels(ctx, scenario_labels)` to verify cheap-path eligibility. Missing labels → cluster-level error in `ValidationReport.errors`.
5. Build `ClusterRoutingDecision` per cluster.

Cross-cluster sanity check: if *no* cluster is affected, append to `warnings` (NOT `errors`) — the run still proceeds via the cheap path on every cluster. Message: `"No clusters intersect any shocked risk factor — every cluster will take the cheap historical-lookup path."`

Populates `self._validation_report`.

**Routing decision dataclass**:
```python
@dataclass(frozen=True)
class ClusterRoutingDecision:
    cluster_id:                  str
    is_affected:                 bool
    intersecting_risk_factors:   List[str]
    n_elementary_trades:         int
    n_target_trades:             int
    missing_scenario_labels:     List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]: ...
```

**Validation report dataclass**:
```python
@dataclass(frozen=True)
class ValidationReport:
    ensemble_version:    str
    n_scenarios:         int
    scenario_labels:     List[str]
    cluster_decisions:   List[ClusterRoutingDecision]
    errors:              List[str]
    warnings:            List[str]
    # Derived properties:
    is_valid:               bool   # == (len(errors) == 0)
    affected_cluster_ids:   List[str]
    unaffected_cluster_ids: List[str]
    affected_count:         int
    unaffected_count:       int
    cheap_path_used:        bool   # == (unaffected_count > 0)
    def to_dict(self) -> Dict[str, Any]: ...
```

**Errors raised**:
- `RuntimeError` if `load_scenarios()` not called yet.

**Stage 3b stub** — `validate_new_trades() -> ValidationReport` exists but raises `NotImplementedError`.

**Current state**: ✅ Verified aligned.

---

## Stage 4 — `_build_member_inputs(input_mode)`

**File**: `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Signature**:
```python
def _build_member_inputs(
    self,
    input_mode: str,                              # "new_scenarios" | "new_trades"
) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...
```

**Behaviour**:
For each cluster, call `HybridGnnRnnInferencePipeline.build_new_scenario_inputs(ctx, new_scenario_shocks, scenario_labels, is_affected=decision.is_affected)`. This single method dispatches internally on `is_affected`:
- **Affected branch** → re-prices elementary PnL through `cluster_assets`, scales via `ctx.elementary_scaler`, packs the 7-key model input dict.
- **Unaffected branch** → looks up pre-scaled historical elementary PnL via `ctx.elementary_pnl.loc[scenario_labels]`, packs the same 7-key dict (skips the re-pricing).

The combined-method shape (rather than two separate ones) keeps the per-cluster routing event emission (`"Building cluster inputs"` → `"Cluster inputs ready"`) and the catch-and-emit failure path centralised in `_build_new_scenarios_inputs` on this side.

**Returns**: `(member_inputs, extra_meta)` where:
- `member_inputs: Dict[cluster_id, model_input_dict]` — the 7-key dict ready for `EnsembleModel.predict`.
- `extra_meta: Dict[str, Any]` — currently `{"sample_ids": {cluster_id: List[trade_id_or_window_id]}}`. Consumed downstream by `_build_result` to populate `InferenceResult.metadata["per_member_sample_ids"]` and flatten into `InferenceResult.sample_ids`.

**Helper signatures (in `HybridGnnRnnInferencePipeline`)** — all static methods:
```python
@staticmethod
def intersecting_risk_factors(
    ctx: InferenceContext,
    shock_risk_factors: Iterable[str],
) -> List[str]: ...

@staticmethod
def missing_scenario_labels(
    ctx: InferenceContext,
    scenario_labels: List[str],
) -> List[str]: ...

@staticmethod
def build_new_scenario_inputs(
    ctx:                 InferenceContext,
    new_scenario_shocks: Dict[str, Dict[Any, Any]],
    scenario_labels:     List[str],
    is_affected:         bool,
) -> Dict[str, Any]:
    """
    Returns a dict with three keys:
      * 'inputs'    — the 7-key model input dict for predict_member_chunked.
      * 'sample_ids' — list of trade-id-or-window-id labels for this cluster.
      * 'metadata'  — at minimum {'n_scenarios': int, 'path': 'affected'|'unaffected'}.
    """
```

**InferenceContext shape** (read-only contract — `dataclass`):
```python
@dataclass
class InferenceContext:
    data_config:       Optional[Dict[str, Any]] = None
    encoder:           Optional[TradeAttributeEncoder] = None
    encoder_results:   Optional[Dict[str, Any]] = None
    graph_builder:     Optional[TradeGraphBuilder] = None
    graph_results:     Optional[Dict[str, Any]] = None
    elementary_pnl:        Optional[pd.DataFrame] = None
    elementary_scaler:     Optional[Any]          = None
    elementary_attributes: Optional[Dict[str, Any]] = None
    target_scaler:         Optional[Any]          = None
    target_attributes:     Optional[Dict[str, Any]] = None   # must include 'trade_id' and 'NotionalSign'
    trade_universe:    Optional[Any] = None
    cluster_info:      Optional[Dict[str, Any]] = None
    cluster_assets:    Optional[Dict[str, Any]] = None       # heavy, lazy-loaded
    cluster_rf_keys:   Optional[Dict[str, List[str]]] = None # sidecar for fast routing
    _cluster_assets_path: Optional[Path] = None              # for lazy load
    cluster_elem_trades:  Optional[Any] = None
```

**Current state**: ✅ Verified aligned.

---

## Stage 5 — Forward pass (inside `run_inference`)

**File**: `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Outer signature**:
```python
def run_inference(self) -> InferenceResult: ...
```

**Behaviour** (the per-cluster loop):
For each `cid` in `self._ensemble.router.cluster_ids`:
1. Skip if no member model (defensive).
2. Skip if `member_inputs[cid] is None`.
3. Call `HybridGnnRnnInferencePipeline.predict_member_chunked(ensemble=self._ensemble, cluster_id=cid, cluster_inputs=member_inputs[cid], batch_size=batch_size)`. Returns predictions in **scaled** space, shape `[n_scenarios, n_trades]`.
4. Store in `member_preds[cid] = cluster_preds`.
5. (Stage 6 work, in same loop) Call `self._post_infer_cluster(...)`, accumulate `ClusterArtifact` into `cluster_artifacts`.
6. Free `member_inputs[cid] = None` to release deep-copied risk_factor_shocks.

After the loop:
- `combined = self._ensemble._combine(member_preds)` — uses the configured aggregation strategy (`concat` or `mean`); accessed via `_combine` because there's no public `combine_predictions()` on `EnsembleModel` yet (deferred).
- `result = self._build_result(combined, infer_meta, extra_meta)`.
- `result.latency_seconds = time.perf_counter() - t0`.
- Call `self.post_infer(result, cluster_artifacts=cluster_artifacts)`.

**`predict_member_chunked` signature** (in `HybridGnnRnnInferencePipeline`):
```python
@staticmethod
def predict_member_chunked(
    ensemble:       "EnsembleModel",
    cluster_id:     str,
    cluster_inputs: Dict[str, Any],          # must contain "pnl_history"
    batch_size:     int = 128,
) -> np.ndarray: ...
```

Splits `pnl_history` along the scenario axis into `batch_size`-sized blocks, runs each through `ensemble.predict_member(cluster_id, chunk_inputs)`, concatenates outputs. Static inputs are passed through unchanged (zero-copy views). Raises `KeyError` if `pnl_history` is missing.

**`batch_size` source**: `config.metadata["inference"].get("batch_size", 128)`.

**Current state**: ✅ Verified aligned.

---

## Stage 6 — Post-inference

### 6a — Per-cluster: `_post_infer_cluster()`

**File**: `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Signature**:
```python
def _post_infer_cluster(
    self,
    cluster_id:      str,
    cluster_preds:   np.ndarray,           # shape [n_scenarios, n_trades], scaled
    scenario_labels: List[str],
    ctx:             "InferenceContext",
    out_dir:         Path,                 # the run's <inference> dir
) -> ClusterArtifact: ...
```

**Behaviour**:
1. Call `HybridGnnRnnInferencePipeline.transform_predictions(cluster_id, cluster_preds, scenario_labels, ctx)`. Returns `(scaled_wide, original_wide, summary_df)`.
2. Write `out_dir / "trade_predictions" / f"{cluster_id}_scaled.parquet"`.
3. Write `out_dir / "trade_predictions" / f"{cluster_id}_original.parquet"`.
4. Emit `event(STAGE_INFERENCE, "Wrote cluster artifacts", target=cluster_id, ...)`.
5. Return `ClusterArtifact(...)`.

**`transform_predictions` signature** (in `HybridGnnRnnInferencePipeline` — single source of truth for the scaled→original chain):
```python
@staticmethod
def transform_predictions(
    cluster_id:      str,
    cluster_preds:   np.ndarray,           # [n_scenarios, n_trades]
    scenario_labels: Optional[List[str]],
    ctx:             InferenceContext,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (scaled_wide, original_wide, summary_df)."""
```

Returns:
- `scaled_wide`: index=`scenario_label`, columns=trade_ids, values in scaled space.
- `original_wide`: same shape, inverse-scaled AND notional-sign-restored.
- `summary_df`: long format, one row per scenario, columns:
  - `scenario_label`, `cluster_id`,
  - `sum_pnl_scaled`, `sum_pnl_original`,
  - `mean_pnl_original`, `std_pnl_original`,
  - `min_pnl_original`, `max_pnl_original`.

Padding/slicing rules for `n_trades ≠ len(scaler.feature_names_in_)`: see `transform_predictions` docstring; covered by the existing implementation.

### 6b — `ClusterArtifact` dataclass

**File**: `src/rade_ml_pt/pipelines/ensemble/infer.py`
```python
@dataclass
class ClusterArtifact:
    cluster_id:    str
    n_trades:      int
    n_scenarios:   int
    scaled_path:   str
    original_path: str
    trade_ids:     List[str] = field(default_factory=list)
    summary_df:    Optional[pd.DataFrame] = field(default=None, repr=False, compare=False)

    def to_manifest_dict(self) -> Dict[str, Any]:
        """JSON-safe view written into manifest.json (excludes summary_df)."""
```

### 6c — Run-level: `post_infer()`

**File**: `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Signature**:
```python
def post_infer(
    self,
    result:            InferenceResult,
    *,
    cluster_artifacts: Optional[List[ClusterArtifact]] = None,
) -> None: ...
```

**Behaviour**:
1. Log summary statistics from `result.predictions` (mean, std, min, max).
2. If `config.artifacts_dir is None` or `cluster_artifacts is None`: return (preserves backwards-compat for programmatic callers).
3. `out_dir = Path(config.artifacts_dir) / INFERENCE_DIRNAME`.
4. If `cluster_artifacts` non-empty:
   - Concat all `a.summary_df` → write to `<out_dir>/cluster_summary/cluster_predictions.parquet`.
   - Group by `scenario_label`, aggregate (`sum_pnl_scaled→sum`, `sum_pnl_original→sum`, `cluster_id→nunique`) → write to `<out_dir>/portfolio_summary/portfolio_predictions.parquet`.
5. Call `self._write_run_manifest(...)`.

### 6d — `_write_run_manifest()`

```python
def _write_run_manifest(
    self,
    out_dir:                Path,
    result:                 InferenceResult,
    cluster_artifacts:      List[ClusterArtifact],
    cluster_summary_path:   Optional[Path],
    portfolio_summary_path: Optional[Path],
) -> Path: ...                          # returns out_dir / MANIFEST_FILENAME
```

Manifest JSON schema:
```jsonc
{
  "schema_version":         "1",
  "generated_at":           "<UTC ISO timestamp>",
  "ensemble_version":       "<version>",
  "input_mode":             "new_scenarios" | "new_trades",
  "n_scenarios":            <int>,
  "scenario_labels":        [<str>, ...],
  "clusters": [
    {
      "cluster_id":    "<cid>",
      "n_trades":      <int>,
      "n_scenarios":   <int>,
      "scaled_path":   "<abs path>",
      "original_path": "<abs path>",
      "trade_ids":     [<str>, ...]
    }
  ],
  "cluster_summary_path":   "<abs path or null>",
  "portfolio_summary_path": "<abs path or null>",
  "validation":             <ValidationReport.to_dict() or null>,
  "scenarios":              <LoadedScenariosReport.to_dict() or null>,
  "latency_seconds":        <float or null>
}
```

**Current state**: ✅ Verified aligned **after** the layout drift fix on 2026-05-15 — see "On-disk layout" below.

---

## On-disk layout (canonical)

Pinned by named constants in `src/rade_ml_pt/pipelines/ensemble/infer.py`:

| Constant | Value |
|---|---|
| `INFERENCE_DIRNAME` | `"inference"` |
| `CLUSTER_SUMMARY_DIRNAME` | `"cluster_summary"` |
| `PORTFOLIO_SUMMARY_DIRNAME` | `"portfolio_summary"` |
| `TRADE_PREDICTIONS_DIRNAME` | `"trade_predictions"` |
| `CLUSTER_SUMMARY_FILENAME` | `"cluster_predictions.parquet"` |
| `PORTFOLIO_SUMMARY_FILENAME` | `"portfolio_predictions.parquet"` |
| `MANIFEST_FILENAME` | `"manifest.json"` |

Per run:

```
<config.artifacts_dir>/                       (= <base_artifacts_dir>/inference_runs/<run_id>/ for API runs)
└── inference/                                  (= INFERENCE_DIRNAME)
    ├── manifest.json
    ├── cluster_summary/
    │   └── cluster_predictions.parquet
    ├── portfolio_summary/
    │   └── portfolio_predictions.parquet
    └── trade_predictions/
        ├── <cluster_id>_scaled.parquet
        └── <cluster_id>_original.parquet
```

**Single source of truth**: the constants above. The result reader
(`src/rade_ml_pt/ensemble/api/services/result_reader.py`) **imports
them directly from `pipelines.ensemble.infer`** so writer ↔ reader
can never drift.

---

## Out-of-scope but relevant references

- **`InferenceResult`**: defined in `src/rade_ml_pt/core/types.py`. Public surface of `EnsembleInferencePipeline.run()` and unchanged by Stage 6. Fields used by the pipeline: `predictions`, `n_samples`, `sample_ids`, `metadata`, `latency_seconds`, `to_json()`.
- **`EnsembleModel._combine`**: private but stable. Called from `run_inference` until a public `combine_predictions()` is added (deferred — Win F).
- **`EnsembleSession`**: provides the warm-load path. Tracked separately in `RADE_UI_DESIGN.md`.

---

## Drift log (this repo only)

| Date | Stage | Drift | Status |
|---|---|---|---|
| 2026-05-15 | 6 | Wrote per-cluster parquets to `clusters/<cid>_<space>.parquet` and run-level summaries directly under `inference/<filename>` instead of the agreed `cluster_summary/`, `portfolio_summary/`, `trade_predictions/` subdirs | **Fixed** — constants now centralized in `pipelines/ensemble/infer.py`, reader imports them |
| 2026-05-15 | 1 | `_load_from_registry` was a serial `for cid in config.cluster_ids` loop; the contract claimed "parallel-load" but the implementation never had `ThreadPoolExecutor` / `as_completed` / a worker-count knob. The previous "✅ Verified aligned" marker on Stage 1 was wrong | **Fixed** — `_load_from_registry` now uses `ThreadPoolExecutor` with worker count resolved from `config.metadata['inference']['loader_max_workers']` → `config.max_workers` → `min(8, n_clusters)`. Failures are aggregated and raised once at the end of the pool. Appendix A.1 in `RADE_UI_DESIGN.md` updated to match |
| 2026-05-15 | 1 | Contract said `load()` emits `event(STAGE_LOAD, ...)` — but `STAGE_LOAD` doesn't exist in `infer_events.py` (the only stages are `ingest` / `validate` / `inference`). The code correctly uses `STAGE_INFERENCE`. | **Fixed** — contract now documents the actual per-layer events (`"Loading ensemble from registry"` → `"Ensemble assembled"` → per-cluster `"Cluster context loading"` / `"loaded"` / `"failed"`), all under `STAGE_INFERENCE`. Stage 1 errors section also corrected to the actual `RuntimeError` / aggregated `ValueError` |
| 2026-05-15 | 3 | Cross-cluster sanity check (no cluster affected) was appended to `report.errors`, blocking the run — even though we explicitly agreed in chat to downgrade it to a warning so the cheap-path-on-every-cluster case still runs. The code had never been updated. | **Fixed** — code now appends to `warnings` with a self-explanatory message; contract Stage 3 section now states the warning behaviour as the canonical contract. Appendix A.1 in `RADE_UI_DESIGN.md` updated to match |
| 2026-05-15 | 4 | Contract documented two separate static methods (`_build_affected_inputs(ctx, shocks)` and `_build_unaffected_inputs(ctx, scenario_labels)`) and an `extra_meta` shape of `{input_mode, n_clusters_affected, n_clusters_unaffected}`. The actual code uses **one** combined method `HybridGnnRnnInferencePipeline.build_new_scenario_inputs(ctx, shocks, scenario_labels, is_affected=...)` that returns `{inputs, sample_ids, metadata}`, and `extra_meta` is `{"sample_ids": {cid: ...}}` consumed by `_build_result`. | **Fixed** — contract Stage 4 rewritten to describe the single combined method and the actual `extra_meta` shape. No code change needed (the current implementation is what your work env has been running) |

> Any future divergence should be appended here. The fix pattern is the same: rename, centralise as a named constant, re-import on the reader side.
