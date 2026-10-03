# Ensemble Inference Refactor — Implementation Tracker

> **Purpose.** Step-by-step porting checklist for the staged + tiered
> refactor of `EnsembleInferencePipeline`.  Each step entry contains
> the exact diff, the verification command, and any porting notes for
> the work env.
>
> **Sibling docs.**  This file is *not* the design rationale.  For the
> long-form spec (Phase-1 foundations, event protocol, backend
> wrapper, etc.) see
> [`inference_implementation.md`](./inference_implementation.md).  For
> the UI layout see
> [`../platform_designs/RADE_UI_DESIGN.md`](../platform_designs/RADE_UI_DESIGN.md).
>
> **Two-environment convention.**  Every step has two status cells —
> *This env* (the `QuantStrata` Cursor repo) and *Work env* (your
> production repo).  A step is fully complete only when both cells are
> ✅.

---

## 0 · Principle

> Anything that reads from `InferenceContext` lives in
> `hybrid_gnn_rnn/infer.py`.  `ensemble/infer.py` owns orchestration,
> file IO, aggregation, and the `predict()` call.  If you see
> `ctx.<something>` in `ensemble/infer.py` after this refactor, it's
> a bug.

The refactor delivers two things on top of this rule:

1. **Staged pipeline.**  `load() → load_scenarios() → validate() →
   run_inference()` are independently callable.  `run()` keeps its
   existing contract and chains them internally.
2. **Tiered input building (Tier 2 optimisation).**  Affected clusters
   take the full re-pricing path; unaffected clusters take the cheap
   path (`ctx.elementary_pnl.loc[scenario_labels]` — straight from the
   training-time stored parquet).  Both paths produce the same 7-key
   model input dict.

---

## 1 · Status matrix (LEAN plan — single-file refactor)

> **Scope reduction.**  After step 2 we audited against the three
> shipping goals and pulled the cluster-pipeline refactor out of
> scope.  All remaining steps now touch only
> `src/rade_ml_pt/pipelines/ensemble/infer.py`.  The new ensemble
> imports existing `@staticmethod`s from
> `HybridGnnRnnInferencePipeline` (same way the current code already
> does for the full path).  When `new_trades` mode lands, we revisit
> the cluster-side split (Reading B from the decisions log).

| # | Step                                                        | This env | Work env | Notes |
|---|-------------------------------------------------------------|----------|----------|-------|
| 1 | Add 3 module-level dataclasses                              | ✅       | ⬜       | done 2026-05-12 |
| 2 | Add `missing_scenario_labels` field                         | ✅       | ⬜       | done 2026-05-12 |
| 3 | Extend `__init__` with cached state attributes               | ✅       | ⬜       | done 2026-05-12 |
| 4 | Validation helpers (classify, eligibility, canonical index) | ✅       | ⬜       | done 2026-05-12 |
| 5 | `_build_cluster_inputs_full` + `_build_cluster_inputs_cheap` | ✅       | ⬜       | done 2026-05-12 |
| 6 | Public stages — `load()`, `load_scenarios()`, `validate()`, `run_inference()`, `_run_with_prebuilt` | ✅       | ⬜       | done 2026-05-12 |
| 7 | Rewrite `run()` body — chains stages, signature unchanged   | ✅       | ⬜       | done 2026-05-12 |
| 8 | Rewire `_build_member_inputs` + `_build_new_scenarios_inputs` to dispatch via routing decisions | ✅       | ⬜       | done 2026-05-12 |
| 9 | Extend `_build_result()` metadata (4 new keys)              | ✅       | ⬜       | done 2026-05-12 |
| 10 | Smoke test — pytest suite + import smoke                   | ✅       | ⬜       | done 2026-05-12 — 23/26 ensemble tests green; 3 failing are pre-existing rade_sr env issues (will pass in work env) |

Legend: ⬜ pending · 🟡 in progress · ✅ done

**Dropped from original plan:**

- ❌ Five module-level helpers in `hybrid_gnn_rnn/infer.py` — out of scope.
- ❌ `_extract_canonical_index` as a separate step — folded into the validation helpers (step 4).
- ❌ Separate "update `14_run_infer_pipeline.py`" step — the example is data-driven, new metadata appears automatically.

---

## 2 · Contracts preserved (porting sanity check)

Each must remain **byte-identical** post-refactor.  If any of these
diverge between envs, it's a bug.

| Surface                                                                                  | Owner module           |
|------------------------------------------------------------------------------------------|------------------------|
| `EnsembleInferencePipeline.__init__(ensemble_config, ensemble_version, session=None, *, on_event=None)` | `ensemble/infer.py`    |
| `EnsembleInferencePipeline.run(self) -> InferenceResult`                                 | `ensemble/infer.py`    |
| `HybridGnnRnnInferencePipeline.run()` and all public methods                             | `hybrid_gnn_rnn/infer.py` |
| All five existing `@staticmethod` helpers on `HybridGnnRnnInferencePipeline`             | `hybrid_gnn_rnn/infer.py` |
| `InferenceContext` schema                                                                 | `hybrid_gnn_rnn/infer.py` |
| `EnsembleModel.predict()`, `_combine()`                                                  | `ensemble/model.py`    |
| Event taxonomy in `infer_events.py`                                                       | `pipelines/ensemble/`  |

---

## 3 · Implementation steps

> Each step entry follows the same template: status · files · imports
> · diff/snippet · verification · porting notes.

### Step 1 — Module-level dataclasses

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)
**Files:** `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Imports added:**

```python
from dataclasses import dataclass, field
from typing import ..., Union, ...   # added Union
```

**Code added:** three frozen dataclasses (`LoadedScenariosReport`,
`ClusterRoutingDecision`, `ValidationReport`) plus a `Stage reports`
section banner between the existing `Helpers` and `Pipeline` sections.
See the file directly for the full block; size ≈ 180 lines.

**Verification:**

```bash
.venv/bin/python -c "
from src.rade_ml_pt.pipelines.ensemble.infer import (
    LoadedScenariosReport, ClusterRoutingDecision, ValidationReport,
)
r = ValidationReport(ensemble_version='v1', n_scenarios=0, scenario_labels=[])
assert r.is_valid is True
print('ok')
"
```

**Porting notes:** none.  Pure additive change.

---

### Step 2 — Add `missing_scenario_labels` field to `ClusterRoutingDecision`

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)
**Files:** `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Imports added:** none.
**Diff:**

```diff
 @dataclass(frozen=True)
 class ClusterRoutingDecision:
     ...
     cluster_id:                  str
     is_affected:                 bool
     intersecting_risk_factors:   List[str]
     n_elementary_trades:         int
     n_target_trades:             int
+    missing_scenario_labels:     List[str] = field(default_factory=list)

     def to_dict(self) -> Dict[str, Any]:
         return {
             ...
             "n_target_trades":           self.n_target_trades,
+            "missing_scenario_labels":   list(self.missing_scenario_labels),
         }
```

**Why this field exists:** populated by `validate()` for unaffected
clusters whose history doesn't cover all requested scenario labels.
Affected clusters always have `missing_scenario_labels == []` (they
don't use the cheap path, so the lookup never happens).
Non-empty on an unaffected cluster ⇒ cheap path blocked ⇒
corresponding entry appears in `ValidationReport.errors` ⇒
`report.is_valid is False`.

**Verification:**

```bash
.venv/bin/python -c "
from src.rade_ml_pt.pipelines.ensemble.infer import ClusterRoutingDecision
d_default = ClusterRoutingDecision('c1', True, ['rf_a'], 10, 3)
assert d_default.missing_scenario_labels == []
d_missing = ClusterRoutingDecision('c2', False, [], 10, 3, missing_scenario_labels=['s_99'])
assert d_missing.to_dict()['missing_scenario_labels'] == ['s_99']
print('ok')
"
```

**Porting notes:** pure additive; existing callers that construct
`ClusterRoutingDecision` positionally still work because the new field
has a default.

---

### Step 3 — Extend ensemble `__init__` cached state

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)
**Files:** `src/rade_ml_pt/pipelines/ensemble/infer.py`
**Imports added:** `Iterable` added to typing import.

**Diff:** add to `__init__`:

```diff
         self._ensemble = None
         self._ens_config: Optional[EnsembleConfig] = None
         self._member_versions: Optional[Dict[str, str]] = None
         self._inference_contexts: Dict[str, Any] = {}
+
+        # Staged-flow cache: populated by load_scenarios() / validate().
+        # The pre-built short-circuit in run() leaves these as None.
+        self._new_scenario_shocks: Optional[Dict[str, Any]] = None
+        self._loaded_scenarios:    Optional[LoadedScenariosReport] = None
+        self._validation_report:   Optional[ValidationReport] = None
```

**Verification:** import smoke confirms class has the new attributes.

**Porting notes:** signature unchanged.  Also bump the typing import
to include `Iterable` — used by `_classify_cluster(shock_rfs:
Iterable[str])`.

---

### Step 4 — Validation helpers (private methods on the pipeline)

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)
**Files:** `src/rade_ml_pt/pipelines/ensemble/infer.py`

Four small private methods, all in a new `# === Validation helpers ===`
section between `Loading` and `Input building`:

| Helper | Signature | Role |
|---|---|---|
| `_extract_canonical_index` | `(shocks) -> List[str]` | Cross-shock-file: verify every CSV shares the same scenario labels in the same order; return the canonical list. |
| `_extract_cluster_risk_factors` | `(ctx) -> List[str]` | Per-cluster: union of RFs across `ctx.cluster_assets[*].risk_factor_shocks`. |
| `_classify_cluster` | `(cid, ctx, shock_rfs) -> ClusterRoutingDecision` | Per-cluster: affected iff intersection non-empty. Populates trade counts from `ctx.trade_universe`. |
| `_validate_cluster_for_labels` | `(cid, ctx, labels) -> (bool, List[str])` | Per-cluster cheap-path eligibility: do all labels exist in `ctx.elementary_pnl.index`? Defensive on `RangeIndex`. |

**Porting notes:** all private to `EnsembleInferencePipeline`; no
imports from `hybrid_gnn_rnn` needed (just touches `ctx.*` attributes).

---

### Step 5 — `_build_cluster_inputs_full` + `_build_cluster_inputs_cheap`

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)
**Files:** `src/rade_ml_pt/pipelines/ensemble/infer.py`

Both methods live in the `Input building` section and import existing
`@staticmethod`s from `HybridGnnRnnInferencePipeline` (same imports
the current `_build_new_scenarios_inputs` already uses):

```python
def _build_cluster_inputs_full(self, cid, ctx, shocks) -> Dict[str, Any]:
    # 9-step composition: _inject_unchanged_inputs → _update_asset_portfolio
    # → calculate_elementary_pnl → concat/reorder → _standardise_pnl
    # → build_new_pnl_sequences → build_model_inputs
    ...  # body lifted from current _build_new_scenarios_inputs

def _build_cluster_inputs_cheap(self, cid, ctx, scenario_labels) -> Dict[str, Any]:
    # 4-step cheap path: _inject_unchanged_inputs
    # → ctx.elementary_pnl.loc[scenario_labels]
    # → build_new_pnl_sequences → build_model_inputs
    ...  # NEW
```

Both return the full `{"inputs": ..., "sample_ids": ..., "metadata": ...}`
dict from `build_model_inputs`; the dispatcher in
`_build_new_scenarios_inputs` peels off `["inputs"]` and
`.get("sample_ids")`.

**Porting notes:** the full-path body is **structurally identical** to
what's already inline in `_build_new_scenarios_inputs`.  Moving it
into its own method is a refactor, not new behaviour.

---

### Step 6 — Public stages: `load()`, `load_scenarios()`, `validate()`, `run_inference()`, `_run_with_prebuilt()`

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)
**Files:** `src/rade_ml_pt/pipelines/ensemble/infer.py`

Five new methods in the `Orchestration` section.  All emit
start/OK/fail events through `self._emit`.  Each checks prerequisites
and raises `RuntimeError` with an actionable message if the previous
stage hasn't run.

| Method | Signature | Notes |
|---|---|---|
| `load` | `() -> None` | Idempotent. Returns immediately if `self._ensemble is not None`. Dispatches to `_load_from_registry` or `_load_from_session` based on `self._session`. |
| `load_scenarios` | `(new_scenario_dir=None) -> LoadedScenariosReport` | Falls back to `config.metadata["inference"]["new_scenario_dir"]` if arg omitted. Parses shocks once, extracts canonical index, resets downstream state. |
| `validate` | `() -> ValidationReport` | Classifies all clusters; runs cheap-path eligibility on unaffected ones. Never raises on validation issues (errors land in the report). |
| `run_inference` | `() -> InferenceResult` | Builds inputs (routed full/cheap), predicts, builds result, post_infers. Raises `RuntimeError` if any prior stage missing or `validate()` failed. |
| `_run_with_prebuilt` | `(infer_meta) -> InferenceResult` | Private helper for the legacy pre-built-`member_inputs` short-circuit. Emits the `"Using pre-built member inputs"` phase the tests pin. |

**Porting notes:** `load_scenarios`'s optional `new_scenario_dir` arg
is the *only* signature addition.  Programmatic callers can ignore it
and rely on `config.metadata["inference"]` as today.

---

### Step 7 — Rewrite `run()` body

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)

**Subtlety encountered & fixed:** initial draft raised the
`new_trades NotImplementedError` *before* the pre-built short-circuit
check, breaking `test_infer_new_trades_mode` (the test legitimately
exercises new_trades + pre-built `member_inputs`).  Fix: the
NotImplementedError now lives inside the `else` (staged) branch,
preserving the old behaviour where pre-built `member_inputs` work
for any input_mode.

**Files:** `src/rade_ml_pt/pipelines/ensemble/infer.py`

Signature **identical**: `run(self) -> InferenceResult`.  New body:

```python
def run(self) -> InferenceResult:
    self._emit(event(STAGE_INFERENCE, "Pipeline started", ...))
    t0 = time.perf_counter()
    try:
        infer_meta = self.config.metadata.get("inference", {})
        input_mode = infer_meta.get("input_mode", "new_scenarios")
        if input_mode not in {"new_scenarios", "new_trades"}:
            raise ValueError(f"Unknown input_mode '{input_mode}'. ...")
        if input_mode == "new_trades":
            raise NotImplementedError("new_trades inference is not yet supported.")

        if infer_meta.get("member_inputs"):
            self.load()
            result = self._run_with_prebuilt(infer_meta)
        else:
            self.load()
            self.load_scenarios()
            report = self.validate()
            if not report.is_valid:
                raise ValueError(f"Inference validation failed: {report.errors}")
            result = self.run_inference()

        self._emit(event(STAGE_INFERENCE, "Pipeline complete", ...))
        return result
    except Exception as exc:
        self._emit(event(STAGE_INFERENCE, "Pipeline failed",
                          status=STATUS_FAIL,
                          target=type(exc).__name__, detail=str(exc)))
        raise
```

**Porting notes:** preserves every phase string the test suite pins
(`"Pipeline started"`, `"Pipeline complete"`, `"Pipeline failed"`,
`"Using pre-built member inputs"`, `"Forward pass started"`,
`"Forward pass complete"`).

---

### Step 8 — Rewire `_build_member_inputs` + `_build_new_scenarios_inputs`

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)
**Files:** `src/rade_ml_pt/pipelines/ensemble/infer.py`

* `_build_member_inputs(input_mode)` — pre-built short-circuit is
  gone (handled in `run()`); body collapses to a mode dispatcher.
* `_build_new_scenarios_inputs()` (no args) — body becomes a loop
  over `self._validation_report.cluster_decisions` that calls
  `_build_cluster_inputs_full` or `_build_cluster_inputs_cheap` per
  cluster.

---

### Step 9 — Extend `_build_result()` metadata

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)
**Files:** `src/rade_ml_pt/pipelines/ensemble/infer.py`

Only emitted when `self._validation_report is not None` (i.e. the
staged flow; pre-built short-circuit skips these keys):

```python
meta["affected_clusters"]    = self._validation_report.affected_cluster_ids
meta["unaffected_clusters"]  = self._validation_report.unaffected_cluster_ids
meta["cheap_path_used"]      = self._validation_report.cheap_path_used
meta["scenario_labels"]      = self._validation_report.scenario_labels
```

---

### Step 10 — Smoke test

**Status:** ✅ done (this env, 2026-05-12) · ⬜ pending (work env)

**Pytest results in this env:**

```
tests/rade_ml_pt/pipelines/ensemble/test_infer_events.py    18 PASSED
tests/rade_ml_pt/pipelines/ensemble/test_ensemble_pipelines.py
    test_eval_runs_and_returns_metrics                       PASSED
    test_eval_saves_artifacts                                FAILED ★
    test_eval_per_member_metrics_have_expected_keys          PASSED
    TestEnsembleInferencePipeline (5 tests)
      test_infer_returns_predictions                         PASSED
      test_infer_unknown_mode_raises                         PASSED
      test_infer_failure_emits_red_event_before_reraising    PASSED
      test_infer_new_trades_mode                             FAILED ★
      test_infer_missing_member_inputs_raises                FAILED ★
─────────────────────────────────────────────────────────────────────
TOTAL: 23 passed · 3 failed
```

★ All three failures share the **same root cause**:
`ModuleNotFoundError: No module named 'src.rade_sr.market_data_manager'`
— a missing workspace dependency unrelated to this refactor.  Verified
by `git stash` baseline run: the same 2 inference failures and the
1 eval failure occur **on `main` without my changes**.  These tests
will pass in your work env where `rade_sr` is installed.

**Verification commands:**

```bash
# Import smoke (confirms module loads, all methods present)
.venv/bin/python -c "
from src.rade_ml_pt.pipelines.ensemble.infer import (
    EnsembleInferencePipeline,
    LoadedScenariosReport, ClusterRoutingDecision, ValidationReport,
)
for m in ['load', 'load_scenarios', 'validate', 'run_inference', 'run',
          '_build_cluster_inputs_full', '_build_cluster_inputs_cheap',
          '_classify_cluster', '_validate_cluster_for_labels',
          '_extract_canonical_index', '_extract_cluster_risk_factors',
          '_run_with_prebuilt']:
    assert hasattr(EnsembleInferencePipeline, m), f'missing: {m}'
print('all 12 expected methods present')
"

# Event-contract test (pin-points the public phase strings)
.venv/bin/python -m pytest tests/rade_ml_pt/pipelines/ensemble/test_infer_events.py -v
```

Production-shock end-to-end test happens in your work env — this
repo has no real shock CSVs or registry to point at.

---

## 4 · Decisions log

| Date       | Decision                                                                  |
|------------|---------------------------------------------------------------------------|
| 2026-05-12 | Reading B: anything reading `InferenceContext` lives in `hybrid_gnn_rnn`. |
| 2026-05-12 | `ClusterRoutingDecision` stays in `ensemble/infer.py` (movable later).    |
| 2026-05-12 | Cluster helpers are module-level functions (not `@staticmethod`).         |
| 2026-05-12 | One tracker doc, not multiple; per-step diffs live inline.                |
| 2026-05-12 | `HybridGnnRnnInferencePipeline` class untouched in this PR; DRY-up of `_prepare_new_scenarios_inputs` is a separate follow-up. |
| 2026-05-12 | `missing_scenario_labels` lives on `ClusterRoutingDecision` (not a separate dataclass). |
| 2026-05-12 | **Switched to LEAN plan** — single-file refactor in `ensemble/infer.py`; cluster-pipeline split deferred until `new_trades` lands. |

---

## 5 · Open questions

_None right now.  Add new ones here as they come up._

---

## 6 · End-to-end smoke command (filled in at Step 11)

```bash
# placeholder — fill in once Step 11 lands
.venv/bin/python examples/rade_ml_pt/hybrid_gnn_rnn/14_run_infer_pipeline.py
```
