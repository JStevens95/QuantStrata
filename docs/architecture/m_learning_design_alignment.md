# machine_learning design alignment plan

**Status:** ✅ IMPLEMENTED (January 2026)

**Source of truth:** `docs/architecture/m_learning_design.md` (Section 4: Proposed layout; Section 2 & 5: design principles and generic vs model-specific).

This document was the **plan-first** checklist used to:
1. **Rename the package** from `m_learning` to `machine_learning` ✅
2. **Align physical structure** with the proposed layout (pipelines/, model.py, etc.) ✅
3. **Align content** so that scripts in each folder match the design principles ✅

All changes have been implemented.

---

## 0. Package rename: machine_learning → machine_learning

**Recommendation:** Yes. `machine_learning` is clearer and more professional; it avoids the shorthand and makes the package purpose obvious to new contributors and tooling. Doing it in this refactor means one breaking change and one pass of updates.

**Scope:** Rename directory `src/machine_learning/` → `src/machine_learning/` and replace every reference (imports, docs, tests, config, docstrings) from `machine_learning` to `machine_learning`. Grep shows ~218 matches in ~43 files; tests and docs under `docs/tutorials/machine_learning/` and any path containing `machine_learning` should be updated (tutorial folder can be renamed to `machine_learning` for consistency).

**Checklist (do this first, then proceed to structure and content):**

- [ ] **0.1** Rename directory: `src/machine_learning/` → `src/machine_learning/`.
- [ ] **0.2** Replace all import paths: `src.machine_learning` → `src.machine_learning` (every Python file under the package and any file that imports it).
- [ ] **0.3** Update docstrings and comments that say `machine_learning` or `src/machine_learning` to `machine_learning` / `src/machine_learning`.
- [ ] **0.4** Rename docs path: `docs/tutorials/machine_learning/` → `docs/tutorials/machine_learning/` (and update any links or references to that path).
- [ ] **0.5** Update all docs: `docs/architecture/machine_learning_design.md`, `machine_learning_design_alignment.md`, `NOTEBOOK_PLAN.md`, `M_LEARNING_STRUCTURE_SCAN.md`, `component_reference.md`, `ecosystem_diagrams.md`, READMEs, roadmap, etc.: replace `machine_learning` with `machine_learning` (and optionally rename `machine_learning_design.md` → `machine_learning_design.md` for consistency).
- [ ] **0.6** Update tests: any test file or config that references `machine_learning` → `machine_learning`.
- [ ] **0.7** Verification: grep for `machine_learning` (no hits in code or docs except in this alignment doc’s “before” context or history).

After this, the rest of this plan uses **`src/machine_learning/`** as the package path.

---

## 1. Current vs design (structure diff)

| Area | Design (Section 4) | Current | Action |
|------|--------------------|---------|--------|
| **Pipelines directory** | `pipelines/` (plural) | `pipeline/` (singular) | Rename directory; update all imports and docs. |
| **models/pricing** | `model.py` (MLPPricer, create_mlp_pricer), `config.py` | `mlp_pricer.py`, `config.py` | Rename `mlp_pricer.py` → `model.py`; update imports. |
| **models/gnn_rnn_hybrid** | `model.py` (HybridGnnRnn), `config.py`, `layers/` | `hybrid_model.py`, `config.py`, `layers/` | Rename `hybrid_model.py` → `model.py`; update imports. |
| **data/** | dataset.py, types.py, common/, pricing/, gnn_rnn_hybrid/ | Same; pricing has build.py; gnn_rnn_hybrid has build.py, synthetic.py, portfolio_builder.py, dataset_utils.py | No structural change. Design allows optional files; `dataset_utils.py` is an implementation detail. |
| **evaluation/** | evaluator.py, metrics.py, results.py (optional), delta_hedging_backtest.py | evaluator.py, metrics.py, delta_hedging_backtest.py, delta_hedging_data.py | No change required. `delta_hedging_data.py` is data for the backtest; design does not forbid it. Optional `results.py` can be added later if desired. |
| **inference/** | Model I/O, Predictor | model_io.py, predictor.py | No change. |
| **tuning/** | Optional top-level “if not only in pipelines” | No top-level tuning/; tuning lives in pipeline/tuning.py | After rename: tuning stays in `pipelines/tuning.py`. No separate top-level `tuning/` unless you decide to add it later. |
| **core/, training/, utilities/, calibration/** | As listed in design | Present | No change. |

---

## 2. Content alignment (design principles)

Scripts in each folder must match the design: **generic** areas have no model-specific logic; **model-specific** logic lives only in `data/<model>/` and `models/<model>/`.

### 2.1 Generic components (no model-specific logic)

| Area | Allowed | Not allowed | Audit action |
|------|---------|------------|--------------|
| **core/** | Base classes, `TrainingConfig`, `Trainable` protocol, `TrainingResult`, `EvaluationResult`, `TuningResult`, callbacks. Types used by pipelines. | No imports of a specific model (e.g. MLPPricer, HybridGnnRnn) for logic; type hints / adapters (e.g. `Trainable`) are fine. | Ensure no `from …models.pricing` or `…models.gnn_rnn_hybrid` for behaviour; only protocols/types. |
| **data/dataset.py**, **data/types.py** | TFDataset, NormalizationStats, split helpers, generic data types (e.g. MLDataset). Used by `data/<model>/` builders. | No pricing- or GNN-specific logic. | Keep dataset.py and types.py generic; move any model-specific helpers into `data/<model>/` or `data/common/`. |
| **data/common/** | Shared building blocks: encoders, graph builders, normalisation helpers used by **multiple** `data/<model>/` modules. | No orchestration that assumes one model (e.g. “pricing only” or “GNN only”); that belongs in `data/<model>/`. | If a builder is used by only one model, move it into that `data/<model>/`. |
| **pipelines/** | `run_training`, `evaluate_model`, `load_model`, `predict`, `save_model`, `run_tuning`. All consume “data + model” via interfaces (e.g. `Trainable`, Keras model). | No branching on model type (e.g. “if pricing do X, if GNN do Y”); use protocols/adapters only. | Remove any model-name checks; rely on protocol or adapter. |
| **training/** | Keras `Trainer` / `fit_model`; works with any Keras model. | No hard-coded use of a specific model class for training logic; examples in docstrings are fine. | Keep trainer generic; example code can mention `create_mlp_pricer` as usage only. |
| **evaluation/** | Evaluator, metrics, canonical `EvaluationResult`, delta_hedging_backtest (evaluation use-case). | No model-specific evaluation logic (e.g. “pricing evaluator” vs “GNN evaluator”); one generic Evaluator. | Unify on one EvaluationResult shape if there are two; keep backtest as a generic “evaluation script” that uses data from `delta_hedging_data`. |
| **inference/** | Model I/O, Predictor, save/load; works with any Keras model. | Implementation may register custom_objects for loading (e.g. MLPPricer, HybridGnnRnn); keep registration in one place (e.g. a single “register custom models” helper) so inference stays otherwise generic. | Audit: no branching on model type for save/load logic; only registration of classes. |
| **utilities/** | Cross-cutting helpers (e.g. trade_attribute_encoder, trade_graph_builder) used by `data/common/` or `data/<model>/`. | No orchestration that builds a full dataset for one model; that lives in `data/<model>/`. | Keep as shared helpers only. |
| **calibration/** | Training manager (orchestration for training/eval); can use a specific model in examples or config, but the manager itself should be generic (e.g. accepts “model” and “data”). | No hard-coded assumption that the only model is HybridGnnRnn; accept model and data from outside. | Prefer dependency injection of model and data so calibration stays generic. |

### 2.2 Model-specific components (only data + model)

| Area | Allowed | Not allowed | Audit action |
|------|---------|------------|--------------|
| **data/pricing/** | Build and preprocess data for models/pricing **only**. Output: `tf.data.Dataset`(s) (train_ds, val_ds, test_ds) and optional normalisation stats. Use `data/dataset.py` and `data/common/` as helpers. | No GNN or other-model logic. No model class definitions. | Ensure build.py (and optional synthetic/loaders) only produce Datasets for the pricing model. |
| **data/gnn_rnn_hybrid/** | Build and preprocess data for models/gnn_rnn_hybrid **only**. Output: `tf.data.Dataset`(s) (e.g. train_ds, val_ds, proj_ds). May contain synthetic.py, portfolio_builder.py, dataset_utils.py as implementation details. | No pricing or other-model logic. No model class definitions. | Ensure all code here only produces Datasets for the GNN-RNN model. |
| **models/pricing/** | Model class (MLPPricer), factory (create_mlp_pricer), config. **No** data construction; consume what `data/pricing/` provides. | No dataset builder, no `build_pricing_data`-style logic. | Ensure only model.py (or model + config) and no data-building code. |
| **models/gnn_rnn_hybrid/** | Model class (HybridGnnRnn), config, layers. **No** data construction; consume what `data/gnn_rnn_hybrid/` provides. | No dataset builder, no `build_gnn_data`-style logic. | Ensure only model, config, layers; no data-building code. |

### 2.3 Content checklist (to run after structure changes)

- [ ] **C1** core/: No imports of `models.pricing` or `models.gnn_rnn_hybrid` for behaviour; only protocols and types.
- [ ] **C2** data/dataset.py, data/types.py: No pricing- or GNN-specific logic; only generic types and helpers.
- [ ] **C3** data/common/: Only shared building blocks; anything used by a single model lives in `data/<model>/`.
- [ ] **C4** pipelines/: No branching on model type; all logic model-agnostic via protocols/adapters.
- [ ] **C5** training/: Trainer generic; no hard-coded model class for logic.
- [ ] **C6** evaluation/: Single canonical EvaluationResult usage; backtest is a generic evaluation script.
- [ ] **C7** inference/: Custom-object registration isolated; no model-type branching for save/load behaviour.
- [ ] **C8** data/pricing/: Only pricing data → tf.data.Dataset; no model definitions.
- [ ] **C9** data/gnn_rnn_hybrid/: Only GNN data → tf.data.Dataset; no model definitions.
- [ ] **C10** models/pricing/: Only model + config; no data construction.
- [ ] **C11** models/gnn_rnn_hybrid/: Only model + config + layers; no data construction.

---

## 3. Implementation checklist (structure)

All paths below use the package name **`machine_learning`** (after Section 0 rename). If you defer the package rename, substitute `machine_learning` → `machine_learning` in paths.

### 3.1 Rename `pipeline/` → `pipelines/`

### 2.1 Rename `pipeline/` → `pipelines/`

- [ ] **2.1.1** Rename directory: `src/machine_learning/pipeline/` → `src/machine_learning/pipelines/`.
- [ ] **2.1.2** Update imports that reference `src.machine_learning.pipeline` or `pipeline.` (machine_learning only):
  - `src/machine_learning/__init__.py` – if it imports from pipeline, change to `pipelines`.
  - `src/machine_learning/pipelines/__init__.py` – internal imports stay as relative (e.g. `from src.machine_learning.pipelines.training import ...`).
  - `src/machine_learning/core/protocols.py` – docstring: `src.machine_learning.pipeline.training` → `src.machine_learning.pipelines.training`.
  - `src/machine_learning/pipelines/evaluation.py` – docstring example: `from src.machine_learning.pipeline` → `from src.machine_learning.pipelines`.
  - `src/machine_learning/pipelines/training.py` – docstring example: same.
- [ ] **2.1.3** Update docs and notebooks:
  - `docs/tutorials/machine_learning/ml_lifecycle_tutorial.ipynb` – any cell that imports or refers to `pipeline.evaluation`, `pipeline.tuning` → `pipelines.evaluation`, `pipelines.tuning`.
  - `docs/tutorials/machine_learning/NOTEBOOK_PLAN.md` – table and text: `pipeline.` → `pipelines.`.
- [ ] **2.1.4** Update tests (if any) that import from `src.machine_learning.pipeline` → `src.machine_learning.pipelines`.
- [ ] **2.1.5** Search repo for remaining `machine_learning.pipeline` or `machine_learning/pipeline` (code, docs, config) and fix.

**Note:** Do **not** change references to “pipeline” in `src/orchestrator/` or other packages; those refer to orchestrator pipelines, not the machine_learning pipelines directory.

---

### 2.2 Rename `models/pricing/mlp_pricer.py` → `models/pricing/model.py`

- [ ] **2.2.1** Rename file: `src/machine_learning/models/pricing/mlp_pricer.py` → `src/machine_learning/models/pricing/model.py`.
- [ ] **2.2.2** Update imports of the pricing model:
  - `src/machine_learning/models/pricing/__init__.py`: `from src.machine_learning.models.pricing.mlp_pricer import` → `from src.machine_learning.models.pricing.model import` (or `.model`).
  - `src/machine_learning/models/__init__.py`: same.
  - `src/machine_learning/inference/model_io.py`: custom-objects registration – `import src.machine_learning.models.pricing.mlp_pricer` → `import src.machine_learning.models.pricing.model` (so Keras can find MLPPricer when loading).
  - `src/machine_learning/training/trainer.py`: if it imports `create_mlp_pricer` or MLPPricer from mlp_pricer, change to import from `models.pricing.model` or `models.pricing`.
- [ ] **2.2.3** Update docstrings/examples inside `model.py` that mention `mlp_pricer` (e.g. “see mlp_pricer”) to “model” or “pricing model” where it improves clarity. Leave class/function names (e.g. `create_mlp_pricer`) unchanged unless the design explicitly renames them.
- [ ] **2.2.4** Run tests that use pricing model; fix any remaining references.

---

### 2.3 Rename `models/gnn_rnn_hybrid/hybrid_model.py` → `models/gnn_rnn_hybrid/model.py`

- [ ] **2.3.1** Rename file: `src/machine_learning/models/gnn_rnn_hybrid/hybrid_model.py` → `src/machine_learning/models/gnn_rnn_hybrid/model.py`.
- [ ] **2.3.2** Update imports:
  - `src/machine_learning/models/gnn_rnn_hybrid/__init__.py`: `from src.machine_learning.models.gnn_rnn_hybrid.hybrid_model import HybridGnnRnn` → `from src.machine_learning.models.gnn_rnn_hybrid.model import HybridGnnRnn`.
  - `src/machine_learning/calibration/training_manager.py`: `from src.machine_learning.models.gnn_rnn_hybrid.hybrid_model import HybridGnnRnn` → `from src.machine_learning.models.gnn_rnn_hybrid.model import HybridGnnRnn`.
- [ ] **2.3.3** If inference/model_io or any other module registers custom objects for GNN model (e.g. for loading saved models), update the import from `hybrid_model` to `model`.
- [ ] **2.3.4** Inside `model.py`, update any internal docstrings that refer to “hybrid_model” as the file name to “model” if desired. Leave class name `HybridGnnRnn` unchanged.
- [ ] **2.3.5** Run tests that use GNN model; fix any remaining references.

---

### 2.4 Root and docs

- [ ] **2.4.1** `src/machine_learning/__init__.py`: ensure public API imports use the new paths (`pipelines`, `models.pricing.model`, `models.gnn_rnn_hybrid.model`). No need to expose internal file names; re-exports like `evaluate_model`, `run_tuning`, `create_mlp_pricer`, `HybridGnnRnn` are enough.
- [ ] **2.4.2** In `docs/architecture/machine_learning_design.md`, Section 1 “Current structure (review)” can be updated after alignment to reflect the new layout (pipelines/, model.py) so the doc stays accurate.
- [ ] **2.4.3** Optional: add a one-line note at the top of `machine_learning_design.md` that the layout in Section 4 is implemented as of [date], and that `machine_learning_design_alignment.md` was used for the alignment.

---

## 3. Out of scope for this pass

- **EvaluationResult unification** (design Section 6): keep as a separate change; not required for directory/file naming alignment.
- **Adding optional `evaluation/results.py`**: design says optional; skip unless you want it in this pass.
- **Adding top-level `tuning/`**: design says “if not only in pipelines”; current choice is tuning only in pipelines; no change unless you decide otherwise.
- **Renaming classes/functions** (e.g. `create_mlp_pricer` → `create_pricing_model`): design specifies file names and layout, not necessarily public API names; leave as-is unless you explicitly want to rename.

---

## 4. Verification after implementation

- [ ] All unit tests under (or referencing) the package pass (use `machine_learning` after Section 0).
- [ ] Tutorial notebook runs top-to-bottom (path: `docs/tutorials/machine_learning/` after Section 0).
- [ ] Grep for `machine_learning` (no hits after rename). Grep for `.pipeline` (should be `.pipelines`) (no hits except in history or “pipelines”).
- [ ] Grep for `mlp_pricer` and `hybrid_model` in import paths (no hits; only in comments or docstrings if desired).
- [ ] Content checklist C1–C11 (Section 2.3): no violations.

---

## 5. Summary

| Step | Action |
|------|--------|
| **0** | **Package rename:** `machine_learning` → `machine_learning`; update all imports, docs, tests, and tutorial path. |
| **1** | **Structure:** Rename `pipeline/` → `pipelines/`; update all references. |
| **2** | **Structure:** Rename `models/pricing/mlp_pricer.py` → `model.py`; update imports. |
| **3** | **Structure:** Rename `models/gnn_rnn_hybrid/hybrid_model.py` → `model.py`; update imports. |
| **4** | **Structure:** Root __init__ and design doc housekeeping. |
| **5** | **Content:** Run content alignment checklist (Section 2.3: C1–C11); fix any generic vs model-specific violations. |
| **6** | **Verify:** Tests, notebook, greps, and content checklist. |

**Implementation note:** After Section 0 (package rename), use the package name `machine_learning` in all paths in Section 3 (e.g. `src/machine_learning/...`). Add a step to run the content checklist (Section 2.3: C1–C11) and fix any violations before final verification.

Once this checklist is done, the package is renamed to `machine_learning`, the layout matches the proposed structure in the design doc (Section 4), and the content of each folder aligns with the design principles (generic vs model-specific).
