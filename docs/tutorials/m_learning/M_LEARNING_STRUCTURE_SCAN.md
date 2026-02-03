# m_learning structure scan vs NOTEBOOK_PLAN / design

This document lists modules and files that **do not align** with the structure set out in `NOTEBOOK_PLAN.md` and `docs/architecture/m_learning_design.md`. Use it to decide what to remove or relocate to keep the library clean.

**Design rules (short):**
- **Data** mirrors **models**: only `data/<model>/` and `models/<model>/` are model-specific (e.g. `data/pricing/`, `data/gnn_rnn_hybrid/`).
- Data builders output **`tf.data.Dataset`(s)**; canonical entry points are `data/<model>/build.py` (e.g. `build_pricing_data()`, `build_gnn_data()`).
- **One model class per ML model**; batching comes from the data pipeline, not a separate “batched” model class.
- Shared data building blocks live in **`data/common/`** (or `utilities/`).
- Generic pipelines: **pipeline/** (training, evaluation, inference, tuning); **inference/** for save/load/artifact; **evaluation/** for Evaluator and metrics.

---

## 1. Scripts / modules not in keeping (candidates for removal or move)

### 1.1 Data layer

| Item | Location | Issue | Recommendation |
|------|----------|--------|-----------------|
| **Legacy pricing (MLDataset API)** | `data/pricing.py` (top-level module) | Returns `MLDataset` (numpy), not `tf.data.Dataset`. Design: pricing data comes from `data/pricing/build.py` → `build_pricing_data()`. Also: with `data/pricing/` package present, `data.pricing` in Python refers to the **package**, so `from src.m_learning.data.pricing import build_pricing_dataset_from_mc` in `data/__init__.py` would fail (package does not export those names). | **Remove** if you drop the legacy API; then remove `build_pricing_dataset_from_mc` / `build_pricing_dataset_from_analytic` from `data/__init__.py` and from `src/m_learning/__init__.py`. Or keep the file but **rename** (e.g. `data/pricing_legacy.py`) and re-export from there so the package name `data.pricing` stays the package only. |
| **Calibration data** | `data/calibration.py` | Not under `data/<model>/`. Design has no `models/calibration/`; calibration is a use case. So this is a standalone data builder not following “data mirrors models”. | **Move** to `data/calibration/build.py` and treat as a future `models/calibration` pair, or **keep** as a special-case “application” builder and document it; or remove if unused. |
| **Delta hedging data** | `data/delta_hedging.py` | Not under `data/<model>/`. Used by `evaluation/delta_hedging_backtest.py` and tests. It’s data for an evaluation/backtest use case, not for a “delta hedging model” in `models/`. | **Move** to `evaluation/` (e.g. `evaluation/delta_hedging_data.py`) as data for the backtest, or to `data/common/` if shared; or keep and document as application-specific. |
| **GNN synthetic** | `data/gnn_synthetic.py` (top-level) | Design: GNN synthetic generation belongs under `data/gnn_rnn_hybrid/` (e.g. `synthetic.py` or inside `build.py`). Currently `data/gnn_rnn_hybrid/build.py` imports from here; `models/gnn_rnn_hybrid/config.py` imports `default_hybrid_model_config` from here. | **Move** into `data/gnn_rnn_hybrid/` (e.g. `data/gnn_rnn_hybrid/synthetic.py`) and update imports; move `default_hybrid_model_config` to `models/gnn_rnn_hybrid/config.py` (already re-exported there). |
| **Portfolio (GNN inputs)** | `data/portfolio.py` | Builds GNN inputs and `gnn_inputs_to_tf_dataset`. Design: GNN-specific orchestration belongs in `data/gnn_rnn_hybrid/`; shared helpers in `data/common/`. `data/gnn_rnn_hybrid/build.py` uses `gnn_inputs_to_tf_dataset` from here. | **Move** `gnn_inputs_to_tf_dataset` (and optionally `build_gnn_dataset_from_portfolio`) into `data/gnn_rnn_hybrid/` (e.g. `build.py` or a small `tf_dataset.py`); then remove or shrink `data/portfolio.py`. |
| **Portfolio builder (FX)** | `data/portfolio_builder.py` (top-level) | FX portfolio → GNN data. Design: per-model data in `data/<model>/`. Used by `data/gnn_rnn_hybrid/build.py`. | **Move** into `data/gnn_rnn_hybrid/` (e.g. `portfolio_builder.py` or `fx_portfolio.py`) so all GNN-RNN data lives under `data/gnn_rnn_hybrid/`. |

### 1.2 Models layer

| Item | Location | Issue | Recommendation |
|------|----------|--------|-----------------|
| **Batched wrapper** | `models/gnn_rnn_hybrid/wrapper.py` (`BatchedHybridGnnRnn`) | Design: **one model class per ML model**; batching is defined by the `tf.data.Dataset` from the data builder, not by a separate “batched” model class. | **Option A:** Remove and have the data builder produce batches in the shape the **single** `HybridGnnRnn` expects (no batch dimension on graph inputs), so one model class is used everywhere. **Option B:** Keep but document as a thin Dataset-adapter layer (not a second “model”), and clarify in design that the single logical model is `HybridGnnRnn`; the wrapper is for Dataset compatibility only. |

### 1.3 Inference (two surfaces)

| Item | Location | Issue | Recommendation |
|------|----------|--------|-----------------|
| **Pipeline inference** | `pipeline/inference.py` | Saves/loads **protocol-based** `Trainable` (parameters as JSON, artifact layout: `config.json`, `parameters.json`, `metadata.json`). No Keras `.keras` or normalization stats. | Design and notebook use **`inference/`** (model_io + Predictor + ModelArtifact with `feature_stats`/`target_stats`, `.keras`). So you have **two** inference APIs. |
| **Inference package** | `inference/model_io.py`, `inference/predictor.py` | Keras save/load, `ModelArtifact`, normalization, `artifact.predict(..., denormalize=True)`. This matches the notebook and design. | **Unify:** Prefer the **`inference/`** package as the canonical API (save_model, load_model → ModelArtifact, predict with denorm). Deprecate or remove **`pipeline/inference.py`** for the Trainable/parameters-only path, or document it as “protocol-based / non-Keras” and keep both only if you need both. |

---

## 2. Summary table (quick reference)

| Category | Item | Action to align |
|----------|------|------------------|
| Data | `data/pricing.py` (legacy) | Remove or rename; fix `data/__init__.py` and root `__init__.py`. |
| Data | `data/calibration.py` | Move under `data/calibration/` or treat as app-specific / remove if unused. |
| Data | `data/delta_hedging.py` | Move to `evaluation/` or `data/common/`; or keep as documented exception. |
| Data | `data/gnn_synthetic.py` | Move into `data/gnn_rnn_hybrid/synthetic.py` (or similar); move config to models. |
| Data | `data/portfolio.py` | Move GNN helpers into `data/gnn_rnn_hybrid/`; then remove or shrink. |
| Data | `data/portfolio_builder.py` | Move into `data/gnn_rnn_hybrid/`. |
| Models | `models/gnn_rnn_hybrid/wrapper.py` | Remove batching from “model” and handle in data, or keep as documented Dataset adapter only. |
| Pipeline | `pipeline/inference.py` | Prefer `inference/` as canonical; deprecate or remove pipeline inference, or document both. |

---

## 3. What is already in keeping

- **core/** — protocols, types, config, callbacks.
- **data/dataset.py**, **data/types.py** — generic.
- **data/common/** — shared building blocks.
- **data/pricing/** — `build_pricing_data()` → `tf.data.Dataset`; matches design.
- **data/gnn_rnn_hybrid/** — `build_gnn_data()` → `tf.data.Dataset`; matches design (though it currently depends on top-level gnn_synthetic, portfolio_builder, portfolio).
- **pipeline/** — training, evaluation, tuning; evaluation returns `EvaluationResult`; tuning returns `TuningResult`.
- **evaluation/** — Evaluator, metrics, delta_hedging_backtest.
- **inference/** (package) — save_model, load_model, ModelArtifact, Predictor; matches notebook and design.
- **training/** — Trainer (Keras-oriented).
- **models/pricing/** — MLPPricer, create_mlp_pricer, config; one model class.
- **models/gnn_rnn_hybrid/** — HybridGnnRnn, config, layers; wrapper is the only exception (see above).
- **utilities/** — trade_attribute_encoder, trade_graph_builder.
- **calibration/** — training_manager (design allows it; can be folded into pipelines later).

---

## 4. Dependency notes before removing

- **data/pricing.py**: `data/__init__.py` and `src/m_learning/__init__.py` re-export `build_pricing_dataset_from_mc`, `build_pricing_dataset_from_analytic`. Tests: `tests/unit/m_learning/data/test_pricing.py` targets the legacy API. If you remove the file, switch tests to `data/pricing/build.py` (build_pricing_data) and fix both `__init__.py` files.
- **data/calibration.py**: Re-exported from `data/__init__.py`; no other internal references found. Safe to move or remove after checking external/repo usage.
- **data/delta_hedging.py**: Used by `evaluation/delta_hedging_backtest.py` and `tests/unit/m_learning/data/test_delta_hedging.py`, `tests/unit/m_learning/evaluation/test_delta_hedging_backtest.py`. Any move must update these imports.
- **data/gnn_synthetic.py**: Used by `data/gnn_rnn_hybrid/build.py`, `models/gnn_rnn_hybrid/__init__.py`, `models/gnn_rnn_hybrid/config.py`. Move together with import updates.
- **data/portfolio.py**: Used by `data/gnn_rnn_hybrid/build.py` for `gnn_inputs_to_tf_dataset`. Move that function into `data/gnn_rnn_hybrid/` and update build.py.
- **data/portfolio_builder.py**: Used by `data/gnn_rnn_hybrid/build.py`. Move into `data/gnn_rnn_hybrid/` and update imports.
- **pipeline/inference.py**: Used by `pipeline/__init__.py` and tests in `tests/unit/m_learning/pipeline/test_inference.py`. Notebook and design use `inference/` package; consolidate and then adjust pipeline exports and tests.

After you review this list, you can remove or relocate the highlighted scripts and update imports and tests accordingly.
