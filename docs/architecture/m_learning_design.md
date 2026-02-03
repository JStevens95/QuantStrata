# Machine Learning Module Design

**Status:** ✅ IMPLEMENTED (January 2026) — See `m_learning_design_alignment.md` for implementation notes.

This document describes the `src/machine_learning` structure that separates **generic ML infrastructure** (pipelines, datasets, evaluation, tuning) from **model-specific** components under `models/`, with a plug-and-play layout per model.

---

## 1. Current structure (review)

```
src/machine_learning/
├── __init__.py
├── core/           # Base classes, config, protocols, types, callbacks
├── data/            # TFDataset, normalization; pricing/calibration/portfolio/gnn data builders
├── pipeline/        # run_training, evaluate_model, load_model, predict, save_model (protocol-based)
├── training/        # Trainer, fit_model (Keras-oriented)
├── evaluation/      # Evaluator, EvaluationResult, metrics, delta_hedging_backtest
├── inference/       # save_model, load_model, ModelArtifact, Predictor
├── calibration/     # training_manager
├── utilities/       # trade_attribute_encoder, trade_graph_builder
└── models/          # Model-specific
    ├── pricing/     # MLPPricer
    └── gnn_rnn_hybrid/  # HybridGnnRnn, layers, wrapper, dataset helpers
```

**What’s already in place**

- **Generic pipelines:** `pipeline/` provides protocol-based training loop, evaluation, and inference; `core/protocols.py` defines `Trainable`; `core/types.py` has `TrainingConfig`, `TrainingResult`, `EvaluationResult`, `CheckpointInfo`.
- **TF datasets:** `data/dataset.py` has `TFDataset`, normalization, splits; `data/portfolio.py` has `gnn_inputs_to_tf_dataset`; GNN model has `create_gnn_tf_dataset`.
- **Standardised results:** `TrainingResult` and `EvaluationResult` in `core/types` with `to_dict`/`to_json`; pipeline `evaluate_model()` returns `EvaluationResult`; `evaluation/evaluator.py` has a richer `EvaluationResult` (predictions, residuals, summary).
- **Evaluation:** Two layers—`evaluation/` (Evaluator, metrics, backtest) and `pipeline/evaluation.py` (standardised `evaluate_model` returning core `EvaluationResult`).

**Gaps vs target design**

1. **Hyperparameter tuning** – No dedicated pipeline or module; needed for grid/random search or Optuna-style tuning.
2. **Result unification** – Two `EvaluationResult`-style concepts (core vs evaluator); benefit from a single canonical shape for “standardised evaluation output”.
3. **Generic TF dataset entry point** – Builders are spread across `data/` and `models/gnn_rnn_hybrid`; a clear place for “generic tf dataset builders” would help.
4. **Models as plug-and-play** – Each model (e.g. `gnn_rnn_hybrid`, `pricing`) should be self-contained: config, model, optional dataset builder, optional layers, and a clear contract with the generic pipelines.

---

## 2. Target design principles

- **Top level (`src/machine_learning/`)**  
  Contains only **generic** components: pipelines (training, evaluation, inference, **tuning**), TF dataset building, core types/protocols, evaluation scripts and **standardised result** types, and shared utilities. No model-specific logic at this level.

- **`models/`**  
  Everything **model-specific** lives here. Each subpackage is a self-contained “model component” that:
  - Exposes a clear **contract** (e.g. `Trainable` or Keras model + optional adapter).
  - Can be trained/evaluated with the **same** generic pipelines and evaluation.
  - Optionally provides its own config schema, layers, and dataset builder (e.g. `create_gnn_tf_dataset`).

- **Standardised results**  
  Training, evaluation, and (future) tuning should return a small set of **canonical** result types (e.g. `TrainingResult`, `EvaluationResult`, `TuningResult`) with a consistent serialisation shape (`to_dict`, `to_json`, optional `from_json`) so that scripts and tooling can consume them uniformly.

- **Plug-and-play**  
  Adding a new model = adding a new subpackage under `models/` and, if needed, registering or importing it in `models/__init__.py`. Generic pipelines stay unchanged.

---

## 3. Data mirrors models: one data section per model

**Principle:** For every ML model run off this structure, the **only** model-dependent parts are:

1. **Data building / preprocessing** for that model → lives under **`data/<model_name>/`**
2. **The ML model itself** → lives under **`models/<model_name>/`**

So **`data/` is structured to match `models/`**: each model has a corresponding data subpackage. Generic pipelines (training, evaluation, inference, tuning), core types, and evaluation scripts stay model-agnostic; they only need “inputs + model”. This keeps the surface area of model-specific code minimal and makes it obvious where to add or change behaviour for a given model.

- **Symmetry:** `data/pricing/` pairs with `models/pricing/`, `data/gnn_rnn_hybrid/` with `models/gnn_rnn_hybrid/`. Everything for “model X” lives in two places: data for X, model for X.
- **Shared building blocks:** Utilities used by multiple data modules (e.g. encoders, market/portfolio builders) live in **`data/common/`** (or `utilities/`) and are *used by* `data/<model>/`. So the *orchestration* of data building is per-model; the *building blocks* are shared and DRY.
- **Data/<model>/ responsibility:** Build and preprocess everything needed for that model: raw data → features/targets → **output as `tf.data.Dataset`** (see below). The model package then contains **only** the network, config, and layers—no data construction.

**Data builder output: `tf.data.Dataset` for all models**

Every `data/<model>/` builder should **output `tf.data.Dataset`** (or a thin wrapper that exposes train/val/test Datasets and optional metadata such as normalisation stats). That way:

- **Single interface:** Training, evaluation, inference, and tuning pipelines all consume the same type; no branching on “arrays vs Dataset”.
- **Repeatable:** Batching, shuffling, and prefetching are handled once in the data builder; pipelines stay generic and predictable.
- **Clean:** New models only need to implement “raw data → `tf.data.Dataset`”; the rest of the pipeline is unchanged.

So the contract for `data/<model>/` is: **build and preprocess model-specific data, then return `tf.data.Dataset`(s)** (e.g. `train_ds`, `val_ds`, `test_ds` or a small holder type). Generic utilities in `data/dataset.py` (TFDataset, NormalizationStats, split helpers) can be used inside each builder to produce those Datasets consistently.

---

## 4. Proposed layout

```
src/machine_learning/
├── __init__.py                 # Public API: core, data, pipelines, evaluation, inference, tuning
├── core/                       # Base, config, protocols, types, callbacks
│   ├── base.py, config.py, callbacks.py, protocols.py, types.py
├── data/                       # Dataset utilities + per-model data (mirrors models/)
│   ├── dataset.py              # Generic: TFDataset, NormalizationStats, split helpers
│   ├── types.py                # Generic data types used across data modules
│   ├── common/                 # Shared building blocks used by data/<model>/
│   │   ├── __init__.py
│   │   └── (e.g. encoders, market/portfolio builders, normalization helpers)
│   ├── pricing/                # Data building & preprocessing for models/pricing → tf.data.Dataset
│   │   ├── __init__.py
│   │   ├── build.py            # build_pricing_data(); outputs train_ds, val_ds, test_ds (tf.data.Dataset)
│   │   └── (optional) synthetic.py, loaders.py
│   └── gnn_rnn_hybrid/         # Data building & preprocessing for models/gnn_rnn_hybrid → tf.data.Dataset
│       ├── __init__.py
│       ├── build.py            # build_gnn_data(); outputs train_ds, val_ds, proj_ds (tf.data.Dataset)
│       ├── synthetic.py        # Synthetic GNN data generation (or under build.py)
│       └── portfolio_builder.py # FX portfolio → features, adjacency, PnL (or in common if shared)
├── pipelines/                  # Generic pipelines only
│   ├── __init__.py
│   ├── training.py             # run_training, TrainingLoop → TrainingResult
│   ├── evaluation.py           # evaluate_model → EvaluationResult (canonical)
│   ├── inference.py            # load_model, predict, save_model
│   └── tuning.py               # Hyperparameter tuning → TuningResult
├── training/                   # Keras-style Trainer, fit_model
│   └── trainer.py
├── evaluation/                 # Evaluator, metrics, standardised result helpers
│   ├── evaluator.py, metrics.py, results.py (optional), delta_hedging_backtest.py
├── inference/                  # Model I/O, Predictor
├── tuning/                     # Tuning strategies, TuningResult (if not only in pipelines)
├── utilities/                  # Cross-cutting helpers (e.g. graph builders used by data/common)
├── calibration/                # Training manager (or fold into pipelines later)
└── models/                     # MODEL-SPECIFIC ONLY – plug and play (no data construction here)
    ├── __init__.py             # Optional: registry or re-export create_*()
    ├── pricing/
    │   ├── __init__.py
    │   ├── model.py            # MLPPricer, create_mlp_pricer
    │   └── config.py           # Config schema / defaults
    └── gnn_rnn_hybrid/
        ├── __init__.py
        ├── model.py            # HybridGnnRnn only (one model instance; batching from data)
        ├── config.py           # default_hybrid_model_config, config schema
        └── layers/             # gnn_layers, rnn_layers, fusion, attention, projection
            ├── __init__.py
            └── (gnn_layers.py, rnn_layers.py, fusion_layer.py, attention_layer.py, projection_layer.py)
```

**Summary:** `data/<model>/` = everything needed to produce training/eval inputs for that model (build, preprocess, output `tf.data.Dataset`). `models/<model>/` = architecture, config, layers only. No dataset builder inside `models/`; the model consumes what `data/<model>/` provides. **One model class per ML model**; batching is defined by the Dataset from the data builder, not by a separate “batched” model class (see below).

**Naming:** `pipeline/` → `pipelines/` is optional; the important part is that training, evaluation, inference, and tuning live in one place as “generic pipelines”.

---

## 5. Generic components (top-level)

| Area | Purpose |
|------|--------|
| **core/** | Base classes, `TrainingConfig`, `Trainable` protocol, `TrainingResult`, `EvaluationResult`, callbacks. Single source of truth for result dataclasses used by pipelines. |
| **data/** | **Generic:** `dataset.py` (TFDataset, NormalizationStats, split helpers), `types.py`. **Per-model:** `data/<model>/` mirrors `models/<model>/`—build and preprocess data for that model and **output `tf.data.Dataset`(s)** (e.g. train_ds, val_ds, test_ds) so pipelines have a single, repeatable interface. **Shared:** `data/common/` for building blocks (encoders, market/portfolio builders) used by multiple `data/<model>/` modules. |
| **pipelines/** | `run_training()` → `TrainingResult`; `evaluate_model()` → `EvaluationResult`; `load_model`, `predict`, `save_model`; and `run_tuning()` → `TuningResult`. All model-agnostic; they consume “data + model”. |
| **training/** | Keras-oriented `Trainer` / `fit_model` for models that use Keras natively; can call into pipeline training or stay as a convenience layer. |
| **evaluation/** | Evaluator class, metrics, and evaluation scripts; all produce or use the **canonical** `EvaluationResult` so that standardised results are consistent. |
| **inference/** | Model save/load and predictor interface; unchanged. |
| **tuning/** | Hyperparameter search (e.g. grid, random, or Optuna). Returns standardised `TuningResult` (best config, best score, trials, optional checkpoints). |
| **utilities/** | Cross-cutting helpers (e.g. graph builders) used by `data/common/` or multiple modules. |

---

## 6. Standardised results

- **TrainingResult** (existing in `core/types`)  
  - `history`, `final_epoch`, `best_epoch`, `best_train_loss`, `best_val_loss`, `checkpoints`, `config`, `training_time_seconds`, `metadata`.  
  - Keep as the single type returned by `run_training` / training pipeline.

- **EvaluationResult**  
  - Unify on one shape: either the current `core/types` version (loss, metrics, loss_curves, pricing_error, metadata) or an extended version that also includes `predictions`, `targets`, `residuals` and a `summary()` method (from current evaluator).  
  - Recommendation: one canonical type in `core/types` (or `evaluation/results.py`) that supports both pipeline and Evaluator use cases, with `to_dict`/`to_json`/`from_json` and optional `summary()`.

- **TuningResult** (new)  
  - e.g. `best_config`, `best_score`, `best_checkpoint_path`, `trials` (list of config + score), `metadata`.  
  - Returned by the new tuning pipeline; same serialisation pattern as above.

---

## 7. Model-specific layout (plug-and-play)

Each model has **two** model-dependent surfaces that mirror each other:

| Surface | Location | Responsibility |
|--------|----------|----------------|
| **Data** | `data/<name>/` | Build and preprocess data for this model; output **`tf.data.Dataset`(s)**. Batch size and structure are defined here. May use `data/common/` for shared builders. |
| **Model** | `models/<name>/` | Architecture, config, layers only. No data construction. Consumes whatever `data/<name>/` produces. |

**One model instance per ML model; batching is a data concern**

There should be **one** model class per ML model (e.g. `HybridGnnRnn`, `MLPPricer`). Batching is **not** part of the model: it is configurable and decided by the **data builder**. The `tf.data.Dataset` produced by `data/<model>/` defines batch size, structure, and any tiling/expansion (e.g. for graph batches); the training manager receives that Dataset and feeds it to the model. The model instance stays the same whether we train with batched or unbatched data—the Dataset passed in defines the batching. So we do **not** have separate “batched” vs “unbatched” model classes (e.g. no `BatchedHybridGnnRnn`); we have one model (`HybridGnnRnn`) and the data pipeline supplies batches in the format the model expects.

**models/<name>/** layout:

- **`__init__.py`** – Re-exports the public API: model class, `create_*`, config builder. No dataset builder (that lives in `data/<name>/`).
- **`model.py`** – Model definition and factory (e.g. `create_mlp_pricer`, `HybridGnnRnn`). One class per model; no separate batched variant.
- **`config.py`** – Model-specific config (e.g. `default_hybrid_model_config`, schema).
- **`layers/`** (optional) – Submodules for layers (e.g. GNN, RNN, fusion, attention, projection).

**Contract with generic pipelines**

- Models either implement the `Trainable` protocol (e.g. via `KerasTrainableAdapter`) or are used with the Keras `Trainer`/`fit_model`; the **same** pipelines handle training, evaluation, and inference.
- **Data** for a model is produced only by `data/<name>/`. **Model** code lives only in `models/<name>/`. Pipelines and evaluation are generic and only need “data (from data/<name>/) + model (from models/<name>/)”.

This keeps the rule: **the only model-dependent parts are data preprocessing (data/<name>/) and the ML model itself (models/<name>/).**

---

## 8. Summary

- **Generic:** core, pipelines (training, evaluation, inference, tuning), generic data utilities (`data/dataset.py`, `data/common/`), evaluation, inference, utilities. No model-specific logic here.
- **Model-dependent (only two places):**  
  - **`data/<model>/`** – Data building and preprocessing for that model; **output is always `tf.data.Dataset`(s)** so pipelines are clean and repeatable.  
  - **`models/<model>/`** – The ML model (architecture, config, layers). No data construction.  
- **Result types:** One canonical `TrainingResult`, one canonical `EvaluationResult`, and `TuningResult`, all with consistent serialisation.

**Next steps (when implementing):** (1) Add `pipelines/tuning.py` and `TuningResult` in core/types; (2) Unify `EvaluationResult` and document it; (3) Restructure `data/` into generic + `data/common/` + `data/pricing/`, `data/gnn_rnn_hybrid/` (move current pricing/calibration/portfolio/gnn_synthetic/portfolio_builder into the appropriate per-model or common); (4) Refactor `models/` so each subpackage contains only model + config + layers (no dataset builder; dataset lives in `data/<model>/`).

---

## 9. Same structure for reinforcement learning (q_learning)

The same design applies to **reinforcement learning** (`src/q_learning/`): generic infrastructure at the top level, with **experience/data** and **agents** as the only agent-dependent parts. Pipelines stay generic and consume “experience source + agent”.

**Mapping from ML to RL**

| ML (machine_learning) | RL (q_learning) |
|-----------------|------------------|
| **data/<model>/** | **data/<agent>/** (or **env/<agent>/**) – environment setup, experience collection, replay/trajectory building for that agent. Output: a **consistent interface** the pipeline consumes (e.g. `tf.data.Dataset` over transitions, or Env + replay buffer sampler). |
| **models/<model>/** | **agents/<agent>/** – one agent class per algorithm (e.g. DQN, Double DQN). Architecture, config, layers only. No environment or data construction. Batching/minibatch size is defined by the experience side. |
| **pipelines/** (training, evaluation, inference, tuning) | **pipelines/** – `run_learning()` (or `run_training()`), `evaluate_policy()`, `load_agent()`, `save_agent()`, tuning. Same idea: generic loops that take “experience + agent”. |
| **core/** (protocols, config, types, results) | **core/** – e.g. `Agent` protocol (act, update, get_parameters, set_parameters), `LearningConfig`, `TrainingResult`, `EvaluationResult` (episode returns, loss curves), `TuningResult`. |
| **evaluation/** | **evaluation/** – metrics (mean return, std, success rate), standardised evaluation results. |

**RL-specific nuance**

- **Experience interface:** In RL, “data” is often **online** (env steps + replay buffer) rather than a pre-built Dataset. The contract can still be “one consistent interface”: e.g. `data/<agent>/` provides an **environment factory** plus a **replay/trajectory sampler** that yields batches of transitions `(s, a, r, s', done)` in the shape the agent expects—either as a `tf.data.Dataset` over stored transitions or a callable the training loop uses. So “data builder output” = whatever the pipeline is defined to consume (Dataset of transitions, or Env + buffer with a standard `.sample(batch_size)`).
- **One agent class per algorithm:** Same as ML: one `DQN` (or `DqnAgent`) class, not separate “batched” vs “unbatched” variants; minibatch size and sampling are decided by the experience/data side.
- **Standardised results:** TrainingResult (episode returns, loss curves, steps), EvaluationResult (mean return, std, etc.), TuningResult (best config, best return). Same serialisation pattern as ML.

**Proposed q_learning layout (mirroring machine_learning)**

```
src/q_learning/
├── __init__.py
├── core/                 # Agent protocol, LearningConfig, TrainingResult, EvaluationResult, TuningResult
├── data/                 # Generic replay/trajectory utilities + per-agent experience (mirrors agents/)
│   ├── dataset.py        # Generic: replay buffer, transition types, maybe tf.data from buffer
│   ├── common/           # Shared: env base, reward shaping, etc.
│   └── <agent>/          # e.g. dqn, ppo – env setup, experience collection, output consistent interface
├── pipelines/            # run_learning, evaluate_policy, load_agent, save_agent, tuning
├── evaluation/           # Metrics, standardised evaluation results
├── inference/            # Load/save agent, act (policy deployment)
└── agents/               # AGENT-SPECIFIC ONLY – one class per algorithm
    ├── __init__.py
    ├── dqn/
    │   ├── __init__.py
    │   ├── agent.py      # DQN agent (one class; batching from data)
    │   ├── config.py
    │   └── layers/       # optional
    └── ...
```

So yes: the same professional structure (generic pipelines + data mirroring models/agents + one model/agent class per algorithm + standardised results) applies to reinforcement learning; the only adaptation is that “data” becomes “experience” (env + replay/trajectories) with a consistent interface the pipeline consumes.
