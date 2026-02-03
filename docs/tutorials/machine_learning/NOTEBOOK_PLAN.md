# Plan: machine_learning Lifecycle Tutorial Notebook

**Goal:** One comprehensive tutorial notebook that walks through the full ML model lifecycle in the library, with detailed explanation of each component, line-by-line guidance, tables, plots, and clear flow between **data → model → training → evaluation → inference → tuning** pipelines.

**Model used:** Pricing (MLPPricer) so the tutorial stays readable and runnable in one sitting; the same patterns apply to GNN-RNN.

---

## 1. Notebook structure (sections)

| Section | Title | Purpose |
|--------|--------|--------|
| 0 | **Overview & pipeline map** | High-level lifecycle diagram; table of pipelines and their inputs/outputs; where each component lives in `src/machine_learning` |
| 1 | **Configuration** | Single config cell; table of parameters (data, split, training, model); how config flows into later sections |
| 2 | **Data pipeline** | Build data → `tf.data.Dataset`; explain each argument; table of outputs; plot: sample features/targets or split sizes |
| 3 | **Model instance** | Build model from config; explain layers and contract; table: config → model attributes; optional: diagram of model architecture |
| 4 | **Training pipeline** | `model.fit(train_ds, val_ds)`; what happens each epoch; table: history keys; plot: loss curves (train/val) |
| 5 | **Evaluation pipeline** | Standardised evaluation; `EvaluationResult`; table: metrics; plot: predicted vs actual, residuals |
| 6 | **Inference pipeline** | Save model + stats; load; predict; table: artifact layout; optional: plot predictions vs actual for a batch |
| 7 | **Tuning pipeline** | Short section: `run_tuning` with a tiny grid; table: `TuningResult`; plot: trials (config vs score) or bar chart of trial scores |
| 8 | **Flow summary & next steps** | Recap: one diagram showing flow data→model→train→eval→inference→tuning; link to design doc and GNN-RNN |

---

## 2. Section-by-section detail

### Section 0: Overview & pipeline map

- **Markdown:**
  - Short intro: "This notebook demonstrates the full ML lifecycle using the pricing model. Every step uses the same generic pipelines; only the **data builder** and **model** are model-specific."
  - **Table 1:** Pipeline overview  
    | Pipeline | Module | Inputs | Output | Standardised type |
    |----------|--------|--------|--------|--------------------|
    | Data | data.pricing | config (n_samples, splits, …) | train_ds, val_ds, test_ds | PricingDataResult |
    | Model | models.pricing | config | Keras model | MLPPricer |
    | Training | Keras / training | model, train_ds, val_ds | history | — |
    | Evaluation | pipeline.evaluation | model, features, targets | metrics, summary | EvaluationResult |
    | Inference | inference | model, path, stats | artifact, predictions | ModelArtifact |
    | Tuning | pipeline.tuning | objective_fn, search_space | best_config, trials | TuningResult |
  - **Visual:** Mermaid or matplotlib diagram: boxes for Data → Model → Training → Evaluation → Inference → Tuning, with arrows and labels "tf.data.Dataset", "TrainingResult", etc.
  - **Tree:** Short code or text showing `src/machine_learning` layout (core, data, pipeline, training, evaluation, inference, models).

**Deliverables:** One markdown cell with table + diagram; one small code cell that prints the package layout (e.g. `import machine_learning; print(dir(...))` or a static tree).

---

### Section 1: Configuration

- **Markdown:** "All parameters in one place. Training and model configs are separate dictionaries so they can easily be loaded from YAML (or other config files) and still work."
- **Table:** Config parameters grouped (Training / data, split, fit | Model / architecture).
- **Code cell 1 — Training parameters (dict):** `TRAINING_CONFIG = {"n_samples": ..., "train_ratio": ..., "val_ratio": ..., "test_ratio": ..., "batch_size": ..., "epochs": ..., "learning_rate": ..., "seed": ...}`. Short comment: can be loaded from YAML.
- **Code cell 2 — Model parameters (dict):** `MODEL_CONFIG = {"n_features": ..., "hidden_units": [...], "dropout_rate": ..., "use_batch_norm": ...}`. Same note re YAML.
- **Optional:** Table "Parameter → Used in" (e.g. n_samples → data, epochs → training).

**Deliverables:** 1 markdown, 2 code cells (training dict, model dict).

---

### Section 2: Data pipeline

- **Markdown:**
  - "The **data pipeline** is model-specific: we use `data.pricing.build_pricing_data()` so the rest of the lifecycle sees only `tf.data.Dataset`."
  - Bullet list: what the function does (generate samples, normalize, split, batch, shuffle).
- **Code:** Call `build_pricing_data(...)` with config; assign to `data`. Print `data.metadata`, `len(list(data.train_ds))`, etc.
- **Table:** Columns: Output | Type | Description (e.g. train_ds | tf.data.Dataset | Batched, shuffled training data).
- **Line-by-line:** Short comments or a follow-up markdown explaining key arguments (n_samples, train_ratio, batch_size, normalize).
- **Plot 1:** Split sizes (pie or bar: train / val / test counts).
- **Plot 2 (optional):** One batch of features (e.g. first feature vs target) or feature distributions (histograms).

**Deliverables:** 1–2 markdown cells, 1–2 code cells, 1 table, 1–2 plots.

---

### Section 3: Model instance

- **Markdown:**
  - "The **model** is built from `models.pricing`: config drives architecture; batching is defined by the data, not the model."
  - Short note: one model class (MLPPricer), no separate "batched" variant.
- **Code:** `default_pricing_config()` or manual config dict; `create_mlp_pricer(...)`; `model.compile(...)`.
- **Table:** Config key → Effect (e.g. hidden_units → layer sizes, dropout_rate → regularization).
- **Line-by-line:** Explain `create_mlp_pricer(n_features=..., hidden_units=..., ...)` and `compile(optimizer=..., loss=..., metrics=...)`.
- **Visual:** Either (a) `model.summary()` output in a code cell, or (b) a simple architecture diagram (boxes: Input → Dense → … → Output).

**Deliverables:** 1 markdown, 1 code cell (config + build + compile + summary), 1 table, optional diagram.

---

### Section 4: Training pipeline

- **Markdown:**
  - "The **training pipeline** fits the model to the data. We use Keras `model.fit()`; the same pattern works for any model that consumes our `tf.data.Dataset`."
  - What happens each epoch (forward pass, loss, backward pass, validation).
- **Code:** `history = model.fit(data.train_ds, validation_data=data.val_ds, epochs=EPOCHS, verbose=1)`.
- **Table:** History keys (loss, val_loss, mae, val_mae) and what they mean.
- **Plot:** Training and validation loss (and optionally MAE) vs epoch (line plot). Mark best epoch if desired.

**Deliverables:** 1 markdown, 1 code cell, 1 table, 1 plot.

---

### Section 5: Evaluation pipeline

- **Markdown:**
  - "The **evaluation pipeline** produces a **standardised** `EvaluationResult`: same shape for any model, so we can compare runs and log metrics."
  - Mention: pipeline.evaluation.evaluate_model (protocol-based) or Evaluator (Keras); we show one path.
- **Code:** Prepare test features/targets from `data.test_ds` (e.g. unbatch, take N); wrap model in `KerasTrainableAdapter`; call `evaluate_model(adapter, X_test, y_test, metrics=[...])`; print `eval_result.summary()`.
- **Table:** EvaluationResult fields (loss, metrics dict, optional predictions/targets/residuals).
- **Plot 1:** Predicted vs actual (scatter); optionally a 45° line.
- **Plot 2:** Residuals (e.g. histogram or residuals vs actual).

**Deliverables:** 1 markdown, 1 code cell, 1 table, 2 plots.

---

### Section 6: Inference pipeline

- **Markdown:**
  - "The **inference pipeline** saves the model and normalization stats, then loads them for prediction. This is what you’d use for deployment or backtesting."
- **Code:** `save_model(model, path, feature_stats=data.feature_stats, target_stats=data.target_stats)`; `artifact = load_model(path)`; `predictions = artifact.predict(X_sample, denormalize=True)`.
- **Table:** Artifact directory layout (model.keras, config.json, metadata.json, normalization.json).
- **Optional plot:** Same as eval (predictions vs actual) for a small batch after load.

**Deliverables:** 1 markdown, 1–2 code cells, 1 table, optional plot.

---

### Section 7: Tuning pipeline

- **Markdown:**
  - "The **tuning pipeline** runs a small hyperparameter search and returns a standardised `TuningResult`. Same pattern for any model."
- **Code:** Define a simple `objective_fn(config)` that builds model, fits for 1–2 epochs on a tiny subset, returns val loss; call `run_tuning(objective_fn, search_space={"lr": [1e-3, 1e-2], "units": [32, 64]}, method="grid")`; print `result.best_config`, `result.best_score`, `result.trials`.
- **Table:** TuningResult fields (best_config, best_score, trials, metadata).
- **Plot:** Trial scores (e.g. bar chart of each trial’s score or scatter: lr vs score coloured by units).

**Deliverables:** 1 markdown, 1 code cell, 1 table, 1 plot.

---

### Section 8: Flow summary & next steps

- **Markdown:**
  - Recap: "We walked through: Data → Model → Training → Evaluation → Inference → Tuning. Each pipeline consumes standardised inputs and produces standardised outputs."
  - **One final diagram:** Full flow (Data → Model → Training → Evaluation → Inference; Tuning as a wrapper that runs Training+Evaluation repeatedly).
  - Next steps: try GNN-RNN (data.gnn_rnn_hybrid + models.gnn_rnn_hybrid), read design doc, run with your own data.

**Deliverables:** 1 markdown with diagram and bullets.

---

## 3. Visualisations checklist

| # | Visual | Section | Description |
|---|--------|--------|-------------|
| 1 | Pipeline map (Mermaid or matplotlib) | 0 | Boxes and arrows: Data → Model → Training → Eval → Inference → Tuning |
| 2 | Split sizes (pie/bar) | 2 | Train / val / test sample counts |
| 3 | Feature/target sample (optional) | 2 | Scatter or histograms from one batch |
| 4 | Model architecture (optional) | 3 | Simple block diagram or summary text |
| 5 | Loss curves | 4 | Train/val loss (and MAE) vs epoch |
| 6 | Predicted vs actual | 5 | Scatter + 45° line |
| 7 | Residuals | 5 | Histogram or residuals vs actual |
| 8 | Tuning trials | 7 | Bar or scatter of trial scores |
| 9 | Full flow diagram | 8 | Same as 1, repeated for recap |

---

## 4. Code style and pedagogy

- **One concept per cell where possible:** e.g. one cell for "build data", one for "inspect data and plot".
- **Comments:** Inline comments for non-obvious lines (e.g. "Adapter wraps Keras model for protocol-based evaluate_model").
- **Print statements:** Print shapes, `metadata`, `result.summary()`, so the reader sees concrete output.
- **Reproducibility:** Use `SEED` everywhere (data, train, tuning) and mention it in the config section.
- **Runtime:** Keep default N_SAMPLES and EPOCHS moderate so the notebook runs in a few minutes; add a note "Increase for production runs."

---

## 5. File and naming

- **Filename:** `ml_lifecycle_tutorial.ipynb` (or `machine_learning_lifecycle.ipynb`).
- **Location:** `docs/tutorials/machine_learning/`.

---

## 6. Implementation order

1. Create notebook with Section 0 (overview + pipeline map + table).
2. Add Section 1 (config) and Section 2 (data pipeline + table + split plot).
3. Add Section 3 (model + table + summary/diagram).
4. Add Section 4 (training + table + loss curves).
5. Add Section 5 (evaluation + table + predicted vs actual + residuals).
6. Add Section 6 (inference + table).
7. Add Section 7 (tuning + table + trial plot).
8. Add Section 8 (flow summary + next steps).
9. Pass: run all cells, fix any errors, and tighten prose/plots.

---

## 7. Open choices

- **Mermaid vs matplotlib for pipeline diagram:** Mermaid renders in many Markdown viewers and is easy to edit; matplotlib is fully controllable and always visible in the notebook. Recommendation: use **matplotlib** for the main flow diagram so it’s guaranteed to render in Jupyter; optional Mermaid in a markdown cell for readers who support it.
- **Second model (GNN-RNN):** This plan uses only the pricing model. A separate notebook "GNN-RNN lifecycle" can mirror the same sections for `data.gnn_rnn_hybrid` and `models.gnn_rnn_hybrid`; not in scope for this single comprehensive notebook.
- **Trainer vs model.fit:** We use Keras `model.fit()` for clarity; the design also has `training.Trainer` and `pipeline.run_training` (protocol-based). The tutorial can mention "the same flow works with the generic Trainer/run_training" and stick to `model.fit()` for the pricing model.

If this plan looks good, next step is to implement the notebook section by section (creating the file, then filling each section with markdown, code, tables, and plots as above).
