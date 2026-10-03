#!/usr/bin/env python3
"""
Full Pipeline Example: Hybrid GNN-RNN from CacheLoader Data
============================================================

Production-style script that runs the complete Hybrid GNN-RNN workflow from
pre-built data files through training, evaluation, registration, experiment
tracking, inference, and diagnostic plots.

Unlike the synthetic-data examples (01-03), this script expects **real data
files** already on disk -- the same files your job scheduler or data pipeline
would produce.  It loads them via ``CacheLoader.load()`` and runs every
framework stage end-to-end.

Required data files (pickle or parquet):

  - ``elementary_pnl_path``   -- DataFrame [scenarios x elementary_trades]
  - ``target_pnl_path``       -- DataFrame [scenarios x target_trades]
  - ``elementary_attribs_path``-- Dict[str, List[Any]] of trade attributes
  - ``target_attribs_path``   -- Dict[str, List[Any]] of trade attributes

Usage::

    python examples/rade_ml_pt/hybrid_gnn_rnn/04_full_pipeline_from_cache.py

Stages
------
  1. Data build        -- PnL standardisation, dim-reduction, attribute
                          encoding, graph construction, DataLoader creation.
  2. Model build       -- HybridGnnRnn instantiation from default config.
  3. Compile + Train   -- Trainer.fit with callbacks.
  4. Training results  -- Loss history, best epoch, timing.
  5. Evaluation        -- Evaluator with RMSE, MAE, R², MAPE + residual stats.
  6. Model registration-- Persist model + inference artifacts to ModelRegistry.
  7. Experiment tracking-- Log config + metrics to ExperimentTracker.
  8. Inference          -- Round-trip via InferenceRunner from registry.
  9. Diagnostic plots  -- Training curves + evaluation residual/prediction plots.
"""
from __future__ import annotations

import sys
import logging
import tempfile
from pathlib import Path

# Add the project root to sys.path so we can import src.rade_ml_pt.*
# parents[3] walks up from examples/rade_ml_pt/hybrid_gnn_rnn/ -> project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Reproducibility & PyTorch environment setup
# ---------------------------------------------------------------------------
# Fix Python's hash seed so dict/set ordering is deterministic across runs.
import os
os.environ["PYTHONHASHSEED"] = "42"

import torch
import numpy as np

# Seed all random number generators for reproducible training.
# Without this, weight initialisation, dropout masks, and data shuffling
# would differ between runs, making results non-comparable.
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)            # seed all GPU devices
    torch.backends.cudnn.deterministic = True  # force deterministic cuDNN ops
    torch.backends.cudnn.benchmark = False     # disable auto-tuner (non-deterministic)

# Set up a consistent log format so every framework module writes timestamps,
# logger names, and severity levels in the same style.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-45s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("example.full_pipeline")


# ======================================================================
# 1.  Configuration
# ======================================================================

def build_configs(job: dict, workdir: Path) -> "PipelineConfig":
    """
    Build the full pipeline configuration pointing at real data files.

    Parameters
    ----------
    job : dict
        ``{"cluster_info": {"elementary_pnl_path": ..., ...}}`` pointing to
        pickle/parquet files already on disk.  This is the same structure
        that a production job scheduler would produce.
    workdir : Path
        Root directory for artifacts, registry, and experiment tracking output.

    Returns
    -------
    PipelineConfig
        The single configuration object that every pipeline stage reads from.
    """
    # Lazy imports -- keeps the top of the file clean and avoids circular
    # imports when the module is imported but not executed.
    from src.rade_ml_pt.pipelines.config import PipelineConfig
    from src.rade_ml_pt.data.hybrid_gnn_rnn.config import (
        HybridGnnRnnDataConfig,
        FolderEnvironmentConfig,
        DimensionalityConfig,
        BasisSelectionConfig,
        GraphBuilderConfig,
        AttributeEncoderConfig,
    )
    from src.rade_ml_pt.core.config import (
        TrainingConfig,
        OptimizerConfig,
        EarlyStoppingConfig,
        ReduceLrConfig,
    )

    # All output (model files, plots, experiment logs) goes under workdir/artifacts/
    artifacts_dir = workdir / "artifacts"
    registry_dir = workdir / "artifacts" / "registry"
    tracking_dir = workdir / "artifacts" / "experiments"

    # ------------------------------------------------------------------
    # Data pipeline config
    # ------------------------------------------------------------------
    # Controls how raw PnL + attributes are processed into train/val/test
    # DataLoaders that the model consumes.
    data_config = HybridGnnRnnDataConfig(
        # Where intermediate files are written (e.g. scaled PnL, graphs)
        folders=FolderEnvironmentConfig(root_folder=str(workdir)),

        # Chronological split ratios: 85% train / 10% val / 5% test
        validation_split=0.10,
        test_split=0.05,

        # seq_length=1 means the RNN sees one time-step per sample.
        # Increase for multi-day PnL forecasting.
        seq_length=1,

        # Batch size for DataLoaders -- 32 is a good starting point.
        # Larger batches smooth gradients but use more memory.
        batch_size=32,
        shuffle=True,
        cache=False,
        drop_remainder=False,

        # "standard" applies sklearn StandardScaler (zero-mean, unit-variance).
        # Alternatives: "minmax", "robust", "none".
        transform_type="standard",

        # PCA-based dimensionality reduction on elementary PnL.
        # Reduces correlated trades into a compact basis so the GNN sees
        # the essential variance without redundancy.
        dimensionality=DimensionalityConfig(
            reduction_mode="basis_selection",
            basis_selection=BasisSelectionConfig(
                var_threshold=0.9999,    # retain 99.99% of variance
                method="pca",
                max_components=200,      # hard cap on PCA components
            ),
        ),

        # k-NN graph builder config.
        # k=5 connects each trade to its 5 nearest neighbours based on a
        # weighted distance over the attribute features below.
        # The alpha_* weights control how much each attribute contributes
        # to the distance metric (higher = more important).
        graph_builder=GraphBuilderConfig(
            k=5,
            distance_metric="euclidean",
            include_quota=False,
            alpha_moneyness=1.0,
            alpha_maturity=1.0,
            alpha_delta=1.0,
            alpha_vega=1.0,
            alpha_prod_type=1.0,
            alpha_prod_subtype=0.5,    # lower weight for subtypes
            alpha_underlying=1.0,
            alpha_underlying_rf=0.5,
        ),

        # Trade attribute encoder -- turns raw attribute dicts into numeric
        # feature vectors for the graph builder and the GNN input layer.
        # numeric_keys    -> StandardScaler normalisation
        # categorical_keys-> one-hot encoding
        # multi_label_keys-> multi-hot encoding (e.g. multiple risk factors)
        attribute_encoder=AttributeEncoderConfig(
            numeric_keys=["moneyness", "yrs_to_maturity", "delta", "vega"],
            categorical_keys=["product_type", "product_subtype", "trade_type"],
            multi_label_keys=["underlying_risk_factors"],
            num_decay_terms=3,  # exponential decay features for maturity
        ),

        # Disable optional plots during the data build (enable for debugging)
        plot_trade_graph=False,
        plot_pnl_distribution=False,
        save_intermediate_files=False,
        seed=42,
    )

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    # Controls the training loop: how many epochs, which optimizer,
    # when to stop, and how to adjust the learning rate.
    training_config = TrainingConfig(
        epochs=500,       # upper bound -- early stopping will cut this short
        loss="mse",       # mean squared error (good for regression tasks)
        metrics=["mse", "mae"],

        # Adam optimiser with standard betas.
        # learning_rate=1e-3 is a safe default; tune via the TunePipeline.
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=1e-3,
            beta_1=0.9,
            beta_2=0.999,
        ),

        # Stop training when val_loss hasn't improved for 30 consecutive
        # epochs.  restore_best_weights rolls the model back to the epoch
        # with the lowest val_loss so you always get the best checkpoint.
        early_stopping=EarlyStoppingConfig(
            patience=30,
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
        ),

        # Reduce learning rate by 20% (factor=0.8) when val_loss plateaus
        # for 10 epochs.  Prevents the optimiser from overshooting once it
        # is close to a minimum.  min_lr is the floor.
        lr_reduction=ReduceLrConfig(
            monitor="val_loss",
            mode="min",
            initial_lr=1e-3,
            patience=10,
            factor=0.8,
            min_lr=1e-6,
        ),

        # "auto" picks the best available device (CUDA > MPS > CPU).
        strategy="auto",
        mixed_precision=False,   # set True for fp16 on CUDA (faster, less memory)
        compile_model=False,     # torch.compile() graph-mode -- experimental
        verbose=True,            # print epoch-level progress
    )

    # ------------------------------------------------------------------
    # Pipeline config (wraps everything together)
    # ------------------------------------------------------------------
    # PipelineConfig is the single object every stage reads from.
    # training_config is serialised to dict so it can be persisted as JSON
    # alongside the model in the registry.
    config = PipelineConfig(
        training_config=training_config.to_dict(),  # dict for JSON serialisability
        data_config=data_config,                    # HybridGnnRnnDataConfig dataclass
        model_config=None,                          # None = use default HybridGnnRnnModelConfig
        registry_dir=str(registry_dir),             # where models are saved
        tracking_dir=str(tracking_dir),             # where experiment runs are logged
        artifacts_dir=str(artifacts_dir),           # where plots/reports are written
        metadata={
            # "job" carries the cluster_info paths that build_data reads
            "job": job,
            "run_name": "hybrid_gnn_rnn_full_pipeline",
            # Tags used by both ModelRegistry and ExperimentTracker
            "tags": ["hybrid_gnn_rnn", "latest"],
            "description": "Hybrid GNN-RNN full pipeline run from cached data",
            "generate_training_report": False,
        },
    )

    logger.info("Configuration built")
    logger.info(f"  artifacts -> {artifacts_dir}")
    logger.info(f"  registry  -> {registry_dir}")
    logger.info(f"  tracking  -> {tracking_dir}")
    return config


# ======================================================================
# 2.  Step-by-Step Pipeline Execution
# ======================================================================

def run_full_pipeline(config: "PipelineConfig") -> None:
    """
    Execute every pipeline stage individually with detailed logging.

    We call each pipeline hook manually (instead of pipeline.run()) so that
    intermediate results can be inspected and printed at each stage.  This
    is the pattern you'd use in a Jupyter notebook or debugging session.
    """

    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline
    from src.rade_ml_pt.training.trainer import Trainer, setup_training_environment

    # Instantiate the concrete training pipeline with our config.
    # The pipeline holds model-specific logic (build_data, build_model, post_train).
    pipeline = HybridGnnRnnTrainPipeline(config)

    # Resolve the TrainingConfig dataclass from the dict stored in PipelineConfig.
    # Also extract the random seed so we can set it before any stochastic operation.
    training_config = pipeline._resolve_training_config()
    seed = pipeline._resolve_seed()

    # Set global seeds & deterministic flags (torch, numpy, cudnn).
    setup_training_environment(training_config, seed)

    # ----------------------------------------------------------------
    # Stage 1: Data Build
    # ----------------------------------------------------------------
    # Loads the 4 data files via CacheLoader, then runs:
    #   - PnL standardisation (fit scaler on train split only)
    #   - PCA dimensionality reduction on elementary trades
    #   - Trade attribute encoding (numeric + categorical + multi-label)
    #   - k-NN graph construction from encoded features
    #   - Sliding-window PnL sequences -> PyTorch DataLoaders
    # Result: HybridGnnRnnResult with .train_ds, .val_ds, .test_ds DataLoaders
    #         plus metadata (scalers, graph builder, encoder, trade IDs).
    print("\n" + "=" * 70)
    print("STAGE 1: DATA BUILD")
    print("=" * 70)

    data_result = pipeline.build_data(config)

    print(f"\n  Data result type: {type(data_result).__name__}")
    if hasattr(data_result, "elementary_pnl") and data_result.elementary_pnl is not None:
        print(f"  Elementary PnL shape: {data_result.elementary_pnl.shape}")
    if hasattr(data_result, "target_pnl") and data_result.target_pnl is not None:
        print(f"  Target PnL shape:     {data_result.target_pnl.shape}")
    print(f"  Metadata keys: {sorted(data_result.metadata.keys())}")

    # Inspect one batch to verify the DataLoader is producing the expected
    # tensor shapes and keys.  The model's forward() method expects a dict
    # with keys like "trade_features", "pnl_history", "adjacency_indices", etc.
    print("\n  --- DataLoader batch inspection ---")
    for batch in data_result.train_ds:
        if isinstance(batch, (tuple, list)):
            inputs_batch, targets_batch = batch[0], batch[1]
        else:
            inputs_batch, targets_batch = batch, None
        if isinstance(inputs_batch, dict):
            print(f"  Input keys: {sorted(inputs_batch.keys())}")
            for k, v in sorted(inputs_batch.items()):
                shape = v.shape if hasattr(v, "shape") else "N/A"
                print(f"    {k:30s} -> {shape}")
        if targets_batch is not None:
            print(f"  Targets shape: {targets_batch.shape}")
        break  # only need one batch for inspection

    # ----------------------------------------------------------------
    # Stage 2: Model Build
    # ----------------------------------------------------------------
    # Instantiates the HybridGnnRnn model using the default model config
    # (since we passed model_config=None in PipelineConfig).
    # The model is composed of 5 blocks:
    #   GNN -> RNN -> Fusion -> TargetAttention -> Projection
    # Each block has LayerNorm between them.
    print("\n" + "=" * 70)
    print("STAGE 2: MODEL BUILD")
    print("=" * 70)

    model = pipeline.build_model(config, data_result)

    print(f"\n  Model class: {type(model).__name__}")
    print(f"  Model name:  {model.model_name}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters:  {total_params:,} total, {trainable:,} trainable")

    # ----------------------------------------------------------------
    # Stage 3: Compile + Train
    # ----------------------------------------------------------------
    # The Trainer handles the full training loop:
    #   1. Builds the optimizer from TrainingConfig.optimizer
    #   2. Builds the loss function from TrainingConfig.loss
    #   3. Creates callbacks (EarlyStopping, ReduceLROnPlateau, MetricsLogger)
    #   4. Iterates over epochs:
    #      - Forward + backward pass on each batch
    #      - Gradient clipping (if configured)
    #      - Validation pass
    #      - LR scheduler step
    #      - Callback hooks (early stopping check, checkpoint save, etc.)
    #   5. Returns a TrainingResult with full loss history
    print("\n" + "=" * 70)
    print("STAGE 3: COMPILE + TRAIN")
    print("=" * 70)

    trainer = Trainer(model=model, config=training_config, seed=seed)

    print(f"\n  Training config:")
    print(f"    Epochs:         {training_config.epochs}")
    print(f"    Loss:           {training_config.loss}")
    print(f"    Optimizer:      {training_config.optimizer.name} (lr={training_config.optimizer.learning_rate})")
    print(f"    Early stopping: patience={training_config.early_stopping.patience}")
    print(f"    LR reduction:   patience={training_config.lr_reduction.patience}, factor={training_config.lr_reduction.factor}")

    print("\n  Starting training...")
    result = trainer.fit(
        train_data=data_result.train_ds,
        val_data=data_result.val_ds,
    )

    # ----------------------------------------------------------------
    # Stage 4: Training Results
    # ----------------------------------------------------------------
    # TrainingResult contains:
    #   - history: dict of lists (e.g. {"loss": [...], "val_loss": [...]})
    #   - best_epoch / final_epoch: when the best and last epochs occurred
    #   - best_train_loss / best_val_loss: lowest losses achieved
    #   - stopped_early: True if EarlyStopping triggered before max epochs
    #   - training_time_seconds: wall-clock training duration
    print("\n" + "=" * 70)
    print("STAGE 4: TRAINING RESULTS")
    print("=" * 70)

    print(f"\n  Training time:   {result.training_time_seconds:.1f}s")
    print(f"  Final epoch:     {result.final_epoch}")
    print(f"  Best epoch:      {result.best_epoch}")
    print(f"  Best train loss: {result.best_train_loss:.6f}")
    print(f"  Best val loss:   {result.best_val_loss:.6f}")
    print(f"  Stopped early:   {result.stopped_early}")

    # ----------------------------------------------------------------
    # Stage 5: Evaluation
    # ----------------------------------------------------------------
    # The Evaluator runs the model in eval mode (no gradients, dropout off)
    # over the test DataLoader in a single pass.  It collects predictions,
    # targets, residuals, and computes:
    #   - Loss (MSE in this case)
    #   - Custom metrics: RMSE, MAE, R², MAPE
    #   - Residual statistics: mean, std, max, P95, P99
    # Use test_ds if available; fall back to val_ds for small datasets.
    print("\n" + "=" * 70)
    print("STAGE 5: EVALUATION")
    print("=" * 70)

    from src.rade_ml_pt.evaluation.evaluator import Evaluator
    from src.rade_ml_pt.evaluation.metrics import rmse, mae, r_squared, mape

    eval_ds = data_result.test_ds if data_result.test_ds is not None else data_result.val_ds
    evaluator = Evaluator(model, loss_fn=torch.nn.MSELoss())
    eval_result = evaluator.run(
        eval_ds,
        additional_metrics={
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "mape": mape,
        },
    )

    print(f"\n  {eval_result.summary()}")
    print(f"\n  Residual statistics:")
    for key in sorted(eval_result.metrics):
        if key.startswith("residual_"):
            print(f"    {key:20s}: {eval_result.metrics[key]:.6f}")

    # ----------------------------------------------------------------
    # Stage 6: Model Registration
    # ----------------------------------------------------------------
    # ModelRegistry saves the full model (torch.save) plus structured
    # metadata (JSON) to a versioned directory.  Each version gets a
    # unique ID (timestamp + hash).  Tags ("latest", "best") provide
    # human-friendly aliases for loading.
    #
    # After registration, we also save inference artifacts alongside
    # the model so that evaluation and inference pipelines can cold-start
    # without re-running the data build.  Artifacts include:
    #   - graph_builder.pkl    (fitted TradeGraphBuilder)
    #   - encoder.pkl          (fitted TradeAttributeEncoder)
    #   - target_scaler.pkl    (fitted PnL StandardScaler)
    #   - data_config.json     (reproducible data pipeline settings)
    #   - trade_universe.json  (elementary/target trade IDs and indices)
    #   - datasets/*.pt        (cached DataLoader datasets for fast eval)
    print("\n" + "=" * 70)
    print("STAGE 6: MODEL REGISTRATION")
    print("=" * 70)

    from src.rade_ml_pt.registry.store import ModelRegistry

    registry = ModelRegistry(config.registry_dir)
    entry = registry.register(
        model=model,
        training_result=result,
        tags=config.metadata.get("tags", ["hybrid_gnn_rnn", "latest"]),
        description=config.metadata.get("description", "Hybrid GNN-RNN model"),
    )

    print(f"\n  Registry dir: {config.registry_dir}")
    print(f"  Version:      {entry.version}")
    print(f"  Tags:         {entry.tags}")

    # Save inference artifacts (graph builder, encoder, scalers, PnL
    # parquets, attribute JSON, cached datasets) into the registry
    # version directory so everything needed for cold-start inference
    # lives alongside the model checkpoint.
    pipeline._registered_entry = entry
    pipeline._save_inference_artifacts(entry, data_result)
    pipeline._save_datasets(Path(entry.model_dir), data_result)
    print("  Inference artifacts saved alongside model")

    # ----------------------------------------------------------------
    # Stage 7: Experiment Tracking
    # ----------------------------------------------------------------
    # ExperimentTracker writes a JSON record for each run, capturing:
    #   - Config snapshot (for reproducibility)
    #   - Training metrics (best_val_loss, final_epoch, etc.)
    #   - Evaluation metrics (RMSE, MAE, R², etc.)
    #   - Link to the registered model version
    # Runs are queryable by tag and comparable via tracker.compare_runs().
    print("\n" + "=" * 70)
    print("STAGE 7: EXPERIMENT TRACKING")
    print("=" * 70)

    from src.rade_ml_pt.tracking.tracker import ExperimentTracker

    tracker = ExperimentTracker(config.tracking_dir)
    run = tracker.start_run(
        name=config.metadata.get("run_name", "hybrid_gnn_rnn"),
        tags=config.metadata.get("tags", []),
    )

    # Log the training config so the run is fully reproducible
    run.log_config(training_config)

    # log_result extracts best_val_loss, best_train_loss, final_epoch,
    # training_time_seconds, and stopped_early from the TrainingResult
    run.log_result(result)

    # Link the experiment run to the registered model version for traceability
    run.set_model_version(entry.version)

    # Add evaluation metrics so they appear in cross-run comparisons
    run.log_metrics({
        "eval_loss": eval_result.loss if eval_result.loss is not None else -1.0,
        "eval_rmse": eval_result.metrics.get("rmse", -1.0),
        "eval_mae": eval_result.metrics.get("mae", -1.0),
        "eval_r_squared": eval_result.metrics.get("r_squared", -1.0),
    })

    # Mark the run as completed and persist to disk
    tracker.end_run(run)

    print(f"\n  Tracking dir: {config.tracking_dir}")
    print(f"  Run ID:       {run.run_id}")
    print(f"  Run name:     {run.name}")
    print(f"  Status:       {run.status}")

    # ----------------------------------------------------------------
    # Stage 8: Inference from Registry
    # ----------------------------------------------------------------
    # This proves the registry round-trip works: load the model back
    # from disk using only its tag, then run a forward pass on a test
    # batch.  In production, this is how a serving layer or analytics
    # dashboard would consume the model.
    #
    # InferenceRunner handles:
    #   - Loading the model from the registry version directory
    #   - Moving inputs to the correct device
    #   - Running torch.no_grad() forward pass
    #   - Computing input hash for audit trail
    #   - Returning an InferenceResult with predictions + provenance
    print("\n" + "=" * 70)
    print("STAGE 8: INFERENCE (via InferenceRunner)")
    print("=" * 70)

    from src.rade_ml_pt.inference.runner import InferenceRunner

    # Load the model back from the registry using the "latest" tag
    runner = InferenceRunner.from_registry(registry, version_or_tag="latest")
    print(f"\n  Loaded model version: {runner.model_version}")

    # Grab one batch of inputs from the test DataLoader to feed to the runner
    for batch in eval_ds:
        if isinstance(batch, (tuple, list)):
            test_inputs = batch[0]
        else:
            test_inputs = batch
        break

    # Use real target trade IDs as sample_ids for provenance tracking
    target_ids = None
    if hasattr(data_result, "target_pnl") and data_result.target_pnl is not None:
        target_ids = list(data_result.target_pnl.columns)

    infer_result = runner.predict(
        inputs=test_inputs,
        sample_ids=target_ids,
        metadata={"source": "full_pipeline_example", "dataset": "test"},
    )

    print(f"\n  InferenceResult:")
    print(f"    n_samples:        {infer_result.n_samples}")
    print(f"    latency:          {infer_result.latency_seconds:.4f}s")
    preds = infer_result.predictions
    print(f"    prediction shape: {preds.shape}")
    print(f"    prediction mean:  {np.mean(preds):.6f}")
    print(f"    prediction std:   {np.std(preds):.6f}")

    # ----------------------------------------------------------------
    # Stage 9: Diagnostic Plots
    # ----------------------------------------------------------------
    # Two categories of plots:
    #
    # A) Training plots (from TrainingResult):
    #    - 2x2 panel: loss curves, train-val gap, val/train ratio, other metrics
    #
    # B) Evaluation plots (from EvaluationResult):
    #    - Predicted vs actual scatter (45-degree reference line)
    #    - Residual distribution histogram + KDE + box plot
    #    - Cumulative error CDF (empirical distribution of absolute errors)
    #    - QQ plot against standard normal (checks residual normality)
    #
    # All plots are saved as PNG files in the artifacts/plots/ directory.
    print("\n" + "=" * 70)
    print("STAGE 9: DIAGNOSTIC PLOTS")
    print("=" * 70)

    plots_dir = Path(config.artifacts_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Training dynamics: loss curves, overfitting gap, and metric evolution
    from src.rade_ml_pt.training.plots import save_training_plots
    training_plot_path = save_training_plots(result, save_dir=plots_dir)
    print(f"\n  Training plots saved: {training_plot_path}")

    from src.rade_ml_pt.evaluation.plots import (
        plot_predicted_vs_actual,
        plot_residual_distribution,
        plot_cumulative_error,
        plot_qq,
    )

    # Scatter of predicted vs actual PnL with a 45-degree reference line.
    # Points close to the line = good predictions.
    plot_predicted_vs_actual(eval_result, save_path=plots_dir / "predicted_vs_actual.png")
    print(f"  Predicted vs actual: {plots_dir / 'predicted_vs_actual.png'}")

    # Histogram + KDE of residuals (predicted - actual).
    # A tight, zero-centred distribution indicates unbiased predictions.
    plot_residual_distribution(eval_result, save_path=plots_dir / "residual_distribution.png")
    print(f"  Residual distribution: {plots_dir / 'residual_distribution.png'}")

    # Empirical CDF of absolute errors.
    # Read as: "X% of predictions have absolute error <= Y".
    plot_cumulative_error(eval_result, save_path=plots_dir / "cumulative_error.png")
    print(f"  Cumulative error CDF: {plots_dir / 'cumulative_error.png'}")

    # QQ plot comparing residual quantiles against a standard normal.
    # Deviations from the diagonal indicate non-normal residual tails.
    plot_qq(eval_result, save_path=plots_dir / "qq_plot.png")
    print(f"  QQ plot: {plots_dir / 'qq_plot.png'}")

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  All artifacts in: {config.artifacts_dir}")
    print(f"  Model version:    {entry.version}")
    print(f"  Run ID:           {run.run_id}")
    print(f"  Best val loss:    {result.best_val_loss:.6f} (epoch {result.best_epoch})")

    return result


# ======================================================================
# 3.  Main
# ======================================================================

def main() -> None:
    """
    Entry point.  Configure your data paths below and run.

    The ``job`` dict mirrors the structure a production job scheduler produces:
    ``{"cluster_info": {<path_keys>: <absolute_file_paths>}}``.

    Replace the placeholder paths with your actual data file locations.
    """

    # -- CONFIGURE THESE PATHS ----------------------------------------
    # Point to your pre-built data files (pickle or parquet).
    # These are the 4 files your data pipeline / job scheduler produces:
    #   - elementary_pnl:     DataFrame [scenarios x elementary_trades]
    #   - target_pnl:         DataFrame [scenarios x target_trades]
    #   - elementary_attribs: Dict of trade attributes (moneyness, delta, ...)
    #   - target_attribs:     Dict of trade attributes for target trades
    DATA_ROOT = Path(r"/path/to/your/data")

    job = {
        "cluster_info": {
            "elementary_pnl_path": str(DATA_ROOT / "elementary_pnl.pkl"),
            "target_pnl_path": str(DATA_ROOT / "target_pnl.pkl"),
            "elementary_attribs_path": str(DATA_ROOT / "elementary_attribs.pkl"),
            "target_attribs_path": str(DATA_ROOT / "target_attribs.pkl"),
        }
    }

    # -- CONFIGURE OUTPUT DIRECTORY -----------------------------------
    # Use a temp directory for experimentation; swap to a fixed path for
    # persistent runs (e.g. Path("./runs/hybrid_gnn_rnn/2026-02-22")).
    # Temp directories are cleaned up by the OS eventually.
    workdir = Path(tempfile.mkdtemp(prefix="rade_ml_pt_full_pipeline_"))
    # -----------------------------------------------------------------

    logger.info(f"Working directory: {workdir}")
    logger.info(f"Data root:         {DATA_ROOT}")

    # Build the full configuration from the job paths and output directory
    config = build_configs(job, workdir)

    # Run all 9 stages of the pipeline
    run_full_pipeline(config)

    logger.info(f"All artifacts in: {workdir}")


if __name__ == "__main__":
    main()
