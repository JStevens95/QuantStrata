"""
Abstract base classes for end-to-end ML pipelines (PyTorch).

Four pipeline archetypes are defined:

* **TrainPipeline**      -- build data -> build model -> train -> (optionally) register & track
* **EvalPipeline**       -- load model -> build data -> evaluate -> report
* **InferencePipeline**  -- load model -> prepare inputs -> predict -> post-process
* **TunePipeline**       -- build data -> define search space -> run Optuna trials -> save results

Each archetype implements a ``run()`` method that orchestrates the steps in
order and delegates model-specific logic to abstract hooks that subclasses
override.

Example (model-specific subclass)::

    class GnnRnnTrainPipeline(TrainPipeline):
        def build_data(self, config):
            return build_gnn_rnn_dataset(config)

        def build_model(self, config, data_result):
            return HybridGnnRnn(model_config)

    pipeline = GnnRnnTrainPipeline(pipeline_config)
    result = pipeline.run()
"""
from __future__ import annotations

import abc
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.rade_ml_pt.pipelines.config import PipelineConfig

if TYPE_CHECKING:
    from src.rade_ml_pt.core.types import TrainingResult, InferenceResult, EvaluationResult
    from src.rade_ml_pt.data.result import DataBuildResult
    from src.rade_ml_pt.registry.store import ModelRegistry
    from src.rade_ml_pt.tracking.tracker import ExperimentTracker
    from src.rade_ml_pt.tracking.run import Run

logger = logging.getLogger(__name__)


# ======================================================================
# Train Pipeline
# ======================================================================

class TrainPipeline(abc.ABC):
    """
    Abstract training pipeline.

    Subclasses provide model-specific data building and model construction;
    the base class handles the generic train -> register -> track flow.

    Parameters
    ----------
    config : PipelineConfig
        Top-level pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._registered_entry: Optional[Any] = None

    # ------------------------------------------------------------------
    # Abstract hooks (model-specific)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        """Build train/val/test datasets from the pipeline config."""
        ...

    @abc.abstractmethod
    def build_model(
        self,
        config: PipelineConfig,
        data_result: "DataBuildResult",
    ) -> nn.Module:
        """Construct the PyTorch model."""
        ...

    def post_train(
        self,
        result: "TrainingResult",
        model: nn.Module,
        registry: Optional["ModelRegistry"] = None,
        tracker: Optional["ExperimentTracker"] = None,
        run: Optional["Run"] = None,
        data_result: Optional["DataBuildResult"] = None,
    ) -> None:
        """
        Optional hook called after training completes.

        Default behaviour: register the model and log the run.  Override for
        custom post-training logic (ensemble checkpointing, alerting, plotting, etc.).
        Use ``data_result`` when custom plotting or reports need model-specific data
        (e.g. test set, graph structure, metadata).

        After registration, ``self._registered_entry`` holds the RegistryEntry
        so subclasses can save additional artifacts alongside the model.
        """
        self._registered_entry = None
        if registry is not None:
            entry = registry.register(
                model=model,
                training_result=result,
                tags=self.config.metadata.get("tags", []),
                description=self.config.metadata.get("description", ""),
            )
            self._registered_entry = entry
            if run is not None:
                run.set_model_version(entry.version)

        if run is not None:
            run.log_result(result)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> "TrainingResult":
        """
        Execute the full training pipeline.

        Steps
        -----
        1. ``build_data``
        2. ``build_model`` → device placement → optional compile / DDP wrap
        3. ``Trainer.fit``
        4. ``post_train`` (registration, tracking, reports — rank 0 only)

        Returns
        -------
        TrainingResult
        """
        from src.rade_ml_pt.training.trainer import Trainer, setup_training_environment
        from src.rade_ml_pt.training.strategy import get_training_strategy, setup_ddp, cleanup_ddp

        logger.info("TrainPipeline: starting")
        t0 = time.perf_counter()

        self._ensure_directories()
        training_config = self._resolve_training_config()
        seed = self._resolve_seed()
        setup_training_environment(training_config, seed)

        use_ddp = getattr(training_config, "strategy", None) == "ddp"
        if use_ddp:
            setup_ddp()

        data_result = self.build_data(self.config)
        logger.info("TrainPipeline: data built")

        device = get_training_strategy(training_config)
        model = self._prepare_model(training_config, data_result, device, use_ddp)

        train_data, val_data = self._prepare_data(data_result, use_ddp)

        trainer = Trainer(model=model, config=training_config, seed=seed)
        result = trainer.fit(train_data=train_data, val_data=val_data)
        logger.info("TrainPipeline: training complete (%.1fs)", time.perf_counter() - t0)

        unwrapped = model.module if use_ddp else model
        self._finalise(result, unwrapped, data_result, use_ddp)

        if use_ddp:
            cleanup_ddp()

        logger.info("TrainPipeline: done")
        return result

    # ------------------------------------------------------------------
    # run() sub-steps
    # ------------------------------------------------------------------

    def _ensure_directories(self) -> None:
        """Create artifacts and registry directories if configured."""
        if self.config.artifacts_dir:
            Path(self.config.artifacts_dir).mkdir(parents=True, exist_ok=True)
        if self.config.registry_dir:
            Path(self.config.registry_dir).mkdir(parents=True, exist_ok=True)

    def _prepare_model(
        self,
        training_config: Any,
        data_result: "DataBuildResult",
        device: torch.device,
        use_ddp: bool,
    ) -> nn.Module:
        """Build the model, move to device, optionally compile and wrap in DDP."""
        model = self.build_model(self.config, data_result)
        model = model.to(device)
        logger.info("TrainPipeline: model built (device=%s)", device)

        if training_config.compile_model and device.type != "cpu":
            model = torch.compile(model)
            logger.info("TrainPipeline: model compiled with torch.compile()")
        elif training_config.compile_model:
            logger.warning(
                "compile_model=True but device is CPU; skipping torch.compile "
                "(inductor backend is slower on CPU than eager mode)"
            )

        if use_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            from src.rade_ml_pt.training.strategy import get_ddp_local_rank
            model = DDP(model, device_ids=[get_ddp_local_rank()])
            logger.info("TrainPipeline: model wrapped in DistributedDataParallel")

        return model

    def _prepare_data(
        self,
        data_result: "DataBuildResult",
        use_ddp: bool,
    ) -> tuple:
        """Return ``(train_data, val_data)``, wrapping for DDP when needed."""
        train_data = data_result.train_ds
        val_data = data_result.val_ds
        if use_ddp:
            train_data = self._wrap_loader_for_ddp(train_data)
            if val_data is not None:
                val_data = self._wrap_loader_for_ddp(val_data)
        return train_data, val_data

    def _finalise(
        self,
        result: "TrainingResult",
        model: nn.Module,
        data_result: "DataBuildResult",
        use_ddp: bool,
    ) -> None:
        """Register model, log run, and generate reports (rank 0 only)."""
        from src.rade_ml_pt.training.strategy import is_main_process

        if use_ddp and not is_main_process():
            return

        registry = self._init_registry()
        tracker, run = self._init_tracker()

        self.post_train(
            result, model, registry=registry, tracker=tracker, run=run,
            data_result=data_result,
        )

        if run is not None and tracker is not None:
            tracker.end_run(run)

        if self.config.artifacts_dir and self.config.metadata.get("generate_training_report", True):
            try:
                self._generate_training_report(result, model, data_result)
            except Exception as exc:
                logger.error("Failed to generate training report: %s", exc)

    def _init_registry(self) -> Optional[Any]:
        """Build a ModelRegistry if ``registry_dir`` is set."""
        if not self.config.registry_dir:
            return None
        try:
            from src.rade_ml_pt.registry.store import ModelRegistry
            return ModelRegistry(self.config.registry_dir)
        except Exception as exc:
            logger.error("Failed to initialise ModelRegistry: %s", exc)
            return None

    def _init_tracker(self) -> tuple:
        """Build an ExperimentTracker + Run if ``tracking_dir`` is set."""
        if not self.config.tracking_dir:
            return None, None
        try:
            from src.rade_ml_pt.tracking.tracker import ExperimentTracker
            tracker = ExperimentTracker(self.config.tracking_dir)
            run = tracker.start_run(
                name=self.config.metadata.get("run_name", "train"),
                tags=self.config.metadata.get("tags", []),
            )
            return tracker, run
        except Exception as exc:
            logger.error("Failed to initialise ExperimentTracker: %s", exc)
            return None, None

    def _generate_training_report(
        self,
        result: "TrainingResult",
        model: nn.Module,
        data_result: "DataBuildResult",
    ) -> None:
        """
        Generate a professional training report when artifacts_dir is set.

        When a model was registered, saves to artifacts_dir/training/{version}/.
        Otherwise falls back to artifacts_dir/training/{run_name}_{timestamp}/.
        """
        from datetime import datetime
        from src.rade_ml_pt.training.reports import generate_training_report

        run_name = self.config.metadata.get("run_name", "train")
        if self._registered_entry is not None:
            subdir = self._registered_entry.version
        else:
            subdir = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir = Path(self.config.artifacts_dir) / "training" / subdir
        generate_training_report(
            result=result,
            config=self.config,
            save_dir=report_dir,
            data_result=data_result,
            model=model,
            run_name=run_name,
            include_loss_plot=True,
            format="markdown",
        )

    @staticmethod
    def _wrap_loader_for_ddp(loader: Any) -> Any:
        """Rebuild a DataLoader with a ``DistributedSampler``.

        If *loader* is not a ``DataLoader`` (e.g. a plain list of batches),
        it is returned unchanged.
        """
        from torch.utils.data import DataLoader, DistributedSampler
        from src.rade_ml_pt.training.strategy import get_ddp_rank, get_ddp_world_size

        if not isinstance(loader, DataLoader):
            return loader

        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=get_ddp_world_size(),
            rank=get_ddp_rank(),
        )
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=getattr(loader, "num_workers", 0),
            collate_fn=loader.collate_fn,
            pin_memory=getattr(loader, "pin_memory", False),
            drop_last=getattr(loader, "drop_last", False),
        )

    def _resolve_training_config(self) -> Any:
        """Resolve the training config from PipelineConfig."""
        from src.rade_ml_pt.core.config import TrainingConfig, sanitize_yaml_values

        tc = self.config.training_config
        if tc is None:
            return TrainingConfig()
        if isinstance(tc, dict):
            return TrainingConfig.from_dict(sanitize_yaml_values(tc))
        return tc

    def _resolve_seed(self) -> int:
        """Get seed from data config; default 42. Seed is a data-processing concern."""
        dc = self.config.data_config
        if dc is None:
            return 42
        if isinstance(dc, dict):
            val = dc.get("seed", 42)
        else:
            val = getattr(dc, "seed", 42)
        if isinstance(val, str):
            val = None if val.lower() in ("none", "null", "~", "") else int(val)
        return val if val is not None else 42


# ======================================================================
# Eval Pipeline
# ======================================================================

class EvalPipeline(abc.ABC):
    """
    Abstract evaluation pipeline.

    Loads a trained model from the registry, builds test data, runs evaluation,
    and produces diagnostics.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._loaded_entry: Optional[Any] = None

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        """Build test dataset."""
        ...

    def load_model(self, config: PipelineConfig) -> tuple:
        """
        Load a model from the registry.

        Returns (model, RegistryEntry).  Override for custom loading.
        """
        from src.rade_ml_pt.registry.store import ModelRegistry

        registry = ModelRegistry(config.registry_dir)
        return registry.load(config.version_or_tag)

    def post_eval(
        self,
        eval_result: "EvaluationResult",
        config: PipelineConfig,
        data_result: Optional["DataBuildResult"] = None,
    ) -> None:
        """
        Optional hook for custom post-evaluation actions.

        The loaded model's RegistryEntry is available via ``self._loaded_entry``.
        When ``config.artifacts_dir`` and ``self._loaded_entry`` are set, standard
        evaluation plots (predicted vs actual, residual distribution, etc.) are
        saved to ``artifacts_dir/evaluation/{version}/``.
        """
        if config.artifacts_dir and self._loaded_entry is not None:
            try:
                from src.rade_ml_pt.evaluation.plots import save_evaluation_plots

                version = self._loaded_entry.version
                eval_dir = Path(config.artifacts_dir) / "evaluation" / version
                save_evaluation_plots(eval_result, eval_dir)
                logger.info("Saved evaluation plots to %s", eval_dir)
            except Exception as exc:
                logger.warning("Could not generate evaluation plots: %s", exc)

    def get_target_scaler(self, data_result: "DataBuildResult") -> Optional[Any]:
        """
        Return scaler for inverse-transforming predictions/targets to original units.

        Override in model-specific pipelines when targets were scaled during data build.
        Default returns None (no inverse transform; metrics stay in scaled space).
        """
        return None

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> "EvaluationResult":
        """
        Execute the full evaluation pipeline.

        Steps
        -----
        1. ``load_model``
        2. ``build_data``
        3. ``Evaluator.run``
        4. ``post_eval``

        Returns
        -------
        EvaluationResult
        """
        from src.rade_ml_pt.evaluation.evaluator import Evaluator

        logger.info("EvalPipeline: starting")

        model, entry = self.load_model(self.config)
        self._loaded_entry = entry
        logger.info(f"EvalPipeline: loaded model version '{entry.version}'")

        data_result = self.build_data(self.config)
        logger.info("EvalPipeline: data built")

        evaluator = Evaluator(model=model)
        eval_result = evaluator.run(data_result.test_ds)
        logger.info("EvalPipeline: evaluation complete")

        self.post_eval(eval_result, self.config, data_result=data_result)

        logger.info("EvalPipeline: done")
        return eval_result


# ======================================================================
# Inference Pipeline
# ======================================================================

class InferencePipeline(abc.ABC):
    """
    Abstract inference pipeline.

    Loads a trained model, applies model-specific input preparation, runs the
    forward pass via ``InferenceRunner``, and performs optional post-processing.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig) -> None:

        # initiate required variables.
        self.config = config

        # initiate placeholder for loaded runner.
        self._loaded_runner: Optional[Any] = None

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def prepare_inputs(self, config: PipelineConfig) -> Dict[str, Any]:
        """
        Build model-ready inputs from the pipeline config.

        Returns a dict containing at minimum an ``"inputs"`` key with the
        tensor(s) to feed the model, and optionally ``"trade_ids"``.
        """
        ...

    def get_result_cls(self) -> type:
        """Return the InferenceResult subclass for this pipeline. Override for model-specific types."""
        from src.rade_ml_pt.core.types import InferenceResult
        return InferenceResult

    def load_runner(self, config: PipelineConfig) -> Any:
        """
        Construct an InferenceRunner from the registry.

        Override for custom loading (e.g. from a direct path).
        """
        from src.rade_ml_pt.inference.runner import InferenceRunner
        from src.rade_ml_pt.registry.store import ModelRegistry

        registry = ModelRegistry(config.registry_dir)
        return InferenceRunner.from_registry(registry, config.version_or_tag)

    def post_infer(
        self,
        result: "InferenceResult",
        config: PipelineConfig,
    ) -> None:
        """Optional hook for custom post-inference actions (result storage, alerting, etc.)."""
        pass

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> "InferenceResult":
        """
        Execute the full inference pipeline.

        Steps
        -----
        1. ``load_runner``
        2. ``prepare_inputs``
        3. ``InferenceRunner.predict``
        4. ``post_infer``

        Returns
        -------
        InferenceResult
        """
        logger.info("InferencePipeline: starting")

        # 1. load inference runner.
        runner = self.load_runner(self.config)
        self._loaded_runner = runner
        logger.info("InferencePipeline: runner loaded")

        # 2. prepare inputs.
        prepared = self.prepare_inputs(self.config)
        inputs = prepared["inputs"]
        sample_ids = prepared.get("sample_ids", prepared.get("trade_ids"))
        metadata = prepared.get("metadata")

        # 3. return cls instance.
        result_cls = self.get_result_cls()
        result = runner.predict(
            inputs=inputs,
            sample_ids=sample_ids,
            metadata=metadata,
            result_cls=result_cls,
        )
        logger.info(
            f"InferencePipeline: inference complete "
            f"({result.n_samples} samples, {result.latency_seconds:.3f}s)"
        )

        # 4. optional post-infer hook.
        self.post_infer(result, self.config)

        logger.info("InferencePipeline: done")
        return result


# ======================================================================
# Tune Pipeline
# ======================================================================

class TunePipeline(abc.ABC):
    """
    Abstract hyperparameter tuning pipeline.

    Builds the data once, then runs the Optuna-backed ``Tuner`` with a
    model-specific search space.  Each trial builds a fresh model from
    sampled hyperparameters, trains it, and returns the validation metric.

    Subclasses must implement:

    * ``build_data``      -- same as TrainPipeline
    * ``search_space``    -- sample hyperparameters from an ``optuna.Trial``
    * ``build_trial_model`` -- construct a model from the sampled params

    Parameters
    ----------
    config : PipelineConfig
    tuner_kwargs : dict, optional
        Forwarded to ``Tuner`` (n_trials, direction, pruner, seed, etc.).
    """

    def __init__(
        self,
        config: PipelineConfig,
        tuner_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.tuner_kwargs = tuner_kwargs or {}

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        """Build train/val datasets (shared across all trials)."""
        ...

    @abc.abstractmethod
    def search_space(self, trial: Any) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for a single trial.

        Parameters
        ----------
        trial : optuna.Trial

        Returns
        -------
        dict
            Sampled hyperparameters (will be passed to ``build_trial_model``).
        """
        ...

    @abc.abstractmethod
    def build_trial_model(
        self,
        params: Dict[str, Any],
        data_result: "DataBuildResult",
    ) -> nn.Module:
        """Construct a model from the sampled hyperparameters."""
        ...

    def post_tune(
        self,
        result: Any,
        config: PipelineConfig,
        data_result: Optional["DataBuildResult"] = None,
    ) -> None:
        """Optional hook called after tuning completes (save plots, register best, etc.)."""
        pass

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> Any:
        """
        Execute the full tuning pipeline.

        Steps
        -----
        1. ``build_data``  (once, shared across trials)
        2. Create ``Tuner``
        3. For each trial:
           a. ``search_space`` -> sampled params
           b. ``build_trial_model`` -> model
           c. ``Trainer.fit`` -> val loss
        4. ``post_tune``
        5. Save tuning results to artifacts

        Returns
        -------
        TuningResult (from ``src.rade_ml_pt.tuning.tuner``)
        """
        from src.rade_ml_pt.tuning.tuner import Tuner
        from src.rade_ml_pt.training.trainer import Trainer, setup_training_environment

        logger.info("TunePipeline: starting")
        t0 = time.perf_counter()

        training_config = self._resolve_training_config()
        seed = self._resolve_seed()
        setup_training_environment(training_config, seed)

        data_result = self.build_data(self.config)
        logger.info("TunePipeline: data built (shared across trials)")

        def objective(trial: Any) -> float:
            params = self.search_space(trial)
            model = self.build_trial_model(params, data_result)

            trial_training_config = self._resolve_trial_training_config(
                training_config, params, trial,
            )
            trainer = Trainer(model=model, config=trial_training_config, seed=seed)
            try:
                result = trainer.fit(
                    train_data=data_result.train_ds,
                    val_data=data_result.val_ds,
                )
            except RuntimeError as exc:
                logger.warning(f"Trial {trial.number} failed: {exc}")
                return float("inf")
            return result.best_val_loss if result.best_val_loss is not None else float("inf")

        tuner_kw = {"seed": seed, **self.tuner_kwargs}
        tuner = Tuner(**tuner_kw)
        tuning_result = tuner.run(objective)

        logger.info(
            f"TunePipeline: complete in {time.perf_counter() - t0:.1f}s | "
            f"best={tuning_result.best_value:.6f} (trial {tuning_result.best_trial_number})"
        )

        if self.config.artifacts_dir:
            self._save_tuning_artifacts(tuning_result, tuner)

        self.post_tune(tuning_result, self.config, data_result=data_result)

        logger.info("TunePipeline: done")
        return tuning_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_training_config(self) -> Any:
        from src.rade_ml_pt.core.config import TrainingConfig

        tc = self.config.training_config
        if tc is None:
            return TrainingConfig()
        if isinstance(tc, dict):
            return TrainingConfig.from_dict(tc)
        return tc

    def _resolve_seed(self) -> int:
        dc = self.config.data_config
        if dc is None:
            return 42
        return dc.get("seed", 42) if isinstance(dc, dict) else getattr(dc, "seed", 42)

    def _resolve_trial_training_config(
        self,
        base_config: Any,
        params: Dict[str, Any],
        trial: Any,
    ) -> Any:
        """
        Merge trial-sampled training params (lr, batch_size, etc.) into the
        base TrainingConfig.  Override for custom merging logic.
        """
        from dataclasses import replace

        overrides = {}
        if "learning_rate" in params:
            overrides["learning_rate"] = params["learning_rate"]
        if "batch_size" in params:
            overrides["batch_size"] = params["batch_size"]
        if "epochs" in params:
            overrides["epochs"] = params["epochs"]

        return replace(base_config, **overrides) if overrides else base_config

    def _save_tuning_artifacts(self, result: Any, tuner: Any) -> None:
        """Save tuning results and plots to artifacts_dir/tuning/{study_name}/."""
        from pathlib import Path
        from src.rade_ml_pt.tuning.plots import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
            plot_contour,
        )

        study_name = result.study_name or "tune"
        tune_dir = Path(self.config.artifacts_dir) / "tuning" / study_name
        tune_dir.mkdir(parents=True, exist_ok=True)

        result.to_json(tune_dir / "tuning_result.json")

        study = getattr(tuner, "study", None)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            plot_optimization_history(result, ax=ax)
            fig.savefig(tune_dir / "optimization_history.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            if study is not None:
                fig, ax = plt.subplots(figsize=(8, 5))
                plot_param_importances(study, ax=ax)
                fig.savefig(tune_dir / "param_importances.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                plot_parallel_coordinate(study)
                fig = plt.gcf()
                fig.savefig(tune_dir / "parallel_coordinate.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                plot_slice(study)
                fig = plt.gcf()
                fig.savefig(tune_dir / "slice.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                import optuna
                top_params = list(optuna.importance.get_param_importances(study).keys())[:2]
                if len(top_params) == 2:
                    plot_contour(study, params=top_params)
                    fig = plt.gcf()
                    fig.savefig(tune_dir / "contour.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

        except Exception as exc:
            logger.warning(f"Could not generate tuning plots: {exc}")

        logger.info(f"Tuning artifacts saved to {tune_dir}")
