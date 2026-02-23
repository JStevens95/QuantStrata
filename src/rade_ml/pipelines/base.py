"""
Abstract base classes for end-to-end ML pipelines.

Three pipeline archetypes are defined:

* **TrainPipeline** -- build data -> build model -> train -> (optionally) register & track
* **EvalPipeline**  -- load model -> build data -> evaluate -> report
* **InferencePipeline** -- load model -> prepare inputs -> predict -> post-process

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

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.rade_ml.pipelines.config import PipelineConfig

if TYPE_CHECKING:
    import tensorflow as tf
    from src.rade_ml.core.types import TrainingResult, InferenceResult, EvaluationResult
    from src.rade_ml.data.result import DataBuildResult
    from src.rade_ml.registry.store import ModelRegistry
    from src.rade_ml.tracking.tracker import ExperimentTracker
    from src.rade_ml.tracking.run import Run

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
    ) -> "tf.keras.Model":
        """Construct and compile the Keras model."""
        ...

    def post_train(
        self,
        result: "TrainingResult",
        model: "tf.keras.Model",
        registry: Optional["ModelRegistry"] = None,
        tracker: Optional["ExperimentTracker"] = None,
        run: Optional["Run"] = None,
    ) -> None:
        """
        Optional hook called after training completes.

        Default behaviour: register the model and log the run.  Override for
        custom post-training logic (ensemble checkpointing, alerting, etc.).
        """
        if registry is not None:
            entry = registry.register(
                model=model,
                training_result=result,
                tags=self.config.metadata.get("tags", []),
                description=self.config.metadata.get("description", ""),
            )
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
        2. ``build_model``
        3. ``Trainer.fit``
        4. ``post_train``

        Returns
        -------
        TrainingResult
        """
        from src.rade_ml.training.trainer import Trainer
        from src.rade_ml.registry.store import ModelRegistry
        from src.rade_ml.tracking.tracker import ExperimentTracker

        logger.info("TrainPipeline: starting")
        t0 = time.perf_counter()

        data_result = self.build_data(self.config)
        logger.info("TrainPipeline: data built")

        model = self.build_model(self.config, data_result)
        logger.info("TrainPipeline: model built")

        training_config = self._resolve_training_config()
        seed = self._resolve_seed()
        trainer = Trainer(model=model, config=training_config, seed=seed)

        result = trainer.fit(
            train_data=data_result.train_ds,
            val_data=data_result.val_ds,
        )
        logger.info(f"TrainPipeline: training complete ({time.perf_counter() - t0:.1f}s)")

        registry = (
            ModelRegistry(self.config.registry_dir) if self.config.registry_dir else None
        )
        tracker = (
            ExperimentTracker(self.config.tracking_dir) if self.config.tracking_dir else None
        )
        run = tracker.start_run(
            name=self.config.metadata.get("run_name", "train"),
            tags=self.config.metadata.get("tags", []),
        ) if tracker else None

        self.post_train(result, model, registry=registry, tracker=tracker, run=run)

        if run is not None:
            tracker.end_run(run)

        logger.info("TrainPipeline: done")
        return result

    def _resolve_training_config(self) -> Any:
        """Resolve the training config from PipelineConfig."""
        from src.rade_ml.core.config import TrainingConfig

        tc = self.config.training_config
        if tc is None:
            return TrainingConfig()
        if isinstance(tc, dict):
            return TrainingConfig(**tc)
        return tc

    def _resolve_seed(self) -> int:
        """Get seed from data config; default 42. Seed is a data-processing concern."""
        dc = self.config.data_config
        if dc is None:
            return 42
        return dc.get("seed", 42) if isinstance(dc, dict) else getattr(dc, "seed", 42)


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
        from src.rade_ml.registry.store import ModelRegistry

        registry = ModelRegistry(config.registry_dir)
        return registry.load(config.version_or_tag)

    def post_eval(
        self,
        eval_result: "EvaluationResult",
        config: PipelineConfig,
    ) -> None:
        """Optional hook for custom post-evaluation actions (report generation, etc.)."""
        pass

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
        from src.rade_ml.evaluation.evaluator import Evaluator

        logger.info("EvalPipeline: starting")

        model, entry = self.load_model(self.config)
        logger.info(f"EvalPipeline: loaded model version '{entry.version}'")

        data_result = self.build_data(self.config)
        logger.info("EvalPipeline: data built")

        evaluator = Evaluator(model=model)
        eval_result = evaluator.run(data_result.test_ds)
        logger.info("EvalPipeline: evaluation complete")

        self.post_eval(eval_result, self.config)

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
        self.config = config

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
        from src.rade_ml.core.types import InferenceResult
        return InferenceResult

    def load_runner(self, config: PipelineConfig) -> Any:
        """
        Construct an InferenceRunner from the registry.

        Override for custom loading (e.g. from a direct path).
        """
        from src.rade_ml.inference.runner import InferenceRunner
        from src.rade_ml.registry.store import ModelRegistry

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

        runner = self.load_runner(self.config)
        logger.info("InferencePipeline: runner loaded")

        prepared = self.prepare_inputs(self.config)
        inputs = prepared["inputs"]
        sample_ids = prepared.get("sample_ids", prepared.get("trade_ids"))
        metadata = prepared.get("metadata")

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

        self.post_infer(result, self.config)

        logger.info("InferencePipeline: done")
        return result
