"""
Hyperparameter tuning pipeline for the Hybrid GNN-RNN model (PyTorch).

Defines the search space over GNN, RNN, fusion, attention, projection,
and training hyperparameters, wiring into the generic ``TunePipeline``
orchestration.

Data-level params (``transform_type``, ``shuffle``) are deliberately excluded
from the default search space because each unique combination requires a
full ``build_dataset`` call.  Evaluate these separately with short preliminary
runs, then fix the best choice in your ``HybridGnnRnnDataConfig`` before
launching the main Optuna study.  A subclass can add them back by overriding
``search_space`` and ``run`` (with a caching strategy).

Usage::

    config = PipelineConfig(
        data_config=HybridGnnRnnDataConfig(...).to_dict(),
        artifacts_dir="./rade_ml_output/artifacts",
        metadata={"run_name": "gnn_rnn_tune"},
    )
    pipeline = HybridGnnRnnTunePipeline(
        config,
        tuner_kwargs={"n_trials": 100, "direction": "minimize", "pruner": "hyperband"},
    )
    result = pipeline.run()

    # To retrain the best trial and register the model:
    config.metadata["retrain_best"] = True
    result = pipeline.run()
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch.nn as nn

from src.rade_ml_pt.pipelines.base import TunePipeline
from src.rade_ml_pt.pipelines.config import PipelineConfig
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml_pt.data.hybrid_gnn_rnn.build import build_dataset

if TYPE_CHECKING:
    import optuna
    from src.rade_ml_pt.data.result import DataBuildResult

logger = logging.getLogger(__name__)


class HybridGnnRnnTunePipeline(TunePipeline):
    """
    Concrete tuning pipeline for Hybrid GNN-RNN.

    Search space covers:
        - GNN: units, layers, dropout, aggregation
        - RNN: units, layers, dropout, layer_type
        - Fusion: units, dropout, fusion_mode
        - Attention: mode, units, num_heads, dropout
        - Projection: units, dropout
        - Training: learning_rate, batch_size, loss
    """

    # ------------------------------------------------------------------
    # Abstract hook implementations
    # ------------------------------------------------------------------

    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        """Build DataLoaders from the pipeline config and job metadata."""
        data_config = self._resolve_data_config(config.data_config)
        job = config.metadata.get("job", {})
        return build_dataset(config=data_config, job=job)

    def search_space(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """
        Sample hyperparameters for one Optuna trial.

        Override this method to customise the search space for your
        specific use case (e.g. narrower ranges after initial exploration).

        Returns a flat dict of param_name -> sampled_value, grouped by
        subsystem (gnn, rnn, fusion, attention, projection, training).

        Note: data-level params (transform_type, shuffle) are excluded from
        the default search space because each combination triggers a full
        data rebuild.  Evaluate those separately and fix the best choice in
        your HybridGnnRnnDataConfig before launching the Optuna study.
        """
        return {
            # ---- GNN layer ----
            # hidden units per GNN message-passing layer.
            "gnn_units": trial.suggest_int("gnn_units", 32, 256, step=32),
            # number of stacked GNN layers (depth of message passing).
            "gnn_layers": trial.suggest_int("gnn_layers", 1, 4),
            # dropout probability applied after each GNN layer.
            "gnn_dropout": trial.suggest_float("gnn_dropout", 0.0, 0.5, step=0.05),
            # neighbourhood aggregation operator for GraphSAGE.
            "gnn_aggregation": trial.suggest_categorical(
                "gnn_aggregation", ["mean", "sum", "max"],
            ),

            # ---- RNN layer ----
            # hidden units per recurrent cell.
            "rnn_units": trial.suggest_int("rnn_units", 32, 256, step=32),
            # number of stacked RNN layers.
            "rnn_layers": trial.suggest_int("rnn_layers", 1, 3),
            # dropout probability between RNN layers.
            "rnn_dropout": trial.suggest_float("rnn_dropout", 0.0, 0.5, step=0.05),
            # recurrent cell type — LSTM retains longer memory, GRU is lighter.
            "rnn_layer_type": trial.suggest_categorical("rnn_layer_type", ["lstm", "gru"]),

            # ---- fusion layer ----
            # hidden units in the GNN–RNN fusion MLP.
            "fusion_units": trial.suggest_int("fusion_units", 32, 256, step=32),
            # dropout probability in the fusion block.
            "fusion_dropout": trial.suggest_float("fusion_dropout", 0.0, 0.5, step=0.05),
            # how GNN and RNN embeddings are combined.
            "fusion_mode": trial.suggest_categorical("fusion_mode", ["gate", "concat", "add"]),

            # ---- attention layer ----
            # attention_mode toggles the entire attention block on/off.
            "attention_mode": trial.suggest_categorical("attention_mode", [True, False]),
            # hidden units inside the attention scoring network.
            "attention_units": trial.suggest_int("attention_units", 16, 128, step=16),
            # number of parallel attention heads (multi-head attention).
            "attention_num_heads": trial.suggest_int("attention_num_heads", 1, 4),
            # dropout probability on attention weights.
            "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.3, step=0.05),

            # ---- projection layer ----
            # hidden units in the final projection MLP before output.
            "projection_units": trial.suggest_int("projection_units", 16, 128, step=16),
            # dropout probability in the projection block.
            "projection_dropout": trial.suggest_float("projection_dropout", 0.0, 0.3, step=0.05),

            # ---- training ----
            # Adam learning rate (log-uniform for orders-of-magnitude search).
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            # mini-batch size for DataLoader.
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            # loss function — mae, huber, or huber_quantile composite.
            "loss": trial.suggest_categorical("loss", ["mae", "huber", "huber_quantile"]),
        }

    def build_trial_model(
        self,
        params: Dict[str, Any],
        data_result: "DataBuildResult",
    ) -> nn.Module:
        """Build a HybridGnnRnn model with trial hyperparameters applied."""
        from src.rade_ml_pt.models.hybrid_gnn_rnn.model import HybridGnnRnn
        from src.rade_ml_pt.models.hybrid_gnn_rnn.config import HybridGnnRnnModelConfig

        # start from the user-supplied base config (or defaults).
        base_model_config = self.config.model_config or {}
        if hasattr(base_model_config, "to_dict"):
            base_model_config = base_model_config.to_dict()

        # round-trip through the dataclass to fill in any missing defaults,
        # then convert back to the mutable dict the model constructor expects.
        model_config_dict = HybridGnnRnnModelConfig.from_dict(base_model_config).to_dict()

        # overwrite architecture values with trial-sampled hyperparameters.
        self._apply_model_params(model_config_dict, params)

        # instantiate the model from the mutated config dict.
        model = HybridGnnRnn(config=model_config_dict)
        logger.debug(
            f"Trial model: gnn={params['gnn_units']}x{params['gnn_layers']}, "
            f"rnn={params['rnn_units']}x{params['rnn_layers']} ({params['rnn_layer_type']}), "
            f"fusion={params['fusion_units']} ({params['fusion_mode']}), "
            f"attn={'on' if params['attention_mode'] else 'off'}, "
            f"lr={params['learning_rate']:.2e}, loss={params['loss']}"
        )
        return model

    # ------------------------------------------------------------------
    # Training config resolution
    # ------------------------------------------------------------------

    def _resolve_trial_training_config(
        self,
        base_training_config: Any,
        params: Dict[str, Any],
        trial: Any,
    ) -> Any:
        """
        Merge trial-sampled training params into a TrainingConfig copy.

        Overrides the base implementation because:
        - ``learning_rate`` lives on ``OptimizerConfig`` (nested inside TrainingConfig).
        - ``batch_size`` lives on ``DataPipelineConfig`` (handled via data config).
        The base ``dataclasses.replace()`` would raise TypeError on both.
        """
        overrides: Dict[str, Any] = {}

        # learning_rate is nested under OptimizerConfig, not a top-level TrainingConfig field.
        if "learning_rate" in params:
            overrides["optimizer"] = replace(
                base_training_config.optimizer, learning_rate=params["learning_rate"],
            )

        # loss is a top-level TrainingConfig field (e.g. "mae", "huber", "huber_quantile").
        if "loss" in params:
            overrides["loss"] = params["loss"]

        # epochs is a top-level TrainingConfig field — not in the default search space
        # but supported so subclass overrides can tune it.
        if "epochs" in params:
            overrides["epochs"] = params["epochs"]

        return replace(base_training_config, **overrides) if overrides else base_training_config

    # ------------------------------------------------------------------
    # Artifact saving — extends base with parallel coordinate plot
    # ------------------------------------------------------------------

    def _save_tuning_artifacts(self, result: Any, tuner: Any) -> None:
        """
        Save tuning results and plots to ``artifacts_dir/tuning/{study_name}/``.

        Extends the base class (which saves optimization_history.png and
        param_importances.png) with an additional parallel coordinate plot.
        """
        from pathlib import Path

        # delegate to base — saves tuning_result.json, optimization_history.png,
        # and param_importances.png.
        super()._save_tuning_artifacts(result, tuner)

        # save additional plots that need the live study.
        study = getattr(tuner, "study", None)
        if study is None or not self.config.artifacts_dir:
            return

        study_name = result.study_name or "tune"
        tune_dir = Path(self.config.artifacts_dir) / "tuning" / study_name

        try:
            import matplotlib.pyplot as plt
            from src.rade_ml_pt.tuning.plots import plot_parallel_coordinate

            # parallel coordinate — multi-dimensional view of params vs objective.
            plot_parallel_coordinate(study)
            fig = plt.gcf()
            fig.savefig(tune_dir / "parallel_coordinate.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved parallel_coordinate.png to {tune_dir}")
        except Exception as exc:
            logger.warning(f"Could not save parallel coordinate plot: {exc}")

    # ------------------------------------------------------------------
    # Post-tuning hooks
    # ------------------------------------------------------------------

    def post_tune(
        self,
        result: Any,
        config: PipelineConfig,
        data_result: Optional["DataBuildResult"] = None,
    ) -> None:
        """Log best trial summary and optionally retrain the winning configuration."""
        logger.info(
            f"Best trial #{result.best_trial_number}: "
            f"val_loss={result.best_value:.6f} | "
            f"params={result.best_params}"
        )
        logger.info(
            f"Trials: {result.n_completed} completed, {result.n_pruned} pruned, "
            f"{result.elapsed_seconds:.1f}s total"
        )

        # if the user opted in via metadata, retrain the best config end-to-end
        # through the full HybridGnnRnnTrainPipeline to produce a registered model.
        if config.metadata.get("retrain_best", False):
            logger.info("Retraining best trial configuration...")
            self._retrain_best(result.best_params, config)

    def _retrain_best(self, best_params: Dict[str, Any], config: PipelineConfig) -> None:
        """
        Retrain a model using the best hyperparameters from tuning.

        Assembles a fresh ``PipelineConfig`` with the winning trial's model,
        training, and data params, then runs ``HybridGnnRnnTrainPipeline``
        end-to-end to produce a registered, artifact-backed model version.
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline
        from src.rade_ml_pt.models.hybrid_gnn_rnn.config import HybridGnnRnnModelConfig
        from src.rade_ml_pt.core.config import TrainingConfig

        # ---- 1. model config — apply best architecture params ----
        base_model_config = config.model_config or {}
        if hasattr(base_model_config, "to_dict"):
            base_model_config = base_model_config.to_dict()

        # round-trip through the dataclass to fill in any missing defaults.
        model_config_dict = HybridGnnRnnModelConfig.from_dict(base_model_config).to_dict()

        # apply best trial's architecture hyperparameters (shared helper).
        self._apply_model_params(model_config_dict, best_params)

        # ---- 2. training config — apply best lr / loss ----
        base_training_config = config.training_config
        if base_training_config is None:
            base_training_config = TrainingConfig()
        elif isinstance(base_training_config, dict):
            base_training_config = TrainingConfig.from_dict(base_training_config)

        # reuse the same merging logic used per-trial (lr, loss, epochs).
        retrain_training_config = self._resolve_trial_training_config(
            base_training_config, best_params, trial=None,
        )

        # ---- 3. data config — apply any data-level params from best trial ----
        # batch_size is the only default search-space param that lives on
        # DataPipelineConfig; transform_type and shuffle are included here for
        # forward-compatibility if a subclass adds them to the search space.
        retrain_data_config = self._resolve_data_config(config.data_config)

        data_overrides = {
            key: best_params[key]
            for key in ("batch_size", "transform_type", "shuffle")
            if key in best_params
        }
        if data_overrides:
            retrain_data_config = replace(retrain_data_config, **data_overrides)

        # ---- 4. assemble pipeline config and run full training ----
        retrain_run_name = f"{config.metadata.get('run_name', 'tune')}_best"
        retrain_metadata = {**config.metadata, "run_name": retrain_run_name}

        retrain_config = PipelineConfig(
            data_config=retrain_data_config,
            model_config=model_config_dict,
            training_config=retrain_training_config,
            artifacts_dir=config.artifacts_dir,
            registry_dir=config.registry_dir,
            tracking_dir=config.tracking_dir,
            metadata=retrain_metadata,
        )

        # run the full train pipeline to produce a registered model.
        train_pipeline = HybridGnnRnnTrainPipeline(retrain_config)
        train_result = train_pipeline.run()
        logger.info(
            f"Retrained best config: val_loss={train_result.best_val_loss:.6f}, "
            f"epoch={train_result.best_epoch}/{train_result.final_epoch}"
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_model_params(model_config_dict: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Apply trial hyperparameters to a ``HybridGnnRnnModelConfig`` dict in-place.

        Maps flat param names (e.g. ``gnn_units``) to the nested config
        structure expected by the model (e.g. ``gnn_layer.parameters.units``).

        Used by both ``build_trial_model`` (per-trial) and ``_retrain_best``
        (post-tuning) to avoid duplicating the mapping logic.
        """
        # GNN — hidden units on parameters; layers, dropout, aggregation on general.
        gnn = model_config_dict["gnn_layer"]
        gnn["parameters"]["units"] = params["gnn_units"]
        gnn["general"]["layers"] = params["gnn_layers"]
        gnn["general"]["dropout_rate"] = params["gnn_dropout"]
        gnn["general"]["aggregator_op"] = params["gnn_aggregation"]

        # RNN — hidden units on parameters; layers, dropout, cell type on general.
        rnn = model_config_dict["rnn_layer"]
        rnn["parameters"]["units"] = params["rnn_units"]
        rnn["general"]["layers"] = params["rnn_layers"]
        rnn["general"]["dropout_rate"] = params["rnn_dropout"]
        rnn["general"]["layer_type"] = params["rnn_layer_type"]

        # Fusion — hidden units on parameters; mode, dropout on general.
        fusion = model_config_dict["fusion_layer"]
        fusion["parameters"]["units"] = params["fusion_units"]
        fusion["general"]["dropout_rate"] = params["fusion_dropout"]
        fusion["general"]["fusion_mode"] = params["fusion_mode"]

        # Attention — hidden units on parameters; mode, heads, dropout on general.
        attention = model_config_dict["attention_layer"]
        attention["general"]["attention_mode"] = params["attention_mode"]
        attention["general"]["num_heads"] = params["attention_num_heads"]
        attention["general"]["dropout_rate"] = params["attention_dropout"]
        attention["parameters"]["units"] = params["attention_units"]

        # Projection — hidden units on parameters; dropout on general.
        projection = model_config_dict["projection_layer"]
        projection["parameters"]["units"] = params["projection_units"]
        projection["general"]["dropout_rate"] = params["projection_dropout"]

    @staticmethod
    def _resolve_data_config(raw_config: Any = None) -> HybridGnnRnnDataConfig:
        """
        Normalise a data config (dict, None, or dataclass) into a concrete
        ``HybridGnnRnnDataConfig`` instance with defaults filled in.
        """
        if isinstance(raw_config, dict):
            return HybridGnnRnnDataConfig.from_dict(raw_config)
        if raw_config is None:
            return HybridGnnRnnDataConfig()
        return raw_config
