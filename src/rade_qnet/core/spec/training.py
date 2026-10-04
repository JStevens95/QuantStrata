"""
Training settings, discriminated by engine.

Each engine's settings are a separate type rather than a bag of optional
fields shared between them. The reason is that a shared bag cannot tell the
difference between "not set" and "not applicable": ``max_depth: null`` on a
neural network is meaningless, and ``gradient_clip_norm: 1.0`` on a tree model
is a setting that will be silently ignored. Both are configuration errors, and
with a discriminated union both are rejected at load time.

Note what is *absent* from every spec here: ``batch_size``. Batching belongs
to :class:`~rade_qnet.core.spec.data.LoaderSpec`, because it describes how data
is presented rather than how parameters are updated. Keeping it there means a
tuning sweep over batch size does not have to know which engine it is
configuring.

Reinforcement learning
----------------------
:class:`RlTrainingSpec` is not a member of the :data:`TrainingSpec` union. It
runs on the Torch engine, so it cannot be discriminated from
:class:`TorchTrainingSpec` by the ``engine`` field; instead it is selected one
level up, by ``task`` on the run spec. That keeps both discriminators
unambiguous.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from .base import Spec

__all__ = [
    "CheckpointSpec",
    "EarlyStoppingSpec",
    "RlTrainingSpec",
    "SchedulerSpec",
    "SklearnTrainingSpec",
    "TorchTrainingSpec",
    "TrainingSpec",
    "XGBoostTrainingSpec",
]


class EarlyStoppingSpec(Spec):
    """
    When to stop training before the epoch budget is exhausted.

    Parameters
    ----------
    enabled
        Whether early stopping is active. Defaults to ``False``, for two
        reasons. A run should train for the epoch budget it was configured
        with -- silently training for fewer, and keeping a different epoch's
        weights, is surprising behaviour to get without asking. And the
        default ``monitor`` is a validation metric, so defaulting to enabled
        would make a bare spec invalid for any run configured without a
        validation split.
    monitor
        Metric to watch. Must be a metric the run actually produces; an
        unknown name is caught when the callback is constructed, where the
        error can list what is available.
    patience
        Epochs without improvement before stopping.
    min_delta
        Improvement below this is treated as no improvement, which stops a
        run from continuing for fifty epochs on fourth-decimal noise.
    mode
        Whether a lower or higher value is better.
    """

    enabled: bool = False
    monitor: str = "val_loss"
    patience: int = Field(default=20, ge=1)
    min_delta: float = Field(default=0.0, ge=0.0)
    mode: Literal["min", "max"] = "min"


class CheckpointSpec(Spec):
    """
    Which parameters are kept, and which are restored at the end.

    Parameters
    ----------
    enabled
        Whether to checkpoint at all.
    monitor, mode
        Metric that identifies the best epoch.
    restore_best
        Whether to restore the best epoch's parameters when training ends.
        Default true: without it, the model that gets saved is the one from
        the last epoch while the metrics reported are from the best, so the
        bundle's weights and its metrics describe different models.
    keep_last
        How many checkpoints to retain on disk.
    """

    enabled: bool = True
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    restore_best: bool = True
    keep_last: int = Field(default=1, ge=1)


class SchedulerSpec(Spec):
    """
    Learning-rate schedule.

    Parameters
    ----------
    kind
        ``none`` holds the rate constant. ``plateau`` reduces it when a
        monitored metric stops improving; the others are time-based.
    factor
        Multiplicative reduction, for ``step`` and ``plateau``.
    patience
        Epochs without improvement before reducing, for ``plateau``.
    step_epochs
        Period between reductions, for ``step``.
    min_learning_rate
        Floor below which the rate will not be reduced.
    """

    kind: Literal["none", "step", "cosine", "plateau"] = "none"
    factor: float = Field(default=0.1, gt=0.0, lt=1.0)
    patience: int = Field(default=10, ge=1)
    step_epochs: int = Field(default=30, ge=1)
    min_learning_rate: float = Field(default=0.0, ge=0.0)


class TorchTrainingSpec(Spec):
    """
    Gradient-based training on the PyTorch engine.

    Parameters
    ----------
    engine
        Discriminator.
    learner
        Registered name of the update rule. ``supervised`` is forward, loss,
        backward. Naming it rather than implying it from the task is what
        lets a reinforcement-learning learner reuse this same engine.
    epochs
        Maximum passes over the training split. Early stopping usually ends
        the run sooner.
    learning_rate, optimiser, weight_decay
        Optimiser settings.
    loss
        Registered loss name.
    gradient_clip_norm
        Global gradient-norm clip. ``None`` disables clipping.
    early_stopping, checkpoint, scheduler
        Callback settings, each with usable defaults.
    """

    engine: Literal["torch"] = "torch"
    learner: str = "supervised"
    epochs: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    optimiser: Literal["adam", "adamw", "sgd"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    loss: str = "mse"
    gradient_clip_norm: float | None = Field(default=None, gt=0.0)
    early_stopping: EarlyStoppingSpec = Field(default_factory=EarlyStoppingSpec)
    checkpoint: CheckpointSpec = Field(default_factory=CheckpointSpec)
    scheduler: SchedulerSpec = Field(default_factory=SchedulerSpec)

    @model_validator(mode="after")
    def _check_monitored_metrics_agree(self) -> TorchTrainingSpec:
        """
        Reject early stopping and checkpointing that disagree.

        Returns
        -------
        TorchTrainingSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if the two callbacks monitor the same
            metric in opposite directions, which would make them select
            different epochs as
            best -- so the saved weights and the stopping decision would
            describe different models.
        """
        if (
            self.early_stopping.enabled
            and self.checkpoint.enabled
            and self.early_stopping.monitor == self.checkpoint.monitor
            and self.early_stopping.mode != self.checkpoint.mode
        ):
            raise SpecError(
                f"early_stopping and checkpoint both monitor "
                f"{self.early_stopping.monitor!r} but disagree on direction "
                f"({self.early_stopping.mode!r} vs {self.checkpoint.mode!r}); "
                f"they would select different epochs as best"
            )
        return self


class XGBoostTrainingSpec(Spec):
    """
    One-shot gradient-boosted tree training.

    No epoch loop: the engine translates this into a single ``train`` call and
    reports the boosting history through the same result contract a neural
    network produces.

    Parameters
    ----------
    engine
        Discriminator.
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
    min_child_weight, reg_lambda
        Boosting settings, named as XGBoost names them so that a user who
        knows the library is not made to learn a translation layer.
    objective
        Training objective.
    early_stopping_rounds
        Boosting rounds without improvement before stopping. ``None``
        disables it. The library's own implementation is used rather than a
        framework reimplementation, because it operates per round inside the
        booster.
    """

    engine: Literal["xgboost"] = "xgboost"
    n_estimators: int = Field(default=500, ge=1)
    max_depth: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.05, gt=0.0)
    subsample: float = Field(default=1.0, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=1.0, gt=0.0, le=1.0)
    min_child_weight: float = Field(default=1.0, ge=0.0)
    reg_lambda: float = Field(default=1.0, ge=0.0)
    objective: str = "reg:squarederror"
    early_stopping_rounds: int | None = Field(default=50, ge=1)


class SklearnTrainingSpec(Spec):
    """
    One-shot scikit-learn estimator training.

    Estimator hyper-parameters belong to the model's own spec, not here -- a
    ridge regression's ``alpha`` is part of what the model *is*. This spec
    carries only what the framework needs to drive the fit.

    Parameters
    ----------
    engine
        Discriminator.
    fit_params
        Extra keyword arguments forwarded to the estimator's ``fit``, such as
        sample weights.
    """

    engine: Literal["sklearn"] = "sklearn"
    fit_params: Mapping[str, object] = Field(default_factory=dict)


#: Supervised training settings, discriminated on ``engine``.
TrainingSpec = Annotated[
    TorchTrainingSpec | XGBoostTrainingSpec | SklearnTrainingSpec,
    Field(discriminator="engine"),
]


class RlTrainingSpec(Spec):
    """
    Interactive training on the PyTorch engine.

    Driven by update count rather than epochs, because the source is
    unbounded: there is no meaningful notion of a pass over an environment.

    Parameters
    ----------
    engine
        Fixed to ``torch``. Present for symmetry with
        :data:`TrainingSpec` and because a future engine would need it.
    learner
        Which update rule. ``pathwise`` requires a differentiable
        environment and back-propagates a risk measure through the simulated
        dynamics; the others are transition-based.
    total_steps
        Total optimisation steps.
    steps_per_update, batch_size
        How much experience is gathered per update, and how much of it each
        update consumes.
    discount
        Reward discount factor. Ignored by ``pathwise``, which optimises a
        risk measure of the terminal outcome directly.
    gradient_clip_norm, learning_rate, optimiser
        Optimiser settings, as for supervised training.
    evaluate_every_steps, evaluation_episodes
        How often to run evaluation episodes, and how many.
    """

    engine: Literal["torch"] = "torch"
    learner: Literal["dqn", "ppo", "sac", "pathwise"] = "ppo"
    total_steps: int = Field(default=100_000, ge=1)
    steps_per_update: int = Field(default=2048, ge=1)
    batch_size: int = Field(default=256, ge=1)
    discount: float = Field(default=0.99, ge=0.0, le=1.0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    optimiser: Literal["adam", "adamw", "sgd"] = "adam"
    gradient_clip_norm: float | None = Field(default=0.5, gt=0.0)
    evaluate_every_steps: int = Field(default=10_000, ge=1)
    evaluation_episodes: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _check_batch_fits_the_update(self) -> RlTrainingSpec:
        """
        Reject a batch larger than the experience gathered per update.

        Returns
        -------
        RlTrainingSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` for an on-policy learner whose batch
            exceeds the experience collected between updates, which would
            require reusing samples the algorithm assumes are fresh.
        """
        on_policy = {"ppo"}
        if self.learner in on_policy and self.batch_size > self.steps_per_update:
            raise SpecError(
                f"learner={self.learner!r} is on-policy, so batch_size "
                f"({self.batch_size}) cannot exceed steps_per_update "
                f"({self.steps_per_update})"
            )
        return self
