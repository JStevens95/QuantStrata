"""
``TorchEngine`` -- the ``Engine`` implementation for PyTorch.

Thin by design. Every decision of substance lives in a focused module --
:mod:`.hardware`, :mod:`.materialise`, :mod:`.loaders`, :mod:`.loops`,
:mod:`.callbacks`, :mod:`.checkpoint`, :mod:`.distributed` -- and this class
sequences them. If a method here grows a second decision, that decision
belongs in one of those modules.

The order in :meth:`TorchEngine.prepare` is the contract
--------------------------------------------------------
``materialise`` is a *pipeline* stage and runs before ``prepare`` is called at
all. Inside ``prepare`` the remaining order is::

    device placement -> compile -> distribute -> optimiser

and every step depends on the one before it. Compiling before placement traces
a graph for the wrong device. Distributing before compiling means the compiler
sees the wrapper rather than the model. Building the optimiser before
distributing hands it parameters the wrapper is about to replace.

The ordering is enforced by it being written once, here, rather than by a
comment asking callers to be careful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...core.runtime.components import engine as register_engine
from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger
from ...core.spec.training import CheckpointSpec, EarlyStoppingSpec, TorchTrainingSpec
from ..base import EngineCapabilities, ModelHandle
from .callbacks import BestCheckpoint, EarlyStopping, GradientNorms, LearningRateSchedule
from .checkpoint import load_weights, save_weights
from .distributed import distribute, shutdown_distributed, undistribute
from .hardware import (
    ResolvedHardware,
    available_accelerators,
    compile_if_requested,
    resolve_hardware,
)
from .learners.supervised import SupervisedLearner
from .loops import fit_epochs
from .losses import build_loss
from .materialise import count_parameters
from .materialise import materialise as materialise_model
from .predictor import predict_batches

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from ...core.contract.data import TensorLike
    from ...core.contract.result import FitOutcome
    from ...core.contract.signature import InputSignature
    from ...core.contract.source import BatchSource
    from ...core.spec.hardware import HardwareSpec
    from .callbacks import Callback

__all__ = ["TorchEngine"]

_LOGGER = get_logger(__name__)

#: Registered name, used by the engine registry and by ``TorchTrainingSpec``'s
#: ``engine`` discriminator.
ENGINE_NAME = "torch"

#: Optimiser constructors by spec name. A table rather than a chain of
#: conditionals, so adding one is a line here and nothing else.
_OPTIMISERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

#: Keys the handle's ``extras`` carries. Named so that :meth:`TorchEngine.fit`
#: and :meth:`TorchEngine.prepare` cannot disagree on a string literal.
_OPTIMISER_KEY = "optimiser"
_HARDWARE_KEY = "hardware"
_SIGNATURE_KEY = "signature"

#: Monitor used in place of a validation metric when a run has no validation
#: split. Always available, since every epoch has a training loss.
_FALLBACK_MONITOR = "train_loss"


def _without_validation_monitor[SpecT: (EarlyStoppingSpec, CheckpointSpec)](
    spec: SpecT,
) -> SpecT:
    """
    Return a spec whose monitor does not require a validation split.

    Parameters
    ----------
    spec
        An early-stopping or checkpoint specification.

    Returns
    -------
    EarlyStoppingSpec or CheckpointSpec
        The spec unchanged when it is disabled or already monitors a training
        metric, and otherwise a copy monitoring :data:`_FALLBACK_MONITOR`.

        A copy rather than a mutation, because the spec belongs to the run
        specification and is shared -- rewriting it in place would change what
        a later report says the run was configured to do.
    """
    if not spec.enabled or not spec.monitor.startswith("val"):
        return spec
    return spec.model_copy(update={"monitor": _FALLBACK_MONITOR})


@register_engine(ENGINE_NAME)
class TorchEngine:
    """
    Trains ``torch.nn.Module`` models.

    Satisfies :class:`~rade_qnet.engines.base.Engine` structurally, without
    inheriting from it, so the conformance suite checks it against the same
    definition a user's engine is held to.

    Parameters
    ----------
    checkpoint_directory
        Where :class:`~.callbacks.BestCheckpoint` writes per-epoch files.
        ``None`` keeps the best weights in memory only, which is the right
        default: a tuning sweep of several hundred trials would otherwise
        leave several hundred checkpoint files behind, and only the final
        bundle's weights are of any interest.
    """

    def __init__(self, *, checkpoint_directory: Path | None = None) -> None:
        self.checkpoint_directory = checkpoint_directory

    def capabilities(self) -> EngineCapabilities:
        """
        Declare what this engine supports.

        Returns
        -------
        EngineCapabilities
            The engine's declared abilities.
        """
        return EngineCapabilities(
            name=ENGINE_NAME,
            supports_epochs=True,
            supports_validation_during_fit=True,
            supports_checkpointing=True,
            supports_distributed=True,
            supports_lazy_materialisation=True,
            accelerators=available_accelerators(),
        )

    def materialise(self, model: object, signature: InputSignature) -> object:
        """
        Give a lazily shaped model its parameters.

        Parameters
        ----------
        model
            The untrained model.
        signature
            The declared interface.

        Returns
        -------
        object
            The model, with parameters.

        Raises
        ------
        EngineError
            If the model is not a Torch module, or the dummy pass fails.
        """
        return materialise_model(self._module(model), signature)

    def prepare(
        self,
        model: object,
        *,
        hardware: HardwareSpec,
        training: object,
        static: Mapping[str, TensorLike] | None = None,
    ) -> ModelHandle:
        """
        Place the model on its device and build its training apparatus.

        Parameters
        ----------
        model
            The materialised model.
        hardware
            Device, precision, compilation and distribution settings.
        training
            A :class:`~rade_qnet.core.spec.training.TorchTrainingSpec`.
        static
            Static inputs from the data build, uploaded once.

        Returns
        -------
        ModelHandle
            The prepared model and its apparatus.

        Raises
        ------
        EngineError
            If ``training`` is not a Torch training spec, or the model is not
            a Torch module.
        """
        module = self._module(model)
        spec = self._training_spec(training)
        resolved = resolve_hardware(hardware)

        # The order below is the contract; see the module docstring.
        module = module.to(device=resolved.device)
        prepared = compile_if_requested(module, spec=hardware)
        prepared, is_distributed = distribute(prepared, spec=hardware, device=resolved.device)

        optimiser = self._build_optimiser(prepared, spec)
        n_parameters = count_parameters(undistribute(prepared), trainable_only=True)
        if n_parameters == 0:
            raise EngineError(
                "the prepared model has no trainable parameters, so the optimiser "
                "would have nothing to update and training would run without "
                "changing anything. Either every parameter has requires_grad=False, "
                "or the model was not materialised before prepare()"
            )

        handle = ModelHandle(
            model=prepared,
            unwrapped=undistribute(prepared),
            device=str(resolved.device),
            precision=resolved.precision,
            is_distributed=is_distributed,
            static=dict(static or {}),
            extras={
                _OPTIMISER_KEY: optimiser,
                _HARDWARE_KEY: resolved,
            },
        )
        _LOGGER.info("prepared %s with %d trainable parameter(s)", handle.describe(), n_parameters)
        return handle

    def fit(
        self,
        handle: ModelHandle,
        sources: Mapping[str, BatchSource],
        training: object,
        *,
        on_epoch_end: object = None,
    ) -> FitOutcome:
        """
        Train the model and report what happened.

        Parameters
        ----------
        handle
            The prepared model.
        sources
            One source per split. ``train`` is required.
        training
            A :class:`~rade_qnet.core.spec.training.TorchTrainingSpec`.
        on_epoch_end
            Currently accepted and not yet forwarded; see the note below.

        Returns
        -------
        FitOutcome
            History, best epoch, and whether the best weights were restored.

        Raises
        ------
        EngineError
            If a required source is missing, or the spec is not a Torch one.

        Notes
        -----
        ``on_epoch_end`` is accepted so the signature matches the protocol,
        and is not yet wired through. The seam exists for the pipeline to
        forward live epoch metrics to hooks and trackers, and those consumers
        arrive with ``orchestration``'s tracker integration -- so wiring it
        now would mean a callback with no reader. It is recorded as an open
        item rather than quietly dropped.
        """
        spec = self._training_spec(training)
        if on_epoch_end is not None:
            _LOGGER.debug("on_epoch_end was supplied but is not yet forwarded")

        module = self._module(handle.model)
        unwrapped = self._module(handle.unwrapped)
        resolved = self._resolved_hardware(handle)
        optimiser = self._optimiser(handle)

        has_validation = "validation" in sources
        early_stopping_spec = spec.early_stopping
        checkpoint_spec = spec.checkpoint
        if not has_validation:
            # Both of these monitor 'val_loss' by default, so a run with no
            # validation split would be refused by the callbacks on the
            # default configuration.  Substituting the training loss here
            # rather than raising is what makes a train-only fit possible at
            # all -- and it is the honest reading of the situation: the run
            # still has a best epoch, it is just chosen on less evidence.
            early_stopping_spec = _without_validation_monitor(early_stopping_spec)
            checkpoint_spec = _without_validation_monitor(checkpoint_spec)
            _LOGGER.warning(
                "no validation source: early stopping and best-epoch selection "
                "fall back to monitoring %r. A best epoch chosen on the training "
                "loss is the epoch that fitted the training data most closely, "
                "which is not the same thing as the one that generalises best",
                _FALLBACK_MONITOR,
            )

        gradient_norms = GradientNorms(clip_norm=spec.gradient_clip_norm)
        checkpoint = BestCheckpoint(
            checkpoint_spec,
            model=unwrapped,
            has_validation=has_validation,
            directory=self.checkpoint_directory,
        )
        callbacks: list[Callback] = [
            EarlyStopping(early_stopping_spec, has_validation=has_validation),
            LearningRateSchedule(
                spec.scheduler,
                optimiser=optimiser,
                epochs=spec.epochs,
                has_validation=has_validation,
            ),
        ]

        learner = SupervisedLearner(
            loss=build_loss(spec.loss),
            optimiser=optimiser,
            hardware=resolved,
            gradient_clip_norm=spec.gradient_clip_norm,
            gradient_norms=gradient_norms,
        )

        try:
            return fit_epochs(
                module,
                sources,
                learner=learner,
                signature=sources["train"].signature,
                device=resolved.device,
                epochs=spec.epochs,
                callbacks=callbacks,
                gradient_norms=gradient_norms,
                checkpoint=checkpoint,
                learning_rate=spec.learning_rate,
            )
        finally:
            # Torn down even when the fit raised.  A process that exits
            # holding a process group leaves its peers blocked on a collective
            # until they time out, which presents as a job set that hangs
            # rather than as the error that actually occurred.
            if handle.is_distributed:
                shutdown_distributed()

    def predict(self, handle: ModelHandle, source: BatchSource) -> NDArray[np.floating]:
        """
        Run a forward pass over a source and return the raw output.

        Returns the model's own output space: inverting the target transform
        is the pipeline's job, using the data build's fitted state. An engine
        that inverted it here would invert it twice.

        Parameters
        ----------
        handle
            The prepared model.
        source
            Batches to predict over. Must be bounded.

        Returns
        -------
        numpy.ndarray
            Predictions in source order, one row per sample.

        Raises
        ------
        EngineError
            If the source is unbounded.
        """
        return predict_batches(
            self._module(handle.model),
            source,
            resolved=self._resolved_hardware(handle),
        )

    def save_weights(self, handle: ModelHandle, path: Path) -> None:
        """
        Write the model's parameters to a path.

        Parameters
        ----------
        handle
            The trained model.
        path
            Destination file.
        """
        save_weights(handle, path)

    def load_weights(self, model: object, path: Path) -> object:
        """
        Load parameters into a freshly built model.

        Parameters
        ----------
        model
            A model of the same architecture.
        path
            File written by :meth:`save_weights`.

        Returns
        -------
        object
            The model with parameters loaded.
        """
        return load_weights(model, path)

    # -- Narrowing helpers -------------------------------------------------

    @staticmethod
    def _module(model: object) -> torch.nn.Module:
        """
        Narrow an engine-opaque model to a Torch module.

        Parameters
        ----------
        model
            The model object.

        Returns
        -------
        torch.nn.Module
            The same object, narrowed.

        Raises
        ------
        EngineError
            If it is not a Torch module. The protocol types model objects as
            ``object`` so one signature serves every engine; this is where
            the Torch engine makes its requirement explicit, and the message
            names the likely cause.
        """
        if not isinstance(model, torch.nn.Module):
            raise EngineError(
                f"the Torch engine requires a torch.nn.Module, received "
                f"{type(model).__name__}. Check that the run spec's training "
                f"engine matches what the model definition builds"
            )
        return model

    @staticmethod
    def _training_spec(training: object) -> TorchTrainingSpec:
        """
        Narrow an engine-opaque training spec to a Torch one.

        Parameters
        ----------
        training
            The training spec.

        Returns
        -------
        TorchTrainingSpec
            The same spec, narrowed.

        Raises
        ------
        EngineError
            If it is another engine's spec.
        """
        if not isinstance(training, TorchTrainingSpec):
            raise EngineError(
                f"the Torch engine requires a TorchTrainingSpec, received "
                f"{type(training).__name__}. The run spec's training.engine "
                f"field selects both, so they cannot normally disagree -- this "
                f"usually means an engine was constructed directly"
            )
        return training

    @staticmethod
    def _optimiser(handle: ModelHandle) -> torch.optim.Optimizer:
        """
        Return the optimiser carried on a handle.

        Parameters
        ----------
        handle
            The prepared model.

        Returns
        -------
        torch.optim.Optimizer
            The optimiser.

        Raises
        ------
        EngineError
            If the handle carries none, which means it was built by a
            different engine.
        """
        optimiser = handle.extras.get(_OPTIMISER_KEY)
        if not isinstance(optimiser, torch.optim.Optimizer):
            raise EngineError(
                "this handle carries no optimiser; it was not prepared by the Torch engine"
            )
        return optimiser

    @staticmethod
    def _resolved_hardware(handle: ModelHandle) -> ResolvedHardware:
        """
        Return the resolved hardware carried on a handle.

        Parameters
        ----------
        handle
            The prepared model.

        Returns
        -------
        ResolvedHardware
            Device, precision and gradient scaler.

        Raises
        ------
        EngineError
            If the handle carries none.
        """
        resolved = handle.extras.get(_HARDWARE_KEY)
        if not isinstance(resolved, ResolvedHardware):
            raise EngineError(
                "this handle carries no resolved hardware; it was not prepared by the Torch engine"
            )
        return resolved

    def _build_optimiser(
        self, model: torch.nn.Module, spec: TorchTrainingSpec
    ) -> torch.optim.Optimizer:
        """
        Construct the optimiser a spec names.

        Parameters
        ----------
        model
            The prepared model, whose parameters the optimiser will track.
        spec
            The training spec.

        Returns
        -------
        torch.optim.Optimizer
            The optimiser.

        Raises
        ------
        EngineError
            If the optimiser name is unrecognised.
        """
        if spec.optimiser not in _OPTIMISERS:
            raise EngineError(
                f"unknown optimiser {spec.optimiser!r}; supported optimisers are "
                f"{sorted(_OPTIMISERS)}"
            )
        return _OPTIMISERS[spec.optimiser](
            model.parameters(),
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )
