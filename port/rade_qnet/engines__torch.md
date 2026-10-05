# `tranql/models/rade/rade_qnet/rade_qnet/engines/torch`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 85 | 3978 | `5bbf3471bf2e365e` |
| 2 | `engine.py` | 892 | 30935 | `7f8c07ab11295508` |
| 3 | `loaders.py` | 288 | 10187 | `3da70099294e818b` |
| 4 | `materialise.py` | 464 | 16380 | `6947dc425a1e454a` |
| 5 | `predictor.py` | 179 | 6469 | `d3bbea51d11a2eef` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/__init__.py`

3978 bytes · SHA-256 `5bbf3471bf2e365e`

```python
"""
The PyTorch engine.

This is the framework's primary engine and the one the flagship model uses.
Its central idea is a split of concerns that most training code leaves tangled:

*The loop* decides when to step, validate, checkpoint and stop.
*The learner* decides what a single update means.

A supervised regression, a deep Q-network and a pathwise hedging objective are
then three learners sharing one loop, rather than three training scripts.

Modules
-------
``engine.py``
    The ``Engine`` implementation, and the only file here that satisfies it:
    build, materialise, fit, checkpoint, predict.  [Phase 2]
``materialise.py``
    Runs one dummy forward pass from the input signature so lazy modules
    acquire concrete shapes before any optimiser, checkpoint or distributed
    wrapper touches them.  [Phase 2]
``loaders.py``
    Conversion of a ``BatchSource``'s NumPy batches into device-resident
    tensors.  Static inputs are kept out of per-sample collation and uploaded
    to the device once.  No ``DataLoader`` is constructed: a ``BatchSource``
    already yields whole batches, so prefetching across processes would mean
    pickling the source for no gain on in-memory arrays.  [Phase 2]
``predictor.py``
    Batched inference, including the precompute path for models that can cache
    an encoding of their static inputs.  [Phase 5]

Sub-packages
------------
``training/``
    How a fit is executed: the two drivers, the callbacks that watch them, the
    losses they optimise and the checkpoints they write.
``learners/``
    What one update means -- one module per algorithm.
``hardware/``
    Where the computation runs, and whether it runs the same way twice:
    device resolution, distributed training, and PyTorch seeding.

The four modules above are the four verbs an engine performs -- build, feed,
fit, predict -- and the three sub-packages are the parts of *fit* that are
large enough to have their own vocabulary.  That shape is the engine template:
``engine.py`` is required and every other name is drawn from this list, which
``tranql/models/rade/rade_qnet/tests/engines/test_engine_layout.py`` enforces.  The progression
across the three engines is itself informative -- xgboost is one file because
it fits in a single call and owns no loop, sklearn adds nothing but shares the
hoisted ``engines/loaders.py``, and only PyTorch needs all of it.

Planned modules
---------------
``training/risk.py``
    Differentiable risk measures (mean-variance, conditional value at risk,
    entropic) used as objectives by the pathwise learner.  [Phase 7]

Importing this package registers its components
-----------------------------------------------
Importing ``rade_qnet.engines.torch`` registers :class:`TorchEngine` under the
name ``"torch"`` and :class:`SupervisedLearner` under ``"supervised"``, which
is what lets a specification name them as strings.  It also registers
:func:`~rade_qnet.engines.torch.hardware.determinism.seed_torch`, without which
``seed_everything`` would leave Torch unseeded -- so two runs of one
configuration would differ in every weight initialisation while both reported
the same seed.

The import is eager rather than lazy on purpose.  A registry populated only
once someone happens to have imported the right module is the classic source
of "no engine named 'torch'" from a run whose configuration is perfectly
correct, and the only reliable cure is for the registration to be a
consequence of importing the package that owns it.  Importing this package
already implies that ``torch`` is installed, so nothing is paid by a host that
does not use it.
"""

from .engine import TorchEngine

# Importing the name is what registers the seeder, since registration happens
# at that module's import.
from .hardware.determinism import seed_torch
from .learners.random import RandomLearner
from .learners.supervised import SupervisedLearner

__all__ = ["RandomLearner", "SupervisedLearner", "TorchEngine", "seed_torch"]
```

---

## 2. `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/engine.py`

30935 bytes · SHA-256 `7f8c07ab11295508`

```python
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

from ...core.lifecycle.components import LEARNERS, ComponentError, get_learner
from ...core.lifecycle.components import engine as register_engine
from ...core.lifecycle.errors import EngineError
from ...core.provenance.logging import get_logger
from ...core.spec.training import (
    CheckpointSpec,
    EarlyStoppingSpec,
    RlTrainingSpec,
    TorchTrainingSpec,
)
from ..base import EngineCapabilities, ModelHandle
from .hardware.devices import (
    ResolvedHardware,
    available_accelerators,
    compile_if_requested,
    resolve_hardware,
)
from .hardware.distributed import distribute, shutdown_distributed, undistribute
from .learners.supervised import SupervisedLearner
from .loaders import to_tensor
from .materialise import count_parameters
from .materialise import materialise as materialise_model
from .materialise import materialise_policy as materialise_policy_model
from .predictor import predict_batches
from .training.callbacks import BestCheckpoint, EarlyStopping, GradientNorms, LearningRateSchedule
from .training.checkpoint import load_weights, save_weights
from .training.loops import fit_epochs, fit_steps
from .training.losses import build_loss

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from ...core.contract.data import TensorLike
    from ...core.contract.result import FitOutcome
    from ...core.contract.signature import InputSignature, PolicySignature
    from ...core.contract.source import BatchSource
    from ...core.spec.hardware import HardwareSpec
    from .training.callbacks import Callback
    from .training.loops import PolicyLearner

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
        spec = self._optimiser_spec(training)
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

    def materialise_policy(self, policy: object, signature: PolicySignature) -> object:
        """
        Give a lazily shaped policy its parameters.

        Parameters
        ----------
        policy
            The untrained policy.
        signature
            The observation and action spaces, from which the dummy
            observation is synthesised.

        Returns
        -------
        object
            The policy, with parameters.

        Raises
        ------
        EngineError
            If the policy is not a Torch module, or the dummy pass fails.
        """
        return materialise_policy_model(self._module(policy), signature)

    def fit_policy(
        self,
        handle: ModelHandle,
        source: BatchSource,
        training: object,
    ) -> FitOutcome:
        """
        Train a policy against a budget of interaction.

        The interactive counterpart of :meth:`fit`, and structurally the
        same: narrow the spec, build the callbacks, resolve the update rule,
        hand everything to a driver.

        One difference is worth naming. The learner here is resolved through
        the registry by the name in the spec, where :meth:`fit` constructs
        :class:`~.learners.supervised.SupervisedLearner` directly. The
        registry is how a third algorithm arrives without this method
        changing, which matters far more on this path than on the supervised
        one: there is exactly one supervised update rule and there will be
        several interactive ones.

        Parameters
        ----------
        handle
            The prepared policy.
        source
            One unbounded source of experience.
        training
            An :class:`~rade_qnet.core.spec.training.RlTrainingSpec`.

        Returns
        -------
        FitOutcome
            The block history and the step budget's outcome.

        Raises
        ------
        EngineError
            If the spec is not an interactive one, or the named learner is
            not registered.
        """
        spec = self._rl_training_spec(training)
        module = self._module(handle.model)
        resolved = self._resolved_hardware(handle)
        optimiser = self._optimiser(handle)

        gradient_norms = GradientNorms(clip_norm=spec.gradient_clip_norm)
        learner = self._policy_learner(
            spec,
            source=source,
            optimiser=optimiser,
            hardware=resolved,
            gradient_norms=gradient_norms,
        )

        # The source collects by acting, and what it acts with is this
        # learner on this policy. Bound here rather than passed to the source
        # at construction because the learner is built *from* the source's
        # policy signature, so the two cannot be created in that order. This
        # is the only place that knows both.
        bind = getattr(source, "bind", None)
        if bind is None:
            raise EngineError(
                f"the {type(source).__name__} handed to fit_policy has no "
                f"bind(), so nothing can tell it how to act. An interactive run "
                f"is driven by a source built over an environment, such as a "
                f"RolloutSource"
            )
        # The observation is placed on the device here, not in the learner.
        # Device placement is the engine's job everywhere else on this path --
        # `to_device_batches` for a dataset, `fit_steps` for experience -- and
        # a learner that had to do its own would get it wrong exactly once per
        # learner. The symptom is specific and unhelpful: the policy is on the
        # accelerator, the environment's observation is a NumPy array on the
        # host, and the forward pass fails inside a linear layer.
        bind(
            lambda observation: learner.act(module, to_tensor(observation, device=resolved.device))
        )

        try:
            return fit_steps(
                module,
                source,
                learner=learner,
                device=resolved.device,
                total_steps=spec.total_steps,
                # The spec's evaluation period doubles as the reporting
                # period, rather than a second field meaning almost the same
                # thing. A block boundary is exactly where an interactive run
                # would evaluate, so two numbers could only ever disagree.
                report_every_steps=spec.evaluate_every_steps,
                gradient_norms=gradient_norms,
                # No best-weight selection. Choosing a best policy needs a
                # metric that says whether the agent improved, and the only
                # honest one is an evaluation return, which nothing can
                # produce until a learner has a greedy mode. Selecting on the
                # training loss instead -- the supervised fallback -- would be
                # worse than not selecting: for a policy-gradient objective a
                # falling loss does not mean a better agent.
                checkpoint=None,
                learning_rate=spec.learning_rate,
            )
        finally:
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
    def _optimiser_spec(training: object) -> TorchTrainingSpec | RlTrainingSpec:
        """
        Narrow a training spec to either kind this engine can prepare from.

        :meth:`prepare` places a module on a device and builds an optimiser
        over it, and that is identical work for a predictor and for a policy.
        So it accepts both specs rather than one, and the two paths diverge
        only where they genuinely differ -- in :meth:`fit` and
        :meth:`fit_policy`, each of which narrows to the one spec it can
        actually drive.

        Parameters
        ----------
        training
            The training spec.

        Returns
        -------
        TorchTrainingSpec or RlTrainingSpec
            The same spec, narrowed.

        Raises
        ------
        EngineError
            If it is another engine's spec.
        """
        if not isinstance(training, TorchTrainingSpec | RlTrainingSpec):
            raise EngineError(
                f"the Torch engine was given a {type(training).__name__}; "
                f"prepare() needs a TorchTrainingSpec or an RlTrainingSpec"
            )
        return training

    @staticmethod
    def _rl_training_spec(training: object) -> RlTrainingSpec:
        """
        Narrow an engine-opaque training spec to an interactive one.

        Parameters
        ----------
        training
            The training spec.

        Returns
        -------
        RlTrainingSpec
            The same spec, narrowed.

        Raises
        ------
        EngineError
            If it is a supervised spec or another engine's. Named separately
            from :meth:`_training_spec` so that handing a supervised spec to
            the interactive path says so, rather than reporting a missing
            field much later.
        """
        if not isinstance(training, RlTrainingSpec):
            raise EngineError(
                f"fit_policy needs an RlTrainingSpec; received a "
                f"{type(training).__name__}. A supervised configuration trains "
                f"through fit(), not here"
            )
        return training

    @staticmethod
    def _policy_learner(
        spec: RlTrainingSpec,
        *,
        source: BatchSource,
        optimiser: torch.optim.Optimizer,
        hardware: ResolvedHardware,
        gradient_norms: GradientNorms,
    ) -> PolicyLearner:
        """
        Resolve and construct the named interactive update rule.

        Every policy learner is constructed with the same keyword set, and
        each takes what it needs. That uniformity is the price of resolving
        by name: the engine cannot know which arguments the algorithm it was
        asked for wants, so the contract is that it is offered all of them.
        :class:`~.learners.random.RandomLearner` uses one and ignores the
        rest, which is what a learner that performs no update should do.

        Parameters
        ----------
        spec
            The interactive training spec, naming the learner.
        source
            The experience source, asked for the signature the learner needs
            to interpret an action.
        optimiser
            Built over the policy's materialised parameters.
        hardware
            Resolved device, precision and gradient scaler.
        gradient_norms
            Tracker for the per-block gradient-norm summary.

        Returns
        -------
        PolicyLearner
            The constructed update rule.

        Raises
        ------
        EngineError
            If the name is not registered, or resolves to something that
            cannot act and update.
        """
        try:
            learner_type = get_learner(spec.learner)
        except ComponentError as error:
            raise EngineError(
                f"no learner is registered as {spec.learner!r}; the interactive "
                f"learners available here are {sorted(LEARNERS.names())}"
            ) from error

        learner = learner_type(
            signature=_policy_signature_of(source),
            training=spec,
            optimiser=optimiser,
            hardware=hardware,
            gradient_norms=gradient_norms,
        )
        if not (hasattr(learner, "act") and hasattr(learner, "update")):
            raise EngineError(
                f"the learner registered as {spec.learner!r} is a "
                f"{type(learner).__name__}, which does not have both act() and "
                f"update(); it is a supervised learner and cannot drive a policy"
            )
        _LOGGER.info("resolved interactive learner %r", spec.learner)
        return learner

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


def _policy_signature_of(source: BatchSource) -> PolicySignature:
    """
    Ask an experience source for the policy signature it was built against.

    A source that yields experience knows the environment behind it, so it is
    the one object in the fit call that can answer. The alternative was to
    thread the signature from the pipeline through
    :meth:`TorchEngine.fit_policy`, which would have put the same value in
    two places -- and two copies of a signature is how a policy comes to be
    built against one action space and driven against another.

    Read by attribute rather than through a declared protocol because it is
    one optional member on an otherwise ordinary ``BatchSource``, in the same
    way ``ordered()`` is.

    Parameters
    ----------
    source
        The experience source.

    Returns
    -------
    PolicySignature
        The observation and action spaces.

    Raises
    ------
    EngineError
        If the source cannot supply one, which means it is a dataset source
        and this is a supervised run reaching the wrong method.
    """
    signature = getattr(source, "policy_signature", None)
    if signature is None:
        raise EngineError(
            f"the {type(source).__name__} handed to fit_policy has no "
            f"policy_signature, so nothing can say what spaces the policy acts "
            f"in. An interactive run is driven by a source built over an "
            f"environment, such as a RolloutSource"
        )
    return signature
```

---

## 3. `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/loaders.py`

10187 bytes · SHA-256 `3da70099294e818b`

```python
"""
Turning a ``BatchSource``'s NumPy batches into device-resident tensors.

The one place in the framework where a NumPy array becomes a
``torch.Tensor``. Everything upstream -- splitting, windowing, batch ordering
-- is engine-neutral, so every engine inherits one implementation of the parts
that are easy to get subtly wrong.

Static inputs leave per-sample collation
----------------------------------------
**Defect 4.** The implementation this framework replaces merged every static
tensor into every sample, and its collation function then ran ``torch.equal``
across the batch for each static key and returned ``values[0]``.

So for every batch of every epoch, a graph adjacency matrix was compared
against itself ``batch_size`` times to confirm something true by
construction. On a batch size of 64 and a 400-node graph that is 64
comparisons of a 160,000-element tensor, per batch, per epoch, to learn
nothing.

:class:`StaticInputs` replaces all of it. The static tensors are uploaded once
at the start of a fit and passed to every forward call by reference. This is
not a behavioural change and is worth being precise about why: the old
collation *already* returned exactly one copy, and the network *already*
received exactly one tensor. The same object reaches the same place; only the
comparison is gone.

Workers, and why the default is none
------------------------------------
``LoaderSpec.num_workers`` defaults to zero, meaning load in the main process.
That is the right default here and not merely a conservative one: the
framework fans a job set out across a process pool, and a worker process that
spawns its own loader workers oversubscribes the machine -- N jobs each
claiming W workers on a C-core box is NW processes competing for C cores, and
throughput collapses rather than degrading.

Because a ``BatchSource`` already yields whole batches, this module does not
construct a ``torch.utils.data.DataLoader`` at all. Prefetching across
processes would mean pickling the source, and the win for an in-memory array
is not worth the failure modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...core.contract.data import TARGET_KEY
from ...core.lifecycle.errors import ContractError
from ...core.provenance.logging import get_logger
from .materialise import torch_dtype

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ...core.contract.signature import InputSignature
    from ...core.contract.source import BatchSource

__all__ = ["TARGET_KEY", "StaticInputs", "to_device_batches", "to_tensor"]

_LOGGER = get_logger(__name__)


def to_tensor(
    value: object,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert one batch entry to a tensor on a device.

    Accepts a tensor as well as an array, and returns an existing tensor
    unchanged when it already has the right dtype and device. That matters for
    a static input: re-uploading a 400-node adjacency matrix on every forward
    call would reintroduce defect 4's cost by a different route.

    Parameters
    ----------
    value
        A NumPy array, a Torch tensor, or anything ``torch.as_tensor``
        accepts.
    dtype
        Target dtype. ``None`` keeps whatever the input has.
    device
        Target device. ``None`` keeps the input where it is.

    Returns
    -------
    torch.Tensor
        The converted tensor.
    """
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None and tensor.device != device:
        # `non_blocking` is deliberately not set.  It is only an advantage
        # from pinned host memory, and setting it without pinning silently
        # does nothing on some backends and returns an unready tensor on
        # others.
        tensor = tensor.to(device=device)
    return tensor


class StaticInputs:
    """
    The inputs that are the same for every batch, uploaded once.

    Parameters
    ----------
    tensors
        Static inputs, already on the target device.

    Notes
    -----
    Held in a small class rather than a bare dict so that the single upload is
    a visible event with a place to log it, and so the set can report itself in
    a run summary. An absent static set is an empty instance rather than
    ``None``, which keeps the forward call free of a branch.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor] | None = None) -> None:
        self.tensors: dict[str, torch.Tensor] = dict(tensors or {})

    @classmethod
    def from_source(
        cls,
        source: BatchSource,
        *,
        signature: InputSignature,
        device: torch.device,
    ) -> StaticInputs:
        """
        Upload a source's static inputs to a device, once.

        Parameters
        ----------
        source
            The batch source.
        signature
            The declared interface, used for the dtype of each static input.
        device
            Where to upload.

        Returns
        -------
        StaticInputs
            The uploaded set, empty when the source has none.

        Raises
        ------
        ContractError
            If the source supplies a static input the signature does not
            declare. A tensor nothing declared will be passed to ``forward``
            as a keyword argument and rejected there, with a message naming
            neither side.
        """
        supplied = dict(source.static)
        if not supplied:
            return cls()

        undeclared = sorted(set(supplied) - set(signature.static))
        if undeclared:
            raise ContractError(
                f"the source supplies static input(s) {undeclared} that the "
                f"signature does not declare; it declares "
                f"{sorted(signature.static)}"
            )

        tensors = {
            name: to_tensor(value, dtype=torch_dtype(signature.static[name].dtype), device=device)
            for name, value in supplied.items()
        }
        _LOGGER.info(
            "uploaded %d static input(s) to %s once: %s",
            len(tensors),
            device,
            {name: tuple(tensor.shape) for name, tensor in sorted(tensors.items())},
        )
        return cls(tensors)

    def __bool__(self) -> bool:
        """Return whether any static input is present."""
        return bool(self.tensors)

    def __len__(self) -> int:
        """Return how many static inputs are present."""
        return len(self.tensors)

    def merge_into(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Add the static inputs to a batch, by reference.

        Parameters
        ----------
        batch
            A batch of dynamic inputs. Mutated and returned, because this runs
            once per batch per epoch and allocating a fresh dict each time is
            measurable at small batch sizes.

        Returns
        -------
        dict
            The same dict, with the static inputs added.
        """
        batch.update(self.tensors)
        return batch

    def describe(self) -> dict[str, list[int]]:
        """
        Return the shape of each static input, for reports and logs.

        Returns
        -------
        dict
            Input name to shape.
        """
        return {name: list(tensor.shape) for name, tensor in sorted(self.tensors.items())}


def to_device_batches(
    source: BatchSource,
    *,
    signature: InputSignature,
    device: torch.device,
    static: StaticInputs | None = None,
    validate_first: bool = True,
) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """
    Convert one pass of a source into device-resident tensor batches.

    Yields the inputs and the target separately rather than as one mapping,
    because the learner needs them apart -- the inputs go to ``forward`` as
    keyword arguments and the target goes to the loss -- and splitting them
    here means the learner never has to know which key the target uses.

    Parameters
    ----------
    source
        The batch source.
    signature
        The declared interface, supplying each input's dtype.
    device
        Where to place the tensors.
    static
        Already-uploaded static inputs. ``None`` means none; pass the result
        of :meth:`StaticInputs.from_source` to avoid re-uploading per epoch.
    validate_first
        Whether to key-check the first batch against the signature. Only the
        first: the check is cheap but not free, and a source that produced a
        correctly keyed first batch and a differently keyed tenth one is a
        failure mode worth catching in the conformance suite rather than on
        every batch of every epoch.

    Yields
    ------
    tuple
        The inputs as a mapping, and the target.

    Raises
    ------
    ContractError
        If a batch carries no target, or -- on the first batch -- disagrees
        with the signature's declared keys.
    """
    statics = static if static is not None else StaticInputs()
    dtypes = {name: torch_dtype(spec.dtype) for name, spec in signature.dynamic.items()}
    target_dtype = torch_dtype(signature.target.dtype)

    for index, batch in enumerate(source.batches()):
        if index == 0 and validate_first:
            signature.validate_batch_keys(batch, where="to_device_batches", target_key=TARGET_KEY)
        if TARGET_KEY not in batch:
            raise ContractError(
                f"batch {index} carries no {TARGET_KEY!r}; a training batch must "
                f"carry the target. An inference stream without one belongs in "
                f"predict(), not here"
            )

        inputs = {
            name: to_tensor(value, dtype=dtypes.get(name), device=device)
            for name, value in batch.items()
            if name != TARGET_KEY
        }
        target = to_tensor(batch[TARGET_KEY], dtype=target_dtype, device=device)
        yield statics.merge_into(inputs), target
```

---

## 4. `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/materialise.py`

16380 bytes · SHA-256 `6947dc425a1e454a`

```python
"""
Giving a lazily shaped model its parameters, before anything else touches it.

**Defect 6.** A module built from ``LazyLinear`` and friends has no parameters
until it has seen one input: the shapes are inferred from the first forward
pass. The implementation this framework replaces wrapped such a model for
distributed training before that point, which hands the wrapper an empty
parameter group to synchronise. The outcomes are a crash deep inside the
distributed library, or -- considerably worse -- a wrapper that synchronises
nothing, so every rank trains its own private copy and the run reports a
plausible loss curve for a model that was never actually distributed.

The same ordering trap catches three other things, which is why this runs
first and not just before the distributed wrapper:

* an **optimiser** constructed over an empty parameter list tracks nothing,
  and ``step()`` is then a no-op that raises no error;
* a **checkpoint** of an unmaterialised module has no tensors in it, and
  loading it back succeeds while restoring nothing;
* ``torch.compile`` traces a graph whose shapes are not yet known.

So the order is fixed: materialise, then hardware, then distribute, then
optimiser. ``ARCHITECTURE.md`` §5 makes ``materialise`` its own pipeline stage
rather than a detail inside the engine, precisely so that this ordering is
visible where runs are defined and cannot be rearranged by accident.

Why a signature is enough
-------------------------
The dummy forward needs exact shapes and dtypes with no data present, and
that is the whole reason :class:`~rade_qnet.core.contract.signature.InputSignature`
exists. Synthesising the batch from the signature rather than borrowing a real
one keeps materialisation independent of the data build, which is what makes a
six-month-old bundle reconstructible: saved signature plus saved weights, no
dataset required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...core.contract.signature import TensorSpec
from ...core.lifecycle.errors import EngineError
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.signature import InputSignature, PolicySignature

__all__ = [
    "count_parameters",
    "dummy_batch",
    "has_lazy_parameters",
    "materialise",
    "materialise_policy",
    "torch_dtype",
]

_LOGGER = get_logger(__name__)

#: Batch size for the dummy forward. Two rather than one, so that a module
#: which collapses a singleton batch dimension -- or a batch-norm layer, which
#: refuses a batch of one in training mode -- is exercised honestly.
_DUMMY_BATCH_SIZE = 2

#: Spec dtype names mapped to Torch dtypes. A fixed table rather than
#: ``getattr(torch, name)``, so an unrecognised name produces a message naming
#: the supported set instead of an ``AttributeError``.
_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

#: Dtypes for which a plain ``ones`` fill is the right dummy value. Integer
#: inputs are usually indices, and zero is the one index guaranteed to be
#: inside any embedding table -- ones would be out of range for a table of
#: size one, which is exactly what a minimal test fixture has.
_INTEGER_DTYPES = frozenset({torch.int8, torch.int16, torch.int32, torch.int64, torch.bool})


def torch_dtype(name: str) -> torch.dtype:
    """
    Return the Torch dtype for a signature's dtype name.

    Parameters
    ----------
    name
        A library-agnostic dtype name, as carried by
        :class:`~rade_qnet.core.contract.signature.TensorSpec`.

    Returns
    -------
    torch.dtype
        The corresponding Torch dtype.

    Raises
    ------
    EngineError
        If the name is not recognised, listing the supported names.
    """
    try:
        return _DTYPES[name]
    except KeyError:
        raise EngineError(
            f"dtype {name!r} is not supported by the Torch engine; supported "
            f"dtypes are {sorted(_DTYPES)}"
        ) from None


def _dummy_tensor(spec: TensorSpec, *, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Build one synthetic tensor from a spec.

    Parameters
    ----------
    spec
        The declared shape and dtype.
    batch_size
        Size to substitute for each wildcard dimension.
    device
        Where to allocate.

    Returns
    -------
    torch.Tensor
        A tensor matching the spec.
    """
    shape = spec.concrete_shape(batch_size)
    dtype = torch_dtype(spec.dtype)
    # Zeros for integers and booleans, which are indices or masks: see the
    # note on `_INTEGER_DTYPES`.  Ones for floats, because a zero-filled float
    # input makes a multiplicative layer's output independent of its weights,
    # so a shape error in the weight initialisation would not show up.
    if dtype in _INTEGER_DTYPES:
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.ones(shape, dtype=dtype, device=device)


def dummy_batch(
    signature: InputSignature,
    *,
    batch_size: int = _DUMMY_BATCH_SIZE,
    device: torch.device | None = None,
    include_static: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Synthesise a batch from a signature, with no data present.

    Parameters
    ----------
    signature
        The declared interface.
    batch_size
        Size for each wildcard dimension.
    device
        Where to allocate. ``None`` means the CPU.
    include_static
        Whether to include the static inputs. False when the caller already
        holds the real static tensors and wants only the dynamic part
        synthesised -- which is the better path when they are available,
        because a graph adjacency matrix of ones is a complete graph and some
        models will not run on one.

    Returns
    -------
    dict
        Input name to synthetic tensor.
    """
    target_device = device or torch.device("cpu")
    batch = {
        name: _dummy_tensor(spec, batch_size=batch_size, device=target_device)
        for name, spec in signature.dynamic.items()
    }
    if include_static:
        batch.update(
            {
                name: _dummy_tensor(spec, batch_size=batch_size, device=target_device)
                for name, spec in signature.static.items()
            }
        )
    return batch


def has_lazy_parameters(model: torch.nn.Module) -> bool:
    """
    Return whether any parameter is still waiting for its shape.

    Parameters
    ----------
    model
        The model to inspect.

    Returns
    -------
    bool
        True if the model contains an uninitialised parameter or buffer.
    """
    uninitialised = (
        torch.nn.parameter.UninitializedParameter,
        torch.nn.parameter.UninitializedBuffer,
    )
    return any(
        isinstance(tensor, uninitialised)
        for tensor in (*model.parameters(recurse=True), *model.buffers(recurse=True))
    )


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    """
    Return the number of parameter elements.

    Zero for an unmaterialised model, which is the signal that makes defect 6
    detectable: an optimiser built over zero parameters is a silent no-op, so
    the count is asserted after materialisation rather than trusted.

    Parameters
    ----------
    model
        The model to inspect.
    trainable_only
        Count only parameters that require a gradient.

    Returns
    -------
    int
        Number of elements.
    """
    if has_lazy_parameters(model):
        return 0
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def materialise(
    model: torch.nn.Module,
    signature: InputSignature,
    *,
    static: Mapping[str, torch.Tensor] | None = None,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Run one dummy forward pass so lazy parameters acquire their shapes.

    Safe to call on an already-materialised model, in which case it returns
    immediately. That makes it callable unconditionally from the pipeline,
    which is what keeps the ordering guarantee from depending on a model
    author remembering to declare that their model is lazy.

    Parameters
    ----------
    model
        The constructed model, possibly unmaterialised.
    signature
        The declared interface, from which the dummy batch is built.
    static
        Real static tensors, when the caller has them. Preferred over
        synthetic ones: a synthetic adjacency matrix of ones is a complete
        graph, and a model that normalises by node degree may not accept one.
    device
        Where to run the pass.

    Returns
    -------
    torch.nn.Module
        The same model, materialised.

    Raises
    ------
    EngineError
        If the dummy pass fails, or if it completes without materialising the
        parameters. The second case is the quiet one: it means the model's
        forward does not route through its lazy submodules, so the run would
        proceed with an optimiser tracking nothing.
    """
    if not has_lazy_parameters(model):
        _LOGGER.debug("model has no lazy parameters; nothing to materialise")
        return model

    batch = dummy_batch(signature, device=device, include_static=not static)
    if static:
        batch.update(static)

    _LOGGER.info(
        "materialising lazy parameters with a dummy batch: %s",
        {name: tuple(tensor.shape) for name, tensor in sorted(batch.items())},
    )

    was_training = model.training
    model.eval()
    try:
        # `no_grad` because this pass exists only to fix shapes.  Without it
        # the dummy batch would build a graph and the first real backward pass
        # could be computed against synthetic activations.
        with torch.no_grad():
            model(**batch)
    except Exception as error:
        raise EngineError(
            f"the dummy forward pass used to materialise lazy parameters failed: "
            f"{type(error).__name__}: {error}. The batch was built from the input "
            f"signature:\n{signature.describe()}\nEither the signature disagrees "
            f"with what forward() accepts, or forward() needs real values rather "
            f"than synthetic ones -- in which case pass the real static inputs"
        ) from error
    finally:
        # Restored rather than left in eval mode: a model that arrived in
        # training mode must leave in training mode, or dropout and batch-norm
        # would behave differently for reasons invisible at the call site.
        model.train(was_training)

    if has_lazy_parameters(model):
        raise EngineError(
            "the dummy forward pass completed but the model still has "
            "uninitialised parameters; its forward() does not route through "
            "every lazy submodule. An optimiser built now would track nothing "
            "and training would appear to run while changing no weights"
        )

    _LOGGER.info("materialised %d parameter element(s)", count_parameters(model))
    return model


def materialise_policy(
    policy: torch.nn.Module,
    signature: PolicySignature,
    *,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Run one dummy forward pass so a lazy policy acquires its shapes.

    The interactive counterpart of :func:`materialise`, and it exists for the
    same defect: an optimiser built over unmaterialised parameters tracks
    nothing, so the run appears to train while changing no weights.

    Separate from :func:`materialise` rather than folded into it, because the
    dummy input is built from a different description. A supervised dummy
    batch is a mapping of named inputs plus a target, synthesised from an
    :class:`~rade_qnet.core.contract.signature.InputSignature`. A policy
    takes one thing -- an observation -- and its signature has no target to
    synthesise. A single function would have had to branch on which
    signature it received, which is the branch the two call sites remove.

    Parameters
    ----------
    policy
        The constructed policy, possibly unmaterialised.
    signature
        The observation and action spaces. Only the observation space is
        read: the action space constrains what the policy's *output* means,
        which a forward pass does not need to know.
    device
        Where to run the pass.

    Returns
    -------
    torch.nn.Module
        The same policy, materialised.

    Raises
    ------
    EngineError
        If the dummy pass fails, or if it completes without materialising the
        parameters.
    """
    if not has_lazy_parameters(policy):
        _LOGGER.debug("policy has no lazy parameters; nothing to materialise")
        return policy

    observation = _dummy_observation(signature, device=device)
    _LOGGER.info(
        "materialising lazy policy parameters with an observation of %s",
        tuple(observation.shape),
    )

    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            # Passed by keyword, matching the convention every model in this
            # framework is called with: a policy declares what it consumes in
            # its own signature rather than depending on positional order.
            policy(observation=observation)
    except Exception as error:
        raise EngineError(
            f"the dummy forward pass used to materialise a lazy policy failed: "
            f"{type(error).__name__}: {error}. The observation was built from the "
            f"policy signature's observation space "
            f"({signature.observation.kind}, shape {signature.observation.shape}). "
            f"Either the space disagrees with what forward() accepts, or forward() "
            f"does not take an 'observation' keyword"
        ) from error
    finally:
        policy.train(was_training)

    if has_lazy_parameters(policy):
        raise EngineError(
            "the dummy forward pass completed but the policy still has "
            "uninitialised parameters; its forward() does not route through "
            "every lazy submodule. An optimiser built now would track nothing "
            "and the run would appear to train while changing no weights"
        )

    _LOGGER.info("materialised %d parameter element(s)", count_parameters(policy))
    return policy


def _dummy_observation(
    signature: PolicySignature, *, device: torch.device | None = None
) -> torch.Tensor:
    """
    Synthesise one batch of observations from a policy signature.

    A batch of one, because the shapes a lazy module needs are fixed by the
    trailing dimensions and a larger batch would only cost more.

    Parameters
    ----------
    signature
        The policy signature, whose observation space gives shape and dtype.
    device
        Where to place the tensor.

    Returns
    -------
    torch.Tensor
        A synthetic observation with a leading batch dimension.

    Raises
    ------
    EngineError
        If the observation space names a kind this function cannot
        synthesise.
    """
    space = signature.observation
    if space.kind == "box":
        spec = TensorSpec(shape=(None, *space.shape), dtype=space.dtype)
    elif space.kind == "discrete":
        # One index per sample, not a one-hot row: a discrete observation
        # reaches a network as something to look up, and widening it here
        # would make every policy undo the widening.
        spec = TensorSpec(shape=(None,), dtype="int64")
    else:
        raise EngineError(
            f"observation space kind {space.kind!r} cannot be synthesised; "
            f"expected 'box' or 'discrete'"
        )

    # Built through the same helper the supervised dummy batch uses, so the
    # choice between zeros and ones is made in one place -- and it matters:
    # a zero-filled float input makes a multiplicative layer's output
    # independent of its weights, hiding a shape error in the initialisation.
    return _dummy_tensor(spec, batch_size=1, device=device or torch.device("cpu"))
```

---

## 5. `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/predictor.py`

6469 bytes · SHA-256 `d3bbea51d11a2eef`

```python
"""
Batched inference, including the precompute path.

Training and inference ask a model for the same thing -- a forward pass --
under opposite constraints. Training updates parameters, so nothing computed
from them survives a step. Inference does not, so anything that does not vary
across batches can be computed once.

For the flagship that distinction is most of the runtime. Its graph encoder
turns a static adjacency into node embeddings, and during training that work
is unavoidable because the encoder's weights are moving. During a prediction
pass the weights are frozen and the graph does not change, so recomputing the
embeddings for every batch repeats identical arithmetic -- which on a long
evaluation pass can dominate everything else.

:class:`~rade_qnet.core.authoring.capabilities.Precomputable` is how a model says
so. This module is where the framework acts on the declaration, and the
critical property is that it must not change any number. A performance path
that quietly perturbs results is worse than no performance path at all,
because the discrepancy shows up as a model that scores differently in
evaluation than it did in training, with nothing pointing at the cause.

So the two routes are kept as close as possible: the same device placement,
the same static inputs, the same batch order, the same dtype on the way out.
The only difference is which method the module is called through, and
``test_precompute_matches_plain_path`` asserts the outputs are identical
rather than merely close.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...core.authoring.capabilities import Precomputable
from ...core.lifecycle.errors import EngineError
from ...core.provenance.logging import get_logger
from .loaders import StaticInputs, to_device_batches

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from ...core.contract.source import BatchSource
    from .hardware.devices import ResolvedHardware

__all__ = ["predict_batches"]

_LOGGER = get_logger(__name__)


def predict_batches(
    module: torch.nn.Module,
    source: BatchSource,
    *,
    resolved: ResolvedHardware,
    allow_precompute: bool = True,
) -> NDArray[np.floating]:
    """
    Run a forward pass over a bounded source and return the raw output.

    Returns the model's own output space. Inverting the target transform is
    the pipeline's job, using the fitted state -- an engine that inverted it
    here would invert it twice, once for the pipeline and once for itself.

    Parameters
    ----------
    module
        The model, already on the right device. Put into evaluation mode
        here rather than by the caller, because a forward pass left in
        training mode would apply dropout and update batch-norm statistics,
        and the resulting predictions would be both wrong and irreproducible.
    source
        Batches to predict over. Must be bounded: an unbounded source has no
        last batch, so there is nothing to return.
    resolved
        The hardware placement, supplying the device and the autocast
        context.
    allow_precompute
        Whether to use the precompute path when the model declares it.
        Defaults to true. The escape hatch exists so a test can run both
        routes over one model and compare them, which is the only way to
        know the optimisation is safe.

    Returns
    -------
    numpy.ndarray
        Predictions in source order, one row per sample.

    Raises
    ------
    EngineError
        If the source is unbounded, or yields no batches at all.
    """
    if source.steps_per_epoch is None:
        raise EngineError(
            "predict needs a bounded source; this one reports "
            "steps_per_epoch=None, so it has no last batch and there is "
            "nothing to return"
        )

    signature = source.signature
    static = StaticInputs.from_source(source, signature=signature, device=resolved.device)

    module.eval()
    blocks: list[NDArray[np.floating]] = []

    with torch.no_grad(), resolved.autocast():
        precomputed = _precompute(module, static, enabled=allow_precompute)

        for inputs, _ in to_device_batches(
            source, signature=signature, device=resolved.device, static=static
        ):
            if precomputed is None:
                output = module(**inputs)
            else:
                output = module.forward_with_precomputed(inputs, precomputed)
            # Cast to float32 before leaving Torch: NumPy has no bfloat16, so
            # a reduced-precision tensor would fail to convert, and a float16
            # one would silently lose decimal digits of a P&L figure.
            blocks.append(output.detach().to(dtype=torch.float32).cpu().numpy())

    if not blocks:
        raise EngineError(
            f"the source yielded no batches, so there is nothing to predict "
            f"(it reports steps_per_epoch={source.steps_per_epoch})"
        )
    return np.concatenate(blocks, axis=0)


def _precompute(
    module: torch.nn.Module,
    static: StaticInputs,
    *,
    enabled: bool,
) -> Mapping[str, torch.Tensor] | None:
    """
    Compute the reusable static encoding, when the model offers one.

    Parameters
    ----------
    module
        The model, already in evaluation mode and inside ``no_grad``. Both
        matter: the encoding is computed once and reused, so it must be
        produced under exactly the conditions the batches will be.
    static
        The static inputs, already on the device.
    enabled
        Whether to try at all.

    Returns
    -------
    Mapping or None
        The precomputed tensors, or ``None`` to take the ordinary forward
        path -- which is also what a model with no static inputs gets, since
        there is nothing for it to encode.
    """
    if not enabled or not isinstance(module, Precomputable):
        return None

    tensors = static.tensors
    if not tensors:
        _LOGGER.debug(
            "%s declares precompute() but the source has no static inputs; "
            "using the ordinary forward path",
            type(module).__name__,
        )
        return None

    precomputed = module.precompute(tensors)
    _LOGGER.debug(
        "precomputed %d static tensor(s) for %s, reused across the pass",
        len(precomputed),
        type(module).__name__,
    )
    return precomputed
```

