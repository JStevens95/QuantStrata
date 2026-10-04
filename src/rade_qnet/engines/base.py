"""
The ``Engine`` protocol -- the only interface a pipeline uses to train.

An engine is the single place in the framework permitted to know about a
specific training library. Everything above it talks to this protocol, which
is what makes "add XGBoost" a new sub-package and one registration line rather
than a change to any pipeline.

Why this is a protocol and not a base class
-------------------------------------------
An engine satisfies this by having the right methods. It need not inherit from
anything of ours, which matters for the same reason it matters for
:class:`~rade_qnet.core.contract.source.BatchSource`: an engine for a library
this repository has never heard of should be writable outside it. The
``runtime_checkable`` decorator lets a pipeline confirm an object is engine
shaped; :func:`rade_qnet.testkit.conformance.check_engine` is what confirms it
*behaves* like one, because an ``isinstance`` check against a protocol
verifies method presence and nothing more.

The contract is deliberately not epoch shaped
---------------------------------------------
:meth:`Engine.fit` takes a mapping of sources and returns a
:class:`~rade_qnet.core.contract.result.FitOutcome`. It does **not** take an
epoch count, expose a step hook, or require a loop. This is the honesty test
named in ``ARCHITECTURE.md`` §7: a gradient-boosted tree fits in a single call
with no loop at all, and if this protocol could only describe a model that
trains over epochs then it would be a PyTorch interface with a generic name.

The tree engine reports one :class:`~rade_qnet.core.contract.result.EpochRecord`
per boosting round, so "learning curve" means the same thing for both and one
report renders either.

Why the handle is separate from the model
-----------------------------------------
:meth:`Engine.prepare` returns a :class:`ModelHandle` rather than a bare model
because preparing a model for training produces things that are *not* the
model: a resolved device, an optimiser, a gradient scaler, a distributed
wrapper. Returning them bundled means :meth:`Engine.fit` has one argument
instead of six, and -- more importantly -- means the handle can keep a
reference to the *unwrapped* model, which is the one that must be checkpointed.
Saving a distributed wrapper's ``state_dict`` produces keys prefixed with
``module.`` that will not load into the bare model, and that mismatch is
discovered months later by whoever tries to load the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.contract.data import TensorLike

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from ..core.contract.result import FitOutcome
    from ..core.contract.signature import InputSignature, PolicySignature
    from ..core.contract.source import BatchSource
    from ..core.spec.hardware import HardwareSpec

__all__ = ["Engine", "EngineCapabilities", "InteractiveEngine", "ModelHandle"]


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """
    What an engine can and cannot do, declared rather than discovered.

    Declared rather than discovered, so that a limitation is stated in one
    place: asking an engine to checkpoint and catching the failure would mean
    finding out after the data build has already run.

    What actually reads these flags
    -------------------------------
    The train pipeline logs :meth:`describe`, reports display it, and
    ``testkit.conformance`` checks that an engine returns one. **No stage
    branches on an individual field.** An earlier version of this docstring
    said a pipeline "needs these answers before it starts, because several of
    them change which stages it runs", and that was never true -- recorded
    here rather than quietly corrected, because the next person to write a
    stage will otherwise trust a flag that nothing enforces.

    The enforcement it looks like it provides comes from the type system
    instead, and more reliably. Each engine has its own training-spec type
    exposing only the settings that engine supports, so
    :class:`~rade_qnet.core.spec.training.SklearnTrainingSpec` has no
    early-stopping field to misuse and an unsupported combination is
    unconstructible rather than rejected at run time. That is the stronger of
    the two mechanisms, which is why these flags have stayed advisory.

    They are still worth declaring. A stage that genuinely does need to
    branch -- the distributed path is the likeliest candidate -- should read
    it from here rather than re-derive it from the engine's name, and a
    reader comparing three engines gets one place to look.

    Parameters
    ----------
    name
        Registered engine name, for messages.
    supports_epochs
        Whether training proceeds in passes the framework can observe between.
        False for a one-shot fit, which is what makes framework-level early
        stopping inapplicable -- the engine's own, which runs per boosting
        round inside the library, is used instead.
    supports_validation_during_fit
        Whether a validation source can be scored as training proceeds. False
        for an engine that scores internally, where nothing the framework
        provides could observe or act on it mid-fit.
    supports_checkpointing
        Whether the best parameters can be captured and restored.
    supports_distributed
        Whether the engine can shard one job across several devices.
    supports_lazy_materialisation
        Whether :meth:`Engine.materialise` does anything. False for engines
        whose models have concrete parameters from construction, where the
        stage is a no-op rather than an error.
    accelerators
        Device kinds the engine can use, as named by
        :class:`~rade_qnet.core.spec.hardware.HardwareSpec`. Always includes
        ``cpu``: an engine that cannot run on a CPU cannot be tested on one.
    """

    name: str
    supports_epochs: bool = True
    supports_validation_during_fit: bool = True
    supports_checkpointing: bool = True
    supports_distributed: bool = False
    supports_lazy_materialisation: bool = False
    accelerators: tuple[str, ...] = ("cpu",)

    def describe(self) -> str:
        """
        Return a compact one-line summary, for logs and reports.

        Returns
        -------
        str
            For example ``torch(epochs, validation, checkpointing,
            accelerators=cpu/cuda)``.
        """
        flags = [
            label
            for label, enabled in (
                ("epochs", self.supports_epochs),
                ("validation", self.supports_validation_during_fit),
                ("checkpointing", self.supports_checkpointing),
                ("distributed", self.supports_distributed),
                ("lazy", self.supports_lazy_materialisation),
            )
            if enabled
        ]
        return (
            f"{self.name}("
            f"{', '.join(flags) or 'no optional features'}, "
            f"accelerators={'/'.join(self.accelerators)})"
        )


@dataclass(frozen=True, slots=True)
class ModelHandle:
    """
    A model that has been prepared for training, with its apparatus.

    Produced by :meth:`Engine.prepare` and consumed by every method after it.

    Parameters
    ----------
    model
        What the training loop should call. May be a wrapper -- a distributed
        wrapper, a compiled graph -- rather than the object the model
        definition returned.
    unwrapped
        The model to checkpoint and to save. Distinct from :attr:`model` on
        purpose: a distributed wrapper's ``state_dict`` keys carry a
        ``module.`` prefix that will not load into a bare model, and a
        compiled graph's carry ``_orig_mod.``. Keeping both references means
        the saved weights are always the portable ones.
    device
        Resolved device, as a string the engine's library understands. This is
        what was *actually* used, which may differ from what the spec
        requested -- an absent accelerator degrades to the CPU with a warning
        rather than failing a run that would otherwise have succeeded.
    precision
        Resolved compute precision.
    is_distributed
        Whether the model is wrapped for multi-device training.
    static
        Static inputs, already moved to :attr:`device`. Uploaded once at
        prepare time rather than collated into every batch -- the whole point
        of the static/dynamic split in
        :class:`~rade_qnet.core.contract.signature.InputSignature`.
    extras
        Engine-owned objects the loop needs and the framework must not
        inspect: an optimiser, a scheduler, a gradient scaler. Typed
        ``object`` rather than ``Any`` so a reader outside the engine cannot
        reach into them without a deliberate cast.
    """

    model: object
    unwrapped: object
    device: str = "cpu"
    precision: str = "fp32"
    is_distributed: bool = False
    static: Mapping[str, TensorLike] = field(default_factory=dict)
    extras: Mapping[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        """
        Return a compact one-line summary, for logs.

        Returns
        -------
        str
            For example ``LinearRegressor on cpu (fp32)``.
        """
        suffix = ", distributed" if self.is_distributed else ""
        return f"{type(self.unwrapped).__name__} on {self.device} ({self.precision}{suffix})"


@runtime_checkable
class Engine(Protocol):
    """
    Everything a pipeline needs from a training library.

    Six questions, in the order the train pipeline asks them. An engine is
    constructed with no arguments -- its configuration arrives per call, from
    the spec -- so a pipeline can resolve one from the registry and
    instantiate it without knowing which it got.

    Implementations must honour two rules the conformance suite checks:

    **Nothing here writes to the run directory.** Only ``reports`` and
    ``storage`` do. :meth:`save_weights` writes to the path it is handed,
    which is inside a bundle's staging directory chosen by ``storage``.

    **Predictions come back in the model's own output space.** Inverting the
    target transform is the pipeline's job, using the fitted state from the
    data build, because the engine does not know what was done to the target.
    An engine that helpfully inverted it would invert it twice.
    """

    def capabilities(self) -> EngineCapabilities:
        """
        Declare what this engine supports.

        Called before any work is done, so a limitation is reported in front
        of the data build rather than behind it.

        Returns
        -------
        EngineCapabilities
            The engine's declared abilities.
        """
        ...

    def materialise(self, model: object, signature: InputSignature) -> object:
        """
        Give a lazily shaped model its parameters.

        A model whose shapes depend on its input has no parameters until it
        has seen one batch. This method synthesises that batch from the
        signature -- which is why the signature carries exact shapes and
        dtypes -- and runs one forward pass under no-grad.

        It runs as its own pipeline stage, *before* ``prepare``, and the order
        is load bearing. Handing an unmaterialised model to an optimiser
        registers an empty parameter group, and handing one to a distributed
        wrapper gives it nothing to synchronise: either a crash deep inside
        the library or, worse, silent non-synchronisation.

        Must be a no-op returning the model unchanged for engines whose models
        are concrete from construction.

        Parameters
        ----------
        model
            The untrained model from the definition's ``build_model``.
        signature
            The declared interface, carrying the shapes to synthesise.

        Returns
        -------
        object
            The model, with parameters. Returning a *new* object is permitted,
            so the caller must use the return value rather than assuming the
            argument was mutated.

        Raises
        ------
        EngineError
            If the dummy forward pass fails, which means the model and the
            signature disagree -- the earliest point at which that is
            detectable, and far cheaper to diagnose here than in the first
            training step.
        """
        ...

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

        Device selection, precision, graph compilation, distribution and
        optimiser construction all happen here, in that order. Static inputs
        are moved to the device once and carried on the handle.

        Parameters
        ----------
        model
            The materialised model.
        hardware
            Device, precision, compilation and distribution settings.
        training
            The engine's own training spec -- a ``TorchTrainingSpec`` for the
            Torch engine. Typed ``object`` because the protocol cannot name
            one engine's spec without excluding the others; each
            implementation narrows it and raises ``EngineError`` on a
            mismatch.
        static
            Static inputs from the data build, to upload once.

        Returns
        -------
        ModelHandle
            The prepared model and its apparatus.

        Raises
        ------
        EngineError
            If ``training`` is not this engine's spec type, or if a requested
            setting cannot be honoured at all. A merely *absent* device is not
            an error: it warns and degrades.
        """
        ...

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
            One source per split, keyed by split name. ``train`` is required.
            A ``validation`` source is used for monitoring when the engine
            declares ``supports_validation_during_fit``; its absence is not an
            error, but it does disable any monitored callback, and the engine
            says so rather than reporting a best epoch that means nothing.
        training
            The engine's own training spec.
        on_epoch_end
            Optional callable invoked with each completed
            :class:`~rade_qnet.core.contract.result.EpochRecord`. The seam the
            pipeline uses to forward epoch metrics to hooks and trackers while
            the run is still going, so a long fit is observable rather than
            silent. Typed ``object`` because ``core`` declares no callback
            type; implementations narrow it.

        Returns
        -------
        FitOutcome
            History, best epoch, and whether the best parameters were
            restored. ``restored_best`` is reported rather than assumed,
            because when it is false the saved weights and the reported
            metrics describe different models.

        Raises
        ------
        EngineError
            If a required source is missing or a spec setting is unusable.
        """
        ...

    def predict(self, handle: ModelHandle, source: BatchSource) -> NDArray[np.floating]:
        """
        Run a forward pass over a source and return the raw output.

        Returns the model's own output space. Inverting the target transform
        is the caller's job -- see the class docstring.

        Batches are consumed in source order, so the returned rows align with
        the source's iteration order. That alignment is what lets a prediction
        be attributed back to a scenario and an entity, and the conformance
        suite checks the row count against the source's ``n_samples``.

        Parameters
        ----------
        handle
            The prepared model.
        source
            Batches to predict over. Must be bounded: an unbounded source has
            no last batch, so there is nothing to return.

        Returns
        -------
        numpy.ndarray
            Predictions, one row per sample, in source order.

        Raises
        ------
        EngineError
            If the source is unbounded.
        """
        ...

    def save_weights(self, handle: ModelHandle, path: Path) -> None:
        """
        Write the model's parameters to a path.

        The callback seam :func:`rade_qnet.storage.bundle.write_bundle` expects.
        ``storage`` deliberately does not serialise weights itself, because
        only the engine knows how -- and because making ``storage`` do it
        would mean the system of record importing a training library.

        Implementations must write a *parameter* payload, not a pickled
        model object. The implementation this framework replaces pickled whole
        modules and loaded them with ``weights_only=False``, which made every
        saved model refactor-fragile and made loading one equivalent to
        executing whatever was in it.

        Parameters
        ----------
        handle
            The trained model. Note that ``handle.unwrapped`` is what should
            be saved -- see :class:`ModelHandle`.
        path
            Destination file, inside a staging directory. Its parent exists.
        """
        ...

    def load_weights(self, model: object, path: Path) -> object:
        """
        Load parameters written by :meth:`save_weights` into a fresh model.

        Takes a model rather than constructing one, because constructing it is
        the definition's job: the spec and signature saved in the bundle are
        passed back through ``build_model``, and only the parameters come from
        here. That division is what makes a bundle loadable without the data
        build that produced it.

        Parameters
        ----------
        model
            A freshly built model of the same architecture, materialised if
            its parameters are lazy.
        path
            File written by :meth:`save_weights`.

        Returns
        -------
        object
            The model with parameters loaded.

        Raises
        ------
        EngineError
            If the payload does not match the model -- a missing or unexpected
            parameter name. Reported rather than tolerated: a partially loaded
            model predicts confidently from random weights.
        """
        ...


@runtime_checkable
class InteractiveEngine(Protocol):
    """
    An opt-in capability: an engine that can also train a policy.

    Two methods, checked with ``isinstance`` and absent from :class:`Engine`
    on purpose. Adding them to the main protocol would make every engine
    declare them -- and the gradient-boosted-tree engine cannot train a
    policy by any amount of plumbing, so its implementation could only be a
    method that raises. A capability an engine opts into lets the pipeline
    say "this engine does not do interactive training" before anything is
    built, which is a better answer than a ``NotImplementedError`` from four
    stages in.

    This is the same pattern as ``core.capability.protocols``, used for the
    same reason, and it is why the interactive pipeline resolves an engine
    through the ordinary registry and then asks one question of it.

    Why these two and not more
    --------------------------
    Everything else an interactive run needs, :class:`Engine` already
    provides and already means the same thing. ``prepare`` places a module on
    a device and builds an optimiser, which is identical work for a policy;
    ``save_weights`` and ``load_weights`` do not care what the weights are
    for. Only two operations genuinely differ, and both differ because a
    policy signature is not an input signature:

    - materialisation has to synthesise an observation rather than a batch of
      named inputs, and
    - fitting drives a step budget against one unbounded source rather than
      epochs over a mapping of splits.

    A third method for evaluation is deliberately absent. Evaluating a policy
    means running episodes with exploration turned off, and nothing in the
    interactive path yet knows how to turn it off -- ``PolicyLearner``
    declares ``act``, not a greedy variant of it. Declaring the method now
    would repeat :class:`EngineCapabilities`, which spent four phases
    asserting a contract nothing honoured. It arrives with the first learner
    that has a meaningful greedy mode.
    """

    def materialise_policy(self, policy: object, signature: PolicySignature) -> object:
        """
        Give a lazily shaped policy its parameters.

        The counterpart of :meth:`Engine.materialise`, separate because the
        dummy forward pass it performs has to be built from an observation
        space rather than from named inputs and a target.

        Parameters
        ----------
        policy
            The untrained policy.
        signature
            The observation and action spaces.

        Returns
        -------
        object
            The policy, with parameters.
        """
        ...

    def fit_policy(
        self,
        handle: ModelHandle,
        source: BatchSource,
        training: object,
    ) -> FitOutcome:
        """
        Train a policy against a budget of interaction.

        Parameters
        ----------
        handle
            The prepared policy, from :meth:`Engine.prepare`.
        source
            One unbounded source of experience. Singular, where
            :meth:`Engine.fit` takes a mapping: an environment has no
            held-out split to hold a second source, and accepting a mapping
            would invite a caller to pass a ``validation`` entry that
            nothing could honestly use.
        training
            An interactive training spec. Typed as ``object`` for the same
            reason :meth:`Engine.fit` is -- ``engines`` must not pin the
            pipeline to one engine's spec type.

        Returns
        -------
        FitOutcome
            The block history and the step budget's outcome, in the same
            shape a supervised run produces, so two runs can be compared.
        """
        ...
