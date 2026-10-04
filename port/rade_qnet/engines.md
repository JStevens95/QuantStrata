# `src/rade_qnet/engines`

2 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 44 | 1782 | `ab977f1168bbb0b0` |
| 2 | `base.py` | 456 | 17734 | `51a0d4d8b27f4e4d` |

---

## 1. `src/rade_qnet/engines/__init__.py`

1782 bytes · SHA-256 `ab977f1168bbb0b0`

```python
"""
How optimisation actually happens -- one adapter per library.

An engine is the only place in the framework that is allowed to know about a
specific training library.  It answers a fixed set of questions for its
library: how is a model instantiated, how is data materialised into the
library's native form, how is a fit run, how is a checkpoint written, how is a
prediction produced.

Because the pipelines above talk only to this interface, adding a backend is a
new sub-package and a registration line -- not a change to any pipeline.

Sub-packages
------------
``torch``
    PyTorch.  The richest engine: gradient-based loops, learners for both
    supervised and reinforcement learning, mixed precision, distributed
    training and lazy-parameter materialisation.
``xgboost``
    Gradient-boosted trees.  Fits in one call, so it implements the engine
    interface with a no-op training loop.
``sklearn``
    scikit-learn estimators, for baselines and sanity checks.

Modules
-------
``base.py``
    The ``Engine`` protocol every adapter satisfies, ``EngineCapabilities``
    (what a pipeline may ask of an engine) and ``ModelHandle`` (a prepared
    model and its apparatus).  Scheduled for Phase 1 and delivered in Phase 2
    instead: Phase 1 built the pipeline skeleton against the *model
    definition*, so nothing consumed an engine interface and writing one would
    have meant designing a contract with no consumer -- the risk that phase's
    own risk table names as its largest.  [Phase 2]

Dependency rule
---------------
May import: ``core``, ``sources``.
May not import: ``orchestration``, ``models``, ``analysis``.
An engine never decides *where* it runs -- that is ``orchestration.compute`` --
and never writes a report.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/engines/base.py`

17734 bytes · SHA-256 `51a0d4d8b27f4e4d`

```python
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
    from ..core.contract.signature import InputSignature
    from ..core.contract.source import BatchSource
    from ..core.spec.hardware import HardwareSpec

__all__ = ["Engine", "EngineCapabilities", "ModelHandle"]


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """
    What an engine can and cannot do, declared rather than discovered.

    A pipeline needs these answers *before* it starts, because several of them
    change which stages it runs. Asking an engine to checkpoint and catching
    the failure would mean discovering a limitation after the data build.

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
        Whether a validation source can be scored as training proceeds. When
        false, a monitored callback has nothing to monitor and the pipeline
        says so rather than silently reporting a best epoch of zero.
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
```

