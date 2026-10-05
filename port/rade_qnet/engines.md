# `tranql/models/rade/rade_qnet/rade_qnet/engines`

3 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 87 | 4136 | `bfa72bb67acebe57` |
| 2 | `base.py` | 575 | 22899 | `373cda51c8e5f5b2` |
| 3 | `loaders.py` | 336 | 12755 | `1e5d33305a6210f6` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/engines/__init__.py`

4136 bytes · SHA-256 `bfa72bb67acebe57`

```python
"""
How optimisation actually happens -- one adapter per library.

An engine is the only place in the framework that is allowed to know about a
specific training library.  It answers a fixed set of questions for its
library: how is a model instantiated, how is data materialised into the
library's native form, how is a fit run, how is a checkpoint written, how is a
prediction produced.

Because the pipelines above talk only to this interface, adding a backend
changes no pipeline.

Engines are framework-owned, models are not
-------------------------------------------
That is the one asymmetry worth knowing before planning work against this
package.  A *model* plugs in entirely from outside: import it and it is
registered, with no edit anywhere in ``rade_qnet`` (``test_extensibility.py``
proves a third-party model gets the whole lifecycle).  An *engine* does not.
Besides the sub-package and the registration line, a new engine needs its own
training-spec type added to the ``TrainingSpec`` union in
``core.spec.training`` -- the union is discriminated on a ``Literal`` engine
name, so a fourth name does not validate until it is declared there.

That is deliberate.  Each engine's settings get their own validated type,
which is what stops a one-shot fit being configured with gradient-descent
options, and it is a stronger guarantee than a free-form settings mapping
could give.  The cost is that a backend cannot arrive from outside the
distribution.  Stated here because the test suite makes it look otherwise:
its synthetic engines register under the name ``sklearn``, reusing a
discriminator that already exists, so none of them exercises a genuinely new
engine name.  ``test_extensibility.py`` covers the refusal explicitly.

One vocabulary, however large the engine
-----------------------------------------
Every engine package draws its filenames from one closed set, enforced by
``tranql/models/rade/rade_qnet/tests/engines/test_engine_layout.py`` exactly as the model layout
test governs ``models``.  ``engine.py`` is required; ``materialise.py``,
``loaders.py`` and ``predictor.py`` are the other three verbs; ``training/``,
``learners/`` and ``hardware/`` are the parts of *fit* large enough to need
their own package.  Nothing else is permitted a name.

The set is optional down to a single file on purpose, so that the shape of a
package reports what its library owns.  A reader can tell from ``ls`` that a
boosted fit has no loop to configure and no device to choose.

Sub-packages
------------
``torch``
    PyTorch.  The richest engine, and the only one that needs all three
    sub-packages: gradient-based drivers, learners for both supervised and
    reinforcement learning, mixed precision, distributed training and
    lazy-parameter materialisation.
``xgboost``
    Gradient-boosted trees.  One module, because the fit is one call.
``sklearn``
    scikit-learn estimators, for baselines and sanity checks.  One module,
    for the same reason.

Modules
-------
``loaders.py``
    Draining a ``BatchSource`` into the single matrix a one-shot fit needs.
    Pure NumPy, and shared.  It lived in ``sklearn`` until the xgboost engine
    imported it from there, which made installing one backend depend on the
    presence of another; the layout test now refuses a cross-engine import
    outright.

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

## 2. `tranql/models/rade/rade_qnet/rade_qnet/engines/base.py`

22899 bytes · SHA-256 `373cda51c8e5f5b2`

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

    This is the same pattern as ``core.authoring.capabilities``, used for the
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
```

---

## 3. `tranql/models/rade/rade_qnet/rade_qnet/engines/loaders.py`

12755 bytes · SHA-256 `1e5d33305a6210f6`

```python
"""
Turn a stream of batches into the single matrix a one-shot fit needs.

Why a one-shot engine is handed a stream at all
-----------------------------------------------
It looks backwards. A tree does not iterate, so handing it an iterator and
asking it to put the pieces back together is work that only exists because
of how the data arrived.

The alternative was a second payload type carrying the whole split at once,
and it was built in Phase 1 and retired in Phase 6. The short version is that
the mismatch it bridged does not exist: ``sources`` imports no training
library, so a batch is already a mapping of NumPy arrays, and the
concatenation here is one copy that XGBoost's ``DMatrix`` construction was
going to make anyway. The long version, including why the second payload
would have forked the lifecycle, is in the Phase 6 charter §3.1.

What matters is that this file is the *only* place the conversion happens.
Both one-shot engines import it, so there is one definition of how a stream
becomes a matrix and one definition of the row order that results — which is
the property every downstream attribution depends on, and the one that would
be quietly violated if two engines each grew their own.

Why order is the whole contract
-------------------------------
``Engine.predict`` promises predictions in source order, because that is what
lets a number be traced back to a scenario and an instrument. The order here
is the order the source yields, and nothing sorts, shuffles or regroups. A
training split's source may well be shuffled — that is its business — which
is why scoring uses :func:`~rade_qnet.orchestration.stages.scoring.
scoring_source` to obtain a stable view first. This module does not try to
fix that, because a module that silently reordered its input would make the
stable-order machinery above it untestable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.contract.data import TARGET_KEY
from ..core.lifecycle.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from ..core.contract.source import BatchSource

__all__ = ["DrainedSplit", "drain", "flatten", "reject_static"]

#: Rank of the matrix a one-shot estimator consumes.
_MATRIX_RANK = 2


class DrainedSplit:
    """
    One split, materialised as a feature matrix and a target vector.

    Deliberately not a frozen dataclass in ``core``: it never crosses a
    pipeline boundary and never reaches a contract. It exists for the few
    lines between draining a source and calling ``fit``, and giving it a
    home in ``core`` would reintroduce exactly the second payload type this
    phase retired.

    Parameters
    ----------
    features
        Two-dimensional, samples by features, in source order.
    target
        One-dimensional, one value per sample, in the same order. Flattened
        because every estimator this engine drives predicts a scalar, and a
        column vector produces a ``DataConversionWarning`` on every fit.
    feature_names
        Column names where the signature supplied them. Carried because tree
        models report importances positionally, and an importance table keyed
        by index is nearly useless.
    """

    __slots__ = ("feature_names", "features", "target")

    def __init__(
        self,
        features: NDArray[np.float64],
        target: NDArray[np.float64],
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        self.features = features
        self.target = target
        self.feature_names = feature_names

    @property
    def n_samples(self) -> int:
        """
        Number of rows.

        Returns
        -------
        int
            The row count, which both arrays agree on by construction.
        """
        return int(self.features.shape[0])

    def describe(self) -> str:
        """
        Return a compact summary, for logs.

        Returns
        -------
        str
            For example ``240 x 12``.
        """
        return f"{self.n_samples} x {self.features.shape[1]}"


def flatten(array: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Reduce a batch's input to two dimensions, samples by features.

    A sequence model's input arrives as ``(samples, timesteps, features)``,
    and a one-shot estimator has no notion of a time axis. Flattening every
    axis after the first turns one timestep-feature pair into one column,
    which is the standard way to put a windowed problem in front of a tree:
    the model loses the knowledge that column 7 and column 19 are the same
    quantity at different lags, and gains the ability to be fitted at all.

    Stating that plainly matters, because it is the main reason a tabular
    baseline underperforms a recurrent model on the same data, and reading
    that difference as "the recurrent architecture is better" when it is
    partly "the baseline was handed a worse representation" would be the
    wrong conclusion to draw from this phase's comparison.

    Parameters
    ----------
    array
        An input of rank one or more.

    Returns
    -------
    numpy.ndarray
        Two-dimensional, with the leading axis preserved.
    """
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim == _MATRIX_RANK:
        return values
    return values.reshape(values.shape[0], -1)


def drain(source: BatchSource, *, require_target: bool = True) -> DrainedSplit:
    """
    Consume a bounded source once and return the whole split.

    Parameters
    ----------
    source
        The split's source. Must be bounded: a one-shot fit over an endless
        stream has no point at which to start.
    require_target
        Whether a missing target is an error. True when fitting, false when
        predicting -- a source built for inference has no labels, and
        demanding them would make a trained model unusable on live data.

    Returns
    -------
    DrainedSplit
        Features, target and column names, in source order.

    Raises
    ------
    EngineError
        If the source is unbounded, yields nothing, yields batches whose
        inputs disagree, or omits a target that was required.
    """
    if source.steps_per_epoch is None:
        raise EngineError(
            "a one-shot engine needs a bounded source: this one reports no "
            "step count, so there is no last batch and nothing to fit against. "
            "An unbounded source belongs to an interactive learner"
        )

    # Checked here as well as in the engines' ``prepare``, because this is
    # the function that actually drops them: ``_input_names`` reads only the
    # signature's dynamic inputs, so a static one would vanish into a run
    # that succeeds with an input missing. The engines check early to fail
    # before the data is built; this checks where the loss would happen.
    reject_static(source.static)

    names = _input_names(source)
    columns: list[list[NDArray[np.float64]]] = [[] for _ in names]
    targets: list[NDArray[np.float64]] = []

    for index, batch in enumerate(source.batches()):
        for position, name in enumerate(names):
            if name not in batch:
                raise EngineError(
                    f"batch {index} is missing the {name!r} input that the "
                    f"source's signature declares; the feature matrix would "
                    f"have a different width than the one before it"
                )
            columns[position].append(flatten(batch[name]))
        if TARGET_KEY in batch:
            targets.append(np.ravel(np.asarray(batch[TARGET_KEY], dtype=np.float64)))

    if not any(columns) or not columns[0]:
        raise EngineError(
            f"the source yielded no batches, so there is nothing to fit. It "
            f"reports {source.steps_per_epoch} step(s), which means the "
            f"count and the iteration disagree"
        )

    features = np.hstack([np.vstack(parts) for parts in columns])
    target = np.concatenate(targets) if targets else np.empty(0, dtype=np.float64)

    if require_target and target.size == 0:
        raise EngineError(
            f"the source yields no {TARGET_KEY!r}, so there is nothing to fit "
            f"towards. A source without labels can be predicted over but not "
            f"trained on"
        )
    if target.size and target.size != features.shape[0]:
        raise EngineError(
            f"the source yielded {features.shape[0]} feature row(s) but "
            f"{target.size} target value(s). A model fitted on a misaligned "
            f"pair learns a permutation and reports a plausible loss while "
            f"doing it"
        )

    return DrainedSplit(features, target, _feature_names(source, features.shape[1]))


def _input_names(source: BatchSource) -> tuple[str, ...]:
    """
    Return the dynamic inputs to concatenate, in a fixed order.

    Sorted rather than taken in declaration order, so that two runs of one
    configuration produce the same column layout. A model whose columns
    depend on dictionary ordering would have weights that silently stop
    matching their own feature names.

    Parameters
    ----------
    source
        The split's source.

    Returns
    -------
    tuple of str
        Input names, excluding the target.

    Raises
    ------
    EngineError
        If the signature declares no dynamic inputs.
    """
    names = tuple(sorted(name for name in source.signature.dynamic if name != TARGET_KEY))
    if not names:
        raise EngineError(
            "the source's signature declares no dynamic inputs, so there are "
            "no features to fit on. A model whose inputs are entirely static "
            "has one sample, not a dataset"
        )
    return names


def _feature_names(source: BatchSource, n_columns: int) -> tuple[str, ...] | None:
    """
    Name the matrix's columns where the signature makes that possible.

    An input wider than one column is expanded into one name per column,
    ``features_0``, ``features_1`` and so on, using the widths the signature
    declares. Naming only the case where every input happens to be a single
    scalar would leave the common case -- one wide feature block --
    permanently unnamed, which is the case a coefficient table is read for.

    The separator is an underscore rather than a bracket subscript because
    XGBoost rejects ``[``, ``]`` and ``<`` in feature names outright, and
    these adapters are shared by both engines. A naming scheme one of them
    refuses is not a naming scheme.

    The widths come from the signature rather than from the matrix, so they
    are checked against it: if the two disagree, something upstream flattened
    differently than declared, and ``None`` is returned. A positional
    importance table is unhelpful; a *mislabelled* one is worse, because it
    is read with confidence.

    Parameters
    ----------
    source
        The split's source.
    n_columns
        Columns the drained matrix actually has.

    Returns
    -------
    tuple of str or None
        One name per column, or ``None`` if they cannot be matched.
    """
    dynamic = source.signature.dynamic
    names: list[str] = []
    for name in _input_names(source):
        # Everything after the sample axis is what flattening collapsed, so
        # its product is the column count this input contributes.
        width = int(np.prod(dynamic[name].shape[1:])) if dynamic[name].shape[1:] else 1
        names.extend([name] if width == 1 else [f"{name}_{i}" for i in range(width)])
    return tuple(names) if len(names) == n_columns else None


def reject_static(static: Mapping[str, object]) -> None:
    """
    Refuse static inputs, which a one-shot estimator cannot consume.

    Parameters
    ----------
    static
        Static inputs carried on the handle.

    Raises
    ------
    EngineError
        If any are present. A graph or an entity table has no column in a
        feature matrix, and quietly dropping it would fit the model on less
        than the specification asked for -- which produces a worse model and
        no indication that anything was lost.
    """
    if static:
        raise EngineError(
            f"this engine cannot consume static inputs, and the data build "
            f"supplied {sorted(static)}. A one-shot estimator has one feature "
            f"matrix and no second channel to put them in. Refused rather "
            f"than dropped, because a dropped input produces a quietly worse "
            f"model that still trains and still reports a loss"
        )
```

