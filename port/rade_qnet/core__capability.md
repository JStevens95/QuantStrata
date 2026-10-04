# `src/rade_qnet/core/capability`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 60 | 2853 | `ce24f36959699033` |
| 2 | `definition.py` | 317 | 12057 | `0cd0a0d1750e8b1c` |
| 3 | `policy.py` | 167 | 6123 | `47bb552c70dba776` |
| 4 | `protocols.py` | 265 | 10203 | `438101cea82e2646` |
| 5 | `supervised.py` | 397 | 14777 | `8e2b5101bac2f3be` |

---

## 1. `src/rade_qnet/core/capability/__init__.py`

2853 bytes · SHA-256 `ce24f36959699033`

```python
"""
The model-facing interface: what a model must provide, and what it may.

A model registers itself with the framework by subclassing a base in
``definition.py``.  The base demands very little -- build a model object from a
spec and a signature -- which is what keeps a simple model to a single file.

Anything beyond that minimum is an *opt-in capability*: a narrow protocol a
model may implement to unlock extra behaviour.  The framework checks for a
capability with a runtime ``isinstance`` test, so a model never pays for a
feature it does not use, and adding a capability never breaks existing models.

Modules
-------
``definition.py``
    ``ModelDefinition`` and its two specialisations, ``PredictorDefinition``
    (learns from a fixed dataset) and ``PolicyDefinition`` (learns by
    interacting with an environment), plus the ``@model`` registration
    decorator re-exported from ``core.runtime.components`` so a model author
    needs one import.  [Phase 1, delivered]
``protocols.py``
    The opt-in capabilities:

    ``StaticInputs``
        The model consumes inputs that are constant across every batch (a
        graph, entity features, index arrays).  Lets the framework upload them
        to the device once instead of once per batch.
    ``Precomputable``
        An expensive encoding of the static inputs can be computed once per
        evaluation pass and reused.
    ``CustomStep``
        The model owns its loss computation, for multi-objective or otherwise
        non-standard training.
    ``Routable``
        The model can serve as a member of a job set, declaring which targets
        it was actually trained on.
    ``Inductive``
        The model can predict for entities that were absent during training.
    [Phase 1 delivered, extended in Phase 3]
``supervised.py``
    ``SupervisedModel``, the base almost every supervised model subclasses:
    it supplies data building, re-loading and the signature, so a model
    writes one line of data wiring and ``build_model``.  Named for the
    learning paradigm rather than the data shape -- tables, sequences and
    graphs all use it.  [Phase 2 as ``simple.py``, renamed after Phase 6]
``policy.py``
    ``PolicyModel``, the equivalent base for reinforcement-learning models:
    it supplies the signature by reading the environment's declared spaces,
    so a model writes ``build_environment`` and ``build_policy``.  [Phase 7]

One base per learning paradigm
------------------------------
Two bases, not three.  An unsupervised base is deliberately absent until a
pipeline exists that can train one -- an empty base with no reader is a
promise the framework cannot keep, and the cost of adding it later is one
module, whereas the cost of publishing it early is every model author who
subclasses something that does not work.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/core/capability/definition.py`

12057 bytes · SHA-256 `0cd0a0d1750e8b1c`

```python
"""
What a model must provide to join the framework.

A model definition is the single object the framework holds on a user's
behalf. It is not the model itself -- it is the recipe: given a validated spec,
it builds the data, declares the interface, and constructs the model object.

The minimum is deliberately small
---------------------------------
:class:`PredictorDefinition` demands three methods: build the data, declare the
signature, build the model. Everything else -- custom losses, static inputs,
precomputation, job-set routing -- is an opt-in protocol from
:mod:`rade_qnet.core.capability.protocols`.

That split is what lets a linear model be forty lines and the flagship
multi-cluster graph model be a package, without the framework treating them
differently. The small model does not stub out methods it has no use for, and
the large one is not constrained by a base class that tried to anticipate it.

Why a definition rather than a subclassed model
-----------------------------------------------
The alternative -- having users subclass a framework ``Model`` base and
override ``forward`` -- would mean the framework owns the model object's type.
That breaks as soon as two engines are involved, because a tree model and a
network have nothing in common to inherit. Keeping the definition separate
from the model object means the model object can be whatever its engine
requires: a ``torch.nn.Module``, a ``Booster``, an sklearn estimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from ..contract.data import DataBundle
from ..contract.requirement import InputRequirement
from ..contract.signature import InputSignature, PolicySignature
from ..contract.state import FittedState
from ..runtime.components import model

if TYPE_CHECKING:
    from ..spec.run import ReinforcementRunSpec, SupervisedRunSpec

__all__ = [
    "ModelDefinition",
    "PolicyDefinition",
    "PredictorDefinition",
    "model",
]


class ModelDefinition(ABC):
    """
    Common base for everything the framework can train.

    Holds only what both learning styles share: a name, the engine that trains
    it, and a hook for declaring what it fitted. The two concrete bases below
    diverge immediately, because learning from a fixed dataset and learning by
    interaction genuinely differ in what the framework must ask for.

    Attributes
    ----------
    component_name
        Registered name, set by the :func:`model` decorator. Present as a
        class attribute so a definition can be looked up by the name that
        appears in a YAML spec.
    component_engine
        Registered name of the engine that trains this model, also set by the
        decorator. Declared by the model rather than chosen by the run,
        because a graph network cannot be trained by the XGBoost engine and
        allowing the spec to ask for it would only defer the error.
    """

    component_name: str
    component_engine: str

    #: What this model consumes, declared in its ``data.py``. Bound here by
    #: ``register.py`` and checked by :meth:`check_signature` between the
    #: data build and the model build.
    #:
    #: Unset is not a legal state for a model in this repository -- the
    #: layout test requires ``data.py`` to export a ``REQUIRES`` -- but the
    #: default exists so that a definition constructed directly in a test
    #: does not have to declare one.
    requires: ClassVar[InputRequirement] = InputRequirement.unconstrained()

    def check_signature(self, signature: InputSignature) -> None:
        """
        Verify the data build produced what this model consumes.

        The one place information flows model to data. Everywhere else in
        the framework the signature travels the other way -- the build
        declares what it made and the model copes -- and coping is what
        turns a wrong data build into a plausible loss curve rather than an
        error. See :mod:`rade_qnet.core.contract.requirement` for the two
        defects that motivated this.

        Called by every pipeline that is about to build a model, including
        the reload paths: a bundle trained against an older version of the
        model can carry a signature the current code no longer consumes,
        and that is worth catching at load rather than at forward.

        Parameters
        ----------
        signature
            What the data build declared.

        Raises
        ------
        ContractError
            If any requirement is unmet, listing every problem at once.
        """
        self.requires.check(signature, model=self.component_name)

    def fitted_state(self) -> FittedState | None:
        """
        Return state fitted by the model itself, beyond its parameters.

        Most models fit nothing here: scalers and encoders belong to the data
        build, and the state it produced travels in the data bundle. This
        hook exists for the minority of models that fit something during
        training -- a learned graph, a selected feature subset -- which the
        data build could not have known.

        Returns
        -------
        FittedState or None
            Additional fitted state, or ``None`` if there is none. When
            non-``None``, it is composed with the data build's state and both
            are saved into the bundle.
        """
        return None


class PredictorDefinition(ModelDefinition):
    """
    A model that learns from a fixed dataset.

    Three abstract methods, in the order the train pipeline calls them.

    Notes
    -----
    The built model object is annotated ``object`` rather than ``Any``, and the
    distinction matters. ``core`` genuinely cannot name the type -- it is a
    ``torch.nn.Module``, a ``Booster``, or an sklearn estimator depending on
    the engine -- but ``object`` says "the framework will not touch this",
    whereas ``Any`` would say "anyone may do anything with this". The
    framework only ever passes the model object through to its declared
    engine, which knows what it accepts and narrows it there. Annotating it
    ``object`` makes a type checker enforce that hands-off treatment.

    The same reasoning gives ``DataBundle[object]``: the payload is engine
    native, and because the bundle is frozen, a model returning
    ``DataBundle[TensorBatchData]`` still satisfies it.
    """

    @abstractmethod
    def build_data(self, spec: SupervisedRunSpec) -> DataBundle[object]:
        """
        Build the dataset, splits and fitted state.

        Called once per run, before anything is constructed. The returned
        bundle's payload type must match what the declared engine consumes:
        ``TensorBatchData`` for every engine in this framework, or whatever a
        one-shot engine.

        Two axes behave differently here, and conflating them is the most
        common correctness error in this kind of code:

        - Along the **scenario (time) axis**, anything fitted must be fitted
          on training rows only. Fitting a scaler on all scenarios leaks the
          test period's distribution into training.
        - Along the **entity (instrument) axis**, the full universe is
          permitted. Knowing which instruments exist is not knowledge of
          their future, and a cross-sectional model that saw only some of
          them would be solving a different problem.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        DataBundle
            Splits, signature, fitted state and lineage.
        """

    @abstractmethod
    def signature(self, bundle: DataBundle[object]) -> InputSignature:
        """
        Declare the interface between the data and the model.

        Separate from the bundle's own signature so a model may narrow it: a
        data build may offer more inputs than a given model consumes, and the
        signature recorded in the saved bundle should describe what the model
        actually used. Returning ``bundle.signature`` unchanged is the normal
        implementation.

        Parameters
        ----------
        bundle
            The data bundle from :meth:`build_data`.

        Returns
        -------
        InputSignature
            The declared interface, saved into the bundle and used to rebuild
            the model later.
        """

    @abstractmethod
    def build_model(self, spec: SupervisedRunSpec, signature: InputSignature) -> object:
        """
        Construct the untrained model object.

        Must be a pure function of the spec and the signature. It receives no
        data, which is what guarantees the same model can be rebuilt from a
        saved bundle without the data build -- the property that makes a
        six-month-old bundle loadable.

        A model with lazily sized parameters may return an unmaterialised
        object; the engine materialises it from the signature before any
        optimiser, checkpoint or distributed wrapper touches it.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared interface from :meth:`signature`.

        Returns
        -------
        object
            An engine-native untrained model.
        """


class PolicyDefinition(ModelDefinition):
    """
    A model that learns by interacting with an environment.

    The structural counterpart of :class:`PredictorDefinition`: an environment
    replaces the dataset, and a policy signature replaces the input signature.
    The pipeline shape is otherwise the same, which is why both derive from
    one base and both train through one loop.

    What makes that possible is
    :class:`~rade_qnet.core.contract.source.BatchSource`. The environment is
    wrapped as a source -- of rollouts, of replayed transitions, of stored
    experience -- and from the loop's perspective it is indistinguishable from
    a dataset. The loop decides *when* an update happens; the learner decides
    what one update means.
    """

    @abstractmethod
    def build_environment(self, spec: ReinforcementRunSpec) -> object:
        """
        Construct the environment the policy learns in.

        Returns an environment object rather than a source: wrapping it as a
        :class:`~rade_qnet.core.contract.source.BatchSource` is the engine's
        job, because how experience is gathered is a property of the learner
        -- on-policy rollouts, a replay buffer, pathwise differentiation --
        and not of the environment.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        object
            An environment satisfying one of the environment protocols in
            ``rade_qnet.sources.environment``.
        """

    @abstractmethod
    def signature(self, environment: object) -> PolicySignature:
        """
        Declare the observation and action spaces.

        Parameters
        ----------
        environment
            The environment from :meth:`build_environment`.

        Returns
        -------
        PolicySignature
            Observation and action spaces, saved into the bundle so the policy
            can be rebuilt without reconstructing the environment.
        """

    @abstractmethod
    def build_policy(self, spec: ReinforcementRunSpec, signature: PolicySignature) -> object:
        """
        Construct the untrained policy.

        As with :meth:`PredictorDefinition.build_model`, this must be a pure
        function of the spec and the signature, so a saved policy can be
        rebuilt without an environment.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared spaces from :meth:`signature`.

        Returns
        -------
        object
            An engine-native untrained policy.
        """
```

---

## 3. `src/rade_qnet/core/capability/policy.py`

6123 bytes · SHA-256 `47bb552c70dba776`

```python
"""
The base class for reinforcement-learning models: learn by acting.

The third of the paradigm bases promised by :mod:`rade_qnet.core.capability`,
and the one ``supervised.py`` says "arrives with Phase 7".

Where :class:`~.supervised.SupervisedModel` learns a mapping from inputs to
known targets, a :class:`PolicyModel` has no targets at all. It is given an
environment, it acts, and the only feedback is a scalar reward for what it
did. That is a different *paradigm*, not a different data shape, which is why
it is a separate base rather than a flag on the supervised one.

What this base saves a model author
-----------------------------------
:class:`~.definition.PolicyDefinition` demands three methods. This class
supplies :meth:`signature` -- read the two spaces off the environment -- so a
model author writes two:

.. code-block:: python

    @model("my_agent")
    class MyAgent(PolicyModel):
        def build_environment(self, spec):
            return MyEnvironment(**spec.environment.params)

        def build_policy(self, spec, signature):
            return MyNetwork(signature)

That is the same bargain :class:`~.supervised.SupervisedModel` strikes, and
it matters for the same reason: a baseline agent that needed fifty lines of
plumbing would not be a useful control.

Why the signature is read, not measured
---------------------------------------
:meth:`signature` asks the environment what its spaces *are*, rather than
resetting it and measuring the observation that comes back. The difference
shows up six months later: a saved bundle has a ``PolicySignature`` and no
environment, and :meth:`~.definition.PolicyDefinition.build_policy` is
required to be a pure function of the spec and that signature so the policy
can be rebuilt regardless. Measuring would also quietly get the action space
wrong, since nothing an environment *returns* describes what it accepts.

An environment whose spaces genuinely are not known until it is constructed
is still fine -- it is constructed before :meth:`signature` is called. What
is not fine is an environment that has to be *stepped* first, and such an
environment should compute its spaces in ``__init__``.

Why ``EnvironmentLike`` is declared here as well as in ``sources``
------------------------------------------------------------------
For the same reason :class:`~.supervised.DataModuleLike` is: ``core`` may
import nothing from its sibling layers, so it cannot name
:class:`~rade_qnet.sources.environment.protocol.Environment`. This protocol
names the two members this base actually reads, and nothing else -- a real
environment satisfies it, and so does a two-property stub in a test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..contract.signature import PolicySignature
from ..runtime.errors import ComponentError
from ..runtime.logging import get_logger
from .definition import PolicyDefinition

if TYPE_CHECKING:
    from ..contract.signature import SpaceSpec

__all__ = ["EnvironmentLike", "PolicyModel"]

_LOGGER = get_logger(__name__)


@runtime_checkable
class EnvironmentLike(Protocol):
    """
    The two properties :class:`PolicyModel` needs from an environment.

    Narrow on purpose; see the module docstring. Note that it names neither
    ``reset`` nor ``step``: this base never drives the environment, it only
    describes it. The engine is what needs the full
    :class:`~rade_qnet.sources.environment.protocol.Environment`, and the
    engine is in a layer allowed to say so.
    """

    @property
    def observation_space(self) -> SpaceSpec:
        """
        What the policy sees.

        Returns
        -------
        SpaceSpec
            The observation space.
        """
        ...

    @property
    def action_space(self) -> SpaceSpec:
        """
        What the policy must produce.

        Returns
        -------
        SpaceSpec
            The action space.
        """
        ...


class PolicyModel(PolicyDefinition):
    """
    A policy that learns from interaction rather than from labelled data.

    Supplies :meth:`signature`, so a subclass implements
    :meth:`~.definition.PolicyDefinition.build_environment` and
    :meth:`~.definition.PolicyDefinition.build_policy`.

    A model whose spaces cannot be read off its environment -- a multi-agent
    setup, say, where the spaces are per-agent -- subclasses
    :class:`~.definition.PolicyDefinition` directly and writes its own
    ``signature``.
    """

    def signature(self, environment: object) -> PolicySignature:
        """
        Read the observation and action spaces off the environment.

        Parameters
        ----------
        environment
            The environment from
            :meth:`~.definition.PolicyDefinition.build_environment`.

        Returns
        -------
        PolicySignature
            The two declared spaces, saved into the bundle so the policy can
            be rebuilt without reconstructing the environment.

        Raises
        ------
        ComponentError
            If the environment does not declare both spaces. Reported as a
            component error, not a contract error, because the fault is in
            the model package's ``build_environment`` rather than in the
            framework or the data.
        """
        if not isinstance(environment, EnvironmentLike):
            raise ComponentError(
                f"{type(self).__name__}.build_environment() returned "
                f"{type(environment).__name__}, which does not declare both "
                f"observation_space and action_space; PolicyModel reads the "
                f"signature off the environment rather than measuring it"
            )

        signature = PolicySignature(
            observation=environment.observation_space,
            action=environment.action_space,
        )
        _LOGGER.info(
            "declared spaces for %s: observation=%s action=%s",
            type(self).__name__,
            signature.observation.kind,
            signature.action.kind,
        )
        return signature
```

---

## 4. `src/rade_qnet/core/capability/protocols.py`

10203 bytes · SHA-256 `438101cea82e2646`

```python
"""
Opt-in capabilities a model may implement.

Every protocol here is ``runtime_checkable``, and the framework tests for them
with ``isinstance``. That choice is deliberate and has three consequences
worth being explicit about.

**A model never pays for a feature it does not use.** A simple regressor does
not implement :class:`StaticInputs`, so the engine never asks about device
placement for static tensors. There is no base-class method to stub out and no
``None`` to return.

**Adding a capability cannot break an existing model.** A new protocol is new
behaviour for models that implement it and a no-op for every model that does
not. Compare the alternative -- a growing abstract base -- where each addition
is a breaking change to every subclass in existence.

**The capability is visible in the type, not in a flag.** The framework never
reads ``model.has_static_inputs``. It asks whether the model satisfies the
protocol, which cannot disagree with what the model actually implements.

A caveat on runtime checking
----------------------------
``isinstance`` against a runtime-checkable protocol verifies *method presence*,
not signatures. A model with a ``static_inputs`` method of the wrong shape
passes the check and then fails when called. This is why
``rade_qnet.testkit.conformance`` exists: it calls each capability it detects and
checks what comes back. The ``isinstance`` test routes; the conformance suite
verifies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from ..contract.data import Batch, TensorLike

__all__ = [
    "CustomStep",
    "Inductive",
    "Precomputable",
    "Routable",
    "StaticInputs",
]


@runtime_checkable
class StaticInputs(Protocol):
    """
    The model consumes inputs that are constant across every batch.

    An adjacency matrix, an entity feature table, an index array. Implementing
    this lets the engine move them to the device once, at the start of
    training, rather than treating them as part of each sample.

    This protocol is what retires a specific inefficiency in the
    implementation being replaced. There, static tensors were merged into
    every sample and then *compared across the batch* during collation, with
    the first returned once they were confirmed equal -- so an adjacency
    matrix was compared against itself once per sample, per batch, per epoch,
    to establish something true by construction. Declaring the inputs static
    deletes both the copying and the comparison.
    """

    def static_inputs(self) -> Mapping[str, TensorLike]:
        """
        Return the inputs that do not vary between batches.

        Called once per fit, before the first batch, and the result is held
        for the duration of the run. Implementations must therefore not
        return something that depends on batch state.

        Returns
        -------
        Mapping
            Static inputs, keyed as declared in the model's signature.
        """
        ...


@runtime_checkable
class Precomputable(Protocol):
    """
    An expensive encoding of the static inputs can be computed once and reused.

    The motivating case is a graph encoder. If the graph does not change
    between batches, neither do its node embeddings, so recomputing them per
    batch repeats identical work -- which during evaluation can dominate the
    runtime entirely.

    The framework calls :meth:`precompute` once at the start of an evaluation
    or inference pass and passes the result to each forward call. It does
    *not* do this during training, where the encoder's parameters are being
    updated and a cached embedding would be stale after the first step.
    """

    def precompute(self, static: Mapping[str, TensorLike]) -> Mapping[str, TensorLike]:
        """
        Compute the reusable encoding of the static inputs.

        Parameters
        ----------
        static
            The model's static inputs, already on the correct device.

        Returns
        -------
        Mapping
            Named intermediate tensors to be reused across the pass.
        """
        ...

    def forward_with_precomputed(
        self,
        batch: Batch,
        precomputed: Mapping[str, TensorLike],
    ) -> TensorLike:
        """
        Run a forward pass using a previously computed encoding.

        Must return the same values the ordinary forward pass would, up to
        floating-point reassociation. The conformance suite checks this
        against the model's standard forward path, because a divergence here
        means evaluation and training silently measure different models.

        Parameters
        ----------
        batch
            One batch of dynamic inputs.
        precomputed
            The output of :meth:`precompute`.

        Returns
        -------
        TensorLike
            Predictions for the batch.
        """
        ...


@runtime_checkable
class CustomStep(Protocol):
    """
    The model owns its loss computation.

    The default training step computes one loss from one output and one
    target, which covers most models and should be left alone when it does.
    Implementing this protocol takes over that computation, which is required
    when the loss is not a function of a single output: multi-objective
    training, auxiliary losses, a regulariser over the model's own
    parameters, or a loss that needs the static inputs.

    What the framework still owns
    -----------------------------
    The step returns a loss; it does not perform the update. Backward passes,
    gradient clipping, optimiser stepping, scheduler stepping, mixed-precision
    scaling and distributed synchronisation all remain the engine's
    responsibility. That boundary is what keeps a custom loss compatible with
    every execution mode -- a model that called ``backward()`` itself would
    quietly break under gradient accumulation and under distributed training.
    """

    def training_step(self, batch: Batch, static: Mapping[str, TensorLike]) -> TensorLike:
        """
        Compute the scalar loss for one batch.

        Parameters
        ----------
        batch
            One batch, carrying the dynamic inputs and the target.
        static
            The model's static inputs, already on the correct device.

        Returns
        -------
        TensorLike
            A scalar loss tensor with a gradient path back to the model's
            parameters. Returning a detached tensor produces a run that
            trains for its full epoch budget and learns nothing, so the
            conformance suite checks that gradients reach the parameters.
        """
        ...


@runtime_checkable
class Routable(Protocol):
    """
    The model can serve as one member of a job set.

    A job set runs the same model across many jobs -- one per instrument,
    region or cluster. The framework needs to know which targets a given
    member is actually responsible for, so that predictions from several
    members can be assembled into one portfolio-level result without
    overlapping or leaving gaps.

    The declaration is required rather than inferred because the honest answer
    is often narrower than the job definition. A member nominally assigned
    twelve instruments may have trained on nine, the other three having been
    dropped for insufficient history. Inferring coverage from the job
    definition would then attribute three instruments' predictions to a model
    that never saw them.
    """

    def covered_targets(self) -> Sequence[str]:
        """
        Return the targets this member actually trained on.

        Returns
        -------
        Sequence of str
            Target identifiers, which must be a subset of those the job
            assigned. The framework checks this and reports any target that
            was assigned to no member.
        """
        ...


@runtime_checkable
class Inductive(Protocol):
    """
    The model can predict for entities absent during training.

    A model that learned a per-entity embedding table cannot do this: a new
    instrument has no row. A model that learned a function of entity
    *features* can, provided those features are available.

    The distinction is declared rather than attempted because the failure mode
    of getting it wrong is quiet. Asking a transductive model about an unseen
    entity tends to yield a default embedding and a confident, meaningless
    prediction -- rather than an error. Declaring the capability lets the
    inference pipeline reject the request instead.
    """

    def supports_unseen_entities(self) -> bool:
        """
        Return whether unseen entities can be predicted for.

        A method rather than a class-level constant because the answer can
        depend on how the instance was configured -- the same class may use
        an embedding table or a feature encoder according to its spec.

        Returns
        -------
        bool
            True if the model generalises to entities it has not seen.
        """
        ...

    # No companion method that *performs* the prediction, deliberately.
    #
    # An earlier draft of this protocol carried a `resolve_unseen` hook so
    # the pipeline would have something to call. It was removed before it
    # shipped, because designing it honestly needs a concrete scenario this
    # framework does not yet have: an entity absent from training is also
    # absent from the fitted state a reloaded model applies, so it has no
    # node, no index and no encoded attributes, and a hook handed only
    # identifiers and the existing static tensors has nothing to build from.
    #
    # The useful signature is therefore not knowable yet, and a protocol
    # method nothing calls and nothing implements is worse than its absence:
    # it reads as a supported path. The declaration alone already pays for
    # itself -- it is what lets the inference pipeline refuse a transductive
    # model rather than return a default embedding -- and the mechanism
    # belongs with the universe handling that can supply the missing rows.
```

---

## 5. `src/rade_qnet/core/capability/supervised.py`

14777 bytes · SHA-256 `8e2b5101bac2f3be`

```python
"""
The base class for supervised models: learn a mapping from inputs to targets.

There is one base per *learning paradigm*, not per data shape:

- :class:`SupervisedModel` (here) -- learns from a fixed dataset of inputs
  and known targets. Ridge, a gradient-boosted tree, an LSTM and the
  graph-plus-recurrent flagship are all supervised, whatever their data
  looks like.
- A reinforcement-learning base -- learns by acting in an environment --
  arrives with Phase 7.
- An unsupervised base is deliberately *not* provided until a pipeline
  exists that can train one; an empty base with no reader would be a promise
  the framework cannot keep.

The shape of the data -- a table, sequences, a graph -- is the data module's
business, not the base class's. An earlier name, ``TabularModel``, suggested
otherwise, and was wrong even then: the flagship is a graph model and has
always used this base.

:class:`SupervisedModel` supplies two of the three stages a
:class:`~.definition.PredictorDefinition` demands, leaving a model author one
method of substance to write: build the network. That is what makes the short
model definition in ``ARCHITECTURE.md`` §1 possible, and it is the mechanism
Phase 6's baselines rely on to keep the framework honest -- a baseline needing
fifty lines of data plumbing would not be a useful control.

Why ``data_module`` is a hook rather than a default
---------------------------------------------------
``PHASE_2_TORCH_ENGINE.md`` §2.4 describes this class as "supplying a standard
data module and split, so a straightforward model needs no data code at all".
Taken literally that is not achievable here, and the reason is the dependency
rule rather than an oversight.

``core`` may import nothing from its sibling layers -- ``ALLOWED_DEPENDENCIES``
gives it an empty set, and the test that enforces it walks nested imports too,
so a lazy import inside a method would not escape it. That rule protects a
property worth more than the convenience: a host that only opens a bundle or
reads a spec does not need the data layer installed.

So :meth:`SupervisedModel.data_module` is a hook the model package fills in with
one line, and the class supplies everything else. The promise holds up to that
one line::

    @model("my_model")
    class MyModel(SupervisedModel):
        def data_module(self, spec):
            return TabularDataModule()

        def build_model(self, spec, signature):
            return MyNet(signature)

A concrete subclass that pre-fills the hook belongs in a layer permitted to
import ``sources`` -- ``models.ridge`` and its siblings, which Phase 6 delivers.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..contract.data import DataBundle, TensorBatchData
from ..contract.source import BatchSource
from ..runtime.errors import ComponentError
from ..runtime.logging import get_logger
from .definition import PredictorDefinition

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..contract.signature import InputSignature
    from ..spec.run import SupervisedRunSpec

__all__ = ["DataModuleLike", "RebuildableDataModule", "SupervisedModel"]

_LOGGER = get_logger(__name__)


@runtime_checkable
class RebuildableDataModule(Protocol):
    """
    The one extra method a data module needs to serve a *saved* model.

    Separate from :class:`DataModuleLike` rather than folded into it, for
    the same reason that one is narrow: a protocol should name what its
    caller needs and no more. Both are ``runtime_checkable``, and an
    ``isinstance`` check against a runtime-checkable protocol tests only that
    the methods are present -- so widening :class:`DataModuleLike` to three
    methods would make every two-method stub stop satisfying it, failing
    training runs over a method only evaluation uses.

    Keeping them apart also gives better errors. A module that can train but
    not be re-loaded fails when something tries to re-load it, with a message
    about re-loading, rather than failing at training time with a message
    about a method the training path never calls.
    """

    def rebuild(self, spec: object, *, state: object, lineage: object) -> object:
        """
        Produce a dataset using a saved state and split rather than new ones.

        Parameters
        ----------
        spec
            The source specification.
        state
            The fitted state loaded from a bundle, to be applied and never
            re-fitted.
        lineage
            The bundle's lineage, supplying the split that must not be
            re-derived.

        Returns
        -------
        object
            Something carrying a ``dataset`` attribute holding a prepared
            dataset. Typed loosely because ``core`` cannot name the concrete
            types in ``sources``.
        """
        ...


@runtime_checkable
class DataModuleLike(Protocol):
    """
    The two methods :class:`SupervisedModel` needs from a data module.

    Narrow on purpose. ``core`` cannot name
    :class:`~rade_qnet.sources.dataset.module.DataModule` without importing
    ``sources``, and it does not need to -- it only builds a dataset and asks
    for batch sources over it. Declaring just that keeps the dependency rule
    intact, and means any object with compatible methods will serve, including
    a stub in a test.
    """

    def build(self, spec: object, *, seed: int = 0) -> object:
        """
        Produce a prepared dataset.

        Parameters
        ----------
        spec
            The source specification.
        seed
            Seed for the stages that randomise.

        Returns
        -------
        object
            A prepared dataset, carrying at least ``signature``, ``state`` and
            ``lineage``.
        """
        ...

    def batch_sources(
        self, prepared: object, spec: object, *, seed: int = 0
    ) -> Mapping[str, BatchSource]:
        """
        Produce one batch source per non-empty split.

        Parameters
        ----------
        prepared
            The prepared dataset from :meth:`build`.
        spec
            The source specification.
        seed
            Base seed for the batch order.

        Returns
        -------
        Mapping
            Split name to batch source.
        """
        ...


class SupervisedModel(PredictorDefinition):
    """
    A supervised predictor fed through the standard prepared dataset.

    Supplies :meth:`build_data`, :meth:`rebuild_data` and :meth:`signature`,
    so a subclass implements :meth:`data_module` -- one line -- and
    :meth:`~.definition.PredictorDefinition.build_model`. Any data shape is
    welcome, provided the data module returns a dataset carrying
    ``signature``, ``state`` and ``lineage`` and batch sources satisfying
    :class:`~rade_qnet.core.contract.source.BatchSource`.

    A model whose data cannot take that shape subclasses
    :class:`~.definition.PredictorDefinition` directly and writes its own
    ``build_data``.
    """

    @abstractmethod
    def data_module(self, spec: SupervisedRunSpec) -> DataModuleLike:
        """
        Return the data module that builds this model's dataset.

        Abstract rather than defaulted, for the dependency reason in the
        module docstring.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        DataModuleLike
            Something with compatible ``build`` and ``batch_sources``.
        """

    def build_data(self, spec: SupervisedRunSpec) -> DataBundle[object]:
        """
        Build the dataset and wrap each split's source as an engine payload.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        DataBundle
            One :class:`~rade_qnet.core.contract.data.TensorBatchData` per
            split, plus the signature, fitted state and lineage.

        Raises
        ------
        ComponentError
            If the data module does not have the expected methods, if the
            prepared dataset is missing a field this base reads, or if a
            source does not satisfy
            :class:`~rade_qnet.core.contract.source.BatchSource`. Reported as a
            component error because each fault is in the model package rather
            than in the framework or the data.
        """
        module = self.data_module(spec)
        if not isinstance(module, DataModuleLike):
            raise ComponentError(
                f"{type(self).__name__}.data_module() returned "
                f"{type(module).__name__}, which does not have both build() and "
                f"batch_sources(); it must return a data module"
            )

        prepared = module.build(spec.source, seed=spec.seed)
        missing = [
            field for field in ("signature", "state", "lineage") if not hasattr(prepared, field)
        ]
        if missing:
            raise ComponentError(
                f"the prepared dataset from {type(module).__name__}.build() has no "
                f"{missing}; SupervisedModel expects the standard PreparedDataset shape"
            )

        return self._wrap(module, prepared, spec)

    def rebuild_data(
        self, spec: SupervisedRunSpec, *, state: object, lineage: object
    ) -> DataBundle[object]:
        """
        Build the dataset a saved model already saw, and wrap it as before.

        The counterpart to :meth:`build_data`, used by evaluation and
        inference. Everything downstream of the dataset is identical -- the
        same batch sources, the same payload type -- because a re-loaded
        model must be fed exactly the way it was fed in training. The one
        difference is upstream: the split and the fitted state are read from
        the bundle rather than derived, which is the whole point.

        Parameters
        ----------
        spec
            The validated run specification, normally the bundle's own.
        state
            The fitted state loaded from the bundle.
        lineage
            The bundle's lineage, supplying the split.

        Returns
        -------
        DataBundle
            The same shape :meth:`build_data` returns, over the rebuilt
            dataset.

        Raises
        ------
        ComponentError
            If the data module cannot rebuild, or if the result is not the
            standard shape.
        """
        module = self.data_module(spec)
        if not isinstance(module, RebuildableDataModule):
            raise ComponentError(
                f"{type(self).__name__}.data_module() returned "
                f"{type(module).__name__}, which has no rebuild(); a model can "
                f"be trained without one but cannot be re-loaded, because "
                f"nothing can reapply the saved transforms to new inputs"
            )

        rebuilt = module.rebuild(spec.source, state=state, lineage=lineage)
        prepared = getattr(rebuilt, "dataset", None)
        if prepared is None:
            raise ComponentError(
                f"{type(module).__name__}.rebuild() returned "
                f"{type(rebuilt).__name__}, which has no dataset attribute"
            )
        return self._wrap(module, prepared, spec)

    def _wrap(
        self, module: DataModuleLike, prepared: object, spec: SupervisedRunSpec
    ) -> DataBundle[object]:
        """
        Turn a prepared dataset into the engine payload a pipeline consumes.

        Shared by :meth:`build_data` and :meth:`rebuild_data` rather than
        written twice. Evaluation is only meaningful if a re-loaded model is
        fed identically to the way it was trained, and the surest way to
        guarantee that is for one piece of code to do the feeding.

        Parameters
        ----------
        module
            The data module, asked for the batch sources.
        prepared
            The prepared dataset, from either route.
        spec
            The validated run specification.

        Returns
        -------
        DataBundle
            One payload per non-empty split, plus signature, state and
            lineage.

        Raises
        ------
        ComponentError
            If a source does not satisfy
            :class:`~rade_qnet.core.contract.source.BatchSource`.
        """
        sources = module.batch_sources(prepared, spec.source, seed=spec.seed)
        splits: dict[str, TensorBatchData] = {}
        for name, source in sources.items():
            if not isinstance(source, BatchSource):
                raise ComponentError(
                    f"the {name!r} source from {type(module).__name__}."
                    f"batch_sources() is a {type(source).__name__}, which does not "
                    f"satisfy BatchSource; a training loop could not consume it"
                )
            splits[name] = TensorBatchData(
                # The source is stored as the loader because a `DatasetSource`
                # satisfies both `BatchSource` and `Iterable[Batch]`.  Keeping
                # one object rather than a bundle plus a parallel source map
                # means the two cannot drift apart.
                loader=source,
                static=source.static,
                n_samples=source.n_samples or 0,
                n_batches=source.steps_per_epoch,
            )

        _LOGGER.info(
            "built data for %s: %s",
            type(self).__name__,
            {name: payload.n_samples for name, payload in splits.items()},
        )
        return DataBundle(
            splits=splits,
            signature=prepared.signature,
            state=prepared.state,
            lineage=prepared.lineage,
            # Carried through rather than dropped: it is the only record of
            # which entities the model was built against, and inference needs
            # it to refuse a request for one that was absent.
            entity_ids=getattr(prepared, "entity_ids", None),
        )

    def signature(self, bundle: DataBundle[object]) -> InputSignature:
        """
        Return the data build's signature unchanged.

        The normal implementation. A model consuming only some of the inputs
        its data build offers should override this and narrow it, so the
        signature saved in the bundle describes what the model actually used
        rather than what was available.

        Parameters
        ----------
        bundle
            The data bundle from :meth:`build_data`.

        Returns
        -------
        InputSignature
            The declared interface.
        """
        return bundle.signature
```

