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
:mod:`rade_qnet.core.authoring.capabilities`.

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
from ..lifecycle.components import model

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
