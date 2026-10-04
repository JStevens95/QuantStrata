"""
The interactive training pipeline: a sibling of the supervised one, not a fork.

``PHASE_7_REINFORCEMENT_LEARNING.md`` §3.1 said this phase should need "no new
pipelines", and the reason it gave is the one that matters: a second lifecycle
means every later improvement to training has to be made twice, and the two
copies diverge at the first deadline.

What is here is therefore deliberately *not* a second lifecycle. It is a
second :class:`~rade_qnet.core.runtime.pipeline.Pipeline` with the same stage
names in the same order, producing the same
:class:`~rade_qnet.core.contract.result.TrainingResult` into the same bundle
layout, resolved through the same
:func:`~.resolve.pipeline_for` mechanism and overridable in the same four
tiers. A reader who knows ``train.py`` knows this file.

Recorded finding: why it is a separate class at all
---------------------------------------------------
Three of the twelve stages genuinely differ, and each difference is a
*different kind of thing* rather than a different parameter:

- ``build_data`` builds a dataset and gets back splits. There is no dataset
  here, and nothing to split: an environment is constructed and acted in.
- ``declare_signature`` produces an ``InputSignature`` with a target.
  ``PolicySignature`` has no target, because a policy has none.
- ``fit`` drives epochs over a mapping of splits. Here there is one unbounded
  source and a step budget.

A single class covering both would branch on ``spec.task`` in those three
stages, and -- this is the part that decides it -- a *model author* overriding
``build_data`` would then have to know which branch they were in. The four
customisation tiers only work if a stage means one thing.

What is deliberately missing
----------------------------
There is no ``evaluate`` stage. Evaluating a policy means running episodes
with exploration turned off, and nothing on this path can turn it off:
``PolicyLearner`` declares ``act``, not a greedy variant of it. A stage that
scored the policy using its *exploring* behaviour would produce a number that
looks like a test metric and is not one, and that is worse than no number --
somebody would compare two runs with it.

So the run's metrics are its episode returns during training, which the
driver already folds into each block's record, and the gap is recorded here
rather than papered over. It closes when the first learner with a meaningful
greedy mode arrives, together with
:class:`~rade_qnet.engines.base.InteractiveEngine`'s third method.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ... import __version__
from ...analysis.reports.base import ReportContext
from ...core.contract.bundle import ModelBundle
from ...core.contract.data import DataLineage
from ...core.contract.result import TrainingResult
from ...core.contract.state import IdentityFittedState
from ...core.runtime.components import get_engine, get_report
from ...core.runtime.errors import ComponentError
from ...core.runtime.hashing import digest_spec
from ...core.runtime.logging import get_logger
from ...core.runtime.pipeline import Pipeline
from ...core.runtime.seeding import seed_everything
from ...engines.base import Engine, InteractiveEngine, ModelHandle
from ...sources.batching.rollout import RolloutSource
from ...storage.bundle import write_bundle

if TYPE_CHECKING:
    from ...analysis.reports.base import Report, ReportOutcome
    from ...core.capability.definition import PolicyDefinition
    from ...core.contract.bundle import SavedBundle
    from ...core.contract.result import FitOutcome
    from ...core.contract.signature import PolicySignature
    from ...core.runtime.context import RunContext
    from ...core.spec.run import ReinforcementRunSpec

__all__ = ["ReinforcePipeline"]

_LOGGER = get_logger(__name__)

#: Bundle version used when no catalog is available to allocate one. Matches
#: ``train.py``: a run without a catalog has nothing to collide with.
_UNVERSIONED = 1

#: Recorded in the lineage's notes so a reader can tell at a glance that a
#: bundle's empty split map is a property of interactive training rather than
#: a failed data build.
_NO_SPLIT_NOTE = (
    "Interactive run: experience was collected from an environment, so there "
    "is no fixed dataset and no split to record."
)


class ReinforcePipeline(Pipeline[TrainingResult]):
    """
    Train one policy by interaction, from a validated spec to a bundle.

    Parameters
    ----------
    context
        Ambient state for the run.
    spec
        The validated interactive run specification.
    definition
        The policy definition, normally resolved from the spec's model name
        through the component registry. Passed in rather than resolved here
        so a test can supply one without registering it globally.

    Attributes
    ----------
    engine
        The resolved engine, available after the ``resolve`` stage.
    handle
        The prepared policy handle, available after ``prepare_hardware``.
    saved
        What ``persist`` wrote, available afterwards -- so a caller that has
        to record where the bundle went does not re-derive the path.
    """

    stages = (
        "resolve",
        "resolve_seed",
        "build_environment",
        "declare_signature",
        "build_policy",
        "materialise",
        "prepare_hardware",
        "collect",
        "fit",
        "persist",
        "report",
    )

    def __init__(
        self,
        context: RunContext,
        spec: ReinforcementRunSpec,
        definition: PolicyDefinition,
    ) -> None:
        """
        Store the spec and the definition.

        Parameters
        ----------
        context
            Ambient state for the run.
        spec
            The validated interactive run specification.
        definition
            The policy definition to train.
        """
        super().__init__(context)
        self.spec = spec
        self.definition = definition
        self.engine: Engine | None = None
        self.handle: ModelHandle | None = None
        self.environment: object | None = None
        self.source: RolloutSource | None = None
        self.saved: SavedBundle | None = None

    def run(self) -> TrainingResult:
        """
        Execute the stage sequence.

        Returns
        -------
        TrainingResult
            The block history and the seed that produced it.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised.
        """
        self.step("resolve", self.resolve)
        seed = self.step("resolve_seed", self.resolve_seed)
        environment = self.step("build_environment", lambda: self.build_environment(self.spec))
        signature = self.step("declare_signature", lambda: self.declare_signature(environment))
        policy = self.step("build_policy", lambda: self.build_policy(self.spec, signature))
        materialised = self.step("materialise", lambda: self.materialise(policy, signature))
        handle = self.step("prepare_hardware", lambda: self.prepare_hardware(materialised))
        source = self.step("collect", lambda: self.collect(environment, handle))
        outcome = self.step("fit", lambda: self.fit(handle, source))

        result = TrainingResult(fit=outcome, evaluations={}, seed=seed)
        bundle = self.step("persist", lambda: self.persist(handle, signature, result))
        self.step("report", lambda: self.report(bundle))

        if self.saved is None:
            return result
        return result.model_copy(update={"bundle_directory": str(self.saved.directory)})

    def resolve(self) -> None:
        """
        Look up every component the run names, before anything expensive runs.

        One check more than the supervised pipeline makes: the engine must
        also satisfy :class:`~rade_qnet.engines.base.InteractiveEngine`.
        Asked here rather than at the first call, so that configuring an
        interactive run against the gradient-boosted-tree engine fails in
        milliseconds with a message about the engine -- not four stages later
        with a missing attribute.

        Raises
        ------
        ComponentError
            If the engine or a report name is not registered, or if the
            engine cannot train a policy.
        """
        engine_name = self.spec.training.engine
        engine = get_engine(engine_name)()
        if not isinstance(engine, Engine):
            raise ComponentError(
                f"the engine registered as {engine_name!r} is a "
                f"{type(engine).__name__}, which does not satisfy the Engine "
                f"protocol; it cannot drive a training run"
            )
        if not isinstance(engine, InteractiveEngine):
            raise ComponentError(
                f"the engine registered as {engine_name!r} cannot train a "
                f"policy: it has no materialise_policy() and fit_policy(). "
                f"Interactive training needs an engine that opts into the "
                f"InteractiveEngine capability"
            )
        self.engine = engine

        names = self.report_names()
        for name in names:
            get_report(name)

        _LOGGER.info("resolved interactive engine %r and report(s) %s", engine_name, list(names))

    def report_names(self) -> tuple[str, ...]:
        """
        Return the reports to render, in order.

        Returns
        -------
        tuple of str
            The spec's selection, unchanged. A hook for the same reason it is
            one in the supervised pipeline: a policy whose diagnostics only
            make sense for that policy can contribute them without the user
            having to know their names.
        """
        return tuple(self.spec.reports.enabled)

    def resolve_seed(self) -> int:
        """
        Apply the run's seed and return what was actually applied.

        Returns
        -------
        int
            The seed applied. Returned rather than echoed back, because a
            job-set member's seed is derived from the context, and recording
            the derived value is what lets one failed member be re-run alone.
        """
        seed = seed_everything(self.context.seed, determinism=self.spec.hardware.determinism)
        _LOGGER.info("applied seed %d", seed)
        return seed

    def build_environment(self, spec: ReinforcementRunSpec) -> object:
        """
        Construct the environment, by delegating to the policy definition.

        A thin delegation, for the same reason ``build_data`` is one: the
        framework does not know how to build any particular environment, and
        a base implementation that tried would be overridden by every model.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        object
            An environment satisfying
            :class:`~rade_qnet.sources.environment.protocol.Environment`.
            Not checked here -- :class:`RolloutSource` refuses a
            non-environment by name in the ``collect`` stage, and one check
            in one place is better than two that can disagree.
        """
        environment = self.definition.build_environment(spec)
        self.environment = environment
        return environment

    def declare_signature(self, environment: object) -> PolicySignature:
        """
        Read the spaces the policy will act in.

        The counterpart of the supervised ``declare_signature``, and it
        carries the same weight: this is the value saved into the bundle, and
        the only thing a later process has to rebuild the policy from.

        Parameters
        ----------
        environment
            The environment from :meth:`build_environment`.

        Returns
        -------
        PolicySignature
            The observation and action spaces.
        """
        return self.definition.signature(environment)

    def build_policy(self, spec: ReinforcementRunSpec, signature: PolicySignature) -> object:
        """
        Construct the untrained policy.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared spaces.

        Returns
        -------
        object
            An engine-native untrained policy.
        """
        return self.definition.build_policy(spec, signature)

    def materialise(self, policy: object, signature: PolicySignature) -> object:
        """
        Give a lazily shaped policy its parameters, before anything wraps it.

        A separate stage for exactly the reason it is one in the supervised
        pipeline: an optimiser built over unmaterialised parameters tracks
        nothing, and the run then appears to train while changing no weights.
        The ordering is visible in :attr:`stages` so it cannot be reordered
        by accident.

        Parameters
        ----------
        policy
            The untrained policy.
        signature
            The declared spaces, from which the dummy observation is built.

        Returns
        -------
        object
            The policy, with its parameters now real.
        """
        return self._interactive().materialise_policy(policy, signature)

    def prepare_hardware(self, policy: object) -> ModelHandle:
        """
        Place the policy on its device, wrap it, and build the optimiser.

        Uses the ordinary :meth:`~rade_qnet.engines.base.Engine.prepare`,
        because placing a module on a device and building an optimiser over
        it is identical work whether the module predicts or acts. No
        ``static`` argument is passed: an environment delivers everything
        through its observations.

        Parameters
        ----------
        policy
            The materialised policy.

        Returns
        -------
        ModelHandle
            The policy, its device, its precision, and its optimiser.
        """
        handle = self._engine().prepare(
            policy,
            hardware=self.spec.hardware,
            training=self.spec.training,
        )
        self.handle = handle
        _LOGGER.info("prepared policy: %s", handle.describe())
        return handle

    def collect(self, environment: object, handle: ModelHandle) -> RolloutSource:
        """
        Wrap the environment as the source the driver consumes.

        A stage of its own, and this is the point at which the framework's
        central claim becomes true: from here down, the loop is handed a
        :class:`~rade_qnet.core.contract.source.BatchSource` and does not
        know an environment is behind it.

        Parameters
        ----------
        environment
            The environment from :meth:`build_environment`.
        handle
            The prepared policy, needed because the source's action selector
            forwards to the engine's learner acting on it.

        Returns
        -------
        RolloutSource
            An unbounded source of on-policy experience.

        Raises
        ------
        ContractError
            If the definition's ``build_environment`` did not return an
            environment.
        """
        del handle  # Acted on by the learner the engine resolves, not here.
        training = self.spec.training
        source = RolloutSource(
            environment,
            # Left unbound on purpose. The action selector is the learner
            # acting on the policy, and the learner cannot be constructed
            # until the source exists -- it is built from the source's own
            # policy signature. So `fit_policy` binds it, and a source driven
            # before then refuses rather than acting arbitrarily.
            batch_size=training.steps_per_update,
            seed=self.context.seed,
        )
        self.source = source
        _LOGGER.info("collecting experience: %s", source.describe())
        return source

    def fit(self, handle: ModelHandle, source: RolloutSource) -> FitOutcome:
        """
        Train the policy against the spec's step budget.

        Parameters
        ----------
        handle
            The prepared policy.
        source
            The experience source.

        Returns
        -------
        FitOutcome
            The block history, in the same shape a supervised run produces.
        """
        return self._interactive().fit_policy(handle, source, self.spec.training)

    def persist(
        self,
        handle: ModelHandle,
        signature: PolicySignature,
        result: TrainingResult,
    ) -> ModelBundle:
        """
        Assemble the bundle and write it.

        The same bundle layout a supervised run writes, through the same
        function, with the engine's weight serialiser passed as a callback --
        the seam that keeps ``storage`` free of any engine import.

        Parameters
        ----------
        handle
            The trained policy.
        signature
            The declared spaces, saved as the bundle's signature.
        result
            The block history and seed.

        Returns
        -------
        ModelBundle
            The bundle that was written.
        """
        engine = self._engine()
        lineage = self._lineage()
        bundle = ModelBundle(
            model=handle.unwrapped,
            # A policy fits nothing beyond its parameters: there is no scaler
            # to invert and no encoder to reapply, because nothing stood
            # between the environment and the network. Named explicitly
            # rather than defaulted, which is the whole point of
            # `IdentityFittedState`.
            state=IdentityFittedState(),
            signature=signature,
            spec=self.spec,
            lineage=lineage,
            result=result,
        )

        saved = write_bundle(
            bundle,
            root=self.context.bundles_directory,
            version=self._next_version(),
            framework_version=lineage.framework_version,
            engine=self.spec.training.engine,
            model_name=self.spec.model.name,
            write_weights=lambda path: engine.save_weights(handle, path),
            job_id=self.context.job_id,
            tags=self.spec.tags,
        )
        _LOGGER.info("wrote bundle to %s", saved.directory)
        self.saved = saved

        if self.context.catalog is not None and saved.manifest is not None:
            self.context.catalog.record(saved.manifest, location=saved.directory)
        return bundle

    def _lineage(self) -> DataLineage:
        """
        Record where the experience came from.

        The same contract a supervised run records, filled in honestly rather
        than reshaped. ``split_indices`` is empty and ``n_scenarios`` counts
        the transitions collected, because that is what an interactive run
        has instead of a split dataset -- and a note says so, so an empty
        split map is not mistaken for a failed data build.

        Returns
        -------
        DataLineage
            The run's lineage.
        """
        episodes = 0 if self.source is None else self.source.describe()["episodes_finished"]
        return DataLineage(
            # The environment reference is what identifies this experience,
            # in the role a file's content hash plays for a dataset: it is
            # the thing that, if it changed, would make two runs
            # incomparable. Digested rather than stored verbatim so the field
            # keeps the fixed width every other fingerprint has.
            source_fingerprint=digest_spec(self.spec.environment),
            spec_digest=digest_spec(self.spec),
            split_indices={},
            n_scenarios=self.spec.training.total_steps,
            framework_version=__version__,
            created_at=datetime.now(UTC),
            notes={"interaction": _NO_SPLIT_NOTE, "episodes": str(episodes)},
        )

    def _next_version(self) -> int:
        """
        Allocate this bundle's version number.

        Returns
        -------
        int
            The next version from the catalog, or :data:`_UNVERSIONED`.
        """
        if self.context.catalog is None:
            return _UNVERSIONED
        return self.context.catalog.next_version(self.spec.model.name, job_id=self.context.job_id)

    def report(self, bundle: ModelBundle) -> dict[str, ReportOutcome]:
        """
        Render every enabled report.

        Parameters
        ----------
        bundle
            The persisted bundle.

        Returns
        -------
        dict
            One outcome per enabled report.

        Raises
        ------
        ComponentError
            If a report failed and the spec set ``reports.fail_fast``.
        """
        directory = self.context.reports_directory
        context = ReportContext(
            bundle=bundle,
            directory=directory,
            figure_format=self.spec.reports.figure_format,
            figure_dpi=self.spec.reports.figure_dpi,
        )

        outcomes: dict[str, ReportOutcome] = {}
        for name in self.report_names():
            report: Report = get_report(name)()
            outcomes[name] = report.render_safely(context, self.context)

        failed = {
            name: outcome.skipped_reason
            for name, outcome in outcomes.items()
            if not outcome.succeeded
        }
        if failed and self.spec.reports.fail_fast:
            raise ComponentError(
                f"report(s) {sorted(failed)} failed and reports.fail_fast is set, "
                f"so the run does not complete: {failed}"
            )
        return outcomes

    def _engine(self) -> Engine:
        """
        Return the resolved engine.

        Returns
        -------
        Engine
            The engine resolved by :meth:`resolve`.

        Raises
        ------
        ComponentError
            If called before :meth:`resolve`.
        """
        if self.engine is None:
            raise ComponentError(
                "no engine has been resolved; ReinforcePipeline.resolve must run "
                "before any stage that needs the engine. An overridden run() "
                "that omits the 'resolve' stage produces this"
            )
        return self.engine

    def _interactive(self) -> InteractiveEngine:
        """
        Return the resolved engine, narrowed to the interactive capability.

        Returns
        -------
        InteractiveEngine
            The same engine. :meth:`resolve` has already established that it
            opts into the capability, so this narrows rather than re-checks.
        """
        engine = self._engine()
        assert isinstance(engine, InteractiveEngine)
        return engine
