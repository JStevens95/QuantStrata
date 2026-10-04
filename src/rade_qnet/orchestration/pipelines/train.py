"""
The training pipeline: the canonical stage sequence, end to end.

The one place the whole framework is threaded together. Everything else in
``rade_qnet`` is a contract, a component or a helper; this is where a spec
becomes a trained, scored, documented, persisted model.

The stage order is the design
-----------------------------
Four orderings here are load-bearing, and each exists because the opposite
order has failed in practice. The full diagram is in
`ARCHITECTURE.md` §5 (``../ARCHITECTURE.md#5-stage-contracts-and-the-train-pipeline``).

- **Resolution precedes the data build.** A misspelled engine or report name
  is a configuration error, and finding it after a twenty-minute data build
  wastes the twenty minutes. Every name the run needs is looked up first.
- **Seeding precedes the data build**, because a data build that samples -- a
  subgraph, a negative set, a shuffled split -- is part of what a seed has to
  reproduce. Seeding afterwards makes a run reproducible in its training and
  not in its data.
- **Materialisation precedes hardware preparation.** A model with lazily
  shaped parameters has no parameters until it has seen one batch. Handing
  that model to an optimiser or a distributed wrapper gives either an empty
  parameter group -- silent non-training -- or a crash inside the distributed
  library. This is defect 6, and the fix is visible in the stage list rather
  than buried in the engine.
- **Persistence is last, and nothing before it writes to the catalog.** A run
  that fails at any point leaves no half-registered bundle behind.

Reports run after persistence, not before
-----------------------------------------
A report reads a finished bundle, so it cannot run before the bundle exists;
and because a report never fails a run, running it last means a report fault
cannot cost a trained model. The ordering follows from the two properties
rather than being a separate decision.

The four customisation tiers, concretely
----------------------------------------
1. **Spec only.** Write a model definition, write YAML, run this class.
2. **Add observation.** Attach a hook, or enable a report in the spec.
3. **Override one stage.** Subclass, replace ``build_data`` or ``evaluate``,
   inherit the rest -- with their timing, logging and error attribution.
4. **Override** ``run``. Change the sequence. Still gets the run-level
   bookkeeping from :meth:`~rade_qnet.core.runtime.pipeline.Pipeline.execute`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...analysis.metrics.quality import quality_metrics
from ...analysis.reports.base import ReportContext
from ...core.capability.definition import PredictorDefinition
from ...core.contract.bundle import ModelBundle
from ...core.contract.data import (
    DataBundle,
)
from ...core.contract.result import EvalResult, FitOutcome, TrainingResult
from ...core.contract.signature import InputSignature
from ...core.runtime.components import get_engine, get_report
from ...core.runtime.errors import ComponentError
from ...core.runtime.logging import get_logger
from ...core.runtime.pipeline import Pipeline
from ...core.runtime.seeding import seed_everything
from ...engines.base import Engine, ModelHandle
from ...storage.bundle import write_bundle
from .scoring import (
    TRAIN_SPLIT,
    feature_matrix,
    score_splits,
    source_for,
    static_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...analysis.reports.base import Report, ReportOutcome
    from ...core.contract.bundle import SavedBundle
    from ...core.contract.data import DataLineage
    from ...core.runtime.context import RunContext
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["TrainPipeline"]

_LOGGER = get_logger(__name__)

#: Bundle version used when no catalog is available to allocate one. A run
#: without a catalog has nothing to collide with, so the first version is
#: always correct for it.
_UNVERSIONED = 1


class TrainPipeline(Pipeline[TrainingResult]):
    """
    Train one model, from a validated spec to a persisted bundle.

    Parameters
    ----------
    context
        Ambient state for the run.
    spec
        The validated run specification.
    definition
        The model definition, normally resolved from the spec's model name
        through the component registry. Passed in rather than resolved here so
        a test can supply a definition without registering it globally.

    Attributes
    ----------
    engine
        The resolved engine, available after the ``resolve`` stage. Held on
        the instance rather than threaded through every signature because
        ``persist`` needs it for the weight serialiser and ``evaluate`` needs
        it for the forward pass, and passing it through four stages to reach
        them would obscure the sequence.
    handle
        The prepared model handle, available after the ``prepare_hardware``
        stage.
    """

    stages = (
        "resolve",
        "resolve_seed",
        "build_data",
        "declare_signature",
        "build_model",
        "materialise",
        "prepare_hardware",
        "fit",
        "evaluate",
        "persist",
        "report",
    )

    def __init__(
        self,
        context: RunContext,
        spec: SupervisedRunSpec,
        definition: PredictorDefinition,
    ) -> None:
        """
        Store the spec and the definition.

        Parameters
        ----------
        context
            Ambient state for the run.
        spec
            The validated run specification.
        definition
            The model definition to train.
        """
        super().__init__(context)
        self.spec = spec
        self.definition = definition
        self.engine: Engine | None = None
        self.handle: ModelHandle | None = None
        # What `persist` wrote, available afterwards. The run returns metrics
        # and history, which is the right return type -- but a caller that
        # has to record *where* the bundle went, as a job set does for every
        # row of its manifest, would otherwise have to re-derive the path
        # from the version-allocation rules or query the catalog back. Both
        # are reconstructions of something this object already knows.
        self.saved: SavedBundle | None = None

    def run(self) -> TrainingResult:
        """
        Execute the stage sequence.

        Each stage goes through
        :meth:`~rade_qnet.core.runtime.pipeline.Pipeline.step`, so each is timed,
        logged, reported to hooks, and -- on failure -- wrapped in a
        :class:`~rade_qnet.core.runtime.errors.StageError` naming the stage.

        Returns
        -------
        TrainingResult
            Metrics and training history.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised.
        """
        self.step("resolve", self.resolve)
        seed = self.step("resolve_seed", self.resolve_seed)
        data = self.step("build_data", lambda: self.build_data(self.spec))
        signature = self.step("declare_signature", lambda: self.declare_signature(data))
        model = self.step("build_model", lambda: self.build_model(self.spec, signature))
        materialised = self.step("materialise", lambda: self.materialise(model, signature))
        handle = self.step("prepare_hardware", lambda: self.prepare_hardware(materialised, data))
        outcome = self.step("fit", lambda: self.fit(handle, data))
        evaluations = self.step("evaluate", lambda: self.evaluate(handle, data))

        result = TrainingResult(fit=outcome, evaluations=evaluations, seed=seed)
        bundle = self.step("persist", lambda: self.persist(handle, data, signature, result))
        self.step("report", lambda: self.report(bundle))
        # Stamped after persisting rather than before, because until the
        # bundle is written there is no directory to name. The result handed
        # to `persist` deliberately does not carry it: a result that claimed
        # a location before anything was saved there would be wrong in the
        # one case -- a failed write -- where it is read most carefully.
        if self.saved is None:
            return result
        return result.model_copy(update={"bundle_directory": str(self.saved.directory)})

    def resolve(self) -> None:
        """
        Look up every component the run names, before anything expensive runs.

        Resolves the engine and each enabled report. Nothing is executed and
        nothing is built; the point is that a name which does not resolve
        fails here, in a stage that takes milliseconds, rather than after the
        data build.

        Raises
        ------
        ComponentError
            If the engine or a report name is not registered, or if the
            resolved engine does not satisfy the
            :class:`~rade_qnet.engines.base.Engine` protocol.
        """
        engine_name = self.spec.training.engine
        engine = get_engine(engine_name)()
        if not isinstance(engine, Engine):
            raise ComponentError(
                f"the engine registered as {engine_name!r} is a "
                f"{type(engine).__name__}, which does not satisfy the Engine "
                f"protocol; it cannot drive a training run"
            )
        self.engine = engine

        # Resolved and discarded.  Instantiating each report now is what turns
        # a misspelled report name into an immediate failure instead of a
        # warning logged after training finished.
        names = self.report_names()
        for name in names:
            get_report(name)

        _LOGGER.info(
            "resolved engine %r (%s) and report(s) %s",
            engine_name,
            engine.capabilities().describe(),
            list(names),
        )

    def report_names(self) -> tuple[str, ...]:
        """
        Return the reports to render, in order.

        A hook rather than a direct read of the spec, so a model whose
        diagnostics only make sense for that model can contribute them
        without the user having to know their names. A graph model's
        neighbourhood diagnostics are not optional extras a user should
        have to remember to enable -- they are how you tell whether the
        graph is any good.

        Overriding this is the lightest possible pipeline override: the
        stage sequence is untouched, so a model adding reports is not also
        quietly taking ownership of how training runs.

        Returns
        -------
        tuple of str
            Registered report names. The spec's selection, unchanged.
        """
        return tuple(self.spec.reports.enabled)

    def resolve_seed(self) -> int:
        """
        Apply the run's seed and return what was actually applied.

        Returns what was applied rather than what was requested, because the
        two differ for a job-set member: the context derives a per-job seed so
        that two members are independent and a single failed job can be
        re-run alone and reproduce. Recording the derived value in the result
        is what makes that re-run possible.

        Returns
        -------
        int
            The seed applied.
        """
        seed = seed_everything(self.context.seed, determinism=self.spec.hardware.determinism)
        _LOGGER.info("applied seed %d with determinism %r", seed, self.spec.hardware.determinism)
        return seed

    def build_data(self, spec: SupervisedRunSpec) -> DataBundle[object]:
        """
        Build the dataset by delegating to the model definition.

        A thin delegation by design: the framework does not know how to build
        any particular model's data, and a base implementation that tried to
        would have to be overridden by every non-trivial model.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        DataBundle
            Splits, signature, fitted state and lineage. A bundle with no
            training split is refused by
            :class:`~rade_qnet.core.contract.data.DataBundle` itself, so there
            is nothing for this stage to add -- a second check here would be
            unreachable code that reads like a safeguard.
        """
        return self.definition.build_data(spec)

    def declare_signature(self, bundle: DataBundle[object]) -> InputSignature:
        """
        Read what the data build produced, and check the model can use it.

        Two halves, and the second is the one that earns the stage its
        place. The signature says what the build made; ``check_signature``
        compares it against what the model declared it consumes, which is
        the only point in a run where information flows model to data.

        It happens here rather than inside ``build_model`` because a
        mismatch is a fault in the *pairing* of a model and a source, not
        in either one -- and because failing before anything is constructed
        means the message is about the inputs rather than about whatever
        shape error they eventually caused.

        Parameters
        ----------
        bundle
            The data bundle from :meth:`build_data`.

        Returns
        -------
        InputSignature
            The declared interface, recorded in the saved bundle.

        Raises
        ------
        ContractError
            If the build did not produce what the model consumes.
        """
        signature = self.definition.signature(bundle)
        self.definition.check_signature(signature)
        return signature

    def build_model(self, spec: SupervisedRunSpec, signature: InputSignature) -> object:
        """
        Construct the untrained model.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared interface.

        Returns
        -------
        object
            An engine-native untrained model.
        """
        return self.definition.build_model(spec, signature)

    def materialise(self, model: object, signature: InputSignature) -> object:
        """
        Give a lazily shaped model its parameters, before anything wraps it.

        A separate stage rather than a call inside the engine's ``prepare``,
        so that the ordering which fixes defect 6 is visible in
        :attr:`stages` and cannot be reordered by accident. Engines whose
        models are never lazy return the model unchanged, which costs one
        no-op stage and keeps the sequence the same for every backend.

        Parameters
        ----------
        model
            The untrained model from :meth:`build_model`.
        signature
            The declared interface, from which the dummy batch is synthesised.
            This is why the signature exists: the dummy forward needs exact
            shapes and dtypes with no data present.

        Returns
        -------
        object
            The same model, with its parameters now real.
        """
        return self._engine().materialise(model, signature)

    def prepare_hardware(self, model: object, data: DataBundle[object]) -> ModelHandle:
        """
        Place the model on its device, wrap it, and build the optimiser.

        Parameters
        ----------
        model
            The materialised model.
        data
            The data bundle, for the training split's static inputs. Taken
            from the training split because static inputs are constant by
            definition -- an adjacency matrix that differed between train and
            test would not be static -- and uploading them once here is what
            removes the per-sample collation of defect 4.

        Returns
        -------
        ModelHandle
            The model, its device, its precision, and whatever state the
            engine needs to train it.
        """
        handle = self._engine().prepare(
            model,
            hardware=self.spec.hardware,
            training=self.spec.training,
            static=static_inputs(data),
        )
        self.handle = handle
        _LOGGER.info("prepared model: %s", handle.describe())
        return handle

    def fit(self, handle: ModelHandle, data: DataBundle[object]) -> FitOutcome:
        """
        Train the model.

        Parameters
        ----------
        handle
            The prepared model from :meth:`prepare_hardware`.
        data
            The data bundle from :meth:`build_data`.

        Returns
        -------
        FitOutcome
            Training history and the best epoch.
        """
        sources = {
            name: source_for(data, name)
            for name in (TRAIN_SPLIT, "validation")
            if name in data.splits
        }
        return self._engine().fit(handle, sources, self.spec.training)

    def evaluate(self, handle: ModelHandle, data: DataBundle[object]) -> dict[str, EvalResult]:
        """
        Score the fitted model on every available split.

        Delegates to :func:`~.scoring.score_splits`, which is also what
        re-evaluating a saved bundle calls. One implementation rather than
        two, so that a bundle's recorded metrics and its re-computed ones
        cannot drift apart -- see that module's docstring.

        Parameters
        ----------
        handle
            The fitted model.
        data
            The data bundle.

        Returns
        -------
        dict
            Metrics per split, each against a naive baseline.
        """
        engine = self._engine()
        return score_splits(data, lambda source: engine.predict(handle, source))

    def persist(
        self,
        handle: ModelHandle,
        data: DataBundle[object],
        signature: InputSignature,
        result: TrainingResult,
    ) -> ModelBundle:
        """
        Assemble the bundle and write it.

        The engine's weight serialiser is passed as the ``write_weights``
        callback rather than being called here, which is the seam that keeps
        ``storage`` free of any engine import: the storage layer decides
        *where* and *when* weights are written, the engine decides *how*.

        Parameters
        ----------
        handle
            The fitted model.
        data
            The data bundle, for its fitted state and lineage.
        signature
            The declared interface.
        result
            Metrics and history.

        Returns
        -------
        ModelBundle
            The bundle that was written.
        """
        engine = self._engine()
        lineage = self._lineage_with_quality(data)
        bundle = ModelBundle(
            model=handle.unwrapped,
            state=data.state,
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

        # Last, and only on success: a run that failed earlier leaves nothing
        # registered, so the catalog never advertises a bundle that is not
        # there.
        if self.context.catalog is not None and saved.manifest is not None:
            self.context.catalog.record(saved.manifest, location=saved.directory)
        return bundle

    def _next_version(self) -> int:
        """
        Allocate this bundle's version number.

        Returns
        -------
        int
            The next version for this model from the catalog, or
            :data:`_UNVERSIONED` when the run has no catalog.
        """
        if self.context.catalog is None:
            return _UNVERSIONED
        return self.context.catalog.next_version(self.spec.model.name, job_id=self.context.job_id)

    def report(self, bundle: ModelBundle) -> dict[str, ReportOutcome]:
        """
        Render every enabled report.

        Each goes through :meth:`~rade_qnet.analysis.reports.base.Report.render_safely`,
        so a report that fails is logged and skipped. A completed, scored,
        persisted training run is not discarded because a figure could not be
        drawn -- unless the spec sets ``reports.fail_fast``, which exists for
        the case where the report *is* the deliverable.

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
            If a report failed and the spec set ``reports.fail_fast``. Wrapped
            in a ``StageError`` by the step runner, like any stage failure.
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
        _LOGGER.info(
            "rendered %d of %d report(s) into %s",
            len(outcomes) - len(failed),
            len(outcomes),
            directory,
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
            If called before :meth:`resolve`, which only happens in a subclass
            that overrode :meth:`run` and dropped the stage. Named explicitly
            rather than left as an ``AttributeError`` on ``None``, because the
            fix -- restore the stage -- is not obvious from the latter.
        """
        if self.engine is None:
            raise ComponentError(
                "no engine has been resolved; TrainPipeline.resolve must run "
                "before any stage that needs the engine. An overridden run() "
                "that omits the 'resolve' stage produces this"
            )
        return self.engine

    @classmethod
    def _static_inputs(cls, data: DataBundle[object]) -> Mapping[str, object]:
        """
        Return the training split's static inputs.

        Parameters
        ----------
        data
            The data bundle.

        Returns
        -------
        Mapping
            Input name to tensor. Empty for a model with no static inputs,
            which is most of them.
        """
        return cls._source_for(data, TRAIN_SPLIT).static

    def _lineage_with_quality(self, data: DataBundle[object]) -> DataLineage:
        """
        Return the data lineage with quality metrics attached.

        Computed here rather than in the data module because ``sources`` may
        not import ``analysis`` -- orchestration is the first layer that can
        see both. Recording them at training time is the point: once the run
        is over the dataset may not be reconstructable, so numbers describing
        it are captured in the bundle or lost.

        Parameters
        ----------
        data
            The data bundle.

        Returns
        -------
        DataLineage
            The lineage, with ``quality`` populated where it could be computed.
        """
        if data.lineage.quality:
            return data.lineage

        features = feature_matrix(data)
        if features is None:
            return data.lineage
        return data.lineage.model_copy(update={"quality": quality_metrics(features)})
