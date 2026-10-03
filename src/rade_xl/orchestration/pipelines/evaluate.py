"""
Scoring a saved model, on the data it was trained against or on newer data.

Training already scores a model at the end of its run and records the result
in the bundle. This pipeline exists for the cases training cannot cover:

- **Reproducing a number.** Someone asks where a recorded metric came from,
  and the only honest answer is to recompute it.
- **Re-scoring on new data.** The model has not changed; the world has. A
  monthly re-score against fresh observations is how a quiet degradation is
  noticed before it costs anything.
- **Comparing models on one split.** Two bundles trained at different times
  cannot be compared on their own recorded metrics unless they were scored
  against the same rows.

All three come down to the same requirement, and it is the one this phase is
built around: a re-loaded model must see its inputs exactly as it saw them
during training. The scoring itself is therefore *not* written here -- it is
:func:`~.scoring.score_splits`, the same function training calls, for the
reason in that module's docstring. What is written here is everything around
it: opening the bundle, rebuilding the data from the saved state and split,
and putting the weights back into a model that can run a forward pass.

Why this is not a `score: bool` on something else
-------------------------------------------------
Evaluation and inference share their entire reload path and differ in one
thing: whether targets exist. It is tempting to make them one pipeline with a
flag. They are kept apart because the flag would not stay in one place -- it
would reappear as an ``if`` in every stage downstream of it, and the two have
genuinely different outputs, different failure modes and different readers.
What they share is shared as *functions* they both call, which cannot drift,
rather than as a base class they both inherit, which can.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.contract.result import EvalResult, EvaluationResult
from ...core.runtime.components import get_engine
from ...core.runtime.errors import ContractError
from ...core.runtime.logging import get_logger
from ...core.runtime.pipeline import Pipeline
from ...engines.base import Engine
from .reload import load_bundle
from .scoring import EVALUATED_SPLITS, score_splits, static_inputs

if TYPE_CHECKING:
    from pathlib import Path

    from ...core.contract.data import DataBundle
    from ...core.runtime.context import RunContext
    from ...core.spec.data import SourceSpec
    from ...engines.base import ModelHandle
    from .reload import LoadedBundle

__all__ = ["EvaluatePipeline"]

_LOGGER = get_logger(__name__)


class EvaluatePipeline(Pipeline[EvaluationResult]):
    """
    Score a saved bundle, on its own data or on newer data.

    Parameters
    ----------
    context
        Ambient state for the run.
    directory
        The bundle to evaluate.
    source
        A source specification to score against. Defaults to ``None``,
        meaning the bundle's own -- which is the reproduction case. Supplying
        a different one re-scores the same model on other data, and the
        result records that it did.
    splits
        Which splits to score. Defaults to all of them. A re-score against
        new data usually wants only ``test``, since on fresh observations the
        training split is no longer a meaningful category.
    verify
        Whether to re-hash the bundle's files against its manifest.

    Attributes
    ----------
    loaded
        The opened bundle, available after the ``load`` stage.
    handle
        The restored model, available after ``restore``.
    """

    stages = (
        "load",
        "rebuild_data",
        "restore",
        "score",
        "report",
    )

    def __init__(
        self,
        context: RunContext,
        directory: Path,
        *,
        source: SourceSpec | None = None,
        splits: tuple[str, ...] = EVALUATED_SPLITS,
        verify: bool = True,
    ) -> None:
        """
        Store what to evaluate and against what.

        Parameters
        ----------
        context
            Ambient state for the run.
        directory
            The bundle directory.
        source
            Source specification to score against, or ``None`` for the
            bundle's own.
        splits
            Splits to score, in report order.
        verify
            Whether to verify the bundle against its manifest.
        """
        super().__init__(context)
        self.directory = directory
        self.source = source
        self.splits = splits
        self.verify = verify
        self.loaded: LoadedBundle | None = None
        self.handle: ModelHandle | None = None

    def run(self) -> EvaluationResult:
        """
        Execute the stage sequence.

        Returns
        -------
        EvaluationResult
            Metrics per split, with the provenance needed to say what was
            scored against what.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised, naming the stage.
        """
        loaded = self.step("load", self.load)
        data = self.step("rebuild_data", lambda: self.rebuild_data(loaded))
        handle = self.step("restore", lambda: self.restore(loaded, data))
        evaluations = self.step("score", lambda: self.score(loaded, handle, data))
        return self.step("report", lambda: self.report(loaded, data, evaluations))

    def load(self) -> LoadedBundle:
        """
        Open the bundle and resolve everything needed to run it again.

        Returns
        -------
        LoadedBundle
            The spec, definition, fitted state, lineage and signature.
        """
        loaded = load_bundle(self.directory, verify=self.verify)
        self.loaded = loaded
        return loaded

    def rebuild_data(self, loaded: LoadedBundle) -> DataBundle[object]:
        """
        Rebuild the dataset using the bundle's saved state and split.

        The stage that makes the whole pipeline trustworthy, and the one
        with the least code in it. Everything it must *not* do -- re-derive
        the split, re-fit the scalers -- is prevented upstream in
        :meth:`~rade_xl.sources.dataset.module.DataModule.rebuild`, so there
        is no path through this pipeline that can reach the naive behaviour
        by accident.

        Parameters
        ----------
        loaded
            The opened bundle.

        Returns
        -------
        DataBundle
            The same payload shape training consumed.
        """
        spec = loaded.spec
        if self.source is not None:
            spec = spec.model_copy(update={"source": self.source})

        return loaded.definition.rebuild_data(
            spec, state=loaded.state, lineage=loaded.lineage
        )

    def restore(self, loaded: LoadedBundle, data: DataBundle[object]) -> ModelHandle:
        """
        Rebuild the architecture, load the saved weights, place it on a device.

        The architecture is rebuilt from the spec rather than deserialised,
        for the reason the manifest stores a model name rather than a class
        path: a pickled module is a version-locked artefact, while a spec
        plus the current code is readable by anything that can still build
        the model.

        The signature is taken from the *bundle*, not from the rebuilt data.
        They should agree, and :meth:`score` checks that they do -- but if
        they disagree, the weights were shaped by the saved one, and building
        against the other would produce a shape error at best and a silently
        mismatched model at worst.

        Parameters
        ----------
        loaded
            The opened bundle.
        data
            The rebuilt data, for the static inputs.

        Returns
        -------
        ModelHandle
            The restored model, ready for a forward pass.

        Raises
        ------
        ContractError
            If the engine the bundle names is not registered here.
        """
        engine = self._engine(loaded)
        # Re-checked on the reload path, not just at training time. The
        # signature here came off disk, and the model code around it has
        # moved on since: a bundle trained against a version that consumed
        # different inputs would otherwise be rebuilt silently and score
        # against tensors the current forward pass does not expect.
        loaded.definition.check_signature(loaded.signature)
        model = loaded.definition.build_model(loaded.spec, loaded.signature)
        model = engine.materialise(model, loaded.signature)
        model = engine.load_weights(model, loaded.saved.weights_path)

        handle = engine.prepare(
            model,
            hardware=loaded.spec.hardware,
            training=loaded.spec.training,
            static=static_inputs(data),
        )
        self.handle = handle
        _LOGGER.info("restored model: %s", handle.describe())
        return handle

    def score(
        self,
        loaded: LoadedBundle,
        handle: ModelHandle,
        data: DataBundle[object],
    ) -> dict[str, EvalResult]:
        """
        Score the restored model, through the same function training uses.

        Parameters
        ----------
        loaded
            The opened bundle, for the signature check.
        handle
            The restored model.
        data
            The rebuilt data.

        Returns
        -------
        dict
            Split name to :class:`~rade_xl.core.contract.result.EvalResult`.

        Raises
        ------
        ContractError
            If the rebuilt data no longer presents the interface the weights
            were shaped against.
        """
        self._check_signature(loaded, data)
        engine = self._engine(loaded)
        return score_splits(
            data,
            lambda source: engine.predict(handle, source),
            splits=self.splits,
        )

    def report(
        self,
        loaded: LoadedBundle,
        data: DataBundle[object],
        evaluations: dict[str, EvalResult],
    ) -> EvaluationResult:
        """
        Assemble the result, recording what was scored against what.

        The provenance is the point. A metric without the bundle version and
        the source fingerprint behind it cannot be compared to another metric
        with any confidence, and a re-score against changed data that does
        not say so is the quiet failure this phase exists to prevent.

        Parameters
        ----------
        loaded
            The opened bundle.
        data
            The rebuilt data, for the fingerprint actually read.
        evaluations
            The scored splits.

        Returns
        -------
        EvaluationResult
            Metrics plus provenance.
        """
        changed = data.lineage.source_fingerprint != loaded.lineage.source_fingerprint
        result = EvaluationResult(
            evaluations=evaluations,
            model_name=loaded.model_name,
            bundle_version=loaded.saved.manifest.version,
            spec_digest=loaded.saved.manifest.spec_digest,
            source_fingerprint=data.lineage.source_fingerprint,
            trained_on_fingerprint=loaded.lineage.source_fingerprint,
            source_changed=changed,
        )
        _LOGGER.info("%s", result.describe())
        return result

    def _engine(self, loaded: LoadedBundle) -> Engine:
        """
        Resolve the engine the bundle was trained with.

        Parameters
        ----------
        loaded
            The opened bundle.

        Returns
        -------
        Engine
            An instance of the registered engine.

        Raises
        ------
        ContractError
            If the resolved component is not an engine.
        """
        name = loaded.spec.training.engine
        engine = get_engine(name)()
        if not isinstance(engine, Engine):
            raise ContractError(
                f"the bundle names the engine {name!r}, which is registered as "
                f"a {type(engine).__name__} and does not satisfy the Engine "
                f"protocol; it cannot run a forward pass"
            )
        return engine

    @staticmethod
    def _check_signature(loaded: LoadedBundle, data: DataBundle[object]) -> None:
        """
        Check the rebuilt data still presents the interface the weights expect.

        A changed source can change more than its values. A column added to
        the input table, a reduction that selects a different basis because
        the data moved -- either changes the feature count, and a model built
        against the saved signature would then be fed something else.

        Checked rather than trusted because of how it fails otherwise. A
        shape mismatch at the first layer raises, which is fine. A mismatch
        that happens to broadcast does not, and the metric that comes out is
        a plausible number computed from the wrong columns.

        Parameters
        ----------
        loaded
            The opened bundle, holding the saved signature.
        data
            The rebuilt data, holding the one just derived.

        Raises
        ------
        ContractError
            If the two signatures disagree.
        """
        saved = loaded.signature
        rebuilt = data.signature
        if saved == rebuilt:
            return
        raise ContractError(
            f"the data rebuilt for this bundle no longer matches the interface "
            f"its weights were shaped against.\n"
            f"  trained against: {saved.describe()}\n"
            f"  rebuilt:         {rebuilt.describe()}\n"
            f"This usually means the source gained or lost a column since the "
            f"model was trained. Re-scoring would compute a metric from the "
            f"wrong inputs, so it is refused"
        )
