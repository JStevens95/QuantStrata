# `src/rade_qnet/orchestration/pipelines`

6 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 63 | 2775 | `67aa54d4ce6a7acf` |
| 2 | `evaluate.py` | 395 | 13992 | `f9f065ad810b76dc` |
| 3 | `infer.py` | 491 | 17074 | `f53febd64a1946b7` |
| 4 | `reinforce.py` | 627 | 22917 | `96b510ee66d5fd62` |
| 5 | `train.py` | 672 | 24664 | `f61cbb9d86175cd0` |
| 6 | `tune.py` | 604 | 20832 | `dd0d38b33938ebe7` |

---

## 1. `src/rade_qnet/orchestration/pipelines/__init__.py`

2775 bytes · SHA-256 `67aa54d4ce6a7acf`

```python
"""
The pipelines that define a model's lifecycle.

Each pipeline is a template method: ``run()`` calls a sequence of small,
individually typed, individually overridable steps.  The granularity is the
point.  A model author who wants a different loss override one step; they do
not reimplement training.

There are four customisation tiers, and a model should use the lowest one that
works:

1. **Spec only.**  Change configuration; write no code.
2. **Add reports or hooks.**  Extra artifacts and instrumentation, no pipeline
   subclass.
3. **Override one step.**  Keep the sequence, replace a single stage.
4. **Override ``run()``.**  Reserved for genuinely different sequences; the
   conformance suite still applies.

One file, one lifecycle
-----------------------
Every module here defines exactly one ``Pipeline`` subclass and nothing else,
so listing this directory lists what the framework can be asked to do.  The
steps those pipelines share live next door in
:mod:`rade_qnet.orchestration.stages`; they used to live here, which made the
folder's name a half-truth.

Modules
-------
``train.py``
    ``TrainPipeline``: resolve spec, build source, build model, materialise,
    fit, evaluate, package bundle, write reports, register.  [Phase 1 skeleton,
    Phase 2 complete]
``evaluate.py``
    ``EvaluatePipeline``: load bundle, rebuild source from saved lineage,
    predict, invert target transforms, compute metrics, write reports.
    [Phase 5]
``infer.py``
    ``InferPipeline``: load bundle, prepare inputs for unseen entities,
    predict, emit predictions with provenance.  [Phase 5]
``tune.py``
    ``TunePipeline``: propose trials, run a short train per trial against a
    cached data build, select the best, optionally refit.  [Phase 5]
``reinforce.py``
    ``ReinforcePipeline``: resolve spec, build environment, declare spaces,
    build policy, materialise, collect experience, fit a step budget, package
    bundle, write reports, register.  [Phase 7]

Two training pipelines, not a fork
----------------------------------
``reinforce.py`` is a *sibling* of ``train.py``: the same stage names in the
same order, the same ``TrainingResult``, the same bundle layout, the same four
customisation tiers.  Three stages genuinely differ -- there is no dataset to
build, no target to declare and no pass to make -- and each is a different
*kind* of thing rather than a different parameter.

A single class covering both would branch on ``spec.task`` in those three
stages, and a model author overriding ``build_data`` would then have to know
which branch they were in.  The tiers above only work if a stage means one
thing.  The full reasoning is in ``reinforce.py`` and in
``PHASE_7_REINFORCEMENT_LEARNING.md`` §3.1.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/orchestration/pipelines/evaluate.py`

13992 bytes · SHA-256 `f9f065ad810b76dc`

```python
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
from ...core.lifecycle.components import get_engine
from ...core.lifecycle.errors import ContractError
from ...core.lifecycle.pipeline import Pipeline
from ...core.provenance.logging import get_logger
from ...engines.base import Engine
from ..stages.reload import load_bundle
from ..stages.scoring import EVALUATED_SPLITS, score_splits, static_inputs

if TYPE_CHECKING:
    from pathlib import Path

    from ...core.contract.data import DataBundle
    from ...core.lifecycle.context import RunContext
    from ...core.spec.data import SourceSpec
    from ...engines.base import ModelHandle
    from ..stages.reload import LoadedBundle

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
        :meth:`~rade_qnet.sources.dataset.module.DataModule.rebuild`, so there
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

        return loaded.definition.rebuild_data(spec, state=loaded.state, lineage=loaded.lineage)

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
            Split name to :class:`~rade_qnet.core.contract.result.EvalResult`.

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
```

---

## 3. `src/rade_qnet/orchestration/pipelines/infer.py`

17074 bytes · SHA-256 `f53febd64a1946b7`

```python
"""
Producing predictions from a saved model, with the provenance to act on them.

Inference is evaluation without targets, and that one difference changes what
the pipeline is for. Evaluation answers "how good is this model"; inference
answers "what does it say about these inputs", and somebody downstream will
do something with the answer.

Which is why provenance is not optional here. A column of numbers with no
record of which model produced them, from which inputs, at what time, cannot
be reconciled with anything later -- and reconciling predictions after the
fact is the ordinary case, not the exception. Somebody will ask why
Tuesday's number differed from Monday's, and the only useful answer names the
bundle version and the source fingerprint behind each.

The reload path is identical to evaluation's, and is shared as functions both
call rather than as a base class both inherit. See
:mod:`~rade_qnet.orchestration.pipelines.evaluate` for why they are two
pipelines rather than one with a flag.

Unseen entities
---------------
A model that learned a per-entity embedding table has no row for an
instrument added last week. Asking it anyway tends to produce a default
embedding and a confident, meaningless number rather than an error -- which
is the worst available outcome, because the number looks exactly like the
real ones.

:class:`~rade_qnet.core.authoring.capabilities.Inductive` is how a model says it
can do better. This pipeline refuses the request when the capability is
absent, and routes through
:class:`~rade_qnet.core.authoring.capabilities.Inductive` when it
is present.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from ...core.authoring.capabilities import Inductive
from ...core.contract.result import Predictions
from ...core.lifecycle.components import get_engine
from ...core.lifecycle.errors import ContractError
from ...core.lifecycle.pipeline import Pipeline
from ...core.provenance.logging import get_logger
from ...engines.base import Engine
from ..stages.reload import load_bundle
from ..stages.scoring import scoring_source, static_inputs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ...core.contract.data import DataBundle
    from ...core.lifecycle.context import RunContext
    from ...core.spec.data import SourceSpec
    from ...engines.base import ModelHandle
    from ..stages.reload import LoadedBundle

__all__ = ["InferPipeline"]

_LOGGER = get_logger(__name__)

#: The split a prediction pass runs over by default. Every row of a source
#: presented for inference is a row somebody wants a number for, and the
#: training run's notion of "test" is the split whose rows the model never
#: learned from -- the honest default for a pass over the bundle's own data.
DEFAULT_SPLIT = "test"

#: How many unseen identifiers to name in the refusal message. Enough to
#: recognise the pattern -- a whole currency pair missing, a stale mapping --
#: without pasting a thousand tickers into a log line.
_NAMED_IN_ERROR = 5


class InferPipeline(Pipeline[Predictions]):
    """
    Predict with a saved bundle, over its own data or over new inputs.

    Parameters
    ----------
    context
        Ambient state for the run.
    directory
        The bundle to predict with.
    source
        A source specification to predict over. Defaults to ``None``,
        meaning the bundle's own.
    split
        Which split of the source to predict over.
    entities
        Identifiers to predict for. Defaults to ``None``, meaning whatever
        the source holds. Naming entities the model never saw requires the
        model to declare
        :class:`~rade_qnet.core.authoring.capabilities.Inductive`.
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
        "prepare_inputs",
        "restore",
        "predict",
        "invert",
        "attribute",
    )

    def __init__(
        self,
        context: RunContext,
        directory: Path,
        *,
        source: SourceSpec | None = None,
        split: str = DEFAULT_SPLIT,
        entities: Sequence[str] | None = None,
        verify: bool = True,
    ) -> None:
        """
        Store what to predict with and over what.

        Parameters
        ----------
        context
            Ambient state for the run.
        directory
            The bundle directory.
        source
            Source specification to predict over, or ``None``.
        split
            Split to predict over.
        entities
            Identifiers to predict for, or ``None`` for all of them.
        verify
            Whether to verify the bundle against its manifest.
        """
        super().__init__(context)
        self.directory = directory
        self.source = source
        self.split = split
        self.entities = tuple(entities) if entities is not None else None
        self.verify = verify
        self.loaded: LoadedBundle | None = None
        self.handle: ModelHandle | None = None

    def run(self) -> Predictions:
        """
        Execute the stage sequence.

        Returns
        -------
        Predictions
            Values in the target's original units, with provenance.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised, naming the stage.
        """
        loaded = self.step("load", self.load)
        data = self.step("prepare_inputs", lambda: self.prepare_inputs(loaded))
        handle = self.step("restore", lambda: self.restore(loaded, data))
        raw = self.step("predict", lambda: self.predict(loaded, handle, data))
        values = self.step("invert", lambda: self.invert(loaded, raw))
        return self.step("attribute", lambda: self.attribute(loaded, data, values))

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

    def prepare_inputs(self, loaded: LoadedBundle) -> DataBundle[object]:
        """
        Rebuild the inputs using the bundle's saved state.

        The same rebuild evaluation uses, and for the same reason: a
        prediction made from inputs standardised by today's mean, under a
        model that learned a response to inputs standardised by the training
        mean, is a confident number computed on the wrong scale.

        Parameters
        ----------
        loaded
            The opened bundle.

        Returns
        -------
        DataBundle
            The same payload shape training consumed.

        Raises
        ------
        ContractError
            If entities were requested that the model cannot predict for.
        """
        spec = loaded.spec
        if self.source is not None:
            spec = spec.model_copy(update={"source": self.source})

        data = loaded.definition.rebuild_data(spec, state=loaded.state, lineage=loaded.lineage)
        self._check_entities(loaded, data)
        return data

    def restore(self, loaded: LoadedBundle, data: DataBundle[object]) -> ModelHandle:
        """
        Rebuild the architecture, load the saved weights, place it on a device.

        Parameters
        ----------
        loaded
            The opened bundle.
        data
            The prepared inputs, for the static inputs.

        Returns
        -------
        ModelHandle
            The restored model.
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

    def predict(
        self,
        loaded: LoadedBundle,
        handle: ModelHandle,
        data: DataBundle[object],
    ) -> np.ndarray:
        """
        Run the forward pass, in the model's own output space.

        Routed through
        :func:`~.scoring.scoring_source` even though nothing is being scored.
        A prediction pass is paired with its identifiers afterwards, so it
        needs the same stable order a two-pass scoring run needs -- and a
        shuffled pass would attribute each number to the wrong instrument,
        which is a worse outcome here than in evaluation because the numbers
        are acted on individually rather than reduced to a mean.

        Parameters
        ----------
        loaded
            The opened bundle.
        handle
            The restored model.
        data
            The prepared inputs.

        Returns
        -------
        numpy.ndarray
            Raw model output.

        Raises
        ------
        ContractError
            If the requested split is not present in the prepared inputs.
        """
        if self.split not in data.splits:
            raise ContractError(
                f"the prepared inputs have no {self.split!r} split; available "
                f"splits are {sorted(data.splits)}"
            )
        engine = self._engine(loaded)
        source = scoring_source(data, self.split)
        return np.asarray(engine.predict(handle, source), dtype=np.float64)

    @staticmethod
    def invert(loaded: LoadedBundle, raw: np.ndarray) -> np.ndarray:
        """
        Put the predictions back into the target's original units.

        A required stage rather than a courtesy. The model's output space is
        whatever the fitted state transformed the target into, and a number
        in that space is not a quantity anyone can act on -- it is not a P&L,
        a return or a price, and acting on it as though it were would be
        wrong by the training standard deviation.

        Parameters
        ----------
        loaded
            The opened bundle, for the fitted state that inverts.
        raw
            The model's raw output.

        Returns
        -------
        numpy.ndarray
            Values in the original target units.
        """
        return np.asarray(loaded.state.inverse_transform_targets(raw), dtype=np.float64)

    def attribute(
        self,
        loaded: LoadedBundle,
        data: DataBundle[object],
        values: np.ndarray,
    ) -> Predictions:
        """
        Pair the values with their identifiers and record where they came from.

        Parameters
        ----------
        loaded
            The opened bundle.
        data
            The prepared inputs, for the scenario indices and entity ids.
        values
            The inverted predictions.

        Returns
        -------
        Predictions
            Values plus everything needed to reconcile them later.
        """
        flat = np.ravel(values)
        indices = self._scenario_indices(data, n_predictions=flat.size)

        predictions = Predictions(
            values=flat,
            in_original_units=True,
            entity_ids=self.entities,
            scenario_indices=indices,
            bundle_version=str(loaded.saved.manifest.version),
            provenance={
                "model_name": loaded.model_name,
                "spec_digest": loaded.saved.manifest.spec_digest,
                "source_fingerprint": data.lineage.source_fingerprint,
                "trained_on_fingerprint": loaded.lineage.source_fingerprint,
                "split": self.split,
                "predicted_at": datetime.now(UTC).isoformat(),
                "framework_version": loaded.lineage.framework_version,
            },
        )
        _LOGGER.info("predicted %d value(s) with %s", predictions.n_predictions, loaded.describe())
        return predictions

    def _scenario_indices(
        self, data: DataBundle[object], *, n_predictions: int
    ) -> np.ndarray | None:
        """
        Return the source rows each prediction corresponds to.

        Returns ``None`` rather than a guess when the count does not line up.
        A sequence model drops the first few rows of a split because no
        complete window ends there, so the saved indices can legitimately
        outnumber the predictions -- and inventing an alignment would
        attribute every number to the wrong row by a constant offset, which
        is both wrong and almost invisible.

        Parameters
        ----------
        data
            The prepared inputs.
        n_predictions
            How many values came back.

        Returns
        -------
        numpy.ndarray or None
            One index per prediction, or ``None`` if they cannot be paired.
        """
        saved = data.lineage.split_indices.get(self.split)
        if saved is None:
            return None
        indices = np.asarray(saved, dtype=np.int64)
        if indices.size != n_predictions:
            _LOGGER.warning(
                "the %r split holds %d row(s) but %d prediction(s) came back, so "
                "the two cannot be paired; scenario indices are omitted rather "
                "than guessed",
                self.split,
                indices.size,
                n_predictions,
            )
            return None
        return indices

    def _check_entities(self, loaded: LoadedBundle, data: DataBundle[object]) -> None:
        """
        Refuse requested entities the model cannot honestly predict for.

        Parameters
        ----------
        loaded
            The opened bundle, whose definition declares the capability.
        data
            The prepared inputs, naming the entities the model knows.

        Raises
        ------
        ContractError
            If entities absent from training were requested and the model
            does not declare
            :class:`~rade_qnet.core.authoring.capabilities.Inductive`.
        """
        if self.entities is None:
            return

        known = tuple(data.entity_ids or ())
        if not known:
            # Nothing to check against. A model with no entity axis predicts
            # per scenario, and the identifiers are the caller's labels for
            # rows rather than something the model resolves.
            return

        unseen = tuple(name for name in self.entities if name not in known)
        if not unseen:
            return

        definition = loaded.definition
        if not (isinstance(definition, Inductive) and definition.supports_unseen_entities()):
            raise ContractError(
                f"{len(unseen)} requested entity(ies) were absent when this "
                f"model was trained: {list(unseen[:_NAMED_IN_ERROR])}"
                f"{' ...' if len(unseen) > _NAMED_IN_ERROR else ''}. "
                f"{type(definition).__name__} does not declare Inductive, so it "
                f"has no row for them and would return a default embedding -- a "
                f"confident number indistinguishable from a real one. Refused "
                f"rather than predicted"
            )
        _LOGGER.info(
            "%d unseen entity(ies) will be resolved by %s's inductive path",
            len(unseen),
            type(definition).__name__,
        )

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
```

---

## 4. `src/rade_qnet/orchestration/pipelines/reinforce.py`

22917 bytes · SHA-256 `96b510ee66d5fd62`

```python
"""
The interactive training pipeline: a sibling of the supervised one, not a fork.

``PHASE_7_REINFORCEMENT_LEARNING.md`` §3.1 said this phase should need "no new
pipelines", and the reason it gave is the one that matters: a second lifecycle
means every later improvement to training has to be made twice, and the two
copies diverge at the first deadline.

What is here is therefore deliberately *not* a second lifecycle. It is a
second :class:`~rade_qnet.core.lifecycle.pipeline.Pipeline` with the same stage
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
from ...core.lifecycle.components import get_engine, get_report
from ...core.lifecycle.errors import ComponentError
from ...core.lifecycle.pipeline import Pipeline
from ...core.provenance.hashing import digest_spec
from ...core.provenance.logging import get_logger
from ...core.provenance.seeding import seed_everything
from ...engines.base import Engine, InteractiveEngine, ModelHandle
from ...sources.batching.rollout import RolloutSource
from ...storage.bundle import write_bundle

if TYPE_CHECKING:
    from ...analysis.reports.base import Report, ReportOutcome
    from ...core.authoring.definition import PolicyDefinition
    from ...core.contract.bundle import SavedBundle
    from ...core.contract.result import FitOutcome
    from ...core.contract.signature import PolicySignature
    from ...core.lifecycle.context import RunContext
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
```

---

## 5. `src/rade_qnet/orchestration/pipelines/train.py`

24664 bytes · SHA-256 `f61cbb9d86175cd0`

```python
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
   bookkeeping from :meth:`~rade_qnet.core.lifecycle.pipeline.Pipeline.execute`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...analysis.metrics.quality import quality_metrics
from ...analysis.reports.base import ReportContext
from ...core.authoring.definition import PredictorDefinition
from ...core.contract.bundle import ModelBundle
from ...core.contract.data import (
    DataBundle,
)
from ...core.contract.result import EvalResult, FitOutcome, TrainingResult
from ...core.contract.signature import InputSignature
from ...core.lifecycle.components import get_engine, get_report
from ...core.lifecycle.errors import ComponentError
from ...core.lifecycle.pipeline import Pipeline
from ...core.provenance.logging import get_logger
from ...core.provenance.seeding import seed_everything
from ...engines.base import Engine, ModelHandle
from ...storage.bundle import write_bundle
from ..stages.scoring import (
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
    from ...core.lifecycle.context import RunContext
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
        :meth:`~rade_qnet.core.lifecycle.pipeline.Pipeline.step`, so each is timed,
        logged, reported to hooks, and -- on failure -- wrapped in a
        :class:`~rade_qnet.core.lifecycle.errors.StageError` naming the stage.

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
```

---

## 6. `src/rade_qnet/orchestration/pipelines/tune.py`

20832 bytes · SHA-256 `dd0d38b33938ebe7`

```python
"""
Searching a space of configurations for the one that scores best.

A search is forty training runs that differ in a handful of values, and
almost everything that makes it useful or useless is about the bookkeeping
around those runs rather than the runs themselves.

What this pipeline is careful about
-----------------------------------
**Building the data once.** Forty trials over one dataset should read it
once. The outline asked for a general step cache to achieve that; this holds
the prepared dataset for the search instead, for the reasons recorded in
[Phase 5 §8.1](../../docs/phases/PHASE_5_EVALUATE_INFER_TUNE.md). Where the
trials vary the *source*, there is nothing to reuse and the pipeline says so
rather than quietly reusing the wrong thing.

**Failing trials are recorded, not swallowed.** A search that drops a failing
trial and reports the best of the rest looks exactly like a search in which
every trial succeeded. If eleven of forty failed because an override was
nonsense, the conclusion drawn from the remaining twenty-nine is probably
wrong, and nothing in a swallowed-failure report would suggest it. Same
decision as a job set, for the same reason.

**The winner is picked once, by a stated rule.** Direction comes from the
specification rather than from guessing at the metric's name, and ties go to
the earlier trial so a re-run picks the same winner.

**Selection is on validation.** The default objective split is validation,
because selecting on test turns the held-out split into part of the training
procedure -- and the test metric then reported for the winner overstates it
by however much the search exploited that split.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ...core.contract.result import TrialRecord, TuningResult
from ...core.lifecycle.components import get_model
from ...core.lifecycle.context import RunContext
from ...core.lifecycle.errors import ComponentError, ContractError, SpecError
from ...core.lifecycle.pipeline import Pipeline
from ...core.provenance.logging import get_logger
from ...core.spec.run import SupervisedRunSpec
from ..stages.search import expand, propose
from .train import TrainPipeline

if TYPE_CHECKING:
    from ...core.authoring.definition import PredictorDefinition
    from ...core.contract.data import DataBundle
    from ...core.spec.tune import TuneSpec

__all__ = ["TunePipeline"]

_LOGGER = get_logger(__name__)


class TunePipeline(Pipeline[TuningResult]):
    """
    Run a search and return its trials, with the best one identified.

    Parameters
    ----------
    context
        Ambient state for the run.
    spec
        The search specification.
    definition
        The model definition. Defaults to ``None``, meaning resolve it from
        the base specification's model name. Passed in rather than always
        resolved so a test can supply a definition without registering it
        globally, exactly as :class:`~.train.TrainPipeline` allows.

    Attributes
    ----------
    shared_data
        The dataset reused across trials, available after ``build_data``.
        ``None`` when the search varies the source, in which case each trial
        builds its own.
    """

    stages = (
        "resolve",
        "propose",
        "build_data",
        "run_trials",
        "select",
        "refit",
    )

    def __init__(
        self,
        context: RunContext,
        spec: TuneSpec,
        definition: PredictorDefinition | None = None,
    ) -> None:
        """
        Store the search and the model it searches over.

        Parameters
        ----------
        context
            Ambient state for the run.
        spec
            The search specification.
        definition
            The model definition, or ``None`` to resolve it by name.
        """
        super().__init__(context)
        self.spec = spec
        self.definition = definition
        self.shared_data: DataBundle[object] | None = None
        self._builds = 0

    @property
    def data_builds(self) -> int:
        """
        Return how many times the dataset was actually built.

        Exposed because "built once per search" is a claim worth testing,
        and a test that could only time the search would be measuring the
        machine rather than the pipeline.

        Returns
        -------
        int
            The count.
        """
        return self._builds

    def run(self) -> TuningResult:
        """
        Execute the stage sequence.

        Returns
        -------
        TuningResult
            Every trial, and which one won.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised, naming the stage.
        """
        definition = self.step("resolve", self.resolve)
        proposals = self.step("propose", self.propose)
        self.step("build_data", lambda: self.build_data(definition))
        trials = self.step("run_trials", lambda: self.run_trials(definition, proposals))
        result = self.step("select", lambda: self.select(trials))
        return self.step("refit", lambda: self.refit(definition, result))

    def resolve(self) -> PredictorDefinition:
        """
        Resolve the model, and validate every proposal's *shape* before any run.

        Resolution first, for the same reason training resolves before
        building: a misspelled model name should cost milliseconds rather
        than a data build.

        Parameters
        ----------
        None

        Returns
        -------
        PredictorDefinition
            The definition every trial trains.

        Raises
        ------
        ContractError
            If the base specification names no model and none was supplied.
        """
        if self.definition is not None:
            return self.definition

        name = self.spec.base.get("model")
        if isinstance(name, dict):
            name = name.get("name")
        if not isinstance(name, str):
            raise ContractError(
                "the search's base specification names no model, and no "
                "definition was supplied; a search has to know what it is "
                "searching over"
            )
        try:
            definition = get_model(name)()
        except ComponentError as error:
            raise ContractError(
                f"the search names the model {name!r}, which is not registered "
                f"in this process; import the package that defines it first"
            ) from error
        self.definition = definition
        return definition

    def propose(self) -> list[dict[str, Any]]:
        """
        Produce every trial's proposal, and validate each one now.

        Validating the whole set before the first trial runs is the cure for
        defect 7. A typo in a dotted path is a mistake nobody typed -- the
        values are generated -- and left unchecked it would be explored,
        measured and reported as a dimension that makes no difference, which
        reads as a finding rather than a bug.

        Returns
        -------
        list of dict
            One flat proposal per trial.

        Raises
        ------
        SpecError
            If any proposal does not merge into a valid run specification.
        """
        proposals = propose(self.spec)
        failures: list[str] = []
        for index, proposal in enumerate(proposals):
            try:
                self.spec.run_spec_for(expand(proposal), trial=index)
            except SpecError as error:
                failures.append(f"  trial {index} ({proposal}): {error}")

        if failures:
            raise SpecError(
                f"{len(failures)} of {len(proposals)} proposed trial(s) do not "
                f"produce a valid run specification. The search space names a "
                f"path the specification does not have, so the search would "
                f"explore it and report that it makes no difference:\n"
                + "\n".join(failures[: self._REPORTED_FAILURES])
            )
        return proposals

    #: How many invalid proposals to spell out. A broken path breaks every
    #: trial, so printing forty identical messages buries the one that
    #: matters.
    _REPORTED_FAILURES = 3

    def build_data(self, definition: PredictorDefinition) -> DataBundle[object] | None:
        """
        Build the dataset once, when every trial shares one.

        Skipped when the space varies anything under ``source``: the trials
        then have genuinely different datasets, and handing them a shared
        one would mean the search measured something other than what it
        proposed -- the quietest possible way for a search to be wrong.

        Parameters
        ----------
        definition
            The model definition.

        Returns
        -------
        DataBundle or None
            The shared dataset, or ``None`` when each trial builds its own.
        """
        if self._varies_the_source():
            _LOGGER.info(
                "the search varies the source, so each trial builds its own "
                "dataset; there is nothing shared to reuse"
            )
            return None

        spec = self.spec.run_spec_for(expand({}), trial=0)
        if not isinstance(spec, SupervisedRunSpec):
            raise ContractError(
                f"tuning builds data for a supervised run; the base "
                f"specification resolves to a {type(spec).__name__}"
            )

        data = definition.build_data(spec)
        self._builds += 1
        self.shared_data = data
        _LOGGER.info(
            "built the dataset once for %d trial(s): %s",
            self.spec.trials,
            {name: data.splits[name] is not None for name in data.split_names},
        )
        return data

    def run_trials(
        self, definition: PredictorDefinition, proposals: list[dict[str, Any]]
    ) -> list[TrialRecord]:
        """
        Train and score one model per proposal.

        Parameters
        ----------
        definition
            The model definition.
        proposals
            The flat proposals.

        Returns
        -------
        list of TrialRecord
            One record per trial, in trial order, successes and failures
            alike.
        """
        records: list[TrialRecord] = []
        for index, proposal in enumerate(proposals):
            records.append(self._trial(definition, proposal, index=index))
        succeeded = sum(1 for record in records if record.succeeded)
        _LOGGER.info("%d of %d trial(s) succeeded", succeeded, len(records))
        return records

    def select(self, trials: list[TrialRecord]) -> TuningResult:
        """
        Pick the winner, by the direction the specification states.

        Parameters
        ----------
        trials
            Every trial.

        Returns
        -------
        TuningResult
            The trials and the winning index.

        Raises
        ------
        ContractError
            If no trial produced the objective. Raised rather than returning
            a result with no winner, because a search that selected nothing
            and said so quietly would be read as a search that found nothing
            good.
        """
        scored = [record for record in trials if record.objective is not None]
        if not scored:
            raise ContractError(
                f"no trial produced the objective {self.spec.objective!r} on the "
                f"{self.spec.objective_split!r} split, so there is nothing to "
                f"select between. {len(trials)} trial(s) ran and "
                f"{sum(1 for record in trials if not record.succeeded)} failed"
            )

        best = scored[0]
        for record in scored[1:]:
            # Ties go to the incumbent, which makes the winner the earliest
            # best trial and therefore stable across a re-run.
            if self.spec.is_better(record.objective, best.objective):
                best = record

        result = TuningResult(
            trials=tuple(trials),
            best_trial=best.trial,
            objective=self.spec.objective,
            direction=self.spec.direction,
            objective_split=self.spec.objective_split,
            name=self.spec.name,
        )
        _LOGGER.info("%s", result.describe())
        return result

    def refit(self, definition: PredictorDefinition, result: TuningResult) -> TuningResult:
        """
        Optionally retrain the winner on train and validation combined.

        Off unless asked for. The extra data usually helps, but the
        resulting model's reported objective was measured on rows it has now
        trained on, so the number beside it is no longer a held-out estimate
        -- and that trade should be a choice someone made rather than a
        default they inherited.

        Not yet implemented beyond recording the intent: combining two
        splits means re-deriving a split, which this phase spent its effort
        making impossible to do by accident. It is Phase 6 work, and the
        specification field exists so the search's shape is settled now.

        Parameters
        ----------
        definition
            The model definition.
        result
            The selected result.

        Returns
        -------
        TuningResult
            The result, unchanged when no refit was asked for.

        Raises
        ------
        ContractError
            If a refit was asked for. Refused rather than silently skipped:
            a user who set ``refit: true`` and received a model trained on
            the training split alone would have no way to tell.
        """
        del definition
        if not self.spec.refit:
            return result
        raise ContractError(
            "refit is not implemented yet. Combining the training and "
            "validation splits means deriving a new split, which this phase "
            "deliberately made hard to do by accident; it is Phase 6 work. "
            "Set refit: false and retrain the winning configuration directly "
            "if you need the combined fit now"
        )

    def _trial(
        self,
        definition: PredictorDefinition,
        proposal: dict[str, Any],
        *,
        index: int,
    ) -> TrialRecord:
        """
        Run one trial, recording its outcome whether it worked or not.

        Parameters
        ----------
        definition
            The model definition.
        proposal
            The flat proposal.
        index
            Trial number.

        Returns
        -------
        TrialRecord
            The outcome.
        """
        started = time.perf_counter()
        spec = self.spec.run_spec_for(expand(proposal), trial=index)

        try:
            pipeline = _SharedDataTrainPipeline(
                context=self._trial_context(index),
                spec=spec,
                definition=definition,
                shared=self.shared_data,
            )
            outcome = pipeline.execute()
            objective = self._objective_of(outcome)
            record = TrialRecord(
                trial=index,
                overrides=dict(proposal),
                status="succeeded",
                objective=objective,
                metrics=dict(
                    outcome.evaluations[self.spec.objective_split].metrics
                    if self.spec.objective_split in outcome.evaluations
                    else {}
                ),
                bundle_directory=(
                    None if pipeline.saved is None else str(pipeline.saved.directory)
                ),
                wall_seconds=time.perf_counter() - started,
            )
        except Exception as error:
            # Recorded rather than raised, for the reason in the module
            # docstring: a search that stopped at the first bad trial would
            # lose thirty-nine good ones, and one that dropped it silently
            # would report a conclusion drawn from a partial sweep.
            record = TrialRecord(
                trial=index,
                overrides=dict(proposal),
                status="failed",
                wall_seconds=time.perf_counter() - started,
                failure_kind=type(error).__name__,
                failure_message=str(error),
            )
            _LOGGER.warning("trial %d failed: %s: %s", index, type(error).__name__, error)

        return record

    def _trial_context(self, index: int) -> RunContext:
        """
        Give one trial its own directory beneath the search's.

        Every trial trains a model, and a trained model is persisted -- so
        without this, forty trials would all claim version 1 of the same
        bundle path and thirty-nine would fail on a refusal to overwrite.
        The refusal is correct; sharing the directory was the mistake.

        A directory per trial also keeps the winner's bundle on disk, which
        is the difference between acting on a search and repeating it. The
        cost is that a long search leaves forty bundles behind, which is the
        honest price of being able to use the one that won.

        Parameters
        ----------
        index
            Trial number.

        Returns
        -------
        RunContext
            The search's context, redirected at the trial's directory and
            carrying the trial number as its job identifier so log lines
            from concurrent-looking output can be told apart.
        """
        return replace(
            self.context,
            output_directory=self.context.output_directory / "trials" / f"trial-{index}",
            job_id=f"trial-{index}",
        )

    def _objective_of(self, outcome: object) -> float | None:
        """
        Read the objective metric from a trial's result.

        Parameters
        ----------
        outcome
            The trial's training result.

        Returns
        -------
        float or None
            The value, or ``None`` if the split or the metric is absent --
            which :meth:`select` then reports rather than treating as a
            score of zero.
        """
        evaluations = getattr(outcome, "evaluations", {})
        evaluation = evaluations.get(self.spec.objective_split)
        if evaluation is None:
            return None
        return evaluation.metrics.get(self.spec.objective)

    def _varies_the_source(self) -> bool:
        """
        Return whether any search axis changes how the data is built.

        Checked on the path prefix rather than by building twice and
        comparing: the point is to decide *before* building anything.

        Returns
        -------
        bool
            True if a shared dataset would be wrong.
        """
        return any(path == "source" or path.startswith("source.") for path in self.spec.space.paths)


class _SharedDataTrainPipeline(TrainPipeline):
    """
    A training pipeline that reuses a dataset instead of building one.

    A subclass overriding one stage rather than a flag on
    :class:`~.train.TrainPipeline`, so that the ordinary training path has
    no branch in it at all. Training is the pipeline every other phase
    depends on, and a conditional there would be reachable from every run
    rather than only from a search.

    Parameters
    ----------
    context
        Ambient state for the run.
    spec
        The trial's validated specification.
    definition
        The model definition.
    shared
        The dataset to reuse, or ``None`` to build one normally.
    """

    def __init__(
        self,
        context: RunContext,
        spec: SupervisedRunSpec,
        definition: PredictorDefinition,
        *,
        shared: DataBundle[object] | None,
    ) -> None:
        """
        Store the shared dataset alongside the usual arguments.

        Parameters
        ----------
        context
            Ambient state for the run.
        spec
            The trial's specification.
        definition
            The model definition.
        shared
            The dataset to reuse, or ``None``.
        """
        super().__init__(context=context, spec=spec, definition=definition)
        self._shared = shared

    def build_data(self, spec: SupervisedRunSpec) -> DataBundle[object]:
        """
        Return the shared dataset, or build one when there is none.

        Parameters
        ----------
        spec
            The trial's specification.

        Returns
        -------
        DataBundle
            The dataset.
        """
        if self._shared is not None:
            return self._shared
        return super().build_data(spec)
```

