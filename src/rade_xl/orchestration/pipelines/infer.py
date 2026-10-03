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
:mod:`~rade_xl.orchestration.pipelines.evaluate` for why they are two
pipelines rather than one with a flag.

Unseen entities
---------------
A model that learned a per-entity embedding table has no row for an
instrument added last week. Asking it anyway tends to produce a default
embedding and a confident, meaningless number rather than an error -- which
is the worst available outcome, because the number looks exactly like the
real ones.

:class:`~rade_xl.core.capability.protocols.Inductive` is how a model says it
can do better. This pipeline refuses the request when the capability is
absent, and routes through
:class:`~rade_xl.core.capability.protocols.Inductive` when it
is present.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from ...core.capability.protocols import Inductive
from ...core.contract.result import Predictions
from ...core.runtime.components import get_engine
from ...core.runtime.errors import ContractError
from ...core.runtime.logging import get_logger
from ...core.runtime.pipeline import Pipeline
from ...engines.base import Engine
from .reload import load_bundle
from .scoring import scoring_source, static_inputs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ...core.contract.data import DataBundle
    from ...core.runtime.context import RunContext
    from ...core.spec.data import SourceSpec
    from ...engines.base import ModelHandle
    from .reload import LoadedBundle

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
        :class:`~rade_xl.core.capability.protocols.Inductive`.
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

        data = loaded.definition.rebuild_data(
            spec, state=loaded.state, lineage=loaded.lineage
        )
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
        _LOGGER.info(
            "predicted %d value(s) with %s", predictions.n_predictions, loaded.describe()
        )
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
            :class:`~rade_xl.core.capability.protocols.Inductive`.
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
