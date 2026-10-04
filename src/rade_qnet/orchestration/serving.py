"""
Holding a saved model open, for a caller that will use it more than once.

Why the pipelines are not enough
---------------------------------
:func:`rade_qnet.api.infer` reads the bundle from disk, verifies it against
its manifest, resolves the model definition, rebuilds the architecture, loads
the weights and places the result on a device -- and then does all of that
again on the next call.  For scheduled batch scoring that is not waste, it is
the guarantee: the artefact on disk is provably the artefact that produced the
numbers, every time, with no possibility of a process having drifted.

For a service answering requests it is unusable.  So this module holds the
work open, and the split between what is cached and what is not is the whole
of its design.

What is cached, and what deliberately is not
----------------------------------------------
Cached for the life of the handle, because it depends on the bundle alone:
the opened :class:`~rade_qnet.orchestration.stages.reload.LoadedBundle` --
the disk read, the manifest verification, the definition lookup, the fitted
state -- and the rebuilt, weight-loaded model.  That is the expensive part,
and nothing about a later request can change it.

*Not* cached, unless the model has no static inputs: the prepared handle.
Static inputs are things like a graph adjacency or an entity-attribute table,
computed from the data build rather than from the bundle, and a model that
has them can see them change between requests -- a new instrument alters the
graph.  Reusing a handle prepared against last request's adjacency would
produce a confident number computed against the wrong neighbourhood, which is
the failure mode this framework spends most of its refusals avoiding.

Most models have no static inputs at all, so most get the full saving.  The
rest pay a device placement per request and keep the correctness, and the
test suite pins both halves of that behaviour.

What this does not promise
---------------------------
**Thread safety.**  A handle holds one model object and the engines mutate it
during a forward pass -- autocast state, cached encodings for a
:class:`~rade_qnet.core.authoring.capabilities.Precomputable` model.  Two
threads calling :meth:`Predictor.predict` on one instance is undefined.  The
supported pattern is one handle per worker, which is also what lets a process
pool scale without the handles contending.

**Freshness.**  A handle is a snapshot.  Promoting a new run to
``production`` does not move an already-open handle onto it, because silently
swapping the model under a running service is worse than requiring a restart:
the numbers would change with no event in the log that explains why.  Ask the
registry again and open a new handle.

The interactive counterpart
----------------------------
``Agent`` -- the same idea for a policy, answering ``act`` rather than
``predict`` -- lands with the public :func:`rade_qnet.api.act`. It is a
separate type rather than a flag, because a predictor is handed inputs and
returns values with provenance while an agent is handed one observation and
returns one action, with no dataset, no split and nothing to attribute a
number to. One class covering both would mean a method that sometimes took a
source and sometimes an observation, and a caller finding out which at run
time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..core.provenance.logging import get_logger
from .pipelines.infer import DEFAULT_SPLIT, InferPipeline
from .stages.reload import load_bundle
from .stages.resolve import pipeline_for

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..core.contract.result import Predictions
    from ..core.lifecycle.context import RunContext
    from ..core.spec.data import SourceSpec
    from .stages.reload import LoadedBundle

__all__ = ["Predictor"]

_LOGGER = get_logger(__name__)


class Predictor:
    """
    A saved supervised model, held open across many predictions.

    Construct one through :func:`rade_qnet.api.load` rather than directly;
    the public function resolves a tag or an alias to a directory and builds
    the run context this needs.

    Parameters
    ----------
    directory
        The bundle directory.
    context
        The run context predictions are made under. Supplies the output
        root and the identifiers that appear in the log.
    verify
        Whether to re-hash the bundle's files against its manifest. Done
        once, at open, rather than per prediction -- which is the point of
        holding the bundle rather than reopening it.

    Attributes
    ----------
    directory
        Where the bundle was loaded from.

    Examples
    --------
    >>> predictor = api.load("hybrid_gnn_rnn@production")  # doctest: +SKIP
    >>> for batch in requests:  # doctest: +SKIP
    ...     predictions = predictor.predict(source=batch)  # doctest: +SKIP
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        context: RunContext,
        verify: bool = True,
    ) -> None:
        """
        Open the bundle once and keep what a prediction needs.

        Parameters
        ----------
        directory
            The bundle directory.
        context
            The run context predictions are made under.
        verify
            Whether to verify the bundle against its manifest.
        """
        self.directory = Path(directory)
        self._context = context
        self._loaded: LoadedBundle = load_bundle(self.directory, verify=verify)
        # Populated after the first prediction, and only for a model whose
        # static inputs are empty -- see the module docstring for why a
        # model with static inputs may not reuse one.
        self._prepared: object | None = None
        _LOGGER.info("opened predictor: %s", self._loaded.describe())

    @property
    def model_name(self) -> str:
        """
        Return the name the model is registered under.

        Returns
        -------
        str
            From the manifest, so it is what was recorded rather than what
            the resolved class happens to be called now.
        """
        return self._loaded.model_name

    def describe(self) -> str:
        """
        Return a one-line description, for a log line or a health endpoint.

        Returns
        -------
        str
            Model name, bundle version and the splits it was trained on.
        """
        return self._loaded.describe()

    def predict(
        self,
        *,
        source: SourceSpec | None = None,
        split: str = DEFAULT_SPLIT,
        entities: Iterable[str] | None = None,
    ) -> Predictions:
        """
        Predict, reusing everything that does not depend on this request.

        The returned predictions carry the same provenance a one-shot
        :func:`rade_qnet.api.infer` produces -- bundle version, spec digest,
        both source fingerprints and a timestamp. A held model must not be
        cheaper to reconcile than a cold one, because reconciling after the
        fact is the ordinary case and the held path is the one that will run
        a thousand times a day.

        Parameters
        ----------
        source
            A source specification to predict over, or ``None`` for the
            bundle's own.
        split
            Which split of the source to predict over.
        entities
            Identifiers to predict for, or ``None`` for whatever the source
            holds. Naming entities the model never saw requires it to
            declare :class:`~rade_qnet.core.authoring.capabilities.Inductive`;
            otherwise the request is refused rather than answered with a
            default embedding.

        Returns
        -------
        Predictions
            Values in the target's original units, with provenance.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised, naming the stage.
        """
        pipeline_type = pipeline_for(self._loaded.definition, "infer", InferPipeline)
        with self._context.activate():
            pipeline = pipeline_type(
                context=self._context,
                directory=self.directory,
                source=source,
                split=split,
                entities=None if entities is None else tuple(entities),
                verify=False,
                preloaded=self._loaded,
                prepared=self._prepared,
            )
            predictions = pipeline.run()

        self._remember(pipeline)
        return predictions

    def _remember(self, pipeline: InferPipeline) -> None:
        """
        Keep the prepared model, but only when doing so is sound.

        A handle is reusable exactly when the static inputs it was prepared
        against cannot differ next time, and the only case where that is
        certain is when there are none. Anything cleverer -- comparing
        adjacency matrices, hashing the static block -- would be a
        correctness argument resting on a comparison, where this rests on
        there being nothing to compare.

        Parameters
        ----------
        pipeline
            The pipeline that just ran, holding the handle it restored.
        """
        if self._prepared is not None or pipeline.handle is None:
            return
        if pipeline.handle.static:
            _LOGGER.debug("%s has static inputs; preparing per prediction", self._loaded.model_name)
            return
        self._prepared = pipeline.handle
