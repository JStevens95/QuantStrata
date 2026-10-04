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
from ..lifecycle.errors import ComponentError
from ..provenance.logging import get_logger
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
