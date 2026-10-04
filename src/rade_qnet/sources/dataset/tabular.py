"""
The standard data build: read a file, scale, reduce, split, package.

Most models need no data code at all, and this module is why. It supplies
every stage :class:`~rade_qnet.sources.dataset.module.DataModule` declares,
so a model's ``data.py`` can be three lines that hand back one of these --
which is what makes the short model definition in ``ARCHITECTURE.md`` section
1 possible, and what Phase 6's baselines rely on to keep the framework
honest.

The split from ``module.py``
-----------------------------
That module defines what a data build *must* do; this one is a single
implementation of it. They were one file, and separating them means the
abstraction can be read without the CSV handling, and the CSV handling can be
changed without touching the abstraction. The name is honest about its scope:
it reads tables. A build whose raw input is a graph or a stream writes its own
subclass and shares nothing here but the base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.signature import InputSignature, TensorSpec
from ...core.runtime.errors import SpecError
from ...core.runtime.logging import get_logger
from .module import ELEMENT_DTYPE, FEATURE_INPUT_NAME, DataModule
from .tables import TableData, read_table
from .transforms.composite import DatasetState
from .transforms.reduction import ReductionState
from .transforms.scaling import ScalingState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.contract.state import FittedState
    from ...core.spec.data import SourceSpec, TabularSourceSpec

__all__ = ["DataModule", "TabularDataModule"]

_LOGGER = get_logger(__name__)


__all__ = ["TabularDataModule"]


class TabularDataModule(DataModule[TableData]):
    """
    The standard data build: read a file, scale, reduce, split, package.

    Supplies every stage, so a straightforward model needs no data code at
    all -- which is what makes the short model definition in
    ``ARCHITECTURE.md`` §1 possible, and what Phase 6's baselines rely on to
    keep the framework honest.
    """

    def load(self, spec: SourceSpec) -> TableData:
        """
        Read the tabular file named by the specification.

        Parameters
        ----------
        spec
            A :class:`~rade_qnet.core.spec.data.TabularSourceSpec`.

        Returns
        -------
        TableData
            The parsed table.

        Raises
        ------
        SpecError
            If the spec is not a tabular source, or names no path. The path is
            optional on the spec so that it is constructible with defaults for
            testing; this is the stage that notices, and it can name itself in
            the message.
        """
        tabular = self._tabular(spec)
        if tabular.path is None:
            raise SpecError(
                "a tabular source needs 'path' to be set; the specification was "
                "constructed with defaults and has no file to read"
            )
        return read_table(
            tabular.path,
            target_column=tabular.target_column,
            feature_columns=tabular.feature_columns,
            attribute_columns=self._attribute_columns(tabular),
        )

    def n_scenarios(self, raw: TableData) -> int:
        """
        Return the number of rows read.

        Parameters
        ----------
        raw
            The parsed table.

        Returns
        -------
        int
            Row count.
        """
        return raw.n_rows

    def group_labels(self, raw: TableData, spec: SourceSpec) -> NDArray[np.int64] | None:
        """
        Return integer group labels when a grouped split asks for them.

        Parameters
        ----------
        raw
            The parsed table.
        spec
            The source specification.

        Returns
        -------
        numpy.ndarray or None
            One label per row for a grouped split, otherwise ``None``.
        """
        if spec.split.kind != "grouped":
            return None
        values = raw.attribute(spec.split.group_key)
        # Sorted so the label a group receives depends on the set of groups and
        # not on which row happened to be read first.
        codes = {name: code for code, name in enumerate(sorted(set(values)))}
        return np.array([codes[value] for value in values], dtype=np.int64)

    def fit_state(
        self, raw: TableData, spec: SourceSpec, *, train_indices: NDArray[np.int64]
    ) -> DatasetState:
        """
        Fit scaling and reduction on training rows.

        Both receive the full matrix *and* the training indices, rather than a
        pre-sliced matrix. That signature is what makes the leakage rule
        unavoidable: there is no way to pass held-out rows in as though they
        were training rows.

        Parameters
        ----------
        raw
            The parsed table.
        spec
            The source specification.
        train_indices
            Scenario indices that may be observed.

        Returns
        -------
        DatasetState
            The composed state, with scaling owning the target inverse.
        """
        transforms = spec.transforms
        scaling = (
            ScalingState.fit(
                raw.features, raw.target, spec=transforms.scaling, train_indices=train_indices
            )
            if transforms.scaling.method != "none"
            else None
        )

        # Reduction is fitted on scaled features, because both of its methods
        # compare columns against each other and an unscaled comparison is
        # dominated by whichever column happens to have the largest units.
        scaled = scaling.transform_features(raw.features) if scaling else raw.features
        reduction = (
            ReductionState.fit(
                scaled, raw.target, spec=transforms.reduction, train_indices=train_indices
            )
            if transforms.reduction.method != "none"
            else None
        )
        return DatasetState.of(scaling=scaling, reduction=reduction)

    def transform(
        self, raw: TableData, state: FittedState
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Apply scaling then reduction to every row.

        Parameters
        ----------
        raw
            The parsed table.
        state
            The state from :meth:`fit_state`.

        Returns
        -------
        tuple
            Transformed features and target.

        Raises
        ------
        SpecError
            If the state is not the composed state this module fits, which
            would mean a subclass overrode one stage and not the other.
        """
        composed = self._composed(state)
        features, target = raw.features, raw.target
        if composed.has_part("scaling"):
            scaling = composed.part("scaling")
            features = scaling.transform_features(features)
            target = scaling.transform_targets(target)
        if composed.has_part("reduction"):
            features = composed.part("reduction").transform_features(features)
        return features, target

    def signature(
        self,
        spec: SourceSpec,
        *,
        features: NDArray[np.floating],
        state: FittedState,
    ) -> InputSignature:
        """
        Declare one dynamic feature input and a scalar target.

        The batch dimension is wildcarded and the feature dimension concrete,
        which is what lets the engine synthesise a dummy batch of any size to
        materialise a lazily shaped model.

        Parameters
        ----------
        spec
            The source specification, carrying the sequence length.
        features
            The transformed feature matrix.
        state
            Unused here: the feature count is read from the transformed
            matrix, which reduction has already narrowed, so the state would
            only restate what the matrix already shows. Accepted because the
            hook is shared with modules whose static inputs are fitted.

        Returns
        -------
        InputSignature
            The declared interface.
        """
        del state
        n_features = int(features.shape[1])
        length = spec.transforms.sequence.length
        # A length of one means a non-sequential model, and a window axis of
        # size one would make every such model carry a pointless dimension.
        shape: tuple[int | None, ...] = (
            (None, n_features) if length == 1 else (None, length, n_features)
        )
        return InputSignature(
            dynamic={
                FEATURE_INPUT_NAME: TensorSpec(
                    shape=shape,
                    dtype=ELEMENT_DTYPE,
                    description=f"{n_features} feature(s)"
                    + ("" if length == 1 else f" over a {length}-scenario window"),
                )
            },
            target=TensorSpec(shape=(None,), dtype=ELEMENT_DTYPE),
        )

    def feature_names(self, raw: TableData, state: FittedState) -> tuple[str, ...] | None:
        """
        Return column names, narrowed to whatever survived reduction.

        Parameters
        ----------
        raw
            The parsed table.
        state
            The fitted state.

        Returns
        -------
        tuple of str or None
            Column names in column order. ``None`` after a projection, where
            a column is a combination of inputs and no original name
            describes it.
        """
        composed = self._composed(state)
        if not composed.has_part("reduction"):
            return raw.feature_names

        reduction = composed.part("reduction")
        if reduction.method == "basis_selection":
            return tuple(raw.feature_names[index] for index in reduction.selected)
        # A principal component is a mixture of every input column, so naming
        # it after one of them would be worse than admitting there is no name.
        return None

    @staticmethod
    def _tabular(spec: SourceSpec) -> TabularSourceSpec:
        """
        Narrow a source spec to a tabular one.

        Parameters
        ----------
        spec
            The source specification.

        Returns
        -------
        TabularSourceSpec
            The same spec, narrowed.

        Raises
        ------
        SpecError
            If the spec names a different source kind.
        """
        if spec.kind != "tabular":
            raise SpecError(
                f"TabularDataModule requires a tabular source, received "
                f"kind={spec.kind!r}; a model source needs its own data module"
            )
        return spec

    @staticmethod
    def _composed(state: FittedState) -> DatasetState:
        """
        Narrow a fitted state to the composed state this module fits.

        Parameters
        ----------
        state
            The fitted state.

        Returns
        -------
        DatasetState
            The same state, narrowed.

        Raises
        ------
        SpecError
            If it is some other state, which means ``fit_state`` was
            overridden without overriding the stages that read its result.
        """
        if not isinstance(state, DatasetState):
            raise SpecError(
                f"expected a DatasetState from fit_state, received "
                f"{type(state).__name__}; override transform and feature_names "
                f"too if fit_state produces a different state"
            )
        return state

    @staticmethod
    def _attribute_columns(spec: TabularSourceSpec) -> tuple[str, ...]:
        """
        Return the non-feature columns this build needs kept aside.

        Derived from the split strategy rather than configured separately, so
        a grouped split cannot be requested against a file whose group column
        was never read.

        Parameters
        ----------
        spec
            The tabular source specification.

        Returns
        -------
        tuple of str
            Column names to keep as strings.
        """
        if spec.split.kind == "grouped":
            return (spec.split.group_key,)
        return ()
