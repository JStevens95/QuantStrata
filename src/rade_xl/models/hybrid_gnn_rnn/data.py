"""
Turns a cluster's raw P&L and instrument attributes into model input.

The six stages, and which axis each one fits on
------------------------------------------------
The framework drives these in order. The distinction that matters is which
*axis* a stage fits along, because the leakage rules are opposite on the two
and conflating them is how a backtest comes to report a number nobody can
reproduce in production.

===============  ====================================  ==================
Stage            What it does                          Fitted on
===============  ====================================  ==================
``load``         Reads P&L, attributes and universe.    nothing
``split``        Chooses train/validation/test rows.    nothing
``fit_state``    Scalers, basis, encoder, graph.        see below
``transform``    Applies the state to every row.        nothing
``signature``    Declares the model's interface.        nothing
``batch_sources``Windows the rows into batches.         nothing
===============  ====================================  ==================

Inside ``fit_state`` the two axes part company:

**The scenario axis is time.** The P&L scalers see training rows only.
Fitting them on the full history would tell the model the mean and variance
of a period it is about to be scored on, and the backtest would flatter
itself by an amount nobody can estimate after the fact.

**The entity axis is the instrument universe.** The attribute encoder and
the graph see every instrument, including those whose P&L lands only in the
test rows. This is *not* leakage, and the distinction is worth being
precise about: which instruments exist, what their strikes and maturities
are, and how similar they are to each other, is all known before a single
day of P&L is observed. A trader on day one can see the whole book.
Restricting the encoder to instruments that happened to trade during the
training window would discard information genuinely available, and would
leave the model unable to encode an instrument at inference at all.

The one leak that is preserved on purpose
-----------------------------------------
Basis selection runs on the scaled P&L, and the baseline ran it over the
*full* history. That is defect 9, and it is reproducible here through the
framework's ``transforms.reduction.fit_on``, which defaults to ``"train"``
and writes a leakage warning into the run lineage when set to ``"all"``.

It is preserved rather than silently fixed because fixing it changes which
instruments are selected, and therefore every number the model produces.
A refactor that changes the answer and the implementation at the same time
cannot be verified.

The index arrays
----------------
``elementary_indices`` and ``target_indices`` say where each block sits in
the combined attribute encoding. They are computed **after** basis
selection, as ``0..n_e`` and ``n_e..n_e + n_t``.

This is the subtlest trap in the whole build. Carrying the pre-reduction
indices forward produces arrays of plausible length holding plausible
values that address the wrong rows of the encoding. The model trains, the
loss falls, and the predictions are attributed to the wrong instruments.
Parity level 1 compares both arrays explicitly for this reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ...core.contract.requirement import InputRequirement, RequiredInput
from ...core.contract.signature import InputSignature, TensorSpec
from ...core.runtime.errors import ContractError
from ...core.runtime.logging import get_logger
from ...core.spec.data import SourceSpec
from ...sources.dataset.module import DataModule
from .features.basis import select_basis
from .features.encoder import EntityEncoderState
from .features.graph import build_graph
from .spec import HybridDataSpec
from .state import HybridState, StandardScalerState, Universe

__all__ = ["REQUIRES", "HybridDataModule", "HybridRawData"]

#: The six inputs the flagship consumes, by name.
#:
#: Named rather than anonymous because this model genuinely cannot be
#: generic: ``adjacency_values`` and ``target_indices`` are not
#: interchangeable, and a build that swapped them would produce a model
#: that trains to a decreasing loss and attributes every prediction to the
#: wrong instrument. That failure is invisible in every metric a run
#: reports, which is why the names are pinned here rather than trusted.
#:
#: Two shapes are constrained beyond their rank. Edge endpoints come in
#: pairs and a dense shape has two dimensions, so the trailing ``2`` is a
#: real requirement rather than a restatement of whatever the build
#: happened to produce. Everything else is left free: the window length,
#: the number of selected bases, the node count and the attribute width
#: are all properties of the book being modelled, and pinning them here
#: would duplicate numbers the signature already carries.
REQUIRES = InputRequirement(
    dynamic=(
        RequiredInput(
            name="pnl_history",
            rank=3,
            dtype="float32",
            description="scaled elementary P&L over the window",
        ),
    ),
    static=(
        RequiredInput(
            name="trade_features",
            rank=2,
            dtype="float32",
            description="encoded instrument attributes, elementary then target",
        ),
        RequiredInput(
            name="adjacency_indices",
            rank=2,
            dtype="int64",
            shape=(None, 2),
            description="graph edge endpoints, row-major",
        ),
        RequiredInput(
            name="adjacency_values",
            rank=1,
            dtype="float32",
            description="row-normalised edge weights",
        ),
        RequiredInput(
            name="adjacency_shape",
            rank=1,
            dtype="int64",
            shape=(2,),
            description="dense shape of the sparse adjacency",
        ),
        RequiredInput(
            name="target_indices",
            rank=1,
            dtype="int64",
            description="rows of the encoding holding target instruments",
        ),
    ),
)

#: File names the loader expects beneath the source directory. Named as
#: constants so the error message and the reader agree.
_ELEMENTARY_PNL = "elementary_pnl.npy"
_TARGET_PNL = "target_pnl.npy"
_UNIVERSE = "universe.json"
_ELEMENTARY_ATTRIBUTES = "elementary_attributes.json"
_TARGET_ATTRIBUTES = "target_attributes.json"

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HybridRawData:
    """
    One cluster's inputs, before anything is fitted.

    Parameters
    ----------
    elementary_pnl
        Shape ``(n_scenarios, n_elementary)``. The hedging instruments.
    target_pnl
        Shape ``(n_scenarios, n_targets)``. What is being predicted.
    elementary_ids, target_ids
        Column identifiers, in column order.
    elementary_attributes, target_attributes
        Static per-instrument attributes, each a mapping from attribute name
        to one value per instrument.
    """

    elementary_pnl: NDArray[np.float64]
    target_pnl: NDArray[np.float64]
    elementary_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    elementary_attributes: dict[str, list[object]]
    target_attributes: dict[str, list[object]]


class HybridDataModule(DataModule[HybridRawData]):
    """
    The flagship's data build.

    Notes
    -----
    Holds no state between stages. Every stage takes what it needs as an
    argument and returns what it produces, so building twice on one instance
    gives the same answer twice -- and a cached dataset cannot end up
    carrying a signature fitted to data it was not built from.
    """

    state_type = HybridState

    # -- Loading ----------------------------------------------------------

    def load(self, spec: SourceSpec) -> HybridRawData:
        """
        Read one cluster's P&L, attributes and universe from disk.

        Parameters
        ----------
        spec
            The source specification, whose ``params`` carry the directory.

        Returns
        -------
        HybridRawData
            The loaded inputs.

        Raises
        ------
        ContractError
            If the directory is unset, absent, missing a file, or holds
            arrays whose shapes disagree with the universe.
        """
        settings = self._settings(spec)
        directory = settings.directory
        if directory is None:
            raise ContractError(
                "the hybrid data module needs a 'directory' in the source spec's "
                "params, naming the folder holding this cluster's P&L and attributes"
            )

        missing = [
            name
            for name in (
                _ELEMENTARY_PNL,
                _TARGET_PNL,
                _UNIVERSE,
                _ELEMENTARY_ATTRIBUTES,
                _TARGET_ATTRIBUTES,
            )
            if not (directory / name).exists()
        ]
        if missing:
            raise ContractError(
                f"{directory} is not a hybrid cluster directory: missing {', '.join(missing)}"
            )

        universe = json.loads((directory / _UNIVERSE).read_text(encoding="utf-8"))
        raw = HybridRawData(
            elementary_pnl=np.load(directory / _ELEMENTARY_PNL).astype(np.float64),
            target_pnl=np.load(directory / _TARGET_PNL).astype(np.float64),
            elementary_ids=tuple(universe["elementary_ids"]),
            target_ids=tuple(universe["target_ids"]),
            elementary_attributes=json.loads(
                (directory / _ELEMENTARY_ATTRIBUTES).read_text(encoding="utf-8")
            ),
            target_attributes=json.loads(
                (directory / _TARGET_ATTRIBUTES).read_text(encoding="utf-8")
            ),
        )
        self._check_shapes(raw, directory=directory)
        _LOGGER.info(
            "loaded %d scenario(s), %d elementary and %d target instrument(s) from %s",
            raw.elementary_pnl.shape[0],
            len(raw.elementary_ids),
            len(raw.target_ids),
            directory,
        )
        return raw

    def n_scenarios(self, raw: HybridRawData) -> int:
        """
        Return the length of the scenario axis.

        Parameters
        ----------
        raw
            The loaded inputs.

        Returns
        -------
        int
            Number of scenarios.
        """
        return int(raw.elementary_pnl.shape[0])

    def n_entities(self, raw: HybridRawData) -> int:
        """
        Return the size of the instrument universe.

        Parameters
        ----------
        raw
            The loaded inputs.

        Returns
        -------
        int
            Elementary plus target instruments.
        """
        return len(raw.elementary_ids) + len(raw.target_ids)

    # -- Fitting ----------------------------------------------------------

    def fit_state(
        self, raw: HybridRawData, spec: SourceSpec, *, train_indices: NDArray[np.int64]
    ) -> HybridState:
        """
        Fit the scalers, choose the basis, and encode the universe.

        Parameters
        ----------
        raw
            The loaded inputs.
        spec
            The source specification.
        train_indices
            Scenario indices the time axis may observe.

        Returns
        -------
        HybridState
            Everything needed to transform data and to invert a prediction.
        """
        settings = self._settings(spec)

        # Scenario axis: training rows only, no flag, no exceptions.
        feature_scaler = StandardScalerState.fit(raw.elementary_pnl[train_indices, :])
        target_scaler = StandardScalerState.fit(raw.target_pnl[train_indices, :])
        scaled = feature_scaler.transform(raw.elementary_pnl)

        # Defect 9. The framework already owns this flag and already writes
        # the leakage warning into the lineage, so there is nothing
        # model-specific to decide here beyond which rows to pass.
        basis_rows = (
            np.arange(scaled.shape[0])
            if spec.transforms.reduction.fit_on == "all"
            else train_indices
        )
        selected_basis = select_basis(
            scaled[basis_rows, :],
            instrument_ids=raw.elementary_ids,
            variance_threshold=settings.variance_threshold,
            weight_tail=settings.weight_tail,
        )

        # Entity axis: the whole universe, and the encoder has no parameter
        # that could restrict it. See this module's docstring.
        attributes = _merge_attributes(
            raw, selected_basis=selected_basis, elementary_ids=raw.elementary_ids
        )
        encoder = EntityEncoderState.fit(attributes, spec=settings.encoder)
        graph = build_graph(
            encoder.transform(attributes),
            spec=settings.graph,
            is_target=np.array([False] * len(selected_basis) + [True] * len(raw.target_ids)),
        )

        n_elementary, n_targets = len(selected_basis), len(raw.target_ids)
        return HybridState(
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            selected_basis=selected_basis,
            encoder=encoder,
            graph=graph,
            universe=Universe(elementary_ids=selected_basis, target_ids=raw.target_ids),
            # Recomputed here, after reduction, and not carried through from
            # the pre-reduction universe. See this module's docstring.
            elementary_indices=np.arange(n_elementary, dtype=np.int64),
            target_indices=np.arange(n_elementary, n_elementary + n_targets, dtype=np.int64),
            scale_targets=settings.scale_targets,
        )

    # -- Applying ---------------------------------------------------------

    def transform(
        self, raw: HybridRawData, state: HybridState
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Scale every row and narrow the features to the selected basis.

        Applied to the whole scenario axis, not only to training rows. That
        is not a leak: the scaler's *parameters* came from training rows
        alone, and a held-out row must be scaled with those parameters or
        the model would see it on a scale it was never trained on.

        Parameters
        ----------
        raw
            The loaded inputs.
        state
            The fitted state.

        Returns
        -------
        tuple
            Scaled features narrowed to the basis, and the scaled target.
        """
        positions = [raw.elementary_ids.index(name) for name in state.selected_basis]
        features = state.feature_scaler.transform(raw.elementary_pnl)[:, positions]
        target = (
            state.target_scaler.transform(raw.target_pnl)
            if state.scale_targets
            else raw.target_pnl.astype(np.float32)
        )
        return features, target

    # -- Declaring --------------------------------------------------------

    def signature(
        self,
        spec: SourceSpec,
        *,
        features: NDArray[np.floating],
        state: HybridState,
    ) -> InputSignature:
        """
        Declare which tensors the model receives and which are constant.

        The static set is what retires the baseline's per-sample collation
        of the graph: the adjacency was merged into every sample and then
        compared across the batch to recover the one copy, every batch, every
        epoch, to establish something true by construction.

        ``elementary_indices`` is deliberately absent. The elementary block
        always occupies rows ``0..n_e`` of the encoding, so the array is
        ``arange`` of the length the model already knows -- and shipping a
        redundant index array invites the two to disagree. ``target_indices``
        is kept because the output layer must select target rows and its
        offset is not implied by anything else in the signature.

        Parameters
        ----------
        spec
            The source specification, carrying the sequence length.
        features
            The transformed feature matrix, for its column count.
        state
            The fitted state, for the shapes that only exist after fitting.

        Returns
        -------
        InputSignature
            Static inputs, dynamic inputs and the target.
        """
        n_selected = int(features.shape[1])
        n_nodes = state.graph.n_nodes
        n_attributes = state.encoder.n_features

        return InputSignature(
            dynamic={
                "pnl_history": TensorSpec(
                    shape=(None, spec.transforms.sequence.length, n_selected),
                    dtype="float32",
                    description="Scaled elementary P&L over the window.",
                )
            },
            static={
                "trade_features": TensorSpec(
                    shape=(n_nodes, n_attributes),
                    dtype="float32",
                    description="Encoded instrument attributes, elementary then target.",
                ),
                "adjacency_indices": TensorSpec(
                    shape=(state.graph.n_edges, 2),
                    dtype="int64",
                    description="Graph edge endpoints, row-major.",
                ),
                "adjacency_values": TensorSpec(
                    shape=(state.graph.n_edges,),
                    dtype="float32",
                    description="Row-normalised edge weights.",
                ),
                "adjacency_shape": TensorSpec(
                    shape=(2,),
                    dtype="int64",
                    description="Dense shape of the sparse adjacency.",
                ),
                "target_indices": TensorSpec(
                    shape=(state.universe.n_targets,),
                    dtype="int64",
                    description="Rows of the encoding holding target instruments.",
                ),
            },
            target=TensorSpec(
                shape=(None, state.universe.n_targets),
                dtype="float32",
                description="Scaled target P&L at the end of the window.",
            ),
        )

    # -- Reporting --------------------------------------------------------

    def static_inputs(
        self, raw: HybridRawData, state: HybridState
    ) -> dict[str, NDArray[np.generic]]:
        """
        Return the graph and the attribute table, which no sample varies.

        Five tensors, none of which depends on the scenario. Supplying them
        here rather than letting them be collated per sample is what keeps
        a 500-instrument adjacency from being copied once per row of every
        batch -- defect 4. The engine uploads them to the device once per
        run and passes the same references to every forward call.

        Parameters
        ----------
        raw
            The loaded inputs, which carry the attribute tables.
        state
            The fitted state, carrying the fitted encoder and graph.

        Returns
        -------
        dict
            The attribute table, the graph in sparse form, and which rows
            are targets. Keyed exactly as :meth:`signature` declares them,
            which the engine checks.
        """
        # Re-encoded rather than stored on the state. The encoder is a
        # pure function of its fitted levels, so this reproduces exactly
        # what the graph was built from, and keeping a second copy of the
        # encoded table on the state would be a thing that could disagree
        # with the encoder it came from.
        attributes = _merge_attributes(
            raw, selected_basis=state.selected_basis, elementary_ids=raw.elementary_ids
        )
        return {
            "trade_features": state.encoder.transform(attributes).features,
            "adjacency_indices": state.graph.indices,
            "adjacency_values": state.graph.values,
            "adjacency_shape": np.asarray(state.graph.dense_shape, dtype=np.int64),
            "target_indices": state.target_indices,
        }

    def feature_names(self, raw: HybridRawData, state: HybridState) -> tuple[str, ...] | None:
        """
        Return the selected instrument identifiers, in column order.

        Narrowed to what survived basis selection. Reporting the
        pre-reduction names against post-reduction columns would mislabel
        every attribution figure in the report.

        Parameters
        ----------
        raw
            The loaded inputs, unused: the names come from the basis.
        state
            The fitted state.

        Returns
        -------
        tuple of str
            One name per feature column.
        """
        del raw
        return state.selected_basis

    def entity_ids(self, raw: HybridRawData) -> tuple[str, ...]:
        """
        Return the target identifiers, which is what predictions attach to.

        Parameters
        ----------
        raw
            The loaded inputs.

        Returns
        -------
        tuple of str
            One identifier per predicted column.
        """
        return raw.target_ids

    # -- Internals --------------------------------------------------------

    @staticmethod
    def _settings(spec: SourceSpec) -> HybridDataSpec:
        """
        Validate and return the model-specific half of the specification.

        Parsed on every use rather than cached on the instance, which keeps
        the module stateless across stages for the reason given in the class
        notes.

        Parameters
        ----------
        spec
            The source specification.

        Returns
        -------
        HybridDataSpec
            The validated settings.
        """
        return HybridDataSpec.model_validate(dict(getattr(spec, "params", {}) or {}))

    @staticmethod
    def _check_shapes(raw: HybridRawData, *, directory: Path) -> None:
        """
        Reject inputs whose shapes contradict each other.

        Checked here rather than left to fail downstream because every one
        of these produces a confusing error much later: a column-count
        mismatch surfaces as an index error inside basis selection, and a
        scenario-count mismatch surfaces as a silent misalignment between
        features and target that trains to a plausible-looking loss.

        Parameters
        ----------
        raw
            The loaded inputs.
        directory
            Where they came from, for the message.

        Raises
        ------
        ContractError
            On the first disagreement found.
        """
        if raw.elementary_pnl.shape[0] != raw.target_pnl.shape[0]:
            raise ContractError(
                f"{directory}: elementary P&L has {raw.elementary_pnl.shape[0]} "
                f"scenario(s) but target P&L has {raw.target_pnl.shape[0]}. The two "
                f"are aligned row by row, so a mismatch means every target is "
                f"attributed to the wrong day"
            )
        for label, pnl, ids in (
            ("elementary", raw.elementary_pnl, raw.elementary_ids),
            ("target", raw.target_pnl, raw.target_ids),
        ):
            if pnl.shape[1] != len(ids):
                raise ContractError(
                    f"{directory}: {label} P&L has {pnl.shape[1]} column(s) but the "
                    f"universe names {len(ids)} {label} instrument(s)"
                )


def _merge_attributes(
    raw: HybridRawData,
    *,
    selected_basis: tuple[str, ...],
    elementary_ids: tuple[str, ...],
) -> dict[str, list[object]]:
    """
    Stack the selected elementary attributes above the target attributes.

    Elementary first, in basis order, then targets. That order defines the
    node order of the graph and the row order of the encoding, and every
    index array downstream is written against it.

    Parameters
    ----------
    raw
        The loaded inputs.
    selected_basis
        The elementary instruments kept, in order.
    elementary_ids
        Every elementary instrument, in input order, used to locate each
        selected one.

    Returns
    -------
    dict
        Attribute name to one value per instrument.

    Raises
    ------
    ContractError
        If an attribute is present for one block and not the other, which
        would otherwise produce a column the encoder fits on half the
        universe.
    """
    only_elementary = set(raw.elementary_attributes) - set(raw.target_attributes)
    only_target = set(raw.target_attributes) - set(raw.elementary_attributes)
    if only_elementary or only_target:
        raise ContractError(
            f"elementary and target instruments must carry the same attributes, but "
            f"{sorted(only_elementary)} are elementary-only and {sorted(only_target)} "
            f"are target-only. The two blocks become one encoded matrix, so an "
            f"attribute present on one side would be fitted over half the universe"
        )

    positions = [elementary_ids.index(name) for name in selected_basis]
    return {
        key: [raw.elementary_attributes[key][position] for position in positions]
        + list(raw.target_attributes[key])
        for key in raw.elementary_attributes
    }
