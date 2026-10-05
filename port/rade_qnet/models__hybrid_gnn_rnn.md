# `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn`

8 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 66 | 2656 | `d99cf28ec67f4ec0` |
| 2 | `data.py` | 686 | 25497 | `73f29f102ea6daaa` |
| 3 | `model.py` | 360 | 13189 | `7d269cda90d695c5` |
| 4 | `register.py` | 143 | 5610 | `589cfb05e6a8b23e` |
| 5 | `reports.py` | 340 | 11958 | `5308b7d914f3b947` |
| 6 | `spec.py` | 393 | 17434 | `a177505b14867755` |
| 7 | `state.py` | 466 | 16887 | `3cfb158385bbfb3f` |
| 8 | `visuals.py` | 261 | 8717 | `c033d0fcfc405089` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/__init__.py`

2656 bytes · SHA-256 `d99cf28ec67f4ec0`

```python
"""
The flagship model: a hybrid graph-temporal network for P&L replication.

The model learns to express a target instrument's profit and loss as a function
of a set of elementary instruments.  Two structures carry the signal and the
architecture keeps them separate before combining them:

*Cross-sectional structure.*  Instruments relate to each other through shared
attributes (currency pair, tenor, product).  A graph block attends over an
instrument's neighbourhood in that space.

*Temporal structure.*  Each instrument carries a P&L history.  A recurrent
block summarises it.

A fusion layer combines the two representations, a target-attention layer lets
each target instrument weight the elementary instruments it actually depends
on, and the output layer produces the replicated P&L.

Why this model drives the framework's design
--------------------------------------------
It exercises almost every hard requirement at once: static inputs that are
constant across batches, lazy parameter shapes known only after the data build,
fitted state beyond model weights, two fitting axes with different leakage
rules, an expensive encoding worth precomputing once per evaluation pass, and
fan-out across many jobs with different architecture complexity per job.  A
framework that hosts this cleanly will host most things.

Modules
-------
``model.py``
    The network.  Architecture only -- composes the blocks in ``layers/`` and
    defines the forward pass.  [Phase 3]
``layers/``
    The architectural blocks, one per file.  [Phase 3]
``register.py``
    The framework declaration: binds the model to its spec, data module, fitted
    state class and pipeline overrides.  [Phase 3]
``spec.py``
    The model and data specifications.  [Phase 3]
``state.py``
    The ``FittedState`` subclass: target scalers, the selected elementary
    basis, the entity encoder and the graph.  [Phase 3]
``data.py``
    The data module: load, scale, reduce, encode, build the graph, split,
    package.  [Phase 3]
``features/``
    Model-specific feature construction (entity attribute encoding, graph
    construction).  [Phase 3]
``pipelines/``
    Stage overrides for this model.  [Phase 3 / Phase 5]
``reports.py``, ``visuals.py``
    Artifacts specific to this model: the graph diagnostics page and the
    three figures behind it.

Reference
---------
The behaviour this model must reproduce is specified by the parity plan in
``docs/phases/PHASE_0_BASELINE.md``, which pins a golden fixture captured from
the existing ``rade_ml_pt`` implementation.
"""

__all__: tuple[str, ...] = ()

from .register import HybridGnnRnnModel

__all__ = ["HybridGnnRnnModel"]
```

---

## 2. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/data.py`

25497 bytes · SHA-256 `73f29f102ea6daaa`

```python
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
from ...core.lifecycle.errors import ContractError
from ...core.provenance.logging import get_logger
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
```

---

## 3. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/model.py`

13189 bytes · SHA-256 `7d269cda90d695c5`

```python
"""
The hybrid graph-temporal network.

This file is the mathematics and nothing else. How the model plugs into the
framework -- its registration, its specs, its pipelines -- lives in
``register.py``, so a reader asking *what does this compute?* finds only the
answer to that question here.

What it computes
----------------
Given a book of elementary instruments with recent P&L, and a set of target
instruments to replicate, predict each target's P&L.

Two questions have to be answered at once, and the architecture is one block
per question::

    attributes ──► GNN ──► what each instrument IS
                            (one vector per instrument,
                             identical across samples)
                                        │
                                        ├──► fusion ──► attention ──► head ──► P&L
                                        │
    P&L history ──► RNN ──► what the book just DID
                            (one vector per sample,
                             identical across instruments)

The GNN propagates each instrument's attributes along the similarity graph,
so an instrument's representation reflects its neighbourhood rather than
only itself. The RNN reads the recent window of elementary P&L into a
summary of the current regime. Fusion crosses the two into one vector per
instrument per sample. Target attention lets the predicted instruments
coordinate. The head turns each into a number.

Why the two streams are separate
---------------------------------
It would be simpler to concatenate attributes onto the P&L history and run
one network. That would be a worse model and a far more expensive one.

Structure does not vary across samples, so computing it per sample repeats
identical work -- for a 500-instrument universe and a 2,000-sample epoch
that is three orders of magnitude of waste, and it is why
:meth:`HybridGnnRnn.precompute` exists. Market state does not vary across
instruments, so computing it per instrument repeats identical work too.
Keeping the streams separate until they must meet is what makes the model
tractable on a real book, and the asymmetry is in the tensor shapes: the
graph stream has no batch axis and the temporal stream has no node axis
until the fusion layer gives them both.

Why nothing here is cached
---------------------------
The network holds parameters and no other state. The reusable encoding of
the static inputs is produced on demand by :meth:`precompute` and handed
back to the caller, rather than being stashed on the module.

This is deliberate and it is a fix. Caching the graph embedding inside the
module means two jobs sharing a process can see each other's results
through it, and a cache that is invalidated on mode changes but not on
parameter changes goes stale in a way that produces plausible numbers
rather than an error. A pure module cannot do either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ...core.lifecycle.errors import ContractError
from .layers.attention import TargetAttentionLayer
from .layers.fusion import FusionLayer
from .layers.gnn import LAYER_NORM_EPS, GnnBlock
from .layers.projection import ProjectionLayer
from .layers.rnn import RnnBlock
from .spec import HybridModelSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.signature import InputSignature

__all__ = ["HybridGnnRnn"]

#: The static inputs that describe the graph, in the order the sparse
#: tensor constructor wants them.
_ADJACENCY_KEYS = ("adjacency_indices", "adjacency_values", "adjacency_shape")

#: What `precompute` returns, and what `forward_with_precomputed` expects.
_PRECOMPUTED_KEY = "graph_embedding"


class HybridGnnRnn(nn.Module):
    """
    Graph-temporal replication network.

    Every layer is sized from the signature at construction, so the module
    is fully parameterised before it is returned. Lazy layers would defer
    that to the first forward pass, by which point an optimiser built over
    the module is tracking an empty parameter list and the model trains
    nothing -- silently, because the loss still decreases via the layers
    that did exist.

    Parameters
    ----------
    spec
        The architecture specification.
    signature
        The declared input interface, which supplies every width.

    Raises
    ------
    ContractError
        If the signature is missing a tensor the network needs, or declares
        one with a shape it cannot use.
    """

    def __init__(self, spec: HybridModelSpec, signature: InputSignature) -> None:
        super().__init__()
        self.spec = spec

        n_attributes = _width(signature, "trade_features", axis=-1, static=True)
        n_elementary = _width(signature, "pnl_history", axis=-1, static=False)
        n_targets = _width(signature, "target_indices", axis=0, static=True)

        self.gnn_block = GnnBlock(spec, in_features=n_attributes)
        self.rnn_block = RnnBlock(spec, in_features=n_elementary)

        # Each block is normalised before the two streams meet. Without
        # this, whichever stream happens to come out larger dominates the
        # fusion for the first several epochs purely through scale, and
        # the model spends that time learning to undo the imbalance.
        self.gnn_block_ln = nn.LayerNorm(spec.width("gnn"), eps=LAYER_NORM_EPS)
        self.rnn_block_ln = nn.LayerNorm(self.rnn_block.out_features, eps=LAYER_NORM_EPS)

        self.fusion_layer = FusionLayer(
            spec,
            gnn_features=spec.width("gnn"),
            rnn_features=self.rnn_block.out_features,
        )
        self.attention_layer = TargetAttentionLayer(spec, in_features=spec.width("fusion"))
        self.projection_layer = ProjectionLayer(
            spec,
            in_features=spec.width("attention"),
            attribute_features=n_attributes,
            n_fitted_targets=n_targets,
        )

    def forward(
        self,
        *,
        pnl_history: Tensor,
        trade_features: Tensor,
        adjacency_indices: Tensor,
        adjacency_values: Tensor,
        adjacency_shape: Tensor,
        target_indices: Tensor,
    ) -> Tensor:
        """
        Predict each target's P&L.

        Parameters
        ----------
        pnl_history
            Recent elementary P&L, shape
            ``(batch, sequence_length, n_elementary)``.
        trade_features
            Encoded attributes for every instrument, shape
            ``(n_nodes, n_attributes)``.
        adjacency_indices, adjacency_values, adjacency_shape
            The similarity graph in sparse COO form.
        target_indices
            Which rows of ``trade_features`` are targets.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_targets)``, in scaled P&L units. Returning
            them to money is the fitted state's job, not the network's.
        """
        adjacency = _sparse_adjacency(adjacency_indices, adjacency_values, adjacency_shape)
        graph_embedding = self.gnn_block_ln(self.gnn_block(trade_features, adjacency))
        return self._head(
            graph_embedding,
            pnl_history=pnl_history,
            trade_features=trade_features,
            adjacency=adjacency,
            target_indices=target_indices,
        )

    def precompute(self, static: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """
        Encode the graph once for reuse across an evaluation pass.

        The graph stream depends only on the static inputs, so its result
        is identical for every batch. The framework calls this once at the
        start of an evaluation or inference pass, and never during
        training -- where the parameters change each step and a reused
        embedding would be one step stale.

        Parameters
        ----------
        static
            The static inputs, already on the correct device.

        Returns
        -------
        dict
            The graph embedding, under a key this class also understands
            in :meth:`forward_with_precomputed`.
        """
        adjacency = _sparse_adjacency(*(static[key] for key in _ADJACENCY_KEYS))
        return {
            _PRECOMPUTED_KEY: self.gnn_block_ln(self.gnn_block(static["trade_features"], adjacency))
        }

    def forward_with_precomputed(
        self, batch: Mapping[str, Tensor], precomputed: Mapping[str, Tensor]
    ) -> Tensor:
        """
        Run a forward pass reusing an already-computed graph embedding.

        Parameters
        ----------
        batch
            One batch, including the static inputs.
        precomputed
            The mapping returned by :meth:`precompute`.

        Returns
        -------
        torch.Tensor
            The same values :meth:`forward` would return, up to
            floating-point reassociation.
        """
        return self._head(
            precomputed[_PRECOMPUTED_KEY],
            pnl_history=batch["pnl_history"],
            trade_features=batch["trade_features"],
            adjacency=_sparse_adjacency(*(batch[key] for key in _ADJACENCY_KEYS)),
            target_indices=batch["target_indices"],
        )

    @property
    def supports_unseen_entities(self) -> bool:
        """
        Whether targets absent from training can be predicted.

        True: the output head gives an unfitted target a baseline borrowed
        from its nearest fitted neighbours in attribute space. The quality
        of that borrow is only as good as the attributes, which is a
        question for the graph diagnostics rather than for the network.
        """
        return True

    def _head(
        self,
        graph_embedding: Tensor,
        *,
        pnl_history: Tensor,
        trade_features: Tensor,
        adjacency: Tensor,
        target_indices: Tensor,
    ) -> Tensor:
        """
        Run everything downstream of the graph stream.

        Shared by the ordinary and the precomputed forward paths, so the
        two cannot drift apart: the only difference between them is where
        the graph embedding came from.

        Parameters
        ----------
        graph_embedding
            Normalised node embeddings, shape ``(n_nodes, gnn_width)``.
        pnl_history, trade_features, adjacency, target_indices
            As for :meth:`forward`, with the graph already assembled.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_targets)``.
        """
        temporal = self.rnn_block_ln(self.rnn_block(pnl_history))
        fused = self.fusion_layer(graph_embedding, temporal, adjacency)
        attended = self.attention_layer(fused, adjacency, target_indices)
        return self.projection_layer(attended, trade_features[target_indices])


def _sparse_adjacency(indices: Tensor, values: Tensor, shape: Tensor) -> Tensor:
    """
    Assemble the similarity graph as a sparse tensor.

    Rebuilt per call rather than held on the module. It is three cheap
    tensor operations, and the alternative -- a cached sparse tensor
    belonging to the model -- is state that outlives the batch it was
    built for and silently follows the model into the next job.

    Parameters
    ----------
    indices
        Edge endpoints, shape ``(n_edges, 2)``, row-major.
    values
        Edge weights, shape ``(n_edges,)``.
    shape
        Dense shape, as a two-element tensor.

    Returns
    -------
    torch.Tensor
        A coalesced sparse COO tensor.
    """
    # Transposed because the signature stores one edge per row, which is
    # how the fixture and every diagnostic reads it, while Torch wants one
    # axis per row.
    return torch.sparse_coo_tensor(
        indices.t().contiguous(), values, tuple(int(size) for size in shape)
    ).coalesce()


def _width(signature: InputSignature, name: str, *, axis: int, static: bool) -> int:
    """
    Read one fixed axis length from the signature.

    Parameters
    ----------
    signature
        The declared interface.
    name
        Which tensor to read.
    axis
        Which axis of it.
    static
        Whether to look among the static inputs or the dynamic ones.

    Returns
    -------
    int
        The axis length.

    Raises
    ------
    ContractError
        If the tensor is absent, or that axis is declared variable. A
        variable axis here is not something to default around: it means
        the data module and the network disagree about what is fixed.
    """
    declared = signature.static if static else signature.dynamic
    if name not in declared:
        raise ContractError(
            f"the signature declares no {'static' if static else 'dynamic'} input "
            f"{name!r}; the hybrid network requires it. Declared: "
            f"{sorted(declared)}"
        )
    size = declared[name].shape[axis]
    if size is None:
        raise ContractError(
            f"axis {axis} of {name!r} is declared variable, but the network must "
            f"size a layer from it. Only the batch axis may vary"
        )
    return int(size)
```

---

## 4. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/register.py`

5610 bytes · SHA-256 `589cfb05e6a8b23e`

```python
"""
The framework declaration for the hybrid graph-temporal network.

``model.py`` answers *what does this compute?*. This file answers *how does
it plug in?* -- which spec validates its settings, which data module builds
its dataset, which engine trains it, and which pipelines it overrides.

Keeping the two apart is not tidiness for its own sake. The network is the
part a quant reads and argues with; the wiring is the part a platform
engineer reads and maintains. Interleaving them means neither reader can
skim their own half, and in practice the wiring wins -- a reader looking
for the mathematics finds registration boilerplate first and stops.

Importing this module is what makes ``hybrid_gnn_rnn`` resolvable by name
in a run specification. The package's ``__init__`` imports it for exactly
that reason, so a user who has imported the model package can name the
model in YAML without knowing this file exists.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from ...core.authoring.supervised import SupervisedModel
from ...core.lifecycle.components import model

# Imported for its import side effect: this is what puts the Torch engine
# and its supervised learner in the registry. The decorator below declares
# `engine="torch"`, and a declaration whose subject may or may not be
# registered depending on what else the process happened to import is the
# classic source of "no engine named 'torch'" from a correct specification.
# Importing this model already implies Torch is installed, so nothing is
# paid by a host that does not use it.
from ...engines import torch as _torch_engine  # noqa: F401
from .data import REQUIRES, HybridDataModule
from .model import HybridGnnRnn
from .pipelines.eval import HybridEvalPipeline
from .pipelines.train import HybridTrainPipeline
from .pipelines.tune import HybridTunePipeline
from .spec import HybridDataSpec, HybridModelSpec
from .state import HybridState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.signature import InputSignature
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["HybridGnnRnnModel"]


@model("hybrid_gnn_rnn", engine="torch")
class HybridGnnRnnModel(SupervisedModel):
    """
    Framework declaration for the hybrid graph-temporal network.

    Attributes
    ----------
    spec
        Validates the ``model.params`` block of a run specification.
    data_spec
        Validates the ``source.params`` block. Separate from ``spec``
        because the data build and the network are configured
        independently: changing the recurrent width should not invalidate
        a cached dataset, and the two specs being distinct types is what
        makes that structurally true rather than merely intended.
    state_cls
        The fitted state this model's data build produces.
    pipelines
        Which lifecycles this model overrides: ``train`` to add reports,
        ``eval`` to break the error down per target, and ``tune`` to drop
        infeasible trials before they are spent. Inference is deliberately
        absent and inherits the framework's -- see
        :mod:`~rade_qnet.models.hybrid_gnn_rnn.pipelines`. Read-only so that
        a caller cannot reach into the class and rebind a stage for every
        run in the process.
    """

    requires = REQUIRES
    spec = HybridModelSpec
    data_spec = HybridDataSpec
    state_cls = HybridState
    pipelines: Mapping[str, type] = MappingProxyType(
        {
            "train": HybridTrainPipeline,
            "eval": HybridEvalPipeline,
            "tune": HybridTunePipeline,
        }
    )

    def data_module(self, spec: SupervisedRunSpec) -> HybridDataModule:
        """
        Return the data module that builds this model's dataset.

        Constructed per call rather than held on the definition. The module
        carries no state of its own -- every setting it uses is read from
        the source spec it is handed -- so a fresh one costs nothing, and a
        shared one would be an object two concurrent jobs could reach.

        Parameters
        ----------
        spec
            The validated run specification. Unused: the module reads its
            settings from ``spec.source`` when the framework calls it,
            which keeps this method a pure constructor.

        Returns
        -------
        HybridDataModule
            The data module.
        """
        del spec
        return HybridDataModule()

    def build_model(self, spec: SupervisedRunSpec, signature: InputSignature) -> HybridGnnRnn:
        """
        Construct the untrained network.

        A pure function of the spec and the signature, with no data in
        sight. That is what makes a bundle reloadable six months later: the
        saved signature plus the saved weights are sufficient to rebuild
        the identical object, with no need to reproduce the dataset that
        originally shaped it.

        The network sizes every layer here rather than lazily on the first
        forward pass, so the object this returns is already fully
        parameterised and the engine's materialise step finds nothing left
        to do.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared interface from the data build.

        Returns
        -------
        HybridGnnRnn
            An untrained, fully parameterised network.
        """
        return HybridGnnRnn(HybridModelSpec.model_validate(dict(spec.model.params)), signature)
```

---

## 5. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/reports.py`

11958 bytes · SHA-256 `5308b7d914f3b947`

```python
"""
Reports only this model can produce.

Every model gets the shared reports: a run summary, training curves, data
quality, baseline comparisons. Those answer *how well did it do?*. This one
answers a question that only applies to a graph model and that no metric
will raise on its own: *is the graph any good?*

Why this is not optional
------------------------
A badly built graph is silent. The loss still falls, because the temporal
stream alone can fit a book reasonably well. The headline metrics still
look acceptable, because they are dominated by instruments the training set
covered densely. What degrades is accuracy on instruments the training set
covered thinly -- and those are exactly the instruments a replication model
is bought for.

So the graph diagnostics are not a nice-to-have appendix. They are the only
place in the run where you can see the model's largest assumption and
decide whether to believe it. This report is registered under
``hybrid_graph`` and is added to the enabled set by the model's own train
pipeline, so it renders whether or not the user thought to ask for it.

What to look for
----------------
Three numbers decide it, and the report states each with the threshold it
should be compared against rather than leaving the reader to judge:

- **isolated instruments** -- nodes whose only edge is their own self-loop.
  They get nothing from the graph, so for them the model is the temporal
  stream and a bias term.
- **weight concentration** -- the share of each node's weight sitting on
  its single best neighbour. Near ``1/degree`` means the kernel has
  flattened into an unweighted average and the similarity information has
  been discarded.
- **worst-covered instruments** -- the bottom of the best-neighbour
  distribution, named individually. This is the list to read before
  pricing an unseen trade, because an unseen target's baseline is borrowed
  from precisely these neighbourhoods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...analysis.reports.base import Report
from ...analysis.visuals.export import save_figure
from ...core.lifecycle.components import report
from .state import HybridState
from .visuals import (
    edge_weight_figure,
    neighbour_similarity_figure,
    node_degree_figure,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ...analysis.reports.base import ReportContext
    from .features.graph import SparseGraphState

__all__ = ["HybridGraphReport"]

#: Filename of the Markdown page.
GRAPH_FILENAME = "graph.md"

#: How many of the worst-covered instruments to name. Enough to spot a
#: pattern -- one currency, one product type -- and few enough to read.
_N_WORST = 10

#: Below this, an instrument's best neighbour is weak enough that a
#: borrowed baseline should not be trusted without a second look. It is a
#: prompt to investigate, not a hard rule, and the report says so.
_WEAK_NEIGHBOUR = 0.10


@report("hybrid_graph")
class HybridGraphReport(Report):
    """
    A Markdown page describing the instrument graph the model was trained on.

    Reads only the bundle's fitted state, so it re-renders from a saved run
    without the data and without the weights.
    """

    def render(self, context: ReportContext) -> Sequence[Path]:
        """
        Write the graph diagnostics page and its figures.

        Parameters
        ----------
        context
            The bundle to report on, and where to write.

        Returns
        -------
        Sequence of Path
            The Markdown page followed by the figures.

        Raises
        ------
        LookupError
            If the bundle's state is not a hybrid state. A deliberate skip
            rather than an incidental failure: this report is enabled by
            the model's own pipeline, so reaching it with another model's
            state means the run was assembled by hand and the reader
            deserves an explanation rather than a stack trace.
        """
        state = context.bundle.state
        if not isinstance(state, HybridState):
            raise LookupError(
                f"the hybrid graph report needs a HybridState, but this bundle "
                f"holds a {type(state).__name__}; nothing to describe"
            )

        figures = self._write_figures(state.graph, context)
        sections = [
            "# Instrument graph",
            self._shape_section(state),
            self._coverage_section(state),
            self._worst_section(state),
            self._figures_section(figures),
        ]
        page = context.directory / GRAPH_FILENAME
        page.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
        return (page, *figures)

    @staticmethod
    def _shape_section(state: HybridState) -> str:
        """
        Return the graph's dimensions.

        Parameters
        ----------
        state
            The fitted state.

        Returns
        -------
        str
            Markdown text.
        """
        graph = state.graph
        degrees = graph.degrees()
        # Self-loops are excluded from the quoted neighbour count because
        # "three neighbours" meaning "two neighbours and itself" is the
        # kind of off-by-one that survives into a conversation with a desk.
        neighbours = degrees - 1
        return "\n".join(
            [
                "## Shape",
                "",
                "| property | value |",
                "| --- | --- |",
                f"| instruments | {graph.n_nodes} |",
                f"| elementary | {state.universe.n_elementary} |",
                f"| targets | {state.universe.n_targets} |",
                f"| edges (self-loops included) | {graph.n_edges} |",
                f"| neighbours per instrument (median) | {int(np.median(neighbours))} |",
                f"| neighbours per instrument (min) | {int(neighbours.min())} |",
                f"| density | {graph.n_edges / max(graph.n_nodes**2, 1):.4f} |",
            ]
        )

    @classmethod
    def _coverage_section(cls, state: HybridState) -> str:
        """
        Return the two findings that decide whether the graph is usable.

        Parameters
        ----------
        state
            The fitted state.

        Returns
        -------
        str
            Markdown text, with a verdict line per finding.
        """
        graph = state.graph
        neighbours = graph.degrees() - 1
        isolated = int((neighbours <= 0).sum())
        best = cls._best_weight_per_node(graph)
        weak = int((best < _WEAK_NEIGHBOUR).sum())

        lines = [
            "## Coverage",
            "",
            "| finding | count | share |",
            "| --- | --- | --- |",
            f"| isolated (self-loop only) | {isolated} | {isolated / max(graph.n_nodes, 1):.1%} |",
            f"| best neighbour below {_WEAK_NEIGHBOUR:.2f} | {weak} | "
            f"{weak / max(graph.n_nodes, 1):.1%} |",
            "",
        ]
        if isolated:
            lines.append(
                f"{isolated} instrument(s) have no neighbour but themselves. For "
                f"these the graph contributes nothing and the model reduces to "
                f"the temporal stream. Either the similarity threshold is too "
                f"strict or these instruments are genuinely unlike the rest of "
                f"the book, and the two call for different responses."
            )
        else:
            lines.append("Every instrument has at least one neighbour.")
        if weak:
            lines.append(
                f"{weak} instrument(s) have no neighbour carrying more than "
                f"{_WEAK_NEIGHBOUR:.0%} of their edge weight. A target in this "
                f"group that was not in the training set will borrow its "
                f"baseline from neighbours that are not very close, so treat "
                f"its prediction as indicative rather than priced."
            )
        return "\n".join(lines)

    @classmethod
    def _worst_section(cls, state: HybridState) -> str:
        """
        Name the instruments with the weakest best neighbour.

        Named individually rather than counted, because the action this
        prompts is per instrument: look at it, decide whether its
        attributes are wrong or whether it is genuinely an outlier.

        Parameters
        ----------
        state
            The fitted state.

        Returns
        -------
        str
            Markdown text.
        """
        best = cls._best_weight_per_node(state.graph)
        identifiers = (
            *state.universe.elementary_ids[: state.graph.n_nodes],
            *state.universe.target_ids,
        )
        order = np.argsort(best, kind="stable")[:_N_WORST]

        rows = [
            "## Worst-covered instruments",
            "",
            f"The {min(_N_WORST, best.size)} instruments whose single best "
            f"neighbour carries the least weight.",
            "",
            "| instrument | role | best neighbour weight | neighbours |",
            "| --- | --- | --- | --- |",
        ]
        neighbours = state.graph.degrees() - 1
        for node in order:
            name = identifiers[node] if node < len(identifiers) else f"node {node}"
            role = "target" if state.graph.is_target[node] else "elementary"
            rows.append(f"| `{name}` | {role} | {best[node]:.4f} | {int(neighbours[node])} |")
        return "\n".join(rows)

    @staticmethod
    def _write_figures(graph: SparseGraphState, context: ReportContext) -> tuple[Path, ...]:
        """
        Render and save the three graph figures.

        Parameters
        ----------
        graph
            The fitted graph.
        context
            Where to write, and in what format.

        Returns
        -------
        tuple of Path
            The saved figures.
        """
        factories = {
            "graph_degrees": node_degree_figure,
            "graph_edge_weights": edge_weight_figure,
            "graph_neighbour_similarity": neighbour_similarity_figure,
        }
        return tuple(
            save_figure(
                factory(graph),
                context.directory,
                stem,
                figure_format=context.figure_format,
                dpi=context.figure_dpi,
            )
            for stem, factory in factories.items()
        )

    @staticmethod
    def _figures_section(paths: Sequence[Path]) -> str:
        """
        Link the figures from the page.

        Parameters
        ----------
        paths
            The saved figures.

        Returns
        -------
        str
            Markdown text.
        """
        # Linked by filename rather than by absolute path, so the page
        # survives the directory being copied or served from elsewhere.
        return "\n".join(["## Figures", "", *(f"![{path.stem}]({path.name})" for path in paths)])

    @staticmethod
    def _best_weight_per_node(graph: SparseGraphState) -> np.ndarray:
        """
        Find each node's largest edge weight, ignoring its self-loop.

        The self-loop is excluded because every node has one and it says
        nothing about that node's relationship to the rest of the book --
        including it would give an isolated instrument a perfect score.

        Parameters
        ----------
        graph
            The fitted graph.

        Returns
        -------
        numpy.ndarray
            One value per node, zero where a node has no neighbour.
        """
        rows, columns = graph.indices[:, 0], graph.indices[:, 1]
        off_diagonal = rows != columns
        best = np.zeros(graph.n_nodes, dtype=np.float64)
        np.maximum.at(
            best,
            rows[off_diagonal],
            np.asarray(graph.values, dtype=np.float64)[off_diagonal],
        )
        return best
```

---

## 6. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/spec.py`

17434 bytes · SHA-256 `a177505b14867755`

```python
"""
Specifications for the hybrid GNN-RNN model and its data build.

Every number the flagship needs that is not a weight lives here, validated at
load time rather than discovered at layer-construction time. The original
carried roughly sixty configuration fields across six nested dataclasses with
no validation; a typo in a layer's ``units`` surfaced as a shape error
somewhere inside a forward pass, several minutes into a run.

What is *not* here
------------------
Sequence length, split fractions, batch size, batch shuffling, scaling
method and whether the reduction basis may see held-out rows are all
framework settings, carried on the ``SourceSpec`` this module is given.
They are deliberately not repeated here.

An earlier draft did repeat them, and that was a mistake worth naming: two
fields meaning the same thing is one field and a bug waiting for someone to
set the wrong one. It also meant re-implementing machinery the framework
already provides and tests -- an explicit-split strategy for parity replay,
a leakage annotation written into the run lineage, and the structural
separation of batch order from scenario split that fixes defect 3.

The one compatibility flag that *is* model-specific is
``AttributeEncoderSpec.numeric_precision``, with its twin
``GraphSpec.precision``. Both default to the correct behaviour, so
forgetting to set one fails safe.

Defect 9 -- the reduction basis being selected over the full scaled history
-- is reproduced through the framework's ``transforms.reduction.fit_on``,
which already defaults to ``"train"`` and already writes a leakage warning
into the lineage when set to ``"all"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ...core.spec.base import Spec

#: Activations the layers accept, matching the set the original supported.
#: A literal rather than a free string, so a typo is a specification error
#: rather than a crash several minutes into a run.
ActivationName = Literal["relu", "leaky_relu", "tanh", "sigmoid", "elu", "selu", "gelu", "linear"]

__all__ = [
    "ActivationName",
    "AttributeEncoderSpec",
    "GraphSpec",
    "HybridDataSpec",
    "HybridModelSpec",
]


class AttributeEncoderSpec(Spec):
    """
    How instrument attributes become a feature matrix.

    Fitted along the **entity axis** over the full universe, which is
    legitimate rather than leakage: which instruments exist, and what their
    attributes are, is known before any P&L is observed. Restricting the
    encoder to training-period instruments would discard information a
    trader has rather than information they lack.

    Parameters
    ----------
    numeric_keys
        Attributes scaled to zero mean and unit variance.
    categorical_keys
        Attributes one-hot encoded.
    multi_label_keys
        Attributes holding a list per instrument, binarised and then
        row-normalised so an instrument with three risk factors is not
        three times as far from the origin as one with a single factor.
    maturity_key
        Which numeric attribute the decay features are derived from.
    numeric_precision
        **Compatibility flag.** Which precision the numeric attributes are
        rounded to and scaled in. The original rounded its attributes to
        ``float32``, computed the mean and standard deviation in ``float64``
        over those rounded values, then downcast both statistics back to
        ``float32`` to centre and scale -- which loses roughly six decimal
        digits for no benefit on a matrix of a few hundred rows.
        ``"float64"`` is correct and is the default; ``"float32"`` reproduces
        the original exactly and is used by the parity tests only.
    n_decay_terms
        How many exponential-decay features to derive from maturity. Each
        is ``exp(-lambda * tau)`` for a lambda on a linear grid from 10.0
        down to 0.1, which gives the model a short, a medium and a long view
        of time to maturity rather than one raw number whose relationship to
        value is strongly non-linear.
    """

    numeric_keys: tuple[str, ...] = ("moneyness", "yrs_to_maturity", "delta", "vega")
    categorical_keys: tuple[str, ...] = ("product_type", "product_subtype", "trade_type")
    multi_label_keys: tuple[str, ...] = ("underlying_risk_factors",)
    maturity_key: str = "yrs_to_maturity"
    numeric_precision: Literal["float64", "float32"] = "float64"
    n_decay_terms: int = Field(default=3, ge=0, le=16)

    @model_validator(mode="after")
    def _check_maturity_key_is_numeric(self) -> AttributeEncoderSpec:
        """
        Reject a maturity key that is not among the numeric attributes.

        Returns
        -------
        AttributeEncoderSpec
            The validated spec.

        Raises
        ------
        ValueError
            If decay terms are requested from an attribute that is not
            numeric. Unchecked, the decay features would be computed from
            whatever that attribute held and the failure would surface as an
            unhelpful cast error inside the encoder.
        """
        if self.n_decay_terms and self.maturity_key not in self.numeric_keys:
            raise ValueError(
                f"maturity_key={self.maturity_key!r} must be one of numeric_keys "
                f"{self.numeric_keys} for its {self.n_decay_terms} decay term(s) to "
                f"be computable"
            )
        return self


class GraphSpec(Spec):
    """
    How the instrument graph is built from encoded attributes.

    Also fitted along the entity axis over the full universe, for the same
    reason the encoder is.

    The ``alpha_*`` weights scale each attribute group before distances are
    measured. They are what stop a one-hot product type with eight levels
    from dominating a single scaled delta purely by occupying more columns.

    Parameters
    ----------
    n_neighbours
        How many neighbours each instrument connects to. Clamped down at
        build time when the universe is smaller than this.
    distance_metric
        How distance between two instruments' attribute vectors is measured.
    include_quota
        Whether a target instrument's neighbourhood is required to contain a
        minimum number of elementary instruments. Without it, targets in a
        dense cluster connect only to each other and the graph carries no
        path from a target to anything that could replicate it.
    min_elementary_neighbours
        The quota's elementary minimum, when enforced.
    min_target_neighbours
        The quota's target minimum, when enforced.
    alpha_moneyness, alpha_maturity, alpha_delta, alpha_vega
        Weights on the numeric attribute groups.
    alpha_product_type, alpha_product_subtype
        Weights on the categorical groups.
    alpha_underlying, alpha_underlying_risk_factors
        Weights on the underlying and its risk-factor set.
    precision
        **Compatibility flag**, and the twin of
        :attr:`AttributeEncoderSpec.numeric_precision`. The original summed
        each row's edge weights and took their reciprocal in ``float32``,
        which costs a last bit on every edge. ``"float64"`` is correct and
        is the default; ``"float32"`` reproduces the original. The parity
        tests set this and the encoder's together, because they are one
        decision -- reproduce the baseline's feature arithmetic -- expressed
        in the two places it applies.
    """

    n_neighbours: int = Field(default=5, ge=1)
    precision: Literal["float64", "float32"] = "float64"
    distance_metric: Literal["euclidean", "manhattan", "cosine"] = "euclidean"
    include_quota: bool = False
    min_elementary_neighbours: int = Field(default=2, ge=0)
    min_target_neighbours: int = Field(default=1, ge=0)

    alpha_moneyness: float = Field(default=1.0, ge=0.0)
    alpha_maturity: float = Field(default=1.0, ge=0.0)
    alpha_delta: float = Field(default=1.0, ge=0.0)
    alpha_vega: float = Field(default=1.0, ge=0.0)
    alpha_product_type: float = Field(default=1.0, ge=0.0)
    alpha_product_subtype: float = Field(default=0.5, ge=0.0)
    alpha_underlying: float = Field(default=1.0, ge=0.0)
    alpha_underlying_risk_factors: float = Field(default=0.5, ge=0.0)


class HybridDataSpec(Spec):
    """
    The model-specific half of the flagship's data build.

    Carried in ``ModelSourceSpec.params`` and validated here. Everything the
    framework already owns -- splitting, sequence length, scaling, batching,
    caching -- stays on the source spec and is not repeated.

    Parameters
    ----------
    directory
        Folder holding this cluster's P&L, attributes and universe. Carried
        here rather than on the source spec because ``ModelSourceSpec`` has
        no path: a model's raw input is not always one file, and for this
        model it is five.
    variance_threshold
        How much of each group's variance the selected basis must span.
        Higher keeps more instruments. Not expressible as the framework's
        ``reduction.n_components``, because the count is derived per group
        rather than fixed across the book.
    weight_tail
        Emphasis placed on tail scenarios when choosing the basis. One
        weights every scenario equally. Above one, a basis is chosen to span
        the days the book actually moved rather than the quiet majority.
    scale_targets
        Whether target P&L is standardised. Recorded on the fitted state so
        that inverting a prediction cannot guess wrong.
    encoder
        Attribute encoding settings.
    graph
        Graph construction settings.
    """

    directory: Path | None = None
    variance_threshold: float = Field(default=0.99, gt=0.0, le=1.0)
    weight_tail: float = Field(default=1.0, ge=1.0)
    scale_targets: bool = True
    encoder: AttributeEncoderSpec = Field(default_factory=AttributeEncoderSpec)
    graph: GraphSpec = Field(default_factory=GraphSpec)


class HybridModelSpec(Spec):
    """
    The network's shape.

    Deliberately flat. The original nested six layer configurations two
    levels deep, each with a ``general`` and a ``parameters`` sub-dictionary,
    which made the common case -- "make every layer wider" -- a six-place
    edit. Here the widths that are almost always equal share one field and
    the per-layer overrides are optional.

    Parameters
    ----------
    units
        The width used by every block that does not override it. One number,
        because tuning these independently is rare and tuning them together
        is constant.
    gnn_units, rnn_units, fusion_units, attention_units, projection_units
        Per-block overrides. ``None`` means ``units``.
    gnn_layers
        How many message-passing rounds. Each round widens an instrument's
        receptive field by one hop, so this is the radius of the
        neighbourhood the model can see.
    gnn_type
        Which message-passing rule. ``mixed_graph_sage`` concatenates an
        instrument with the mean *and* the maximum over its neighbours, so
        the layer sees both the typical neighbour and the extreme one --
        which for a risk graph is the one that matters.
        ``graph_sage`` uses a single aggregate.
    gnn_aggregation
        Which aggregate ``graph_sage`` uses. Ignored by
        ``mixed_graph_sage``, which uses both.
    gnn_activation
        Non-linearity between message-passing rounds and after the residual
        add. The sublayers themselves are linear, so this is applied by the
        block rather than inside each round.
    gnn_normalise
        Whether to normalise each round's output. Message passing repeatedly
        averages, which shrinks activations towards the graph mean; without
        normalisation a deep stack converges to a constant and the
        instruments become indistinguishable.
    rnn_type
        Which recurrence encodes the P&L history. ``bilstm`` doubles the
        output width, which the fusion block accounts for.
    rnn_layers
        Stacked recurrent layers.
    fusion_mode
        How the attended graph stream and the recurrent stream are
        combined. ``gate`` learns a per-feature blend; ``add`` sums them.
        A gate is the default because the right blend is not the same for
        every instrument -- a liquid instrument's own history is worth more
        than its neighbours', and a thinly traded one's is worth less.
    fusion_heads, attention_heads
        How many attention heads each block uses. Defaulted to the values
        the original ran in production rather than to a textbook four: a
        refactor that silently changes a hyperparameter is no longer
        comparable with the model it replaced, and that is a modelling
        decision to take deliberately rather than inherit from a default.
    attention_activation
        Non-linearity inside the target block's feed-forward network.
    attention_normalise
        Whether the target block normalises after each sublayer. A
        transformer block without it is unstable at any depth.
    neighbour_cap
        Most neighbours any node attends over. Bounds attention at
        ``O(n·k)`` rather than ``O(n²)``, which is what keeps a large
        universe tractable.
        Heads in the target-attention block.
    projection_activation
        Non-linearity inside the output head's residual network.
    baseline_weight_norm
        Whether each target's baseline kernel is split into a unit-norm
        direction and a positive magnitude. Decoupling the two makes the
        gradient on direction independent of how large the kernel has
        grown, which keeps a target whose baseline has learnt a big
        amplitude from also learning slowly.
    new_target_blend
        How an instrument with no fitted baseline borrows one from the
        instruments that have: by softmaxed cosine similarity in attribute
        space, or by inverse distance.
    new_target_neighbours
        How many fitted instruments a new one borrows from.
    new_target_temperature
        Sharpness of the cosine blend. Larger concentrates the borrow on
        the single closest instrument.
    new_target_distance_power
        Exponent on the inverse-distance blend.
    new_target_residual_damping
        Scales the residual correction for new instruments. Below one
        because the residual network never saw them, so its correction is
        extrapolation rather than fit.
    dropout
        Applied in the GNN, attention and projection blocks.
    use_residual
        Whether message-passing rounds carry a residual connection. With
        several rounds and no residual, early-layer signal is lost.
    """

    units: int = Field(default=16, ge=1)
    gnn_units: int | None = Field(default=None, ge=1)
    rnn_units: int | None = Field(default=None, ge=1)
    fusion_units: int | None = Field(default=None, ge=1)
    attention_units: int | None = Field(default=None, ge=1)
    projection_units: int | None = Field(default=None, ge=1)

    gnn_layers: int = Field(default=2, ge=1)
    gnn_type: Literal["graph_sage", "mixed_graph_sage"] = "mixed_graph_sage"
    gnn_aggregation: Literal["mean", "max"] = "mean"
    gnn_activation: ActivationName = "relu"
    gnn_normalise: bool = True
    rnn_type: Literal["lstm", "gru", "bilstm"] = "lstm"
    rnn_layers: int = Field(default=2, ge=1)
    fusion_mode: Literal["gate", "add"] = "gate"
    fusion_heads: int = Field(default=1, ge=1)
    neighbour_cap: int = Field(default=50, ge=1)
    attention_heads: int = Field(default=1, ge=1)
    attention_activation: ActivationName = "tanh"
    attention_normalise: bool = True
    projection_activation: ActivationName = "gelu"
    baseline_weight_norm: bool = False
    new_target_blend: Literal["cosine_softmax", "inverse_distance"] = "cosine_softmax"
    new_target_neighbours: int = Field(default=5, ge=1)
    new_target_temperature: float = Field(default=5.0, gt=0.0)
    new_target_distance_power: float = Field(default=2.0, gt=0.0)
    new_target_residual_damping: float = Field(default=1.0, ge=0.0, le=1.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_residual: bool = True

    def width(self, block: str) -> int:
        """
        Return the width for one block, falling back to the shared default.

        Parameters
        ----------
        block
            Block name: ``gnn``, ``rnn``, ``fusion``, ``attention`` or
            ``projection``.

        Returns
        -------
        int
            The resolved width.

        Raises
        ------
        ValueError
            If the block name is not one of the five. Raised rather than
            silently returning the default, because a typo would otherwise
            produce a model that is quietly the wrong shape.
        """
        override = getattr(self, f"{block}_units", _MISSING)
        if override is _MISSING:
            raise ValueError(
                f"unknown block {block!r}; expected one of gnn, rnn, fusion, attention, projection"
            )
        return self.units if override is None else int(override)


#: Sentinel distinguishing "no override set" from "block does not exist".
#: `None` cannot serve, because `None` is the legitimate value meaning
#: "fall back to the shared width".
_MISSING = object()
```

---

## 7. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/state.py`

16887 bytes · SHA-256 `3cfb158385bbfb3f`

```python
"""
Everything the flagship fits at training time and needs again at inference.

What this replaces
------------------
The original scattered this across eleven loose files written by the train
pipeline, with the relationship between them encoded only in the code that
loaded them::

    graph_builder.pkl         a pickled TradeGraphBuilder
    graph_results.joblib      its output, pickled separately
    encoder.pkl               a pickled TradeAttributeEncoder
    encoder_results.joblib    its output, pickled separately
    elementary_scaler.pkl     a pickled sklearn StandardScaler
    target_scaler.pkl         another one
    data_config.json          the settings all of the above were fitted under
    trade_universe.json       which instruments the columns refer to
    elementary_attributes.json
    target_attributes.json
    cluster_rf_keys.json

Three problems followed from that, and all three are the reason this module
exists.

The first is that nothing tied them together. Loading ten of the eleven
produced a model that ran and was wrong, because the eleventh -- most often
the target scaler -- was the one that put predictions back into currency.

The second is that six of them were pickles of third-party objects. A saved
model was therefore pinned to the scikit-learn and SciPy versions that wrote
it, and upgrading either silently broke every model in the archive. The
encoder and the graph are reimplemented in plain NumPy precisely so that
this state is arrays and JSON all the way down.

The third is that the split between "the fitted object" and "its results"
was arbitrary. Both were needed, always, and keeping them apart just created
a way to load half of a thing.

The ordering trap
-----------------
Two fields here are ordered sequences that look like sets, and treating
either as a set produces a model that trains happily and predicts nonsense:

* ``selected_basis`` is the ordered list of elementary instruments the basis
  selection kept. Its order *is* the column order of the feature matrix.
* ``universe.elementary_ids`` is what those columns refer to.

Both are compared element by element against the baseline, in order, by
parity level 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ...core.contract.state import FittedState
from ...core.lifecycle.errors import BundleError
from .features.encoder import EntityEncoderState
from .features.graph import SparseGraphState

__all__ = ["HybridState", "StandardScalerState", "Universe"]

#: Replaces a zero standard deviation. A constant feature carries no
#: information, so the choice only has to avoid dividing by zero; one leaves
#: the centred column at zero, which is the honest encoding of "no variation".
_ZERO_SCALE_REPLACEMENT = 1.0


@dataclass(frozen=True, slots=True)
class Universe:
    """
    Which instruments the feature and target columns refer to.

    A replication problem has two sides. The **elementary** instruments are
    the liquid things a desk can actually trade -- the hedging basis. The
    **target** instruments are the illiquid or structured positions whose P&L
    is being replicated. This model predicts the second from the first, and a
    universe is the record of which is which, in which order.

    Why the ordering is the dangerous part
    --------------------------------------
    The ordering is the contract between a matrix column and a real
    instrument. A prediction is a vector of numbers; without a universe it is
    a vector of numbers about nothing. Getting the order wrong does not raise
    -- it produces plausible predictions attributed to the wrong instruments,
    which is the most expensive mistake available in this problem and the
    hardest to see.

    Frozen for that reason: a universe is the thing every downstream index is
    interpreted against. A fitted state, a prediction and a report all read
    positions out of it, and one that could be reordered after a fit would
    silently reattribute every one of them.

    Why this lives in the model rather than the framework
    -----------------------------------------------------
    "Elementary" and "target" are P&L replication vocabulary, not framework
    vocabulary. The framework's job is to carry whatever labels a model
    declares, which it does through
    :class:`~rade_qnet.core.contract.signature.InputSignature`; it has no
    opinion about what the labels mean. Read from this model's own
    ``universe.json`` by its ``data.py``, so a second model over the same
    data reads the same file rather than importing from here.

    Parameters
    ----------
    elementary_ids
        The elementary instruments, **in column order** and already reduced
        to the selected basis. Already reduced, because the alternative --
        carrying the full set and a separate index of survivors -- means
        every consumer has to apply the reduction itself, and one that
        forgets produces results that are wrong rather than absent.
    target_ids
        The target instruments, in column order.
    """

    elementary_ids: tuple[str, ...]
    target_ids: tuple[str, ...]

    @property
    def n_elementary(self) -> int:
        """
        How many elementary instruments survived basis selection.

        Returns
        -------
        int
            The count.
        """
        return len(self.elementary_ids)

    @property
    def n_targets(self) -> int:
        """
        How many target instruments are predicted.

        Returns
        -------
        int
            The count.
        """
        return len(self.target_ids)

    @property
    def instrument_ids(self) -> tuple[str, ...]:
        """
        Every instrument, elementary block first.

        The row order of the combined attribute matrix, which is why the
        concatenation happens here rather than at each call site: two places
        choosing the same order by convention is a convention that will
        eventually be broken by someone who did not know it existed.

        Returns
        -------
        tuple of str
            Elementary identifiers followed by target identifiers.
        """
        return self.elementary_ids + self.target_ids

    def position_of(self, instrument_id: str) -> int:
        """
        Return an instrument's row in the combined attribute matrix.

        Parameters
        ----------
        instrument_id
            The identifier.

        Returns
        -------
        int
            Its position.

        Raises
        ------
        KeyError
            If the universe does not contain it. Raised rather than returning
            ``-1``, which indexes the last row perfectly happily and would
            attribute a prediction to whichever instrument happened to be
            there.
        """
        try:
            return self.instrument_ids.index(instrument_id)
        except ValueError:
            raise KeyError(
                f"{instrument_id!r} is not in this universe of "
                f"{self.n_elementary} elementary and {self.n_targets} target instrument(s)"
            ) from None


@dataclass(frozen=True, slots=True)
class StandardScalerState:
    """
    A fitted mean and scale, as two plain arrays.

    Replaces a pickled ``sklearn.preprocessing.StandardScaler``. The
    arithmetic is four lines, and holding it here rather than in a pickled
    estimator is what lets a model saved today load in five years.

    Parameters
    ----------
    centre
        Per-column mean, held in float64 even when the data is float32. A
        mean accumulated in float32 over thousands of scenarios is
        measurably wrong, and nothing downstream can recover the lost
        digits.
    scale
        Per-column standard deviation, with zeros already replaced.
    """

    centre: NDArray[np.float64]
    scale: NDArray[np.float64]

    @classmethod
    def fit(cls, values: NDArray[np.floating]) -> Self:
        """
        Fit a mean and scale over the rows given.

        The caller is responsible for passing *training rows only*. This is
        the scenario axis, where fitting over the validation or test rows
        would leak the future into the past -- unlike the entity axis, where
        fitting over the whole universe is correct.

        Parameters
        ----------
        values
            Shape ``(n_rows, n_columns)``.

        Returns
        -------
        StandardScalerState
            The fitted statistics.
        """
        wide = np.asarray(values, dtype=np.float64)
        scale = wide.std(axis=0)
        return cls(
            centre=wide.mean(axis=0),
            scale=np.where(scale > 0.0, scale, _ZERO_SCALE_REPLACEMENT),
        )

    def transform(self, values: NDArray[np.floating]) -> NDArray[np.float32]:
        """
        Centre and scale, returning float32 for the network.

        Parameters
        ----------
        values
            Shape ``(n_rows, n_columns)``.

        Returns
        -------
        numpy.ndarray
            The standardised values.
        """
        wide = np.asarray(values, dtype=np.float64)
        return ((wide - self.centre) / self.scale).astype(np.float32)

    def inverse_transform(self, values: NDArray[np.floating]) -> NDArray[np.float64]:
        """
        Undo :meth:`transform`.

        Parameters
        ----------
        values
            Standardised values.

        Returns
        -------
        numpy.ndarray
            Values in the original units.
        """
        return np.asarray(values, dtype=np.float64) * self.scale + self.centre


@dataclass(frozen=True, slots=True)
class HybridState(FittedState):
    """
    The flagship's complete fitted state.

    Parameters
    ----------
    feature_scaler
        Standardises the elementary P&L columns. Fitted on training
        scenarios only.
    target_scaler
        Standardises the target P&L columns, and the only thing that can put
        a prediction back into currency.
    selected_basis
        The elementary instruments basis selection kept, **in order**. The
        order is the column order of the feature matrix.
    encoder
        The fitted attribute encoder, over the full universe.
    graph
        The fitted instrument graph, over the full universe.
    universe
        Which instrument each column refers to.
    elementary_indices, target_indices
        Where the elementary and target blocks sit in the combined attribute
        matrix. Recomputed **after** basis selection, as ``0..n_e`` and
        ``n_e..n_e + n_t``. Carrying the pre-reduction indices forward
        instead produces arrays that look entirely plausible and index the
        wrong rows of the encoding, which is why parity level 1 compares
        them explicitly.
    scale_targets
        Whether the target scaler was applied. Recorded rather than
        inferred, so that :meth:`inverse_transform_targets` cannot guess
        wrong.
    """

    feature_scaler: StandardScalerState
    target_scaler: StandardScalerState
    selected_basis: tuple[str, ...]
    encoder: EntityEncoderState
    graph: SparseGraphState
    universe: Universe
    elementary_indices: NDArray[np.int64]
    target_indices: NDArray[np.int64]
    scale_targets: bool = True

    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Put predictions back into the target's original currency units.

        Called by the evaluate and infer pipelines before any metric is
        computed. Without it a mean absolute error is reported in
        standardised units -- a number that looks fine, compares fine
        between runs, and means nothing to anyone sizing a hedge.

        Parameters
        ----------
        predictions
            Model output, shape ``(n_rows, n_targets)``.

        Returns
        -------
        numpy.ndarray
            The same shape, in original units.
        """
        if not self.scale_targets:
            return predictions
        return self.target_scaler.inverse_transform(predictions)

    def save(self, directory: Path) -> None:
        """
        Write the state as arrays and JSON, with no pickles anywhere.

        Parameters
        ----------
        directory
            Destination, created by the caller but created here too so the
            state can be saved standalone in a test.
        """
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "scalers.npz",
            feature_centre=self.feature_scaler.centre,
            feature_scale=self.feature_scaler.scale,
            target_centre=self.target_scaler.centre,
            target_scale=self.target_scaler.scale,
            elementary_indices=self.elementary_indices,
            target_indices=self.target_indices,
        )
        (directory / "universe.json").write_text(
            json.dumps(
                {
                    # Lists, not sets, and the order is the column order.
                    "selected_basis": list(self.selected_basis),
                    "elementary_ids": list(self.universe.elementary_ids),
                    "target_ids": list(self.universe.target_ids),
                    "scale_targets": self.scale_targets,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.encoder.save(directory / "encoder")
        self.graph.save(directory / "graph")

    @classmethod
    def load(cls, directory: Path) -> Self:
        """
        Read a state written by :meth:`save`.

        Parameters
        ----------
        directory
            Directory previously written by :meth:`save`.

        Returns
        -------
        HybridState
            A state equal to the one saved.

        Raises
        ------
        BundleError
            If any component is missing. Reported as one message listing
            everything absent, because the original's failure mode -- load
            ten of eleven files and predict confidently in the wrong units
            -- is exactly what a partial load must not be allowed to become.
        """
        required = {
            "scalers.npz": directory / "scalers.npz",
            "universe.json": directory / "universe.json",
            "encoder/": directory / "encoder",
            "graph/": directory / "graph",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise BundleError(
                f"cannot load the hybrid state from {directory}: missing "
                f"{', '.join(sorted(missing))}. Every component is required -- a state "
                f"loaded without its target scaler predicts in standardised units and "
                f"reports metrics that look reasonable and are not"
            )

        payload = json.loads(required["universe.json"].read_text(encoding="utf-8"))
        with np.load(required["scalers.npz"]) as arrays:
            return cls(
                feature_scaler=StandardScalerState(
                    centre=arrays["feature_centre"], scale=arrays["feature_scale"]
                ),
                target_scaler=StandardScalerState(
                    centre=arrays["target_centre"], scale=arrays["target_scale"]
                ),
                selected_basis=tuple(payload["selected_basis"]),
                encoder=EntityEncoderState.load(required["encoder/"]),
                graph=SparseGraphState.load(required["graph/"]),
                universe=Universe(
                    elementary_ids=tuple(payload["elementary_ids"]),
                    target_ids=tuple(payload["target_ids"]),
                ),
                elementary_indices=arrays["elementary_indices"],
                target_indices=arrays["target_indices"],
                scale_targets=bool(payload["scale_targets"]),
            )

    def describe(self) -> dict[str, object]:
        """
        Summarise what was fitted, for the run report.

        The basis-selection ratio is the headline: it is the first place a
        misconfigured variance threshold becomes visible, and a run that
        kept every instrument has not reduced anything.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {
            "type": type(self).__name__,
            "n_elementary_selected": len(self.selected_basis),
            "n_targets": self.universe.n_targets,
            "n_encoded_features": self.encoder.n_features,
            "n_graph_nodes": self.graph.n_nodes,
            "n_graph_edges": self.graph.n_edges,
            "scale_targets": self.scale_targets,
        }
```

---

## 8. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/visuals.py`

8717 bytes · SHA-256 `c033d0fcfc405089`

```python
"""
Figures only this model can produce.

The shared figures in :mod:`rade_qnet.analysis.visuals` work for any model
because they only look at predictions and losses. These look at the graph,
which no other model has.

The same rule applies here as there: **a figure factory returns a figure and
writes nothing.** A test asserts on the axes without a temporary directory,
a notebook displays the result without producing a file, and the report
writer decides where it lands.

Why the graph deserves its own figures
---------------------------------------
The graph is the model's single largest modelling assumption and the one
least visible in any metric. A poorly built graph does not announce itself:
the loss still falls, the metrics still look reasonable, and the damage
shows up only on instruments the training set under-covered -- which is
precisely the population the model exists to price.

These three figures are the diagnostics that make that failure visible
before it costs anything:

- the degree distribution says whether the neighbourhood size is uniform or
  whether a handful of instruments are carrying the whole book;
- the edge-weight distribution says whether neighbours are genuinely
  similar or whether the kernel has flattened into an average;
- the neighbour-similarity profile says, per instrument, how good its
  *best* neighbour actually is -- which is what an unfitted target's
  borrowed baseline will be built from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from ...analysis.visuals.style import figure_style
from ...core.lifecycle.errors import ContractError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .features.graph import SparseGraphState

__all__ = [
    "edge_weight_figure",
    "neighbour_similarity_figure",
    "node_degree_figure",
]

#: Default figure size, in inches. Matches the shared primitives so a report
#: mixing both does not have figures of two different widths.
DEFAULT_FIGSIZE = (8.0, 4.5)

#: How many histogram bins. Enough to show a bimodal weight distribution --
#: the signature of a graph that has split into tight cliques -- without
#: being so fine that a 200-instrument cluster shows mostly empty bins.
_N_BINS = 40

#: Quantiles reported alongside the neighbour-similarity profile. The low
#: tail is the interesting half: the median instrument's best neighbour is
#: rarely the problem.
_QUANTILES = (0.01, 0.05, 0.25, 0.50)


def node_degree_figure(graph: SparseGraphState, *, title: str = "Neighbourhood size") -> Figure:
    """
    Plot how many neighbours each instrument has.

    A graph built with a fixed neighbour count produces a near-vertical
    line and is uninteresting; the figure earns its place when a threshold
    or a kernel cutoff was used, where it reveals the instruments that
    ended up nearly isolated. Those instruments get almost no information
    from the graph, so for them the model degenerates to the temporal
    stream alone.

    Parameters
    ----------
    graph
        The fitted graph.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.
    """
    degrees = graph.degrees()
    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        # Integer-aligned bins: a degree histogram on continuous bins
        # straddles whole numbers and reads as though degrees were
        # fractional.
        edges = np.arange(degrees.min(), degrees.max() + 2) - 0.5
        axes.hist(degrees, bins=edges, edgecolor="white", linewidth=0.5)
        axes.axvline(
            float(np.median(degrees)),
            color="#808080",
            linestyle="--",
            linewidth=1.0,
            label=f"median ({int(np.median(degrees))})",
        )
        axes.set_xlabel("neighbours")
        axes.set_ylabel("instruments")
        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure


def edge_weight_figure(graph: SparseGraphState, *, title: str = "Edge weights") -> Figure:
    """
    Plot the distribution of normalised edge weights.

    Rows are normalised, so the weights on one instrument's edges sum to
    one. Mass concentrated near ``1/degree`` means the kernel is treating
    every neighbour alike, which makes the graph an unweighted average and
    discards the similarity information it was built to carry. A long right
    tail is the healthy shape: each instrument has a few neighbours that
    genuinely matter.

    Parameters
    ----------
    graph
        The fitted graph.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the graph has no edges, which would make the figure empty and
        is a far more serious finding than a missing plot.
    """
    weights = np.asarray(graph.values, dtype=np.float64)
    if weights.size == 0:
        raise ContractError(
            "the graph has no edges, so there are no weights to plot; this is a "
            "failure of the graph build rather than of the figure"
        )

    uniform = 1.0 / np.maximum(graph.degrees().mean(), 1.0)
    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        axes.hist(weights, bins=_N_BINS, edgecolor="white", linewidth=0.5)
        axes.axvline(
            uniform,
            color="#808080",
            linestyle="--",
            linewidth=1.0,
            label=f"uniform ({uniform:.3f})",
        )
        axes.set_xlabel("normalised edge weight")
        axes.set_ylabel("edges")
        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure


def neighbour_similarity_figure(
    graph: SparseGraphState, *, title: str = "Best-neighbour weight by instrument"
) -> Figure:
    """
    Plot each instrument's largest edge weight, sorted.

    This is the figure to read before trusting a prediction for an
    instrument that was not in the training set. Such an instrument has no
    fitted baseline and borrows one from its nearest neighbours, so the
    quality of that borrow is bounded by how close its closest neighbour
    actually is. The left end of this curve is the population at risk.

    Parameters
    ----------
    graph
        The fitted graph.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the graph has no edges.
    """
    best = _best_weight_per_node(graph)
    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        axes.plot(np.sort(best), marker="")
        for quantile in _QUANTILES:
            value = float(np.quantile(best, quantile))
            axes.axhline(
                value,
                color="#808080",
                linestyle=":",
                linewidth=0.8,
            )
            # Annotated on the axis rather than in a legend: four legend
            # entries of the same style are harder to read than four
            # labels sitting on the lines they describe.
            axes.annotate(
                f"q{int(quantile * 100):02d} = {value:.3f}",
                xy=(0, value),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize="small",
                color="#505050",
            )
        axes.set_xlabel("instrument (sorted)")
        axes.set_ylabel("largest edge weight")
        axes.set_title(title)
        figure.tight_layout()
        return figure


def _best_weight_per_node(graph: SparseGraphState) -> NDArray[np.float64]:
    """
    Find each node's largest edge weight.

    Parameters
    ----------
    graph
        The fitted graph.

    Returns
    -------
    numpy.ndarray
        One value per node. A node with no edges scores zero, which is the
        honest answer: it has no best neighbour.

    Raises
    ------
    ContractError
        If the graph has no edges.
    """
    values = np.asarray(graph.values, dtype=np.float64)
    if values.size == 0:
        raise ContractError(
            "the graph has no edges, so no instrument has a best neighbour; this "
            "is a failure of the graph build rather than of the figure"
        )
    rows = np.asarray(graph.indices, dtype=np.int64)[:, 0]
    best = np.zeros(graph.n_nodes, dtype=np.float64)
    np.maximum.at(best, rows, values)
    return best
```

