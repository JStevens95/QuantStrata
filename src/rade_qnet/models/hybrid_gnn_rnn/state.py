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
