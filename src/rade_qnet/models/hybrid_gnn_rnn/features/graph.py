"""
Builds the sparse instrument graph the GNN passes messages over.

Like the attribute encoder, this is fitted along the **entity axis** over the
whole universe. Which instruments exist and how similar their attributes are
is known before any P&L is observed, so a graph spanning the full universe
uses no information a trader lacks.

The construction, and why each step is there
--------------------------------------------
1. **Weight the attribute groups.** Each group is scaled by the square root
   of its alpha, so squared Euclidean distance in the weighted space equals
   the alpha-weighted sum of squared differences. Without the square root
   the weights would apply to distance rather than to squared distance, and
   doubling an alpha would not double that attribute's influence.
2. **Divide one-hot blocks by the square root of their width.** A product
   type with eight levels otherwise contributes eight columns against a
   single scaled delta and dominates every distance purely by occupying more
   space.
3. **Find each instrument's nearest neighbours.**
4. **Convert distance to similarity with a Gaussian kernel**, using each
   node's own median neighbour distance as its bandwidth. A single global
   bandwidth would saturate in dense regions and vanish in sparse ones, so a
   thinly populated corner of the universe would end up with no usable edges
   at all.
5. **Add self-loops**, so a node's own features survive aggregation. Without
   them, message passing replaces each instrument with the average of its
   neighbours and the instrument's own attributes are lost after one round.
6. **Row-normalise.** Each row sums to one, which makes aggregation a mean
   rather than a sum and stops high-degree nodes from producing activations
   proportional to their degree.

Why this is implemented here rather than imported
-------------------------------------------------
The original used scikit-learn's neighbour search and SciPy's sparse
matrices. Both are reimplemented here in plain NumPy, for the same reasons
the encoder is: the framework's data path stays free of two heavy
dependencies, and the fitted state becomes plain arrays that serialise
without pickling an estimator.

The search is dense: it materialises the full pairwise distance matrix.
That is deliberate for the sizes this model runs at -- a cluster is tens to
low hundreds of instruments, where a dense matrix is a few hundred kilobytes
and is faster than any tree. It would be the wrong choice at tens of
thousands, and `n_instruments` is logged so that becoming the case is
visible.

The port is exact, and the golden fixture proves it: the edge list and the
edge weights are compared element for element at zero tolerance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ....core.runtime.errors import ContractError
from ....core.runtime.logging import get_logger
from ..spec import GraphSpec
from .encoder import EncodedAttributes

__all__ = ["SparseGraphState", "build_graph"]

#: The default working precision, as a module-level singleton so it can be a
#: parameter default without constructing a dtype on every call.
_FLOAT64 = np.dtype(np.float64)

#: Floor on a node's kernel bandwidth. A node whose neighbours all sit at
#: distance zero -- duplicate instruments -- would otherwise divide by zero
#: and produce NaN weights on every one of its edges.
_BANDWIDTH_FLOOR = 1e-9

#: Floor on the multi-label row norm, matching the original. Distinct from
#: the encoder's floor only because the two were written separately; the
#: value is immaterial since the rows arriving here are already unit norm.
_NORM_FLOOR = 1e-8

#: Above this many instruments the dense distance matrix stops being a
#: sensible choice. Warned rather than refused: a large universe should
#: still run, just not silently at quadratic cost.
_DENSE_SEARCH_WARNING_THRESHOLD = 5_000

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SparseGraphState:
    """
    The fitted graph, as a coordinate-format sparse matrix.

    Stored as an edge list rather than as a dense matrix because the graph
    is a few neighbours per node: at a hundred instruments and five
    neighbours the dense form is a hundred times larger and is almost
    entirely zeros.

    Parameters
    ----------
    indices
        Edge endpoints, shape ``(n_edges, 2)`` as ``(row, column)`` pairs,
        sorted row-major. The order is part of the state: it is what the
        comparison against the baseline checks, and two graphs with the same
        edges in a different order are the same graph but not the same
        array.
    values
        Edge weights, one per row of ``indices``, each row summing to one.
    n_nodes
        How many instruments. Carried explicitly because an instrument with
        no edges would otherwise be invisible in the edge list and the
        matrix would silently come out too small.
    is_target
        Per-node flag marking target instruments, needed by the neighbour
        quota and by the reports.
    """

    indices: NDArray[np.int64]
    values: NDArray[np.float32]
    n_nodes: int
    is_target: NDArray[np.bool_]

    @property
    def n_edges(self) -> int:
        """How many edges, self-loops included."""
        return int(self.indices.shape[0])

    @property
    def dense_shape(self) -> tuple[int, int]:
        """The shape the sparse matrix would have if densified."""
        return (self.n_nodes, self.n_nodes)

    def degrees(self) -> NDArray[np.int64]:
        """
        Return how many edges leave each node.

        Returns
        -------
        numpy.ndarray
            One count per node, self-loop included.
        """
        return np.bincount(self.indices[:, 0], minlength=self.n_nodes).astype(np.int64)

    def save(self, directory: Path) -> None:
        """
        Write the graph as arrays and JSON.

        Parameters
        ----------
        directory
            Destination, created if absent.
        """
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "graph.npz",
            indices=self.indices,
            values=self.values,
            is_target=self.is_target,
        )
        (directory / "graph.json").write_text(
            json.dumps({"n_nodes": self.n_nodes}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """
        Read a previously saved graph.

        Parameters
        ----------
        directory
            Where :meth:`save` wrote.

        Returns
        -------
        SparseGraphState
            The restored graph.
        """
        payload = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
        with np.load(directory / "graph.npz") as arrays:
            return cls(
                indices=arrays["indices"],
                values=arrays["values"],
                n_nodes=int(payload["n_nodes"]),
                is_target=arrays["is_target"],
            )


def build_graph(
    encoded: EncodedAttributes,
    *,
    spec: GraphSpec,
    is_target: NDArray[np.bool_],
) -> SparseGraphState:
    """
    Build the row-normalised, kernel-weighted neighbour graph.

    Parameters
    ----------
    encoded
        The encoder's output, whose block ranges say which columns belong to
        which attribute group.
    spec
        Neighbour count, metric, quota and the per-group weights.
    is_target
        Per-instrument flag. Needed before the graph exists because the
        quota constrains a target's neighbourhood.

    Returns
    -------
    SparseGraphState
        The fitted graph.

    Raises
    ------
    ContractError
        If every attribute group is weighted to zero, which would leave no
        features to measure distance over. Refused rather than returning a
        graph of arbitrary structure.
    """
    features = weighted_features(encoded, spec=spec)
    n_nodes = int(features.shape[0])

    if n_nodes > _DENSE_SEARCH_WARNING_THRESHOLD:
        _LOGGER.warning(
            "building a graph over %d instruments with a dense distance matrix, which "
            "costs %.1f GB and grows quadratically. This path is tuned for the tens-to-"
            "hundreds a cluster holds; a universe this size wants an approximate index",
            n_nodes,
            n_nodes * n_nodes * 4 / 1e9,
        )

    # Clamped to what exists. Asking for more neighbours than there are
    # other instruments is not an error -- a small cluster is legitimate --
    # but silently returning fewer edges than requested would make the
    # graph's density depend on the universe size in a way nothing reports.
    n_neighbours = min(spec.n_neighbours, max(n_nodes - 1, 1))
    if n_neighbours < spec.n_neighbours:
        _LOGGER.info(
            "reduced n_neighbours from %d to %d: the universe holds %d instrument(s)",
            spec.n_neighbours,
            n_neighbours,
            n_nodes,
        )

    distances, neighbours = _nearest_neighbours(
        features, n_neighbours=n_neighbours, metric=spec.distance_metric
    )
    weights = _kernel_weights(distances)

    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), neighbours.shape[1])
    columns = neighbours.ravel().astype(np.int64)
    data = weights.ravel().astype(np.float32)

    # Self-loops, so aggregation keeps a node's own features. Appended after
    # the neighbour edges and before normalisation, so each node's own
    # weight participates in its row sum.
    diagonal = np.arange(n_nodes, dtype=np.int64)
    rows = np.concatenate([rows, diagonal])
    columns = np.concatenate([columns, diagonal])
    data = np.concatenate([data, np.ones(n_nodes, dtype=np.float32)])

    indices, values = _normalise_rows(
        rows, columns, data, n_nodes=n_nodes, precision=np.dtype(spec.precision)
    )
    _LOGGER.info(
        "built a graph over %d instrument(s) with %d edge(s), %d target(s)",
        n_nodes,
        len(values),
        int(is_target.sum()),
    )
    return SparseGraphState(
        indices=indices, values=values, n_nodes=n_nodes, is_target=np.asarray(is_target, dtype=bool)
    )


def weighted_features(encoded: EncodedAttributes, *, spec: GraphSpec) -> NDArray[np.float32]:
    """
    Scale each attribute group so Euclidean distance becomes alpha-weighted.

    Each group is multiplied by the square root of its weight, so that the
    *squared* distance between two instruments is the weighted sum of
    squared per-attribute differences. Multiplying by the weight itself
    would make the weights apply to distance rather than to squared
    distance, and an alpha of four would quadruple an attribute's influence
    rather than double it.

    One-hot groups are additionally divided by the square root of their
    width, so a categorical attribute's influence depends on its weight
    rather than on how many levels it happens to have.

    Parameters
    ----------
    encoded
        The encoder's output.
    spec
        The per-group weights.

    Returns
    -------
    numpy.ndarray
        The weighted feature matrix.

    Raises
    ------
    ContractError
        If no group survives weighting.
    """
    parts: list[NDArray[np.float32]] = []

    for key, weight in _group_weights(spec).items():
        span = encoded.blocks.get(key)
        if span is None or weight <= 0.0:
            continue
        block = encoded.features[:, span[0] : span[1]].astype(np.float32)

        if key in _MULTI_LABEL_GROUPS:
            # Already unit norm from the encoder, so this is idempotent. Kept
            # because the graph's guarantee -- that label count does not
            # drive distance -- should not depend on a detail of how the
            # encoder happened to normalise.
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            block = block / np.maximum(norms, _NORM_FLOOR)
        elif block.shape[1] > 1:
            block = block / np.sqrt(block.shape[1])

        parts.append(np.sqrt(np.float32(weight)) * block)

    if not parts:
        raise ContractError(
            "every attribute group is either absent from the encoding or weighted to "
            "zero, so there are no features to measure instrument distance over. Check "
            "the graph spec's alpha weights against the encoder's attribute keys"
        )
    return np.hstack(parts).astype(np.float32)


def _group_weights(spec: GraphSpec) -> Mapping[str, float]:
    """
    Map each attribute group to its weight, in column order.

    Ordered, and the order is load-bearing: it fixes the column layout of
    the weighted matrix. Distance is invariant to column order, so a
    reordering would not change the graph -- but it would change the
    captured feature matrix, and a comparison that passes for the wrong
    reason is worth avoiding.

    Parameters
    ----------
    spec
        The graph spec.

    Returns
    -------
    Mapping
        Attribute name to weight.
    """
    return {
        "moneyness": spec.alpha_moneyness,
        "yrs_to_maturity": spec.alpha_maturity,
        "delta": spec.alpha_delta,
        "vega": spec.alpha_vega,
        "product_type": spec.alpha_product_type,
        "product_subtype": spec.alpha_product_subtype,
        "underlying": spec.alpha_underlying,
        "underlying_risk_factors": spec.alpha_underlying_risk_factors,
    }


#: Groups normalised per row rather than per block width.
_MULTI_LABEL_GROUPS = frozenset({"underlying_risk_factors"})


def _nearest_neighbours(
    features: NDArray[np.float32], *, n_neighbours: int, metric: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Find each node's nearest neighbours, excluding itself.

    Parameters
    ----------
    features
        The weighted feature matrix.
    n_neighbours
        How many neighbours to return per node.
    metric
        ``euclidean``, ``manhattan`` or ``cosine``.

    Returns
    -------
    tuple
        Distances and indices, each shape ``(n_nodes, n_neighbours)``,
        sorted nearest first.
    """
    distances = _pairwise_distances(features, metric=metric)

    # A node is always its own nearest neighbour at distance zero, so the
    # search asks for one extra and drops the first column. Setting the
    # diagonal to infinity instead is more robust: with duplicate
    # instruments, two rows sit at distance zero from each other and the
    # self column is no longer guaranteed to sort first -- dropping column
    # zero would then drop a genuine neighbour and keep the self-loop twice.
    np.fill_diagonal(distances, np.inf)

    # A stable sort, so that instruments equidistant from a node are taken
    # in index order. With the default quicksort the tie order depends on
    # the array's layout, which would make the graph differ between two runs
    # on identical data.
    order = np.argsort(distances, axis=1, kind="stable")[:, :n_neighbours]
    rows = np.arange(features.shape[0])[:, None]
    return distances[rows, order], order.astype(np.int64)


def _pairwise_distances(features: NDArray[np.float32], *, metric: str) -> NDArray[np.float64]:
    """
    Compute the full distance matrix.

    Parameters
    ----------
    features
        The weighted feature matrix.
    metric
        Which distance to use.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_nodes, n_nodes)``, in float64 so that the subtraction
        below does not lose precision on nearly-identical instruments.

    Raises
    ------
    ContractError
        If the metric is not one the builder implements.
    """
    values = features.astype(np.float64)

    if metric == "euclidean":
        # Computed from the explicit difference rather than from the
        # expanded `|a|^2 - 2ab + |b|^2`. The expansion is faster and
        # cancels catastrophically for near-identical rows, which is
        # precisely the case here: instruments in the same group differ in
        # one attribute out of thirteen, and the expansion can return a
        # small negative number whose square root is NaN.
        difference = values[:, None, :] - values[None, :, :]
        return np.sqrt(np.einsum("ijk,ijk->ij", difference, difference))

    if metric == "manhattan":
        return np.abs(values[:, None, :] - values[None, :, :]).sum(axis=2)

    if metric == "cosine":
        norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), _NORM_FLOOR)
        unit = values / norms
        return 1.0 - unit @ unit.T

    raise ContractError(
        f"unknown distance metric {metric!r}; the graph builder implements "
        f"'euclidean', 'manhattan' and 'cosine'"
    )


def _kernel_weights(distances: NDArray[np.float64]) -> NDArray[np.float32]:
    """
    Turn distances into similarities with a per-node Gaussian kernel.

    Each node's bandwidth is the median of its own neighbour distances. A
    single global bandwidth would saturate in dense regions -- every weight
    near one, so the graph carries no information about which neighbour is
    closer -- and vanish in sparse ones, leaving isolated instruments with
    weights that underflow to zero and no usable edges at all.

    Parameters
    ----------
    distances
        Neighbour distances, shape ``(n_nodes, n_neighbours)``.

    Returns
    -------
    numpy.ndarray
        Similarities of the same shape.
    """
    bandwidth = np.median(np.maximum(distances, _BANDWIDTH_FLOOR), axis=1, keepdims=True).astype(
        np.float32
    )
    return np.exp(-(distances.astype(np.float32) ** 2) / (2.0 * bandwidth**2)).astype(np.float32)


def _normalise_rows(
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    data: NDArray[np.float32],
    *,
    n_nodes: int,
    precision: np.dtype = _FLOAT64,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """
    Sum duplicate edges, sort row-major, and scale each row to sum to one.

    Duplicates are summed rather than overwritten, which is what a
    coordinate-format matrix means by convention and what the original's
    sparse library did. They arise when the quota adds an edge the
    neighbour search already found.

    Row normalisation makes aggregation a mean. Without it a node with
    twenty neighbours produces activations roughly twenty times larger than
    one with a single neighbour, and the network spends its capacity
    learning to undo the degree.

    Parameters
    ----------
    rows, columns
        Edge endpoints.
    data
        Edge weights.
    n_nodes
        Matrix order.
    precision
        Which precision the row sums and their reciprocals are computed in.
        ``float32`` reproduces the original; ``float64`` is the default and
        loses nothing.

    Returns
    -------
    tuple
        The ``(n_edges, 2)`` index array and the matching weights.
    """
    # One integer key per cell, so duplicates become equal keys and a single
    # sort puts the whole matrix in row-major order. int64 holds this for
    # any universe up to three billion instruments.
    keys = rows * np.int64(n_nodes) + columns
    order = np.argsort(keys, kind="stable")
    keys, data = keys[order], data[order]

    unique_keys, start = np.unique(keys, return_index=True)
    summed = np.add.reduceat(data, start).astype(np.float32)

    unique_rows = (unique_keys // n_nodes).astype(np.int64)
    unique_columns = (unique_keys % n_nodes).astype(np.int64)

    # Accumulated in float64 and then rounded to the working precision, so
    # that `float32` matches a float32 row sum rather than a float32
    # accumulation -- the original summed through a sparse matrix whose
    # internal accumulation is not reproducible element by element, but
    # whose *result* is a float32 value.
    row_sums = np.bincount(unique_rows, weights=summed, minlength=n_nodes).astype(precision)
    # A node with no edges at all would divide by zero. It cannot happen
    # while self-loops are added, but the guard costs nothing and the
    # alternative is a NaN that propagates through every message-passing
    # round.
    inverse = np.divide(
        precision.type(1.0), row_sums, out=np.zeros_like(row_sums), where=row_sums > 0
    )
    normalised = (summed * inverse[unique_rows].astype(np.float32)).astype(np.float32)

    return np.column_stack([unique_rows, unique_columns]).astype(np.int64), normalised
