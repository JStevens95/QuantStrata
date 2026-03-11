"""
Trade graph builder for Hybrid GNN-RNN model (PyTorch).

Builds k-NN adjacency from encoded trade attributes for both training and inference.
The graph is deterministic for a given trade universe — reuse the same adjacency
when the trade set is unchanged (e.g. same trades, new PnL scenarios).

Mathematical summary
--------------------
1. Feature space:  f_i = [sqrt(a1)*x1, sqrt(a2)*x2, ...]
   so that ||f_i - f_j||^2 = sum_d( alpha_d * (x_id - x_jd)^2 )  (diagonal Mahalanobis).

2. k-NN graph:     For each trade i, find k nearest neighbours N(i) in feature space.

3. Edge weights:   w_ij = exp(-d_ij^2 / (2 * sigma_i^2))   (Gaussian RBF kernel)
   where sigma_i = median(d_i1, ..., d_ik) adapts to local trade density.

4. Self-loops:     A_ii = 1.0 (before normalisation) so each node includes its own
   features in the neighbourhood aggregation.

5. Row norm:       A_hat = D^{-1} A  =>  each row sums to 1  =>  A_hat @ X gives the
   weighted mean of neighbour features (GraphSAGE mean aggregator).

Inference extension
-------------------
New target trades are projected into the trained graph by finding their k nearest
original neighbours, applying the same RBF kernel, adding self-loops, and row-
normalising only the new rows. Original rows are preserved bit-for-bit so the
trained GNN embeddings are unchanged (GraphSAGE is inductive — learned weight
matrices generalise to unseen nodes).
"""
from __future__ import annotations

import time
import pickle
import logging

import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix

logger = logging.getLogger(__name__)


class TradeGraphBuilder:
    """
    Builds and extends k-NN trade relationship graphs for GNN consumption.

    Training (`build_graph`): constructs a full k-NN adjacency with Gaussian RBF
    kernel weights, optional elementary/target quota constraints, and D^-1 A
    row-normalisation for the GraphSAGE mean aggregator.

    Inference (`build_graph_projection`): extends the trained adjacency with new
    target trade rows while preserving the original graph structure.
    """

    # Maps encoded trade attribute keys to the corresponding alpha weight attribute
    # on this class. Used by _weighted_features to build the feature vector via a
    # data-driven loop instead of per-key if-statements.
    _EMBEDDING_ALPHA_MAP = {
        "product_type_embedding": "alpha_prod_type",
        "product_subtype_embedding": "alpha_prod_subtype",
        "underlying_embedding": "alpha_underlying",
    }

    _SCALAR_ALPHA_MAP = {
        "moneyness": "alpha_moneyness",
        "time_to_maturity": "alpha_maturity",
        "normalised_delta": "alpha_delta",
        "normalised_vega": "alpha_vega",
    }

    # Keys serialised in save() / deserialised in load(). Keeping this as a class
    # constant ensures save and load stay in sync automatically.
    _SAVE_CONFIG_KEYS = (
        "k", "distance_metric", "alpha_moneyness", "alpha_maturity", "alpha_delta",
        "alpha_vega", "alpha_prod_type", "alpha_prod_subtype", "alpha_underlying",
        "alpha_underlying_rf", "p_min_elementary", "q_min_target",
    )

    def __init__(
        self,
        distance_metric: str = "euclidean",
        k: int = 10,
        alpha_moneyness: float = 1.0,
        alpha_maturity: float = 1.0,
        alpha_delta: float = 1.0,
        alpha_vega: float = 1.0,
        alpha_prod_type: float = 1.0,
        alpha_prod_subtype: float = 1.0,
        alpha_underlying: float = 1.0,
        alpha_underlying_rf: float = 1.0,
        p_min_elementary: int = 2,
        q_min_target: int = 0,
    ) -> None:
        """
        Initialise the graph builder.

        :param distance_metric: metric for sklearn NearestNeighbors (e.g. 'euclidean', 'cosine').
        :param k: number of nearest neighbours per node.
        :param alpha_*: feature importance weights (squared distance contribution = alpha * dx^2).
        :param p_min_elementary: minimum elementary neighbours per target trade (quota mode).
        :param q_min_target: minimum target neighbours per target trade (quota mode).
        """
        self.distance_metric = distance_metric
        self.k = k
        self.alpha_moneyness = alpha_moneyness
        self.alpha_maturity = alpha_maturity
        self.alpha_delta = alpha_delta
        self.alpha_vega = alpha_vega
        self.alpha_prod_type = alpha_prod_type
        self.alpha_prod_subtype = alpha_prod_subtype
        self.alpha_underlying = alpha_underlying
        self.alpha_underlying_rf = alpha_underlying_rf
        self.p_min_elementary = p_min_elementary
        self.q_min_target = q_min_target

        # Populated during graph building — None until build_graph() is called.
        self.features: NDArray | None = None
        self._adjacency_csr: csr_matrix | None = None
        self._adjacency_dense_cache: NDArray | None = None
        self.sparse_values: NDArray | None = None
        self.sparse_indices: NDArray | None = None
        self.sparse_shape: list | None = None
        self.is_target_trade: NDArray | None = None

        # Clamp quota constraints so p + q <= k (otherwise some target trades
        # would be forced to have more constrained neighbours than total k allows).
        self._validate_constraints(p_min_elementary, q_min_target, k)

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def adjacency_matrix(self) -> NDArray | None:
        """Dense adjacency matrix (backward-compatible alias for adjacency_dense)."""
        return self.adjacency_dense

    @property
    def adjacency_dense(self) -> NDArray | None:
        """
        Dense adjacency matrix, computed lazily from CSR and cached.

        Call this only when you need a dense view (plotting, get_similarity,
        save). The GNN forward pass should use the torch sparse tensor returned
        by build_graph() / build_graph_projection().
        """
        if self._adjacency_csr is None:
            return None
        if self._adjacency_dense_cache is None:
            self._adjacency_dense_cache = self._adjacency_csr.toarray().astype(np.float32)
        return self._adjacency_dense_cache

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def build_graph(
        self, encoded_trades: dict[str, NDArray], include_quota: bool = True,
    ) -> Dict[str, Any]:
        """
        Build the full training adjacency matrix.

        Pipeline:
          1. Build alpha-weighted feature vectors from encoded trade attributes.
          2. Classify each trade as elementary or target.
          3. Safety-clamp k, p_min, q_min to actual trade counts.
          4. Run k-NN -> RBF kernel -> self-loops -> D^{-1}A normalisation.
          5. Return sparse-first result for GNN consumption; dense available
             via builder.adjacency_dense property for plotting.

        :param encoded_trades: output of TradeAttributeEncoder.transform().
        :param include_quota: enforce elementary/target neighbour quotas for target trades.
        :return: dict with adjacency_matrix (torch sparse), sparse_indices, values, shape, is_target.
        """
        # Step 1: combine encoded features into a single weighted vector per trade.
        features = self._weighted_features(encoded_trades)

        # Step 2: boolean mask — True where trade is a target (not elementary).
        is_target = self._identify_trade_types(encoded_trades)
        self.features = features
        self.is_target_trade = is_target

        # Step 3: clamp k and quotas to the actual number of available trades
        # so we don't request more neighbours than exist.
        num_target = int(np.sum(is_target))
        num_elementary = len(is_target) - num_target
        self._adjust_k_and_quotas(num_elementary, num_target)

        # Step 4: core graph construction.
        start = time.time()
        indices, values, csr = self._build_knn_adjacency(
            features, k=self.k, add_self_loops=True, include_quota=include_quota,
        )
        logger.info(
            f"Built trade graph: {len(values)} edges, {num_target} target / "
            f"{num_elementary} elementary in {time.time() - start:.2f}s"
        )
        if num_target > 0:
            logger.info(
                f"Quota constraints: p_min_elementary={self.p_min_elementary}, "
                f"q_min_target={self.q_min_target}"
            )

        # Step 5: persist results on the instance for save/load and get_similarity().
        self._adjacency_csr = csr
        self._adjacency_dense_cache = None
        self.sparse_values = values
        self.sparse_indices = indices
        self.sparse_shape = list(csr.shape)

        return self._pack_result(csr, indices, values, is_target)

    def build_graph_projection(
        self,
        adjacency_matrix: NDArray | csr_matrix,
        encoded_trades: dict[str, NDArray],
        new_targets: int,
        k: int = 8,
        include_new_new: bool = False,
        freeze_original: bool = True,
    ) -> Dict[str, Any]:
        """
        Extend the trained adjacency with new target trade rows.

        This is the inference-time entry point. New trades are projected into the
        trained graph structure so the (frozen) GNN weights can produce embeddings
        for unseen trades.

        The extended adjacency has block structure:

            [ A_orig   |   0      ]      (freeze_original=True)
            [ B_new    |  C_self  ]

        - A_orig: original trained adjacency, copied verbatim.
        - 0:      no edges from original to new (original embeddings are unchanged).
        - B_new:  RBF-weighted k-NN edges from each new trade to its nearest originals.
        - C_self: self-loops for new nodes.

        :param adjacency_matrix: row-normalised adjacency from training [n_orig, n_orig].
        :param encoded_trades: ALL encoded trades (original + new).
        :param new_targets: number of new trades to append.
        :param k: neighbours per new trade.
        :param include_new_new: allow edges between new trades.
        :param freeze_original: if True (default), original rows stay exactly as trained.
        :return: dict with adjacency_matrix, sparse_tensor, indices, values, shape, is_target.
        """
        features = self._weighted_features(encoded_trades)
        is_target = self._identify_trade_types(encoded_trades)

        num_elementary = len(is_target) - int(np.sum(is_target))
        if self.k > num_elementary:
            self.k = max(1, num_elementary // 2) if num_elementary > 1 else num_elementary
            logger.warning(f"Adjusted k to {self.k} (only {num_elementary} elementary trades).")

        # New trades occupy indices [n_orig, n_orig + new_targets).
        new_idx = np.arange(adjacency_matrix.shape[0], adjacency_matrix.shape[0] + new_targets)
        indices, values, csr = self._extend_adjacency(
            adjacency_matrix, features, new_idx,
            k=k, include_new_new=include_new_new, freeze_original=freeze_original,
        )

        return self._pack_result(csr, indices, values, is_target)

    def get_similarity(self, trade_idx_1: int, trade_idx_2: int) -> float:
        """Get normalised edge weight between two trades (O(1) via CSR lookup)."""
        if self._adjacency_csr is None:
            raise ValueError("Graph not built yet.")
        return float(self._adjacency_csr[trade_idx_1, trade_idx_2])

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, file_path: str | Path) -> None:
        """Serialise graph builder state to disk (config + built graph artefacts)."""
        state = {k: getattr(self, k) for k in self._SAVE_CONFIG_KEYS}
        state.update({
            "features": self.features, "adjacency_matrix": self._adjacency_csr,
            "sparse_values": self.sparse_values, "sparse_indices": self.sparse_indices,
            "sparse_shape": self.sparse_shape, "is_target": self.is_target_trade,
        })
        with open(file_path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"TradeGraphBuilder saved to {file_path}")

    @classmethod
    def load(cls, file_path: str | Path) -> TradeGraphBuilder:
        """Load a persisted graph builder (restores config + all graph artefacts)."""
        with open(file_path, "rb") as f:
            d = pickle.load(f)
        builder = cls(
            distance_metric=d["distance_metric"], k=d["k"],
            alpha_moneyness=d["alpha_moneyness"], alpha_maturity=d["alpha_maturity"],
            alpha_delta=d["alpha_delta"], alpha_vega=d["alpha_vega"],
            alpha_prod_type=d["alpha_prod_type"], alpha_prod_subtype=d["alpha_prod_subtype"],
            alpha_underlying=d["alpha_underlying"], alpha_underlying_rf=d["alpha_underlying_rf"],
            p_min_elementary=d.get("p_min_elementary", 2),
            q_min_target=d.get("q_min_target", 0),
        )
        builder.features = d["features"]
        loaded_adj = d["adjacency_matrix"]
        builder._adjacency_csr = csr_matrix(loaded_adj) if loaded_adj is not None else None
        builder._adjacency_dense_cache = None
        builder.sparse_values = d["sparse_values"]
        builder.sparse_indices = d["sparse_indices"]
        builder.sparse_shape = d["sparse_shape"]
        builder.is_target_trade = d["is_target"]
        logger.info(f"TradeGraphBuilder loaded from {file_path}")
        return builder

    # ------------------------------------------------------------------ #
    #  Core: training graph                                               #
    # ------------------------------------------------------------------ #

    def _build_knn_adjacency(
        self, features: NDArray, k: int, add_self_loops: bool = True, include_quota: bool = False,
    ) -> tuple[NDArray, NDArray, csr_matrix]:
        """
        Build a row-normalised, RBF-weighted k-NN adjacency matrix.

        Algorithm:
          1. Find k+1 nearest neighbours per node (includes self at position 0).
          2. Drop self, leaving k true neighbours with distances.
          3. Convert distances to similarity via Gaussian RBF kernel.
          4. Build sparse COO matrix with k edges per node.
          5. Add self-loops with weight 1.0 (so nodes aggregate their own features).
          6. Row-normalise: D^{-1}A so each row sums to 1 (GraphSAGE mean aggregator).
          7. Export as (indices, values, csr) for torch.sparse_coo_tensor.

        :param features: weighted feature matrix [n, d].
        :param k: neighbours per node (excluding self).
        :param add_self_loops: include diagonal entries with unit weight.
        :param include_quota: use quota-based neighbour selection for target trades.
        :return: (indices [nnz, 2], values [nnz], csr_matrix [n, n]).
        """
        n = features.shape[0]

        # --- Step 1-2: k-NN search ---
        if include_quota:
            dists, inds = self._generate_neighbours_quota(
                n_neighbours=(k + 1), features=features,
                is_target=self.is_target_trade, num_trades=n,
            )
        else:
            nbrs = NearestNeighbors(n_neighbors=(k + 1), metric=self.distance_metric).fit(features)
            dists, inds = nbrs.kneighbors(features)

        # Position 0 is always self (distance=0). Drop it to get k true neighbours.
        dists, inds = dists[:, 1:], inds[:, 1:]
        k_actual = inds.shape[1]

        if not np.isfinite(dists).all():
            raise ValueError("Non-finite distances after neighbour search.")

        # --- Step 3: Gaussian RBF kernel ---
        w = self._rbf_weights(dists)

        # --- Step 4: Assemble sparse COO ---
        rows = np.repeat(np.arange(n), k_actual)
        cols = inds.ravel()
        data = w.ravel()

        # --- Step 5: Self-loops ---
        if add_self_loops:
            diag = np.arange(n)
            rows = np.concatenate([rows, diag])
            cols = np.concatenate([cols, diag])
            data = np.concatenate([data, np.ones(n, dtype=np.float32)])

        adj = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

        # --- Step 6: Row normalisation ---
        adj = self._row_normalise(adj)

        # --- Step 7: Export ---
        return self._to_sparse_output(adj)

    # ------------------------------------------------------------------ #
    #  Core: inference graph extension                                    #
    # ------------------------------------------------------------------ #

    def _extend_adjacency(
        self,
        adjacency_matrix: NDArray | csr_matrix,
        features: NDArray,
        new_idx: NDArray,
        k: int = 8,
        include_new_new: bool = False,
        freeze_original: bool = True,
    ) -> tuple[NDArray, NDArray, csr_matrix]:
        """
        Extend trained adjacency with new trade rows (fully vectorised).

        :param adjacency_matrix: trained row-normalised adj [n_orig, n_orig].
        :param features: all trade features [n_orig + n_new, d].
        :param new_idx: global indices of new nodes.
        :param k: neighbours per new node.
        :param include_new_new: allow new-to-new edges.
        :param freeze_original: preserve original rows bit-for-bit.
        :return: (indices [nnz, 2], values [nnz], csr_matrix [n_total, n_total]).
        """
        n_orig = adjacency_matrix.shape[0]
        n_new = len(new_idx)
        n_total = n_orig + n_new

        # Early exit: no new trades — just reformat the existing adjacency.
        if n_new == 0:
            self.sparse_shape = [n_orig, n_orig]
            return self._to_sparse_output(csr_matrix(adjacency_matrix))

        # k-NN for new nodes against the FULL trade universe
        k_query = min(k + 1, n_total)
        nbrs = NearestNeighbors(n_neighbors=k_query, metric=self.distance_metric).fit(features)
        dists, inds = nbrs.kneighbors(features[new_idx])
        dists, inds = dists[:, 1:], inds[:, 1:]
        k_actual = inds.shape[1]

        # same RBF kernel as training for consistency
        w = self._rbf_weights(dists)

        # forward edges: new_trade -> its k neighbours
        rows_fwd = np.repeat(new_idx, k_actual)
        cols_fwd = inds.ravel().astype(np.int64)
        data_fwd = w.ravel()

        # drop new->new edges when not allowed
        if not include_new_new:
            keep = cols_fwd < n_orig
            rows_fwd, cols_fwd, data_fwd = rows_fwd[keep], cols_fwd[keep], data_fwd[keep]

        # reverse edges
        rev_mask = (cols_fwd >= n_orig) if freeze_original else np.ones(len(cols_fwd), dtype=bool)
        rows_rev = cols_fwd[rev_mask]
        cols_rev = rows_fwd[rev_mask]
        data_rev = data_fwd[rev_mask]

        # self-loops for new nodes
        rows_self = new_idx
        cols_self = new_idx
        data_self = np.ones(n_new, dtype=np.float32)

        # concatenate all edge types
        all_rows = np.concatenate([rows_fwd, rows_rev, rows_self])
        all_cols = np.concatenate([cols_fwd, cols_rev, cols_self])
        all_data = np.concatenate([data_fwd, data_rev, data_self])

        # assemble extended adjacency
        ext = csr_matrix((n_total, n_total), dtype=np.float32)
        ext[:n_orig, :n_orig] = csr_matrix(adjacency_matrix)
        if all_rows.size:
            ext = ext + coo_matrix(
                (all_data, (all_rows, all_cols)), shape=(n_total, n_total), dtype=np.float32,
            )

        # row normalisation
        if freeze_original:
            orig_block = ext[:n_orig, :].copy()
            ext = self._row_normalise(ext)
            ext[:n_orig, :] = orig_block
        else:
            ext = self._row_normalise(ext)

        self.sparse_shape = [n_total, n_total]
        return self._to_sparse_output(ext)

    # ------------------------------------------------------------------ #
    #  Quota-based neighbour generation                                   #
    # ------------------------------------------------------------------ #

    def _generate_neighbours_quota(
        self, n_neighbours: int, features: NDArray, is_target: NDArray, num_trades: int,
    ) -> tuple[NDArray, NDArray]:
        """
        k-NN with elementary/target quota constraints for target nodes.

        :param n_neighbours: total neighbours including self (column 0 = self).
        :param features: weighted feature matrix [n, d].
        :param is_target: boolean mask, True = target trade.
        :param num_trades: expected number of trades (validated against features).
        :return: (distances [n, n_neighbours], indices [n, n_neighbours]).
        """
        n = features.shape[0]
        if n != num_trades:
            raise ValueError(f"Feature rows ({n}) != num_trades ({num_trades}).")
        if n_neighbours < 1:
            raise ValueError(f"n_neighbours must be >= 1, got {n_neighbours}.")

        k = min(n_neighbours - 1, max(0, n - 1))
        n_neighbours = k + 1

        # output arrays: column 0 = self (distance 0, index = node)
        dists_out = np.full((n, n_neighbours), np.inf, dtype=np.float32)
        inds_out = np.full((n, n_neighbours), -1, dtype=np.int32)
        dists_out[:, 0] = 0.0
        inds_out[:, 0] = np.arange(n, dtype=np.int32)

        if k == 0:
            return dists_out, inds_out

        # fetch a generous candidate pool (3x k) so partitioning fills both quotas
        candidate_k = min(
            max(k * 3, self.p_min_elementary + self.q_min_target + k), n,
        )
        nbrs = NearestNeighbors(n_neighbors=candidate_k, metric=self.distance_metric).fit(features)
        all_dists, all_inds = nbrs.kneighbors(features)

        is_elem = ~is_target

        for node in range(n):
            # remove self from candidate list
            mask = all_inds[node] != node
            cand_i = all_inds[node][mask]
            cand_d = all_dists[node][mask]

            if not is_target[node]:
                # elementary nodes: no quota — take k nearest overall
                sel_i = cand_i[:k]
                sel_d = cand_d[:k]
            else:
                # target nodes: partition candidates into elementary vs target
                cand_is_elem = is_elem[cand_i]
                e_i, e_d = cand_i[cand_is_elem], cand_d[cand_is_elem]
                t_i, t_d = cand_i[~cand_is_elem], cand_d[~cand_is_elem]

                need_e = min(self.p_min_elementary, len(e_i), k)
                need_t = min(self.q_min_target, len(t_i), max(0, k - need_e))

                parts_i, parts_d = [], []
                if need_e > 0:
                    parts_i.append(e_i[:need_e]); parts_d.append(e_d[:need_e])
                if need_t > 0:
                    parts_i.append(t_i[:need_t]); parts_d.append(t_d[:need_t])

                sel_i = np.concatenate(parts_i) if parts_i else np.array([], dtype=np.int64)
                sel_d = np.concatenate(parts_d) if parts_d else np.array([], dtype=np.float32)

                # fill remaining slots with nearest unused candidates
                remaining = k - len(sel_i)
                if remaining > 0:
                    used = set(sel_i.tolist())
                    used.add(node)
                    avail_mask = np.array([c not in used for c in cand_i])
                    fill_idx = np.where(avail_mask)[0][:remaining]
                    if fill_idx.size:
                        sel_i = np.concatenate([sel_i, cand_i[fill_idx]])
                        sel_d = np.concatenate([sel_d, cand_d[fill_idx]])

                # sort by (distance, index) for deterministic output
                if sel_i.size > 1:
                    order = np.lexsort((sel_i, sel_d))
                    sel_i, sel_d = sel_i[order], sel_d[order]

            # pad with farthest neighbour if fewer than k candidates exist
            if 0 < len(sel_i) < k:
                pad = k - len(sel_i)
                sel_i = np.concatenate([sel_i, np.repeat(sel_i[-1], pad)])
                sel_d = np.concatenate([sel_d, np.repeat(sel_d[-1], pad)])

            # write to output arrays (columns 1..k, since column 0 = self)
            w = min(len(sel_i), k)
            if w > 0:
                inds_out[node, 1:w + 1] = sel_i[:w].astype(np.int32)
                dists_out[node, 1:w + 1] = sel_d[:w].astype(np.float32)

        return dists_out, inds_out

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _identify_trade_types(encoded_trades: dict[str, NDArray]) -> NDArray:
        """
        Boolean mask: True = target trade.

        Expects trade_type to be one-hot encoded by sklearn's OneHotEncoder with
        categories sorted alphabetically (e.g. ["elementary", "target"]).  The
        "target" column is the last one after alphabetical sorting.
        """
        n = len(encoded_trades["moneyness"])
        is_target = np.zeros(n, dtype=bool)
        if "trade_type" in encoded_trades:
            emb = encoded_trades["trade_type"]
            if emb.ndim == 2 and emb.shape[1] >= 2 and emb.shape[0] > 0:
                is_target = emb[:, -1] == 1
            elif emb.ndim == 2 and emb.shape[1] == 1:
                logger.warning(
                    "trade_type has only 1 category — cannot distinguish "
                    "elementary vs target. Treating all trades as elementary."
                )
        logger.info(f"Identified {np.sum(is_target)} target / {n} total trades.")
        return is_target

    def _weighted_features(self, encoded_trades: dict[str, NDArray]) -> NDArray:
        """
        Build alpha-weighted feature matrix for distance calculation.

        Each feature is scaled by sqrt(alpha) so that the squared Euclidean distance
        in this space equals sum_d( alpha_d * (x_id - x_jd)^2 ).
        """
        parts: list[NDArray] = []

        # scalar features: moneyness, time_to_maturity, delta, vega
        for key, alpha_attr in self._SCALAR_ALPHA_MAP.items():
            alpha = getattr(self, alpha_attr)
            if key in encoded_trades and alpha > 0.0:
                parts.append(np.sqrt(alpha) * encoded_trades[key].reshape(-1, 1))

        # one-hot / embedding features: product_type, product_subtype, underlying
        for key, alpha_attr in self._EMBEDDING_ALPHA_MAP.items():
            alpha = getattr(self, alpha_attr)
            if key in encoded_trades and alpha > 0.0:
                parts.append(np.sqrt(alpha) * self._scale_embedding(encoded_trades[key]))

        # multi-label: underlying_risk_factors. L2-normalised per trade
        if "underlying_risk_factors_embedding" in encoded_trades and self.alpha_underlying_rf > 0.0:
            urf = encoded_trades["underlying_risk_factors_embedding"]
            if urf.ndim == 1:
                urf = urf.reshape(-1, 1)
            urf = urf / np.maximum(np.linalg.norm(urf, axis=1, keepdims=True), 1e-8)
            parts.append(np.sqrt(self.alpha_underlying_rf) * urf)

        if not parts:
            raise ValueError("No features selected for graph building. Check alpha weights and input data.")

        return np.hstack(parts).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Static helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rbf_weights(dists: NDArray) -> NDArray:
        """
        Gaussian RBF kernel: w_ij = exp(-d_ij^2 / (2 * sigma_i^2)).

        sigma_i = median of node i's k neighbour distances (adaptive bandwidth).
        """
        sigma = np.median(np.maximum(dists, 1e-9), axis=1, keepdims=True).astype(np.float32)
        return np.exp(-(dists.astype(np.float32) ** 2) / (2.0 * sigma ** 2))

    @staticmethod
    def _row_normalise(adj) -> csr_matrix:
        """D^{-1}A row normalisation: multiply each row by 1/row_sum so rows sum to 1."""
        csr = csr_matrix(adj)
        row_sum = np.asarray(csr.sum(axis=1)).ravel().astype(np.float32)
        inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
        return csr.multiply(inv.reshape(-1, 1)).tocsr()

    @staticmethod
    def _scale_embedding(emb: NDArray) -> NDArray:
        """Prepare a one-hot or embedding feature for distance calculation."""
        if emb.ndim == 1:
            emb = emb.reshape(-1, 1)
        return emb.astype(np.float32) / np.sqrt(max(1, emb.shape[1]))

    @staticmethod
    def _to_sparse_output(adj: Any) -> tuple[NDArray, NDArray, csr_matrix]:
        """
        Convert scipy sparse matrix to sparse arrays + CSR for downstream use.

        indices [nnz, 2]: row/col pairs for torch.sparse_coo_tensor.
        values  [nnz]:    normalised edge weights.
        csr:              compressed sparse row for efficient storage and element access.
        """
        coo = adj.tocoo()
        indices = np.column_stack([coo.row, coo.col]).astype(np.int64)
        values = coo.data.astype(np.float32)
        return indices, values, csr_matrix(adj)

    def _pack_result(
        self, csr: csr_matrix, indices: NDArray, values: NDArray, is_target: NDArray,
    ) -> Dict[str, Any]:
        """
        Assemble the standard return dictionary.

        ``adjacency_matrix`` is a torch.sparse_coo_tensor used directly by the GNN
        forward pass. For dense access (plotting, debugging), use the builder's
        ``adjacency_dense`` property which lazily calls .toarray() and caches.
        """
        shape = list(csr.shape)

        # build torch sparse COO tensor from scipy indices and values
        # torch.sparse_coo_tensor expects indices as [2, nnz] (transposed from [nnz, 2])
        torch_indices = torch.from_numpy(indices.T)
        torch_values = torch.from_numpy(values)
        sparse_tensor = torch.sparse_coo_tensor(
            torch_indices, torch_values, size=shape
        )

        return {
            "adjacency_matrix": sparse_tensor, "sparse_tensor": sparse_tensor,
            "sparse_indices": indices, "sparse_values": values,
            "sparse_shape": shape, "is_target": is_target,
        }

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #

    def _adjust_k_and_quotas(self, num_elementary: int, num_target: int) -> None:
        """Clamp k, p_min_elementary, q_min_target to available trade counts."""
        if self.k > num_elementary:
            self.k = max(1, num_elementary // 2) if num_elementary > 1 else num_elementary
            logger.warning(f"Adjusted k to {self.k} (only {num_elementary} elementary trades).")
        if num_elementary < self.p_min_elementary and num_target > 0:
            self.p_min_elementary = num_elementary
            logger.warning(f"Adjusted p_min_elementary to {self.p_min_elementary}.")
        if num_target - 1 < self.q_min_target and num_target > 0:
            self.q_min_target = max(0, num_target - 1)
            logger.warning(f"Adjusted q_min_target to {self.q_min_target}.")

    def _validate_constraints(self, p: int, q: int, k: int) -> None:
        """Ensure p + q <= k at init time."""
        if p + q > k:
            logger.warning(f"p_min ({p}) + q_min ({q}) > k ({k}), adjusting.")
            self.p_min_elementary = min(p, k - 1)
            self.q_min_target = min(q, k - self.p_min_elementary)
