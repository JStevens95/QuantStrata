import logging
import time
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Union, Optional
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, diags, csr_matrix, bmat


class TradeGraphBuilder:
    """
    Class for building and updating trade relationship graphs based on parameter similarity.

    This class constructs k-nearest neighbours graph where trades are connected based on their parameter similarity.
    The graph is represented using sparse tensors for computational efficiency.
    """

    def __init__(
            self, distance_metric: str = 'euclidean', k: int = 10, alpha_moneyness: float = 1.0,
            alpha_maturity: float = 1.0, alpha_delta: float = 1.0, alpha_vega: float = 1.0,
            alpha_prod_type: float = 1.0, alpha_prod_subtype: float = 1.0, alpha_underlying: float = 1.0,
            alpha_underlying_rf: float = 1.0, p_min_elementary: int = 2, q_min_target: int = 0

    ) -> None:
        """
        Initialise the graph builder.

        :param distance_metric:
        :param k:
        :param alpha_moneyness:
        :param alpha_maturity:
        :param alpha_delta:
        :param alpha_vega:
        :param alpha_prod_type:
        :param alpha_prod_subtype:
        :param alpha_underlying:
        :param alpha_underlying_rf:
        :param p_min_elementary:
        :param q_min_target:
        """
        # initiate required variables.
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

        # features to be set during graph building.
        self.features = None
        self.adjacency_matrix = None
        self.sparse_values = None
        self.sparse_indices = None
        self.sparse_shape = None
        self._is_target_trade = None

    def _weighted_features(self, encoded_trades: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract and weight features for distance calculation.

        :param encoded_trades:
        :return:
        """
        features_to_stack = []
        # always include moneyness and maturity
        if 'moneyness' in encoded_trades and self.alpha_moneyness > 0:
            features_to_stack.append(np.sqrt(self.alpha_moneyness) * encoded_trades['moneyness'].reshape(-1, 1))
        if 'time_to_maturity' in encoded_trades and self.alpha_maturity > 0:
            features_to_stack.append(np.sqrt(self.alpha_maturity) * encoded_trades['time_to_maturity'].reshape(-1, 1))

        # optionally include other features.
        if 'normalised_delta' in encoded_trades and self.alpha_delta > 0:
            features_to_stack.append(np.sqrt(self.alpha_delta) * encoded_trades['normalised_delta'].reshape(-1, 1))
        if 'normalised_vega' in encoded_trades and self.alpha_vega > 0:
            features_to_stack.append(np.sqrt(self.alpha_vega) * encoded_trades['normalised_vega'].reshape(-1, 1))

        # handle embedding features (potentially multi-dimensional)
        if 'product_type_embedding' in encoded_trades and self.alpha_prod_type > 0:
            # ensure its 2D
            prod_type_emb = encoded_trades['product_type_embedding']
            if prod_type_emb.ndim == 1:
                prod_type_emb = prod_type_emb.reshape(-1, 1)
            features_to_stack.append(np.sqrt(self.alpha_prod_type) * prod_type_emb)
        if 'product_subtype_embedding' in encoded_trades and self.alpha_prod_subtype > 0:
            # ensure its 2D
            prod_subtype_emb = encoded_trades['product_subtype_embedding']
            if prod_subtype_emb.ndim == 1:
                prod_subtype_emb = prod_subtype_emb.reshape(-1, 1)
            features_to_stack.append(np.sqrt(self.alpha_prod_subtype) * prod_subtype_emb)
        if 'underlying_embedding' in encoded_trades and self.alpha_underlying > 0:
            # ensure its 2D
            underlying_emb = encoded_trades['underlying_embedding']
            if underlying_emb.ndim == 1:
                underlying_emb = underlying_emb.reshape(-1, 1)
            features_to_stack.append(np.sqrt(self.alpha_underlying) * underlying_emb)

        # handle multi label binary encoded features.
        if 'underlying_risk_factors_embedding' in encoded_trades and self.alpha_underlying_rf > 0:
            # ensure its 2D
            urf_emb = encoded_trades['underlying_risk_factors_embedding']
            if urf_emb.ndim == 1:
                urf_emb = urf_emb.reshape(-1, 1)
            features_to_stack.append(np.sqrt(self.alpha_underlying_rf) * urf_emb)

        if not features_to_stack:
            raise ValueError("No features selected for graph building. Check alpha weights and input")

        # stack weighted features.
        features = np.hstack(features_to_stack)
        return features.astype(np.float32)