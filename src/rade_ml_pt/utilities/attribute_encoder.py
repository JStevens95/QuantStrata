"""
Trade attribute encoder: transforms raw trade parameters into model-friendly features.
"""
from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Union

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer

# define module level logging.
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class TradeAttributeEncoder:
    """
    Class responsible for encoding trade attributes into model-friendly representations.

    This encoder transforms raw trade parameters (strike, maturity, product type/subtype, sensitivities) into
    features that better capture financial meaning and enable the model to generalise across the parameter space.
    """

    def __init__(
            self, numeric_keys: Sequence[str] = ('moneyness', 'delta', 'vega', 'yrs_to_maturity'),
            categorical_keys: Sequence[str] = ('product_type', 'product_subtype', 'trade_type'),
            multi_label_keys: Sequence[str] = ('underlying_risk_factors',),
            ttm_key: str = 'yrs_to_maturity', num_decay_terms: Union[int, List[float]] = 1
    ) -> None:
        """
        Initiate Trade Attribute Encoder.

        :param numeric_keys: base numeric attribute keys (decay keys are appended by fit).
        :param categorical_keys: categorical attribute keys for one-hot encoding.
        :param multi_label_keys: multi-label attribute keys for binarisation.
        :param ttm_key: attribute key containing time-to-maturity values.
        :param num_decay_terms: number of exponential-decay features to derive from ttm,
            or an explicit list of lambda values. Default 1.
        """
        self.numeric_keys = list(numeric_keys)
        self.categorical_keys = list(categorical_keys)
        self.multi_label_keys = list(multi_label_keys)
        self.ttm_key = ttm_key
        self.num_decay_terms = num_decay_terms

        # Derived in _build_decay_terms(), called by fit().
        self.ttm_decay_lambdas: List[float] = []

        # Fitted state — populated by fit().
        self.num_scaler: Optional[StandardScaler] = None
        self.cat_encoders: Dict[str, OneHotEncoder] = {}
        self.mlb_encoders: Dict[str, MultiLabelBinarizer] = {}
        self.output_features_names: List[str] = []
        self.last_attribs_: Optional[Dict[str, List[Any]]] = None
        self.encode_names = {
            'yrs_to_maturity': 'time_to_maturity', 'delta': 'normalised_delta', 'vega': 'normalised_vega',
            'product_type': 'product_type_embedding', 'product_subtype': 'product_subtype_embedding',
            'underlying_asset': 'underlying_embedding', 'underlying_risk_factors': 'underlying_risk_factors_embedding'
        }

    def _build_decay_terms(self) -> None:
        """
        Derive ttm_decay lambdas from num_decay_terms and extend numeric_keys.

        Idempotent: strips any existing ttm_decay_* keys before appending so
        calling twice (or calling on a loaded encoder) is safe.
        """
        self.numeric_keys = [k for k in self.numeric_keys if not k.startswith('ttm_decay_')]

        if isinstance(self.num_decay_terms, int):
            self.ttm_decay_lambdas = list(np.linspace(10.0, 0.1, self.num_decay_terms))
        elif isinstance(self.num_decay_terms, (list, tuple)):
            self.ttm_decay_lambdas = list(self.num_decay_terms)
        else:
            self.ttm_decay_lambdas = []

        for lam in self.ttm_decay_lambdas:
            self.numeric_keys.append(f'ttm_decay_{lam}')

    def _compute_decays(self, trade_attrs: Dict[str, List[Any]]) -> None:
        """
        Compute exponential-decay features for the time-to-maturity and add to trade attributes.
        :param trade_attrs:
        :return:
        """
        if not self.ttm_decay_lambdas:
            return
        tau = np.array(trade_attrs[self.ttm_key], dtype=np.float32)
        for lam in self.ttm_decay_lambdas:
            trade_attrs[f'ttm_decay_{lam}'] = (np.exp(-lam * tau)).tolist()

    @staticmethod
    def _ensure_mlb_structure(col: Sequence[Any]) -> List[List[Any]]:
        """
        Coerce a column into a list of lists for MultiLabelBinariser.
        - If element is already a list, keep as is.
        - Else wrap element in a list.

        :param col:
        :return:
        """
        out: List[List[Any]] = []
        for x in col:
            if isinstance(x, (list, tuple, set)):
                out.append(list(x))
            else:
                out.append([x])
        return out

    @staticmethod
    def _merge_attrs(e_attrs: Dict[str, List[Any]], t_attrs: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Concatenate two attribute dictionaries, ensuring the same keys.

        :param e_attrs: elementary trade attributes
        :param t_attrs: target trade attributes
        :return: merged attributes
        """
        return {k: list(e_attrs[k]) + list(t_attrs[k]) for k in e_attrs}

    def fit(self, elem_attrs: Dict[str, List[Any]], target_attrs: Dict[str, List[Any]]) -> None:
        """
        Fit the numeric scaler on number keys and fir one-hot encoder on categorical keys.

        :param elem_attrs:
        :param target_attrs:
        :return:
        """
        # merge dictionaries and store for inference (re-encoding / merging with new trades).
        trade_attrs = self._merge_attrs(elem_attrs, target_attrs)
        self.last_attribs_ = {k: list(v) for k, v in trade_attrs.items()}
        logger.info(f"Fitting TradeAttributEncoder with parameters...: {self.__dict__.keys()}")
        logger.info(
            f"Number of trades for fitting: {len(trade_attrs[self.ttm_key])} --> {len(elem_attrs[self.ttm_key])} "
            f"elementary & {len(target_attrs[self.ttm_key])} target trades"
        )

        # 1. build decay lambdas and extend numeric_keys, then compute decay features.
        self._build_decay_terms()
        self._compute_decays(trade_attrs)

        # 2. numeric scaler.
        x_n = np.vstack([trade_attrs[key] for key in self.numeric_keys]).T.astype(np.float32)
        self.num_scaler = StandardScaler().fit(x_n)
        logger.info(f"Numerical encoders fitted for fields: {self.numeric_keys}.")

        # 3. categorical encoders.
        self.cat_encoders.clear()
        for key in self.categorical_keys:
            col = np.array(trade_attrs[key], dtype=str).reshape(-1, 1)
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(col)
            self.cat_encoders[key] = ohe
        logger.info(f"Categorical encoders fitted for fields: {self.categorical_keys}.")


        # 4. multi-label encoders.
        self.mlb_encoders.clear()
        for key in self.multi_label_keys:
            col = trade_attrs[key]
            col_mlb = self._ensure_mlb_structure(col)
            mlb = MultiLabelBinarizer(sparse_output=False)
            mlb.fit(col_mlb)
            self.mlb_encoders[key] = mlb
        logger.info(f"Multi-label encoders fitted for fields: {self.multi_label_keys}.")

        # 4. create list of feature names.
        for n in self.numeric_keys:
            self.output_features_names.append(n)
        for key in self.cat_encoders:
            self.output_features_names.append(key)
        for key in self.mlb_encoders:
            self.output_features_names.append(key)

    def transform(self, trade_attrs: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
        """
        Transform trade attrs into static feature matrix using fitted encoders.

        :param trade_attrs:
        :return:
        """
        # 0. check encoder has been fitted and define encoded output.
        if self.num_scaler is None or not self.cat_encoders:
            raise RuntimeError("TradeAttributeEncoder not fitted. Call fit() first...")
        output: Dict[str, np.ndarray] = {}

        # 1. compute decay features
        self._compute_decays(trade_attrs)

        # 2. numeric transformation.
        x_num = np.vstack([trade_attrs[key] for key in self.numeric_keys]).T.astype(np.float32)
        x_num_scaled = self.num_scaler.transform(x_num)
        for i, name in enumerate(self.numeric_keys):
            out_name = self.encode_names[name] if name in self.encode_names else name
            output[out_name] = x_num_scaled[:, i]

        # 3. categorical transformation.
        for key in self.categorical_keys:
            col = np.array(trade_attrs[key], dtype=str).reshape(-1, 1)
            ohe = self.cat_encoders[key]
            out_name = self.encode_names[key] if key in self.encode_names else key
            output[out_name] = ohe.transform(col).astype(np.float32)

        # 4. multi-label transformation.
        for key in self.multi_label_keys:
            col = trade_attrs[key]
            col_mlb = self._ensure_mlb_structure(col)
            mlb = self.mlb_encoders[key]
            out_name = self.encode_names[key] if key in self.encode_names else key
            out = mlb.transform(col_mlb).astype(np.float32)
            # normalise each row to preserve distance to multi-rf vs single-rf trades.
            output[out_name] = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

        # 5. combine all features into a single array for model input.
        mats = []
        for name in self.numeric_keys:
            key = self.encode_names[name] if name in self.encode_names else name
            mats.append(output[key][:, None])
        for name in self.cat_encoders:
            key = self.encode_names[name] if name in self.encode_names else name
            mats.append(output[key])
        for name in self.mlb_encoders:
            key = self.encode_names[name] if name in self.encode_names else name
            mats.append(output[key])
        combined = np.hstack(mats)
        output['combined_features'] = combined
        return output

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, file_path: str | Path) -> None:
        """Serialise encoder config + fitted state to disk."""
        state: Dict[str, Any] = {
            # constructor config (enough to recreate a blank encoder via cls(...))
            "num_decay_terms": self.num_decay_terms,
            "categorical_keys": self.categorical_keys,
            "multi_label_keys": self.multi_label_keys,
            "ttm_key": self.ttm_key,
            # derived + fitted state (restored on top of the blank encoder)
            "numeric_keys": self.numeric_keys,
            "ttm_decay_lambdas": self.ttm_decay_lambdas,
            "num_scaler": self.num_scaler,
            "cat_encoders": self.cat_encoders,
            "mlb_encoders": self.mlb_encoders,
            "output_features_names": self.output_features_names,
            "encode_names": self.encode_names,
            "last_attribs_": self.last_attribs_,
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"TradeAttributeEncoder saved to {file_path}")

    @classmethod
    def load(cls, file_path: str | Path) -> "TradeAttributeEncoder":
        """Load a persisted encoder (restores config + all fitted state).

        Creates a clean instance via cls(...) — safe because __init__ is now a
        pure config store — then overwrites with the saved fitted state.
        """
        with open(file_path, "rb") as f:
            d: Dict[str, Any] = pickle.load(f)

        encoder = cls(
            numeric_keys=d["numeric_keys"],
            categorical_keys=d["categorical_keys"],
            multi_label_keys=d["multi_label_keys"],
            ttm_key=d["ttm_key"],
            num_decay_terms=d["num_decay_terms"],
        )

        # Restore derived + fitted state (numeric_keys already includes decay names).
        encoder.numeric_keys = d["numeric_keys"]
        encoder.ttm_decay_lambdas = d.get("ttm_decay_lambdas", [])
        encoder.num_scaler = d["num_scaler"]
        encoder.cat_encoders = d["cat_encoders"]
        encoder.mlb_encoders = d["mlb_encoders"]
        encoder.output_features_names = d.get("output_features_names", [])
        encoder.encode_names = d.get("encode_names", encoder.encode_names)
        encoder.last_attribs_ = d.get("last_attribs_")
        logger.info(f"TradeAttributeEncoder loaded from {file_path}")
        return encoder
