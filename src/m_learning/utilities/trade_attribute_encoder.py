import numpy as np
from typing import List, Dict, Any, Optional, Sequence, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer


class TradeAttributeEncoder:
    """
    Class responsible for encoding trade attributes into model-friendly representations.

    This encoders transforms raw trade parameters (strike, maturity, product type/subtype, sensitivities) into
    features that better capture financial meaning and enable the model to generalise across the parameter space.
    """

    def __init__(
            self, numeric_keys: Sequence[str] = ('moneyness', 'delta', 'vega', 'yrs_to_maturity'),
            categorical_keys: Sequence[str] = ('product_type', 'product_subtype', 'trade_type'),
            multi_label_keys: Sequence[str] = ('underlying_risk_factors',),
            ttm_key: str = 'yrs_to_maturity', num_decay_terms: Optional[List[Union[Sequence[float], int]]] = None
    ) -> None:
        """
        Initiate Trade Attribute Encoder.

        :param numeric_keys:
        :param categorical_keys:
        """
        # initiate required variables.
        self.numeric_keys = list(numeric_keys)
        self.categorical_keys = list(categorical_keys)
        self.multi_label_keys = list(multi_label_keys)
        self.ttm_key = ttm_key

        # configure exponential decay lambdas for tenor.
        self.ttm_decay_lambdas = list(np.linspace(10.0, 0.1, num_decay_terms))
        # insert decay feature names into numeric keys.
        for lam in self.ttm_decay_lambdas:
            self.numeric_keys.append(f'ttm_decay_{lam}')

        # initiate variables to be set during fit.
        self.num_scaler: Optional[StandardScaler] = None
        self.cat_encoders: Dict[str, OneHotEncoder] = {}
        self.mlb_encoders: Dict[str, MultiLabelBinarizer] = {}
        self.output_features_names: List[str] = []
        self.encode_names = {
            'yrs_to_maturity': 'time_to_maturity', 'delta': 'normalised_delta', 'vega': 'normalised_vega',
            'product_type': 'product_type_embedding', 'product_subtype': 'product_subtype_embedding',
            'underlying_asset': 'underlying_embedding', 'underlying_risk_factors': 'underlying_risk_factors_embedding'
        }

    def _compute_decays(self, trade_attrs: Dict[str, List[Any]]) -> None:
        """
        Compute exponential-decay features for the time-to-maturity and add to trade attributes.
        :param trade_attrs:
        :return:
        """
        tau = np.array(trade_attrs[self.ttm_key], dtype=np.float32)
        for lam in self.ttm_decay_lambdas:
            trade_attrs[f'ttm_decay_{lam}'] = (np.exp(-lam * tau)).tolist()

    @staticmethod
    def _ensure_mlb_structure(col: Sequence[Any]) -> List[List[Any]]:
        """
        Coerce a column into a list of lists for MultiLabelBinariser.

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
        # merge dictionaries.
        trade_attrs = self._merge_attrs(elem_attrs, target_attrs)

        # 1. calculate decay terms.
        self._compute_decays(trade_attrs)

        # 2. numeric scaler.
        x_n = np.vstack([trade_attrs[key] for key in self.numeric_keys]).T.astype(np.float32)
        self.num_scaler = StandardScaler().fit(x_n)

        # 3. categorical encoders.
        self.cat_encoders.clear()
        for key in self.categorical_keys:
            col = np.array(trade_attrs[key], dtype=str).reshape(-1, 1)
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(col)
            self.cat_encoders[key] = ohe

        # 4. multi-label encoders.
        self.mlb_encoders.clear()
        for key in self.multi_label_keys:
            col = trade_attrs[key]
            col_mlb = self._ensure_mlb_structure(col)
            mlb = MultiLabelBinarizer(sparse_output=False)
            mlb.fit(col_mlb)
            self.mlb_encoders[key] = mlb

        # 4. create list of feature names.
        for n in self.numeric_keys:
            self.output_features_names.append(n)
        for key in self.cat_encoders:
            self.output_features_names.append(key)
        for key in self.mlb_encoders:
            self.output_features_names.append(key)

    def transform(self, trade_attrs: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Transform trade attrs into static feature matrix using fitted encoders.

        :param trade_attrs:
        :return:
        """
        # 0. check encoder has been fitted and define encoded output.
        if self.num_scaler is None or not self.cat_encoders:
            raise RuntimeError("TradeAttributeEncoder not fitted. Call fit() first...")
        output = {}

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