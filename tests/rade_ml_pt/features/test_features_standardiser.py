"""Unit tests for rade_ml_pt.features.transforms.standardiser."""
import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, Normalizer

from src.rade_ml_pt.features.transforms.standardiser import get_transformer
from src.rade_ml_pt.validation.exceptions import UndefinedTransformerType


class TestGetTransformer:
    @pytest.mark.parametrize("name, expected_type", [
        ("standard", StandardScaler),
        ("zscore", StandardScaler),
        ("minmax", MinMaxScaler),
        ("robust", RobustScaler),
        ("power", PowerTransformer),
        ("norm", Normalizer),
    ])
    def test_returns_correct_type(self, name, expected_type):
        t = get_transformer(name)
        assert isinstance(t, expected_type)

    def test_case_insensitive(self):
        assert isinstance(get_transformer("Standard"), StandardScaler)
        assert isinstance(get_transformer("MINMAX"), MinMaxScaler)

    def test_unknown_raises(self):
        with pytest.raises(UndefinedTransformerType):
            get_transformer("nonexistent_scaler")
