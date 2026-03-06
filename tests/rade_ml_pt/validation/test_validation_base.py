"""Unit tests for rade_ml_pt.validation.base -- validate_dict_keys."""
import pytest

from src.rade_ml_pt.validation.base import validate_dict_keys
from src.rade_ml_pt.validation.exceptions import MissingKeyFields


class TestValidateDictKeys:
    def test_all_present(self):
        validate_dict_keys({"a": 1, "b": 2, "c": 3}, ["a", "b"])

    def test_missing_raises(self):
        with pytest.raises(MissingKeyFields, match="Missing keys"):
            validate_dict_keys({"a": 1}, ["a", "b", "c"])

    def test_empty_keys_passes(self):
        validate_dict_keys({"a": 1}, [])

    def test_empty_dict_with_required_keys_raises(self):
        with pytest.raises(MissingKeyFields):
            validate_dict_keys({}, ["a"])

    def test_error_message_contains_missing_key_names(self):
        with pytest.raises(MissingKeyFields, match="x"):
            validate_dict_keys({"a": 1}, ["a", "x", "y"])
