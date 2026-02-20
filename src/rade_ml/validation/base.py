"""
Base validation checks for ML models.

This module provides validation checks shared across all ML models.
"""
from __future__ import annotations

import logging

from typing import Dict, Any, List

from src.rade_ml.validation.exceptions import MissingKeyFields

# define module level logging.
logger = logging.getLogger(__name__)


def validate_dict_keys(input_dict: Dict[str, Any], keys: List[str]) -> None:
    """
    Validate if all required keys are present in dictionary keys.

    :param input_dict: dictionary to validate
    :param keys: keys required to be in dictionary
    :return:
    """
    missing_keys = [k for k in keys if k not in input_dict]
    if missing_keys:
        raise MissingKeyFields(f"Missing keys from dictionary: {missing_keys}")