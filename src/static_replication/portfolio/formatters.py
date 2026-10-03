"""Pure pandas transformations on raw SAGE / file (raw Sage saved down) responses"""
from __future__ import annotations

import re
import time
import logging
import pandas as pd

from typing import List, Sequence, Mapping

from src.static_replication.core.trade_schema import LINEAR_DEFAULTS

# define module level logging.
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Constants.
# --------------------------------------------------------------------------------------