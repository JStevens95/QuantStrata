"""
Scenario-set base.

A scenario set holds the shocked market states (resolved to absolute levels) for one
risk factor, aligned to an ordered ``scenario_ids`` sequence. The abstract base lives
here; asset-class-specific sets (``FXScenarioSet``, ``RatesScenarioSet``) live in
``marketdata/fx`` and ``marketdata/rates``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioSet:
    """Base class for shocked market states for one risk factor."""
    factor_id: str
    asset_class: str
    scenario_ids: np.ndarray  # (n_scenarios,)

    @property
    def n_scenarios(self) -> int:
        return int(np.asarray(self.scenario_ids).shape[0])
