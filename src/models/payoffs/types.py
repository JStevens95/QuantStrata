from __future__ import annotations

import numpy as np
from typing import Protocol, runtime_checkable, Literal


@runtime_checkable
class TerminalPayoff(Protocol):
    def __call__(self, spot_t: np.ndarray) -> np.ndarray: ...

@runtime_checkable
class PathPayoff(Protocol):
    def __call__(self, spot_path: np.ndarray) -> np.ndarray: ...

# define option types.
OptionType = Literal["call", "put"]

# definition of barrier characteristics.
BarrierDirection = Literal["up", "down"]
BarrierStyle = Literal['knock_out', 'knock_in']

# definition of rebate characteristic.
RebateTiming = Literal["at_hit", "at_expiry"]