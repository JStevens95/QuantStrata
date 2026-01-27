from __future__ import annotations

from typing import Literal

# -----------------------------------------------------------------------------
# Core instrument definitions (type aliases)
# -----------------------------------------------------------------------------
# Define averaging type for Asian options
AsianAveragingType = Literal["arithmetic", "geometric"]

# Define lookback type: floating strike or fixed strike
LookbackType = Literal["floating_strike", "fixed_strike"]

# Touch style: one-touch pays if hit, no-touch pays if NOT hit.
TouchStyle = Literal["one_touch", "no_touch"]