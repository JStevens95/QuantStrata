from __future__ import annotations

from typing import Literal

# -----------------------------------------------------------------------------
# Core marketdata conventions & small enums (type aliases)
# -----------------------------------------------------------------------------

# How a curve/surface extrapolates beyond its calibration grid.
ExtrapolationMode = Literal["flat", "linear"]

# Deposit compounding convention used by simple deposit quotes / generators.
DepositCompounding = Literal["simple", "continuous"]

# Instrument kind tags for curve bootstrapping inputs.
InstrumentType = Literal["deposit", "swap"]

# FX vol quote delta convention ("spot" vs "forward") for strike inversion.
DeltaType = Literal["forward", "spot"]

# ATM quote convention for FX vols (V1/V2: forward ATM).
AtmType = Literal["forward"]

# Which backend should perform heavy lifting for bootstrapping/calibration.
# (Keep "native" as default so the library works without QuantLib installed.)
BootstrapEngine = Literal["native", "quantlib"]

# curve method for bootstrapping.
CurveMethod = Literal["zeros", "bootstrap"]

# define panel type.
PanelKind = Literal["quote", "curve_params", "vol_params"]

# Volatility type for IR products (normal = Bachelier, lognormal = Black)
VolType = Literal["normal", "lognormal"]

# Metadata about what the "strike axis" represents.
StrikeSpace = Literal[
    "absolute",              # K in price units (canonical for pricing)
    "spot_moneyness",        # K = m * S0  (stored values are m, not K)  [NOT recommended to store long-term]
    "forward_moneyness",     # K = m * F0(T) (stored values are m)       [NOT recommended to store in 2D strike grid]
    "log_forward_moneyness", # log(K/F0(T))                              [NOT recommended to store in 2D strike grid]
]
