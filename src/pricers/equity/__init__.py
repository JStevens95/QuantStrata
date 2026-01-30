# src/pricers/equity/__init__.py
"""
Equity Pricers Module.

Pricing engines for equity derivatives:
- BSM (Black-Scholes-Merton) analytic pricing
- Monte Carlo simulation
- Finite Difference (PDE) methods
- American options with early exercise
- Exotic options (barrier, digital, Asian, lookback)

Author: QuantStrata Team
"""

# BSM Pricers (vanilla + digital)
from src.pricers.equity.european_bsm import (
    EquityEuropeanVanillaBsmPricer,  # Analytic BSM for vanilla options.
    EquityEuropeanDigitalBsmPricer,  # Analytic BSM for digital options.
)

# MC Simulation Artifacts
from src.pricers.equity.european_mc import (
    # Simulation artifacts (for analysis/plotting without rerunning).
    EquityMcSimulation,  # Vanilla simulation artifact.
    EquityBarrierMcSimulation,  # Barrier simulation artifact.
    EquityAsianMcSimulation,  # Asian simulation artifact.
    EquityLookbackMcSimulation,  # Lookback simulation artifact.
    # MC Pricers.
    EquityEuropeanVanillaMcPricer,  # MC for vanilla options.
    EquityEuropeanBarrierMcPricer,  # MC for barrier options.
    EquityEuropeanAsianMcPricer,  # MC for Asian options.
    EquityEuropeanLookbackMcPricer,  # MC for lookback options.
)

# FDE Pricers (finite difference / PDE)
from src.pricers.equity.european_fde import EquityEuropeanVanillaFdPricer  # FD for European vanilla.
from src.pricers.equity.american_fde import EquityAmericanVanillaFdPricer  # FD for American vanilla.

# Linear pricers (forwards and spots)
from src.pricers.equity.forward import EquityForwardPricer  # Forward pricer.
from src.pricers.equity.spot import EquitySpotPricer  # Spot pricer.

__all__ = [
    # BSM (analytic)
    "EquityEuropeanVanillaBsmPricer",
    "EquityEuropeanDigitalBsmPricer",
    # MC Simulation Artifacts
    "EquityMcSimulation",
    "EquityBarrierMcSimulation",
    "EquityAsianMcSimulation",
    "EquityLookbackMcSimulation",
    # MC Pricers
    "EquityEuropeanVanillaMcPricer",
    "EquityEuropeanBarrierMcPricer",
    "EquityEuropeanAsianMcPricer",
    "EquityEuropeanLookbackMcPricer",
    # FDE (PDE)
    "EquityEuropeanVanillaFdPricer",
    "EquityAmericanVanillaFdPricer",
    # Linear
    "EquityForwardPricer",
    "EquitySpotPricer",
]
