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
    EquityVanillaEuropeanOptionBsmPricer,  # Analytic BSM for vanilla options.
    EquityDigitalEuropeanOptionBsmPricer,  # Analytic BSM for digital options.
)

# MC Simulation Artifacts (BSM/GBM dynamics)
from src.pricers.equity.european_bsm_mc import (
    # Simulation artifacts (for analysis/plotting without rerunning).
    EquityVanillOptionMcSimulation,  # Vanilla simulation artifact.
    EquityBarrierOptionMcSimulation,  # Barrier simulation artifact.
    EquityAsianOptionMcSimulation,  # Asian simulation artifact.
    EquityLookbackOptionMcSimulation,  # Lookback simulation artifact.
    # MC Pricers.
    EquityVanillaEuropeanOptionMcPricer,  # MC for vanilla options.
    EquityBarrierEuropeanOptionMcPricer,  # MC for barrier options.
    EquityAsianEuropeanOptionMcPricer,  # MC for Asian options.
    EquityLookbackEuropeanOptionMcPricer,  # MC for lookback options.
)

# FDE Pricers (finite difference / PDE, BSM/GBM dynamics)
from src.pricers.equity.european_bsm_fde import EquityVanillaEuropeanOptionFdPricer  # FD for European vanilla.
from src.pricers.equity.american_bsm_fde import EquityVanillaAmericanOptionFdPricer  # FD for American vanilla.

# Linear pricers (forwards and spots)
from src.pricers.equity.forward import EquityForwardPricer  # Forward pricer.
from src.pricers.equity.spot import EquitySpotPricer  # Spot pricer.

__all__ = [
    # BSM (analytic)
    "EquityVanillaEuropeanOptionBsmPricer",
    "EquityDigitalEuropeanOptionBsmPricer",
    # MC Simulation Artifacts
    "EquityVanillOptionMcSimulation",
    "EquityBarrierOptionMcSimulation",
    "EquityAsianOptionMcSimulation",
    "EquityLookbackOptionMcSimulation",
    # MC Pricers
    "EquityVanillaEuropeanOptionMcPricer",
    "EquityBarrierEuropeanOptionMcPricer",
    "EquityAsianEuropeanOptionMcPricer",
    "EquityLookbackEuropeanOptionMcPricer",
    # FDE (PDE)
    "EquityVanillaEuropeanOptionFdPricer",
    "EquityVanillaAmericanOptionFdPricer",
    # Linear
    "EquityForwardPricer",
    "EquitySpotPricer",
]
