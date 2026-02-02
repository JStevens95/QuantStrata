"""
Value-at-Risk (VaR): Historical, Parametric (delta-normal), Monte Carlo.

Exports
-------
VarConfig, VarResult
historical_var, parametric_var, mc_var
compute_var (facade)
DiagonalFactorModel, FactorModel (protocol)
"""

from src.risk.var.config import VarConfig, VarResult
from src.risk.var.historical import historical_var
from src.risk.var.mc import DiagonalFactorModel, mc_var
from src.risk.var.parametric import parametric_var
from src.risk.var.runner import compute_var

__all__ = [
    "VarConfig",
    "VarResult",
    "historical_var",
    "parametric_var",
    "mc_var",
    "compute_var",
    "DiagonalFactorModel",
]
