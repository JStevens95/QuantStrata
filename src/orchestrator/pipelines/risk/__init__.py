"""
Risk Pipelines Package.

Contains pipelines for risk analytics:
- run_scenarios: Run scenario analysis (spot/vol/rate shocks)
- compute_sensitivities: Compute portfolio Greeks with aggregation
- compute_var: Compute Value-at-Risk using multiple methods
- pnl_attribution: Attribute P&L to risk factors
- validate_greeks: Validate analytic Greeks against bump-and-reprice
"""
from __future__ import annotations
