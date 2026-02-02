"""Parametric (delta-normal) VaR from sensitivities and factor covariance."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from src.risk.sensitivities.result import SensitivityKey, SensitivitiesReport
from src.risk.var.config import VarConfig, VarResult


def parametric_var(
    sensitivities_report: SensitivitiesReport,
    factor_volatilities: dict[SensitivityKey, float],
    config: VarConfig,
    *,
    factor_correlation: Optional[np.ndarray] = None,
    factor_order: Optional[Sequence[SensitivityKey]] = None,
    include_cvar: bool = True,
) -> VarResult:
    """
    Compute Parametric (delta-normal) VaR from portfolio sensitivities and factor volatilities.

    VaR = z(confidence) * sqrt(Γ' Σ Γ) for single-period; scaled by sqrt(horizon_days)
    for multi-day. Γ = sensitivity vector, Σ = factor covariance (from vols and optional correlation).

    Parameters
    ----------
    sensitivities_report : SensitivitiesReport
        Portfolio sensitivities (from compute_sensitivities).
    factor_volatilities : dict
        Mapping SensitivityKey -> daily factor volatility (e.g. daily return vol).
    config : VarConfig
        confidence, horizon_days; method must be "parametric".
    factor_correlation : np.ndarray, optional
        Correlation matrix of factor returns (same order as factor_order / report rows).
        If None, factors are assumed uncorrelated.
    factor_order : sequence of SensitivityKey, optional
        Order of factors for Γ and Σ. If None, use order of report.rows (only keys in factor_volatilities).
    include_cvar : bool
        If True, set cvar using normal assumption: CVaR = φ(z)/(1-confidence) * sigma_PL.

    Returns
    -------
    VarResult
    """
    if config.method != "parametric":
        raise ValueError("VarConfig.method must be 'parametric' for parametric_var.")
    if not sensitivities_report.rows:
        raise ValueError("sensitivities_report must not be empty.")
    if not factor_volatilities:
        raise ValueError("factor_volatilities must not be empty.")

    # Build ordered list of keys and sensitivity vector Γ
    if factor_order is not None:
        order = [k for k in factor_order if k in factor_volatilities]
    else:
        order = [r.key for r in sensitivities_report.rows if r.key in factor_volatilities]
    if not order:
        raise ValueError("No overlap between report keys and factor_volatilities.")

    report_by_key = {r.key: r.value for r in sensitivities_report.rows}
    gamma = np.array([float(report_by_key.get(k, 0.0)) for k in order], dtype=float)
    vols = np.array([float(factor_volatilities[k]) for k in order], dtype=float)

    # Covariance: Σ = diag(vols) @ R @ diag(vols)
    if factor_correlation is not None:
        r = np.asarray(factor_correlation, dtype=float)
        if r.shape != (len(order), len(order)):
            raise ValueError("factor_correlation shape must match number of factors.")
        sigma = np.diag(vols) @ r @ np.diag(vols)
    else:
        sigma = np.diag(vols * vols)

    # Portfolio variance of P&L: var = Γ' Σ Γ
    var_pl = float(np.dot(gamma, np.dot(sigma, gamma)))
    if var_pl <= 0.0:
        std_pl = 0.0
    else:
        std_pl = math.sqrt(var_pl)

    # VaR = z * std_pl (single period); scale for horizon
    z = float(stats.norm.ppf(config.confidence))
    var_one = z * std_pl
    scale = math.sqrt(max(1, config.horizon_days))
    var_scaled = var_one * scale

    cvar_val: Optional[float] = None
    if include_cvar and std_pl > 0:
        # CVaR for normal: ES = sigma * phi(z) / (1 - confidence) where z = Phi^{-1}(confidence)
        phi_z = float(stats.norm.pdf(z))
        cvar_one = std_pl * phi_z / (1.0 - config.confidence)
        cvar_val = cvar_one * scale

    return VarResult(
        var=var_scaled,
        cvar=cvar_val,
        method="parametric",
        confidence=config.confidence,
        horizon_days=config.horizon_days,
        metadata={"n_factors": len(order)},
    )
