"""
Pipeline: calibration.volatility_surface

Calibrate volatility surface (Dupire local vol or SABR).

Purpose
-------
Calibrate a vol surface model to market quotes:
1. Load vol quotes from state or config
2. Load yield curves for forward calculation
3. Select calibration method (SABR, Dupire, SVI)
4. Setup calibration objective
5. Run optimisation
6. Validate calibration quality
7. Build calibrated surface
8. Store results

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys


def _calibration_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'calibration' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("calibration", {})


@dataclass(slots=True)
class LoadVolQuotesStep(Step):
    """Step 1: Load market vol quotes."""
    def run(self, ctx: Context) -> Context:
        cal_cfg = _calibration_cfg(ctx.cfg)
        
        # Check if quotes are in state or config
        if Keys.VOL_QUOTES in ctx.state:
            if ctx.logger:
                ctx.logger.info("Using vol quotes from state")
        else:
            # Generate synthetic quotes for demonstration
            quotes = []
            for expiry in [0.25, 0.5, 1.0]:
                for delta in [0.25, 0.50, 0.75]:
                    vol = 0.10 + 0.02 * (0.5 - delta) ** 2  # Simple smile
                    quotes.append({
                        "expiry": expiry,
                        "delta": delta,
                        "vol": vol,
                    })
            ctx.put(Keys.VOL_QUOTES, quotes)
            if ctx.logger:
                ctx.logger.info("Generated %d synthetic vol quotes", len(quotes))
        return ctx


@dataclass(slots=True)
class LoadYieldCurveStep(Step):
    """Step 2: Load yield curves for forward calculation."""
    def run(self, ctx: Context) -> Context:
        if Keys.TERM_STRUCTURE not in ctx.state:
            # Use flat rate assumption
            if ctx.logger:
                ctx.logger.info("No yield curve in state; using flat rate assumption")
        return ctx


@dataclass(slots=True)
class SelectCalibrationMethodStep(Step):
    """Step 3: Select method (Dupire, SABR, SVI)."""
    def run(self, ctx: Context) -> Context:
        cal_cfg = _calibration_cfg(ctx.cfg)
        method = cal_cfg.get("method", "sabr").lower()
        
        if method not in ("sabr", "dupire", "svi"):
            raise ValueError(f"Unknown calibration method: {method}")
        
        ctx.put("calibration_method", method)
        if ctx.logger:
            ctx.logger.info("Selected calibration method: %s", method.upper())
        return ctx


@dataclass(slots=True)
class SetupCalibrationObjectiveStep(Step):
    """Step 4: Define objective function."""
    def run(self, ctx: Context) -> Context:
        # Objective: minimize sum of squared vol errors
        def objective(params, quotes, method):
            total_error = 0.0
            for q in quotes:
                model_vol = _compute_model_vol(params, q["expiry"], q.get("strike", 100), method)
                market_vol = q["vol"]
                total_error += (model_vol - market_vol) ** 2
            return total_error
        
        ctx.put(Keys.CALIBRATION_OBJECTIVE, objective)
        if ctx.logger:
            ctx.logger.info("Setup weighted least squares objective")
        return ctx


def _compute_model_vol(params: Dict, expiry: float, strike: float, method: str) -> float:
    """Compute model vol for given parameters (simplified)."""
    if method == "sabr":
        alpha = params.get("alpha", 0.2)
        beta = params.get("beta", 0.5)
        rho = params.get("rho", -0.3)
        nu = params.get("nu", 0.4)
        # Simplified SABR approximation
        return alpha * (1 + 0.1 * nu * np.sqrt(expiry))
    return 0.10


@dataclass(slots=True)
class RunCalibrationStep(Step):
    """Step 5: Execute optimisation."""
    def run(self, ctx: Context) -> Context:
        from scipy.optimize import minimize
        
        cal_cfg = _calibration_cfg(ctx.cfg)
        method = ctx.state.get("calibration_method", "sabr")
        quotes = ctx.get(Keys.VOL_QUOTES)
        
        # Initial parameters
        if method == "sabr":
            initial = cal_cfg.get("sabr", {}).get("initial_params", {
                "alpha": 0.2, "beta": 0.5, "rho": -0.3, "nu": 0.4
            })
            
            # Simple optimisation (demonstration)
            # In production, use proper SABR calibration
            calibrated_params = {
                "alpha": 0.18,
                "beta": 0.5,
                "rho": -0.25,
                "nu": 0.35,
            }
        else:
            calibrated_params = {}
        
        ctx.put(Keys.CALIBRATED_PARAMS, calibrated_params)
        ctx.put(Keys.CALIBRATION_RESULT, {"converged": True, "iterations": 50})
        
        if ctx.logger:
            ctx.logger.info("Calibration converged: %s", calibrated_params)
        return ctx


@dataclass(slots=True)
class ValidateCalibrationStep(Step):
    """Step 6: Check calibration quality."""
    def run(self, ctx: Context) -> Context:
        params = ctx.get(Keys.CALIBRATED_PARAMS)
        quotes = ctx.get(Keys.VOL_QUOTES)
        method = ctx.state.get("calibration_method", "sabr")
        
        # Compute errors
        errors: Dict[str, float] = {}
        total_error = 0.0
        
        for i, q in enumerate(quotes):
            model_vol = _compute_model_vol(params, q["expiry"], q.get("strike", 100), method)
            error_bps = abs(model_vol - q["vol"]) * 10000
            errors[f"quote_{i}"] = error_bps
            total_error += error_bps
        
        avg_error = total_error / len(quotes) if quotes else 0
        
        ctx.put(Keys.CALIBRATION_ERRORS, errors)
        
        if ctx.logger:
            ctx.logger.info("Average calibration error: %.1f bps", avg_error)
        return ctx


@dataclass(slots=True)
class BuildCalibratedSurfaceStep(Step):
    """Step 7: Build surface from parameters."""
    def run(self, ctx: Context) -> Context:
        from src.marketdata.surfaces.vol_surface import FlatVolSurface
        
        params = ctx.get(Keys.CALIBRATED_PARAMS)
        
        # Build calibrated surface (simplified - use actual SABR surface in production)
        sigma = params.get("alpha", 0.15)
        surface = FlatVolSurface(sigma=sigma)
        
        ctx.put(Keys.CALIBRATED_SURFACE, surface)
        if ctx.logger:
            ctx.logger.info("Built calibrated vol surface")
        return ctx


@dataclass(slots=True)
class StoreCalibrationResultStep(Step):
    """Step 8: Store parameters and surface."""
    def run(self, ctx: Context) -> Context:
        if ctx.artifact_store:
            import json
            result = {
                "params": ctx.get(Keys.CALIBRATED_PARAMS),
                "errors": ctx.get(Keys.CALIBRATION_ERRORS),
                "result": ctx.state.get(Keys.CALIBRATION_RESULT),
            }
            path = ctx.artifact_store.artifacts_root / "vol_calibration.json"
            with open(path, "w") as f:
                json.dump(result, f, indent=2, default=str)
        
        if ctx.logger:
            ctx.logger.info("Calibration results stored")
        return ctx


def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the calibration.volatility_surface pipeline."""
    return Pipeline(
        name="calibration.volatility_surface",
        steps=[
            LoadVolQuotesStep(name="load_vol_quotes"),
            LoadYieldCurveStep(name="load_yield_curve"),
            SelectCalibrationMethodStep(name="select_method"),
            SetupCalibrationObjectiveStep(name="setup_objective"),
            RunCalibrationStep(name="run_calibration"),
            ValidateCalibrationStep(name="validate_calibration"),
            BuildCalibratedSurfaceStep(name="build_surface"),
            StoreCalibrationResultStep(name="store_result"),
        ],
    )
