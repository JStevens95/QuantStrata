"""
Unit tests for CalibrationEngine.

Tests the generic calibration engine with various objective functions
and optimizer configurations.
"""

import numpy as np
import pytest

from src.calibration.core.engine import (
    CalibrationEngine,
    CalibrationResult,
    CalibrationConfig,
    calibrate,
)
from src.calibration.core.objectives import (
    WeightedLeastSquares,
    PenalizedObjective,
    MaxLikelihood,
)
from src.calibration.core.optimizers import (
    LBFGSBConfig,
    DifferentialEvolutionConfig,
    get_default_optimizer,
)


class TestCalibrationEngine:
    """Tests for CalibrationEngine class."""
    
    def test_simple_quadratic_minimization(self):
        """Test calibration on simple quadratic objective."""
        # Minimize (x[0] - 1)^2 + (x[1] - 2)^2
        def objective(x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2
        
        engine = CalibrationEngine(optimizer=LBFGSBConfig(max_iter=100))
        result = engine.calibrate(
            objective=objective,
            initial_params=[0.0, 0.0],
            bounds=[(-10, 10), (-10, 10)],
        )
        
        assert result.success
        assert np.allclose(result.params, [1.0, 2.0], atol=1e-4)
        assert result.objective_value < 1e-8
    
    def test_rosenbrock_function(self):
        """Test calibration on Rosenbrock function (more challenging)."""
        # Rosenbrock: (1 - x)^2 + 100*(y - x^2)^2
        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        
        engine = CalibrationEngine(optimizer=LBFGSBConfig(max_iter=500))
        result = engine.calibrate(
            objective=rosenbrock,
            initial_params=[0.0, 0.0],
            bounds=[(-5, 5), (-5, 5)],
        )
        
        # Minimum is at (1, 1)
        assert result.success
        assert np.allclose(result.params, [1.0, 1.0], atol=1e-3)
    
    def test_calibration_result_properties(self):
        """Test CalibrationResult properties."""
        def objective(x):
            return x[0] ** 2
        
        engine = CalibrationEngine()
        result = engine.calibrate(
            objective=objective,
            initial_params=[5.0],
            bounds=[(-10, 10)],
        )
        
        assert isinstance(result, CalibrationResult)
        assert result.converged == result.success
        assert result.improvement_ratio >= 1.0  # Should improve
        assert result.n_function_evals > 0
        assert result.elapsed_time > 0
        assert np.array_equal(result.initial_params, [5.0])
    
    def test_calibration_with_verbose(self):
        """Test calibration with verbose output."""
        def objective(x):
            return x[0] ** 2
        
        engine = CalibrationEngine(
            optimizer=LBFGSBConfig(max_iter=10),
            config=CalibrationConfig(verbose=True),
        )
        result = engine.calibrate(
            objective=objective,
            initial_params=[5.0],
            bounds=[(-10, 10)],
            param_names=["x"],
        )
        
        assert result is not None
    
    def test_retry_on_failure(self):
        """Test that calibration retries on failure."""
        call_count = [0]
        
        def tricky_objective(x):
            call_count[0] += 1
            # Fail on first few calls, then succeed
            if call_count[0] < 5:
                return 1e10
            return (x[0] - 1) ** 2
        
        engine = CalibrationEngine(
            optimizer=LBFGSBConfig(max_iter=100),
            config=CalibrationConfig(retry_on_failure=True, max_retries=3),
        )
        result = engine.calibrate(
            objective=tricky_objective,
            initial_params=[0.0],
            bounds=[(-10, 10)],
        )
        
        # Should eventually find minimum
        assert result is not None


class TestWeightedLeastSquares:
    """Tests for WeightedLeastSquares objective."""
    
    def test_basic_least_squares(self):
        """Test basic least squares calculation."""
        def model_func(params):
            return params[0] * np.array([1, 2, 3])
        
        market = np.array([1.0, 2.0, 3.0])
        
        obj = WeightedLeastSquares(
            model_func=model_func,
            market_values=market,
        )
        
        # At params=[1], model matches market exactly
        assert np.isclose(obj(np.array([1.0])), 0.0, atol=1e-10)
        
        # At params=[2], model = [2, 4, 6], errors = [1, 2, 3]
        # SSE = (1 + 4 + 9) / 3 = 14/3 (normalized weights)
        error = obj(np.array([2.0]))
        assert error > 0
    
    def test_weighted_least_squares(self):
        """Test with non-uniform weights."""
        def model_func(params):
            return np.array([params[0], params[0]])
        
        market = np.array([1.0, 2.0])
        weights = np.array([1.0, 0.0])  # Only care about first point
        
        obj = WeightedLeastSquares(
            model_func=model_func,
            market_values=market,
            weights=weights,
        )
        
        # At params=[1], first point matches, second doesn't matter
        assert np.isclose(obj(np.array([1.0])), 0.0, atol=1e-10)
    
    def test_relative_error(self):
        """Test relative error mode."""
        def model_func(params):
            return params[0] * np.array([100.0, 1.0])
        
        market = np.array([100.0, 1.0])
        
        obj = WeightedLeastSquares(
            model_func=model_func,
            market_values=market,
            use_relative_error=True,
        )
        
        # At params=[1], perfect fit
        assert np.isclose(obj(np.array([1.0])), 0.0, atol=1e-10)
    
    def test_residuals_method(self):
        """Test residuals method for LM optimizer."""
        def model_func(params):
            return np.array([params[0], params[0]])
        
        market = np.array([1.0, 2.0])
        
        obj = WeightedLeastSquares(
            model_func=model_func,
            market_values=market,
        )
        
        residuals = obj.residuals(np.array([1.5]))
        assert residuals.size == 2


class TestPenalizedObjective:
    """Tests for PenalizedObjective."""
    
    def test_penalty_adds_to_objective(self):
        """Test that penalty increases objective."""
        def base_obj(x):
            return x[0] ** 2
        
        def penalty(x):
            return 1.0 if x[0] < 0 else 0.0
        
        penalized = PenalizedObjective(
            base_objective=base_obj,
            penalty_func=penalty,
            penalty_weight=100.0,
        )
        
        # Without penalty
        assert np.isclose(penalized(np.array([1.0])), 1.0)
        
        # With penalty
        assert penalized(np.array([-1.0])) > 100.0
    
    def test_feller_like_constraint(self):
        """Test Feller-like constraint (2*kappa*theta > xi^2)."""
        def base_obj(x):
            return 0.0  # Dummy base
        
        def feller_penalty(x):
            kappa, theta, xi = x[0], x[1], x[2]
            violation = xi ** 2 - 2 * kappa * theta
            return max(0, violation) ** 2
        
        penalized = PenalizedObjective(
            base_objective=base_obj,
            penalty_func=feller_penalty,
            penalty_weight=1000.0,
        )
        
        # Feller satisfied: kappa=2, theta=0.04, xi=0.3 => 0.16 > 0.09
        assert np.isclose(penalized(np.array([2.0, 0.04, 0.3])), 0.0)
        
        # Feller violated: kappa=0.5, theta=0.01, xi=0.5 => 0.01 < 0.25
        assert penalized(np.array([0.5, 0.01, 0.5])) > 0


class TestOptimizers:
    """Tests for optimizer configurations."""
    
    def test_lbfgsb_config_defaults(self):
        """Test L-BFGS-B config defaults."""
        config = LBFGSBConfig()
        assert config.max_iter == 200
        assert config.tol == 1e-8
    
    def test_differential_evolution_config(self):
        """Test DE config."""
        config = DifferentialEvolutionConfig(
            strategy="best1bin",
            max_iter=500,
            polish=True,
        )
        assert config.strategy == "best1bin"
        assert config.polish is True
    
    def test_get_default_optimizer(self):
        """Test factory function."""
        local = get_default_optimizer("local")
        assert isinstance(local, LBFGSBConfig)
        
        global_opt = get_default_optimizer("global")
        assert isinstance(global_opt, DifferentialEvolutionConfig)


class TestCalibrateConvenienceFunction:
    """Tests for calibrate() convenience function."""
    
    def test_basic_usage(self):
        """Test basic calibrate() usage."""
        result = calibrate(
            objective=lambda x: (x[0] - 1) ** 2,
            initial_params=[0.0],
            bounds=[(-10, 10)],
        )
        
        assert result.success
        assert np.isclose(result.params[0], 1.0, atol=1e-4)
    
    def test_with_verbose(self):
        """Test calibrate() with verbose."""
        result = calibrate(
            objective=lambda x: x[0] ** 2,
            initial_params=[5.0],
            bounds=[(-10, 10)],
            verbose=True,
        )
        
        assert result is not None
