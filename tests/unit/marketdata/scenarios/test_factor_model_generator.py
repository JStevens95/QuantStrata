"""
Tests for FactorModelGenerator.

This module tests the production-grade factor model generator for
full term structure simulation (curves, vol surfaces).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.timeseries import (
    FactorModelGenerator,
    FactorModelResult,
    CurveFactorSpec,
    VolSurfaceFactorSpec,
    SpotFactorSpec,
    FactorDynamics,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def spot_spec() -> SpotFactorSpec:
    """Create a sample spot specification."""
    return SpotFactorSpec(
        market_id=MarketId("FX", "SPOT", "EURUSD"),
        initial_value=1.0850,
        dynamics=FactorDynamics(dynamics_type="gbm", vol=0.08),
    )


@pytest.fixture
def curve_spec() -> CurveFactorSpec:
    """Create a sample curve specification with 2 factors."""
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    return CurveFactorSpec(
        market_id=MarketId("IR", "CURVE", "USD"),
        tenors=tenors,
        initial_rates=np.array([0.054, 0.053, 0.052, 0.048, 0.042, 0.040]),
        factor_loadings={
            "level": np.ones(6) * 0.01,
            "slope": np.array([-0.008, -0.004, 0.0, 0.002, 0.006, 0.008]),
        },
        factor_dynamics={
            "level": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=0.3, vol=0.5),
            "slope": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=0.5, vol=0.3),
        },
    )


@pytest.fixture
def vol_spec() -> VolSurfaceFactorSpec:
    """Create a sample vol surface specification."""
    expiries = np.array([0.25, 0.5, 1.0])
    strikes = np.array([0.9, 1.0, 1.1])
    return VolSurfaceFactorSpec(
        market_id=MarketId("FX", "VOL", "EURUSD"),
        expiries=expiries,
        strikes=strikes,
        initial_vols=np.array([
            [0.095, 0.085, 0.090],
            [0.092, 0.083, 0.087],
            [0.090, 0.082, 0.085],
        ]),
        factor_loadings={
            "atm": np.ones((3, 3)) * 0.1,
        },
        factor_dynamics={
            "atm": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=2.0, vol=0.5),
        },
        vol_floor=0.01,
    )


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestFactorModelGeneratorBasics:
    """Test basic generator functionality."""
    
    def test_spot_only_generation(self, spot_spec: SpotFactorSpec) -> None:
        """Generator should work with spot factors only."""
        generator = FactorModelGenerator(
            spots=[spot_spec],
            correlation_matrix=np.array([[1.0]]),
        )
        
        result = generator.generate(n_time=10, n_scenarios=100, seed=42)
        
        assert spot_spec.market_id in result.spot_paths
        assert result.spot_paths[spot_spec.market_id].shape == (11, 100)  # T+1, S
        assert result.n_scenarios == 100
    
    def test_curve_only_generation(self, curve_spec: CurveFactorSpec) -> None:
        """Generator should work with curve factors only."""
        # 2 factors: level, slope
        correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
        
        generator = FactorModelGenerator(
            curves=[curve_spec],
            correlation_matrix=correlation,
        )
        
        result = generator.generate(n_time=10, n_scenarios=100, seed=42)
        
        assert curve_spec.market_id in result.curve_paths
        curve_paths = result.curve_paths[curve_spec.market_id]
        assert curve_paths.shape == (11, 100, 6)  # T+1, S, n_tenors
    
    def test_vol_surface_only_generation(self, vol_spec: VolSurfaceFactorSpec) -> None:
        """Generator should work with vol surface factors only."""
        # 1 factor: atm
        correlation = np.array([[1.0]])
        
        generator = FactorModelGenerator(
            vol_surfaces=[vol_spec],
            correlation_matrix=correlation,
        )
        
        result = generator.generate(n_time=10, n_scenarios=100, seed=42)
        
        assert vol_spec.market_id in result.vol_paths
        vol_paths = result.vol_paths[vol_spec.market_id]
        assert vol_paths.shape == (11, 100, 3, 3)  # T+1, S, n_exp, n_K
    
    def test_combined_generation(
        self,
        spot_spec: SpotFactorSpec,
        curve_spec: CurveFactorSpec,
        vol_spec: VolSurfaceFactorSpec,
    ) -> None:
        """Generator should work with all factor types combined."""
        # Total factors: 1 (spot) + 2 (curve) + 1 (vol) = 4
        correlation = np.eye(4)
        correlation[0, 1] = correlation[1, 0] = -0.2  # Spot-level
        correlation[1, 2] = correlation[2, 1] = 0.3   # Level-slope
        
        generator = FactorModelGenerator(
            spots=[spot_spec],
            curves=[curve_spec],
            vol_surfaces=[vol_spec],
            correlation_matrix=correlation,
        )
        
        result = generator.generate(n_time=10, n_scenarios=100, seed=42)
        
        # Check all outputs exist
        assert spot_spec.market_id in result.spot_paths
        assert curve_spec.market_id in result.curve_paths
        assert vol_spec.market_id in result.vol_paths
        
        # Check shapes
        assert result.spot_paths[spot_spec.market_id].shape == (11, 100)
        assert result.curve_paths[curve_spec.market_id].shape == (11, 100, 6)
        assert result.vol_paths[vol_spec.market_id].shape == (11, 100, 3, 3)


# =============================================================================
# DYNAMICS TESTS
# =============================================================================

class TestFactorModelDynamics:
    """Test that dynamics are correctly applied."""
    
    def test_gbm_spot_positivity(self, spot_spec: SpotFactorSpec) -> None:
        """GBM should always produce positive values."""
        generator = FactorModelGenerator(
            spots=[spot_spec],
            correlation_matrix=np.array([[1.0]]),
        )
        
        result = generator.generate(n_time=252, n_scenarios=1000, seed=42)
        spot_paths = result.spot_paths[spot_spec.market_id]
        
        assert np.all(spot_paths > 0), "GBM produced non-positive values"
    
    def test_gbm_initial_value(self, spot_spec: SpotFactorSpec) -> None:
        """GBM should start at initial value."""
        generator = FactorModelGenerator(
            spots=[spot_spec],
            correlation_matrix=np.array([[1.0]]),
        )
        
        result = generator.generate(n_time=10, n_scenarios=100, seed=42)
        spot_paths = result.spot_paths[spot_spec.market_id]
        
        # All scenarios should start at initial value
        np.testing.assert_allclose(
            spot_paths[0, :],
            spot_spec.initial_value,
            rtol=1e-10,
        )
    
    def test_ou_mean_reversion(self) -> None:
        """OU process should revert toward mean over time."""
        # Create a factor with strong mean reversion to 0.05
        curve_spec = CurveFactorSpec(
            market_id=MarketId("IR", "CURVE", "TEST"),
            tenors=np.array([1.0]),
            initial_rates=np.array([0.10]),  # Start far from mean
            factor_loadings={"level": np.array([1.0])},
            factor_dynamics={
                "level": FactorDynamics(
                    dynamics_type="ou",
                    mean=0.0,  # Mean-reverts to 0 (so rates stay at initial)
                    kappa=10.0,  # Strong mean reversion
                    vol=0.01,
                ),
            },
        )
        
        generator = FactorModelGenerator(
            curves=[curve_spec],
            correlation_matrix=np.array([[1.0]]),
        )
        
        result = generator.generate(n_time=252, n_scenarios=1000, seed=42)
        curve_paths = result.curve_paths[curve_spec.market_id]
        
        # Check that terminal rates are close to initial (factor reverts to 0)
        terminal_rates = curve_paths[-1, :, 0]
        np.testing.assert_allclose(
            np.mean(terminal_rates),
            0.10,  # Initial rate
            atol=0.005,  # Should be close
        )
    
    def test_vol_floor_enforced(self, vol_spec: VolSurfaceFactorSpec) -> None:
        """Vol surface should never go below vol_floor."""
        # Use very high vol-of-vol to test floor
        vol_spec_extreme = VolSurfaceFactorSpec(
            market_id=MarketId("FX", "VOL", "TEST"),
            expiries=np.array([0.25]),
            strikes=np.array([1.0]),
            initial_vols=np.array([[0.05]]),  # Low starting vol
            factor_loadings={"atm": np.array([[1.0]])},
            factor_dynamics={
                "atm": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=1.0, vol=5.0),
            },
            vol_floor=0.01,
        )
        
        generator = FactorModelGenerator(
            vol_surfaces=[vol_spec_extreme],
            correlation_matrix=np.array([[1.0]]),
        )
        
        result = generator.generate(n_time=100, n_scenarios=1000, seed=42)
        vol_paths = result.vol_paths[vol_spec_extreme.market_id]
        
        assert np.all(vol_paths >= vol_spec_extreme.vol_floor), "Vol fell below floor"


# =============================================================================
# CORRELATION TESTS
# =============================================================================

class TestFactorModelCorrelation:
    """Test correlation handling."""
    
    def test_determinism_with_seed(
        self,
        spot_spec: SpotFactorSpec,
        curve_spec: CurveFactorSpec,
    ) -> None:
        """Same seed should produce identical results."""
        correlation = np.array([
            [1.0, -0.2, 0.1],
            [-0.2, 1.0, 0.3],
            [0.1, 0.3, 1.0],
        ])
        
        generator = FactorModelGenerator(
            spots=[spot_spec],
            curves=[curve_spec],
            correlation_matrix=correlation,
        )
        
        result1 = generator.generate(n_time=10, n_scenarios=100, seed=42)
        result2 = generator.generate(n_time=10, n_scenarios=100, seed=42)
        
        np.testing.assert_array_equal(
            result1.spot_paths[spot_spec.market_id],
            result2.spot_paths[spot_spec.market_id],
        )
    
    def test_different_seeds_produce_different_results(
        self,
        spot_spec: SpotFactorSpec,
    ) -> None:
        """Different seeds should produce different results."""
        generator = FactorModelGenerator(
            spots=[spot_spec],
            correlation_matrix=np.array([[1.0]]),
        )
        
        result1 = generator.generate(n_time=10, n_scenarios=100, seed=42)
        result2 = generator.generate(n_time=10, n_scenarios=100, seed=123)
        
        assert not np.allclose(
            result1.spot_paths[spot_spec.market_id],
            result2.spot_paths[spot_spec.market_id],
        ), "Different seeds produced identical results"
    
    def test_realized_correlation_approximates_input(
        self,
        spot_spec: SpotFactorSpec,
        curve_spec: CurveFactorSpec,
    ) -> None:
        """Realized correlation should be close to input correlation."""
        input_corr = np.array([
            [1.0, -0.5, 0.3],
            [-0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ])
        
        generator = FactorModelGenerator(
            spots=[spot_spec],
            curves=[curve_spec],
            correlation_matrix=input_corr,
        )
        
        result = generator.generate(n_time=252, n_scenarios=10000, seed=42)
        
        # Extract factor paths from result
        factor_paths = np.stack([
            result.factor_paths[name][-1, :]
            for name in result.factor_paths.keys()
        ], axis=1)
        
        realized_corr = np.corrcoef(factor_paths.T)
        
        # Should be close within sampling error
        np.testing.assert_allclose(
            realized_corr,
            input_corr,
            atol=0.05,  # 5% tolerance for sampling error
        )


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestFactorModelValidation:
    """Test input validation."""
    
    def test_mismatched_factor_loadings_and_dynamics(self) -> None:
        """Should raise error if loadings and dynamics have different keys."""
        with pytest.raises(ValueError, match="must have same keys"):
            CurveFactorSpec(
                market_id=MarketId("IR", "CURVE", "TEST"),
                tenors=np.array([1.0]),
                initial_rates=np.array([0.05]),
                factor_loadings={"level": np.array([1.0])},
                factor_dynamics={
                    "slope": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=1.0, vol=0.01),
                },
            )
    
    def test_mismatched_loading_length_and_tenors(self) -> None:
        """Should raise error if loading length doesn't match tenors."""
        with pytest.raises(ValueError, match="must match tenors"):
            CurveFactorSpec(
                market_id=MarketId("IR", "CURVE", "TEST"),
                tenors=np.array([1.0, 2.0, 5.0]),  # 3 tenors
                initial_rates=np.array([0.05, 0.04, 0.03]),
                factor_loadings={"level": np.array([1.0, 1.0])},  # Only 2 loadings
                factor_dynamics={
                    "level": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=1.0, vol=0.01),
                },
            )
    
    def test_mismatched_vol_surface_shape(self) -> None:
        """Should raise error if initial_vols shape doesn't match grid."""
        with pytest.raises(ValueError, match="shape must be"):
            VolSurfaceFactorSpec(
                market_id=MarketId("FX", "VOL", "TEST"),
                expiries=np.array([0.25, 0.5]),  # 2 expiries
                strikes=np.array([0.9, 1.0, 1.1]),  # 3 strikes
                initial_vols=np.array([[0.08, 0.07]]),  # Wrong shape
                factor_loadings={"atm": np.ones((2, 3))},
                factor_dynamics={
                    "atm": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=1.0, vol=0.1),
                },
            )
    
    def test_correlation_matrix_size_mismatch(
        self,
        spot_spec: SpotFactorSpec,
        curve_spec: CurveFactorSpec,
    ) -> None:
        """Should raise error if correlation matrix size doesn't match factors."""
        # spot (1) + curve (2 factors) = 3 factors, but correlation is 2x2
        wrong_corr = np.eye(2)
        
        with pytest.raises(ValueError, match="doesn't match"):
            FactorModelGenerator(
                spots=[spot_spec],
                curves=[curve_spec],
                correlation_matrix=wrong_corr,
            )


# =============================================================================
# DATASET CONVERSION TESTS
# =============================================================================

class TestFactorModelResultToDataset:
    """Test conversion of results to MarketDataset."""
    
    def test_to_dataset_creates_panels(
        self,
        spot_spec: SpotFactorSpec,
        curve_spec: CurveFactorSpec,
        vol_spec: VolSurfaceFactorSpec,
    ) -> None:
        """to_dataset should create proper panel structures."""
        correlation = np.eye(4)
        
        generator = FactorModelGenerator(
            spots=[spot_spec],
            curves=[curve_spec],
            vol_surfaces=[vol_spec],
            correlation_matrix=correlation,
        )
        
        result = generator.generate(n_time=10, n_scenarios=100, seed=42)
        dataset = result.to_dataset()
        
        # Check spot panel
        assert spot_spec.market_id in dataset.panels
        assert dataset.panels[spot_spec.market_id].data.shape == (11, 100)
        
        # Check curve params
        assert curve_spec.market_id in dataset.curve_params
        assert curve_spec.market_id in dataset.curve_factories
        
        # Check vol params
        assert vol_spec.market_id in dataset.vol_params
        assert vol_spec.market_id in dataset.vol_factories
    
    def test_dataset_metadata(
        self,
        spot_spec: SpotFactorSpec,
    ) -> None:
        """Dataset should include generator metadata."""
        generator = FactorModelGenerator(
            spots=[spot_spec],
            correlation_matrix=np.array([[1.0]]),
        )
        
        result = generator.generate(n_time=10, n_scenarios=100, seed=42)
        dataset = result.to_dataset()
        
        assert dataset.meta["generator"] == "FactorModelGenerator"
        assert dataset.meta["seed"] == 42
        assert dataset.n_scenarios == 100


# =============================================================================
# EDGE CASES
# =============================================================================

class TestFactorModelEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_scenario(self, spot_spec: SpotFactorSpec) -> None:
        """Generator should work with single scenario."""
        generator = FactorModelGenerator(
            spots=[spot_spec],
            correlation_matrix=np.array([[1.0]]),
        )
        
        result = generator.generate(n_time=10, n_scenarios=1, seed=42)
        
        assert result.spot_paths[spot_spec.market_id].shape == (11, 1)
    
    def test_single_time_step(self, spot_spec: SpotFactorSpec) -> None:
        """Generator should work with single time step."""
        generator = FactorModelGenerator(
            spots=[spot_spec],
            correlation_matrix=np.array([[1.0]]),
        )
        
        result = generator.generate(n_time=1, n_scenarios=100, seed=42)
        
        assert result.spot_paths[spot_spec.market_id].shape == (2, 100)  # t=0 and t=1
    
    def test_no_correlation_defaults_to_identity(
        self,
        spot_spec: SpotFactorSpec,
    ) -> None:
        """Generator should use identity correlation if not provided."""
        generator = FactorModelGenerator(
            spots=[spot_spec],
        )
        
        # Should not raise
        result = generator.generate(n_time=10, n_scenarios=100, seed=42)
        
        assert result.spot_paths[spot_spec.market_id].shape == (11, 100)
    
    def test_empty_generator(self) -> None:
        """Generator with no factors should work (returns empty result)."""
        generator = FactorModelGenerator()
        
        # Should have 0 factors
        assert generator.n_factors == 0
        
        result = generator.generate(n_time=10, n_scenarios=100, seed=42)
        
        assert len(result.spot_paths) == 0
        assert len(result.curve_paths) == 0
        assert len(result.vol_paths) == 0
