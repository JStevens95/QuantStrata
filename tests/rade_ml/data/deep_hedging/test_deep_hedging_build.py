"""Unit tests for rade_ml.data.deep_hedging.build -- build_deep_hedging_data."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.data.deep_hedging.build import build_deep_hedging_data, _build_feature_tensor
from src.rade_ml.data.deep_hedging.config import (
    DeepHedgingDataConfig, MarketDynamicsConfig, OptionConfig, SimulationConfig,
)
from src.rade_ml.data.deep_hedging.simulators import GBMSimulator, SimulationResult
from src.rade_ml.data.result import DataBuildResult


@pytest.fixture
def small_config():
    return DeepHedgingDataConfig(
        market=MarketDynamicsConfig(model="gbm", spot_0=100.0, risk_free_rate=0.05, volatility=0.2),
        option=OptionConfig(option_type="call", strike=100.0, maturity_years=0.25),
        simulation=SimulationConfig(num_paths=200, num_steps=10, seed=42),
        batch_size=32,
        shuffle=False,
    )


class TestBuildDeepHedgingData:
    def test_returns_data_build_result(self, small_config):
        result = build_deep_hedging_data(small_config)
        assert isinstance(result, DataBuildResult)

    def test_has_all_datasets(self, small_config):
        result = build_deep_hedging_data(small_config)
        assert result.train_ds is not None
        assert result.val_ds is not None
        assert result.test_ds is not None

    def test_dataset_yields_dict_and_targets(self, small_config):
        result = build_deep_hedging_data(small_config)
        for batch_x, batch_y in result.train_ds.take(1):
            assert isinstance(batch_x, dict)
            assert "price_paths" in batch_x
            assert "payoffs" in batch_x

    def test_price_paths_shape(self, small_config):
        result = build_deep_hedging_data(small_config)
        for batch_x, _ in result.train_ds.take(1):
            paths = batch_x["price_paths"]
            assert len(paths.shape) == 3  # [batch, timesteps, features]
            assert paths.shape[1] == 11  # num_steps + 1
            assert paths.shape[2] == 5   # 5 features

    def test_payoffs_shape(self, small_config):
        result = build_deep_hedging_data(small_config)
        for batch_x, _ in result.train_ds.take(1):
            payoffs = batch_x["payoffs"]
            assert len(payoffs.shape) == 1

    def test_targets_are_zeros(self, small_config):
        result = build_deep_hedging_data(small_config)
        for _, targets in result.train_ds.take(1):
            np.testing.assert_allclose(targets.numpy(), 0.0)

    def test_metadata_keys(self, small_config):
        result = build_deep_hedging_data(small_config)
        m = result.metadata
        assert "n_train" in m
        assert "n_val" in m
        assert "n_test" in m
        assert "num_steps" in m
        assert "num_features" in m
        assert "spot_paths_test" in m
        assert "payoffs_test" in m
        assert "bs_deltas_test" in m

    def test_split_sizes(self, small_config):
        result = build_deep_hedging_data(small_config)
        total = result.metadata["n_train"] + result.metadata["n_val"] + result.metadata["n_test"]
        assert total == 200

    def test_default_config(self):
        """build_deep_hedging_data(None) should use defaults."""
        config = DeepHedgingDataConfig(
            simulation=SimulationConfig(num_paths=100, num_steps=5, seed=1),
            batch_size=16,
        )
        result = build_deep_hedging_data(config)
        assert isinstance(result, DataBuildResult)


class TestBuildDeepHedgingDataHeston:
    def test_heston_model(self):
        config = DeepHedgingDataConfig(
            market=MarketDynamicsConfig(model="heston"),
            option=OptionConfig(option_type="call", strike=100.0, maturity_years=0.25),
            simulation=SimulationConfig(num_paths=100, num_steps=10, seed=42),
            batch_size=16,
        )
        result = build_deep_hedging_data(config)
        assert isinstance(result, DataBuildResult)
        for batch_x, _ in result.train_ds.take(1):
            assert batch_x["price_paths"].shape[2] == 4  # no BS delta for Heston


class TestBuildDeepHedgingDataInvalidModel:
    def test_unknown_model_raises(self):
        config = DeepHedgingDataConfig(
            market=MarketDynamicsConfig(model="sabr"),
            simulation=SimulationConfig(num_paths=100, num_steps=5, seed=42),
        )
        with pytest.raises(ValueError, match="Unknown market model"):
            build_deep_hedging_data(config)


class TestBuildFeatureTensor:
    def test_shape(self):
        sim = GBMSimulator(spot_0=100.0, risk_free_rate=0.05, volatility=0.2)
        res = sim.simulate(maturity=0.25, num_steps=10, num_paths=50, seed=42, strike=100.0)
        config = DeepHedgingDataConfig(
            market=MarketDynamicsConfig(model="gbm", spot_0=100.0, volatility=0.2),
            option=OptionConfig(strike=100.0, maturity_years=0.25),
        )
        features = _build_feature_tensor(res, config)
        assert features.shape == (50, 11, 5)

    def test_features_finite(self):
        sim = GBMSimulator(spot_0=100.0, risk_free_rate=0.05, volatility=0.2)
        res = sim.simulate(maturity=0.25, num_steps=10, num_paths=50, seed=42, strike=100.0)
        config = DeepHedgingDataConfig(
            market=MarketDynamicsConfig(model="gbm", spot_0=100.0, volatility=0.2),
            option=OptionConfig(strike=100.0, maturity_years=0.25),
        )
        features = _build_feature_tensor(res, config)
        assert np.all(np.isfinite(features))

    def test_normalised_spot_at_t0(self):
        sim = GBMSimulator(spot_0=100.0, risk_free_rate=0.05, volatility=0.2)
        res = sim.simulate(maturity=0.25, num_steps=10, num_paths=50, seed=42, strike=100.0)
        config = DeepHedgingDataConfig(
            market=MarketDynamicsConfig(model="gbm", spot_0=100.0, volatility=0.2),
            option=OptionConfig(strike=100.0, maturity_years=0.25),
        )
        features = _build_feature_tensor(res, config)
        np.testing.assert_allclose(features[:, 0, 0], 1.0, rtol=1e-5)

    def test_time_to_expiry_range(self):
        sim = GBMSimulator(spot_0=100.0, risk_free_rate=0.05, volatility=0.2)
        res = sim.simulate(maturity=0.25, num_steps=10, num_paths=50, seed=42, strike=100.0)
        config = DeepHedgingDataConfig(
            market=MarketDynamicsConfig(model="gbm", spot_0=100.0, volatility=0.2),
            option=OptionConfig(strike=100.0, maturity_years=0.25),
        )
        features = _build_feature_tensor(res, config)
        tte = features[:, :, 2]
        assert np.all(tte >= -0.01)
        assert np.all(tte <= 1.01)
