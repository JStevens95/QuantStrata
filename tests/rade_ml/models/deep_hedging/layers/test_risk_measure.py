"""Unit tests for rade_ml.models.deep_hedging.layers.risk_measure -- CVaRLoss, EntropicRiskLoss."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.deep_hedging.layers.risk_measure import CVaRLoss, EntropicRiskLoss


class TestCVaRLoss:
    def test_output_is_scalar(self):
        loss_fn = CVaRLoss(alpha=0.95)
        pnl = tf.constant([1.0, -2.0, 0.5, -3.0, 0.0, 1.5, -1.0, 0.2])
        y_true = tf.zeros_like(pnl)
        val = loss_fn(y_true, pnl)
        assert val.shape == ()

    def test_higher_alpha_means_more_tail_focus(self):
        """CVaR at higher alpha should be >= CVaR at lower alpha for losses."""
        pnl = tf.constant(np.random.RandomState(42).randn(500).astype(np.float32) * 5)
        y_true = tf.zeros_like(pnl)
        cvar_90 = CVaRLoss(alpha=0.90)(y_true, pnl).numpy()
        cvar_99 = CVaRLoss(alpha=0.99)(y_true, pnl).numpy()
        assert cvar_99 >= cvar_90 - 0.1  # 99th percentile tail should be at least as bad

    def test_positive_for_bad_pnl(self):
        """When P&L is heavily negative the CVaR loss should be positive."""
        pnl = tf.constant([-10.0, -8.0, -12.0, -9.0, -11.0])
        y_true = tf.zeros_like(pnl)
        val = CVaRLoss(alpha=0.95)(y_true, pnl)
        assert val.numpy() > 0

    def test_gradient_flows(self):
        loss_fn = CVaRLoss(alpha=0.95)
        pnl = tf.Variable(np.random.randn(32).astype(np.float32))
        with tf.GradientTape() as tape:
            y_true = tf.zeros_like(pnl)
            loss = loss_fn(y_true, pnl)
        grad = tape.gradient(loss, pnl)
        assert grad is not None
        assert not tf.reduce_any(tf.math.is_nan(grad)).numpy()

    def test_get_config(self):
        loss_fn = CVaRLoss(alpha=0.99)
        cfg = loss_fn.get_config()
        assert cfg["alpha"] == 0.99

    def test_no_nan_output(self):
        loss_fn = CVaRLoss(alpha=0.95)
        pnl = tf.constant(np.random.RandomState(42).randn(100).astype(np.float32))
        val = loss_fn(tf.zeros_like(pnl), pnl)
        assert not tf.math.is_nan(val).numpy()


class TestEntropicRiskLoss:
    def test_output_is_scalar(self):
        loss_fn = EntropicRiskLoss(risk_aversion=1.0)
        pnl = tf.constant([1.0, -2.0, 0.5, -3.0, 0.0])
        y_true = tf.zeros_like(pnl)
        val = loss_fn(y_true, pnl)
        assert val.shape == ()

    def test_higher_risk_aversion_higher_loss(self):
        """Higher lambda penalises tail outcomes more."""
        pnl = tf.constant(np.random.RandomState(42).randn(500).astype(np.float32) * 5)
        y_true = tf.zeros_like(pnl)
        low = EntropicRiskLoss(risk_aversion=0.5)(y_true, pnl).numpy()
        high = EntropicRiskLoss(risk_aversion=5.0)(y_true, pnl).numpy()
        assert high >= low - 0.1

    def test_gradient_flows(self):
        loss_fn = EntropicRiskLoss(risk_aversion=1.0)
        pnl = tf.Variable(np.random.randn(32).astype(np.float32))
        with tf.GradientTape() as tape:
            loss = loss_fn(tf.zeros_like(pnl), pnl)
        grad = tape.gradient(loss, pnl)
        assert grad is not None
        assert not tf.reduce_any(tf.math.is_nan(grad)).numpy()

    def test_get_config(self):
        loss_fn = EntropicRiskLoss(risk_aversion=2.5)
        cfg = loss_fn.get_config()
        assert cfg["risk_aversion"] == 2.5

    def test_no_nan_output(self):
        loss_fn = EntropicRiskLoss(risk_aversion=1.0)
        pnl = tf.constant(np.random.RandomState(42).randn(100).astype(np.float32))
        val = loss_fn(tf.zeros_like(pnl), pnl)
        assert not tf.math.is_nan(val).numpy()

    def test_numerically_stable_large_values(self):
        """Should not overflow even with large P&L values."""
        loss_fn = EntropicRiskLoss(risk_aversion=0.1)
        pnl = tf.constant(np.random.RandomState(42).randn(100).astype(np.float32) * 100)
        val = loss_fn(tf.zeros_like(pnl), pnl)
        assert not tf.math.is_nan(val).numpy()
        assert not tf.math.is_inf(val).numpy()
