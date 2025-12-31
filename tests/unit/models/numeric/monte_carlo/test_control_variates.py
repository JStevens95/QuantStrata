from __future__ import annotations

import pytest
import numpy as np

from src.models.numeric.monte_carlo.control_variates import apply_control_variate


def test_apply_control_variate_rejects_non_1d() -> None:
    y = np.zeros((2, 2), dtype=np.float64)
    c = np.zeros((4,), dtype=np.float64)
    with pytest.raises(ValueError, match="1D"):
        apply_control_variate(y=y, c=c, c_expectation=0.0)


def test_apply_control_variate_rejects_length_mismatch() -> None:
    y = np.zeros((3,), dtype=np.float64)
    c = np.zeros((4,), dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        apply_control_variate(y=y, c=c, c_expectation=0.0)


def test_apply_control_variate_rejects_empty() -> None:
    y = np.array([], dtype=np.float64)
    c = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="non-empty"):
        apply_control_variate(y=y, c=c, c_expectation=0.0)


def test_apply_control_variate_degenerate_control_returns_copy_beta_zero() -> None:
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    c = np.array([5.0, 5.0, 5.0], dtype=np.float64)  # var(c)=0

    res = apply_control_variate(y=y, c=c, c_expectation=5.0)

    assert res.beta == pytest.approx(0.0, abs=0.0)
    assert np.allclose(res.adjusted, y)
    # must be a copy (so caller can mutate without touching original)
    assert res.adjusted is not y


def test_apply_control_variate_beta_matches_cov_over_var_ddof1() -> None:
    # small deterministic data where we can compute beta the same way as the function
    y = np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float64)
    c = np.array([10.0, 11.0, 9.0, 12.0], dtype=np.float64)

    res = apply_control_variate(y=y, c=c, c_expectation=float(np.mean(c)))

    # match implementation: centered dot products / (n-1)
    yc = y - float(np.mean(y))
    cc = c - float(np.mean(c))
    cov = float(np.dot(yc, cc)) / (len(c) - 1)
    var = float(np.dot(cc, cc)) / (len(c) - 1)
    expected_beta = cov / var

    assert res.beta == pytest.approx(expected_beta, rel=0.0, abs=1e-12)


def test_apply_control_variate_preserves_mean_when_expectation_is_correct() -> None:
    rng = np.random.default_rng(123)

    n = 200_000
    c = rng.standard_normal(n).astype(np.float64)
    eps = 0.2 * rng.standard_normal(n).astype(np.float64)

    # y is strongly correlated with c
    y = 2.5 * c + eps

    res = apply_control_variate(y=y, c=c, c_expectation=0.0)  # E[c]=0 for standard normal

    # Adjustment should be unbiased in expectation, so sample mean should be very close.
    # Allow a small tolerance because we're comparing Monte Carlo estimates.
    assert float(np.mean(res.adjusted)) == pytest.approx(float(np.mean(y)), abs=5e-3)


def test_apply_control_variate_reduces_variance_when_control_is_informative() -> None:
    rng = np.random.default_rng(7)

    n = 200_000
    c = rng.standard_normal(n).astype(np.float64)
    eps = rng.standard_normal(n).astype(np.float64)

    # make y correlated with c
    y = 4.0 * c + 0.5 * eps

    res = apply_control_variate(y=y, c=c, c_expectation=0.0)

    var_y = float(np.var(y, ddof=1))
    var_adj = float(np.var(res.adjusted, ddof=1))

    assert var_adj < var_y
    assert res.beta > 0.0