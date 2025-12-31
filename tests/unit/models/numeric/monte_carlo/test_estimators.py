from __future__ import annotations

import math
import pytest
import numpy as np

from src.models.numeric.monte_carlo.estimators import (
    OnlineStats, mean_confidence_interval, mean_stderr,
)


def test_mean_stderr_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1D"):
        mean_stderr(np.zeros((2, 2), dtype=np.float64))


def test_mean_stderr_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mean_stderr(np.array([], dtype=np.float64))


def test_mean_stderr_n1_returns_zero_stderr() -> None:
    m, se, n = mean_stderr(np.array([1.234], dtype=np.float64))
    assert n == 1
    assert m == pytest.approx(1.234, abs=0.0)
    assert se == pytest.approx(0.0, abs=0.0)


def test_mean_stderr_matches_numpy_ddof1() -> None:
    x = np.array([0.0, 2.0], dtype=np.float64)  # mean=1, var(ddof=1)=2, stderr=sqrt((2/2))=1
    m, se, n = mean_stderr(x)

    expected_m = float(np.mean(x))
    expected_var = float(np.var(x, ddof=1))
    expected_se = math.sqrt(expected_var / x.size)

    assert n == 2
    assert m == pytest.approx(expected_m, abs=1e-12)
    assert se == pytest.approx(expected_se, abs=1e-12)


def test_mean_confidence_interval_is_centered() -> None:
    mean = 10.0
    stderr = 2.0
    lo, hi = mean_confidence_interval(mean, stderr, z=2.0)

    assert (lo + hi) / 2.0 == pytest.approx(mean, abs=1e-12)
    assert (hi - lo) == pytest.approx(2.0 * 2.0 * stderr, abs=1e-12)


def test_online_stats_update_matches_numpy() -> None:
    rng = np.random.default_rng(123)
    x = rng.standard_normal(50_000).astype(np.float64)

    s = OnlineStats()
    s.update_many(x)

    # mean
    assert s.n == x.size
    assert s.mean == pytest.approx(float(np.mean(x)), rel=0.0, abs=5e-13)

    # unbiased variance (ddof=1)
    assert s.variance == pytest.approx(float(np.var(x, ddof=1)), rel=0.0, abs=5e-11)

    # stderr = sqrt(var/n)
    expected_stderr = math.sqrt(float(np.var(x, ddof=1)) / x.size)
    assert s.stderr == pytest.approx(expected_stderr, rel=0.0, abs=5e-13)


def test_online_stats_empty_has_zero_variance_and_stderr() -> None:
    s = OnlineStats()
    assert s.n == 0
    assert s.variance == 0.0
    assert s.stderr == 0.0


def test_online_stats_singleton_has_zero_variance_and_stderr() -> None:
    s = OnlineStats()
    s.update(3.0)
    assert s.n == 1
    assert s.mean == 3.0
    assert s.variance == 0.0
    assert s.stderr == 0.0


def test_online_stats_conf_int_95_matches_helper() -> None:
    s = OnlineStats()
    s.update_many([1.0, 2.0, 3.0, 4.0])

    lo1, hi1 = s.conf_int_95()
    lo2, hi2 = mean_confidence_interval(s.mean, s.stderr)

    assert lo1 == pytest.approx(lo2, abs=0.0)
    assert hi1 == pytest.approx(hi2, abs=0.0)