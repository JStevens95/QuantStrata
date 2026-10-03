"""Unit tests for ``rade_ml_pt.monitoring.drift``.

Three rings of coverage:
* Single-feature primitives (PSI, JSD, severity classifier).
* Per-cluster ``build_drift_table`` shape + edge cases.
* Portfolio summary aggregator.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from src.rade_ml_pt.monitoring.drift import (
    PSI_CRITICAL_THRESHOLD,
    PSI_WARN_THRESHOLD,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_NO_DATA,
    SEVERITY_WARN,
    build_drift_table,
    build_portfolio_drift_summary,
    classify_severity,
    js_divergence,
    population_stability_index,
)


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts


def _baseline_row(name: str, values: np.ndarray, n_bins: int = 50) -> dict:
    """Build a single baseline-DataFrame row in the same shape ``load_baseline`` produces."""
    edges  = np.linspace(values.min(), values.max(), n_bins + 1)
    counts = _hist(values, edges)
    return {
        "feature_name": name,
        "mean":         float(values.mean()),
        "std":          float(values.std(ddof=0)),
        "hist_edges":   edges,
        "hist_counts":  counts,
    }


# ═════════════════════════════════════════════════════════════════════
# population_stability_index
# ═════════════════════════════════════════════════════════════════════

class TestPSI:
    def test_identical_distributions_returns_near_zero(self):
        rng    = np.random.default_rng(0)
        x      = rng.normal(0, 1, 5000)
        edges  = np.linspace(-4, 4, 51)
        counts = _hist(x, edges)
        psi    = population_stability_index(counts, edges, x)
        assert psi < 0.01

    def test_disjoint_distributions_returns_large(self):
        rng         = np.random.default_rng(1)
        base        = rng.normal(-3, 0.5, 5000)
        cur         = rng.normal(+3, 0.5, 5000)
        edges       = np.linspace(-5, 5, 51)
        base_counts = _hist(base, edges)
        psi         = population_stability_index(base_counts, edges, cur)
        assert psi > 1.0

    def test_one_sigma_shift_lands_in_warn_or_critical(self):
        rng         = np.random.default_rng(2)
        base        = rng.normal(0, 1, 5000)
        cur         = rng.normal(1, 1, 5000)
        edges       = np.linspace(-4, 5, 51)
        base_counts = _hist(base, edges)
        psi         = population_stability_index(base_counts, edges, cur)
        assert psi > PSI_WARN_THRESHOLD
        assert psi < 2.0  # sanity ceiling for 1-sigma shift

    def test_empty_current_returns_nan(self):
        edges  = np.linspace(-1, 1, 51)
        counts = np.ones(50, dtype=np.int64)
        assert np.isnan(population_stability_index(counts, edges, np.array([])))

    def test_zero_baseline_counts_returns_nan(self):
        edges  = np.linspace(-1, 1, 51)
        counts = np.zeros(50, dtype=np.int64)
        assert np.isnan(population_stability_index(counts, edges, np.array([0.0, 0.1])))

    def test_mismatched_shapes_returns_nan(self):
        # 50-element counts paired with 50-element edges (should be 51)
        edges  = np.linspace(-1, 1, 50)
        counts = np.ones(50, dtype=np.int64)
        assert np.isnan(population_stability_index(counts, edges, np.array([0.0])))

    def test_outlier_current_values_clipped_not_dropped(self):
        rng         = np.random.default_rng(3)
        base        = rng.normal(0, 1, 5000)
        edges       = np.linspace(-3, 3, 51)
        base_counts = _hist(base, edges)
        # Current values entirely outside the baseline range — naive
        # np.histogram would silently drop them.  The clip step lumps
        # them into the outermost bin, so PSI still reflects "today
        # looks nothing like training".
        cur = rng.normal(10, 1, 5000)
        psi = population_stability_index(base_counts, edges, cur)
        assert np.isfinite(psi)
        assert psi > 1.0

    def test_non_finite_current_values_dropped(self):
        rng         = np.random.default_rng(4)
        base        = rng.normal(0, 1, 5000)
        edges       = np.linspace(-4, 4, 51)
        base_counts = _hist(base, edges)
        # Half NaN, half OK; should still return a finite (small) PSI
        cur = np.concatenate([rng.normal(0, 1, 2500), np.full(2500, np.nan)])
        psi = population_stability_index(base_counts, edges, cur)
        assert np.isfinite(psi)
        assert psi < 0.05


# ═════════════════════════════════════════════════════════════════════
# js_divergence
# ═════════════════════════════════════════════════════════════════════

class TestJSD:
    def test_identical_returns_near_zero(self):
        rng    = np.random.default_rng(5)
        edges  = np.linspace(-3, 3, 51)
        counts = _hist(rng.normal(0, 1, 5000), edges)
        assert js_divergence(counts, counts) < 1e-3

    def test_symmetric(self):
        rng   = np.random.default_rng(6)
        edges = np.linspace(-3, 3, 51)
        a     = _hist(rng.normal(0, 1, 5000), edges)
        b     = _hist(rng.normal(1, 1, 5000), edges)
        assert js_divergence(a, b) == pytest.approx(js_divergence(b, a), abs=1e-9)

    def test_bounded_in_unit_interval(self):
        rng   = np.random.default_rng(7)
        edges = np.linspace(-3, 3, 51)
        a     = _hist(rng.normal(-2, 0.5, 5000), edges)
        b     = _hist(rng.normal(+2, 0.5, 5000), edges)
        jsd   = js_divergence(a, b)
        assert 0.0 <= jsd <= 1.0

    def test_shape_mismatch_returns_nan(self):
        assert np.isnan(js_divergence(np.ones(10), np.ones(20)))

    def test_zero_inputs_return_nan(self):
        assert np.isnan(js_divergence(np.zeros(10), np.ones(10)))
        assert np.isnan(js_divergence(np.ones(10), np.zeros(10)))


# ═════════════════════════════════════════════════════════════════════
# classify_severity
# ═════════════════════════════════════════════════════════════════════

class TestSeverity:
    @pytest.mark.parametrize("psi,expected", [
        (0.0,                                 SEVERITY_INFO),
        (0.09,                                SEVERITY_INFO),
        (PSI_WARN_THRESHOLD - 1e-9,           SEVERITY_INFO),
        (PSI_WARN_THRESHOLD,                  SEVERITY_WARN),
        (0.20,                                SEVERITY_WARN),
        (PSI_CRITICAL_THRESHOLD - 1e-9,       SEVERITY_WARN),
        (PSI_CRITICAL_THRESHOLD,              SEVERITY_CRITICAL),
        (0.50,                                SEVERITY_CRITICAL),
        (10.0,                                SEVERITY_CRITICAL),
    ])
    def test_thresholds(self, psi, expected):
        assert classify_severity(psi) == expected

    @pytest.mark.parametrize("v", [
        None,
        float("nan"),
        -0.1,
        float("inf"),
        -float("inf"),
    ])
    def test_no_data_cases(self, v):
        assert classify_severity(v) == SEVERITY_NO_DATA

    def test_non_numeric_returns_no_data(self):
        assert classify_severity("not a number") == SEVERITY_NO_DATA  # type: ignore[arg-type]


# ═════════════════════════════════════════════════════════════════════
# build_drift_table
# ═════════════════════════════════════════════════════════════════════

class TestBuildDriftTable:
    @staticmethod
    def _baseline_df(rng: np.random.Generator) -> pd.DataFrame:
        rows = [
            _baseline_row("rf_a", rng.normal(0, 1, 5000)),
            _baseline_row("rf_b", rng.normal(0, 1, 5000)),
        ]
        return pd.DataFrame(rows)

    def test_shape_and_columns(self):
        rng         = np.random.default_rng(8)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(0, 1, 1000),
        })
        out = build_drift_table(baseline_df, cur, cluster_id="c0")
        assert list(out.columns) == [
            "cluster_id", "feature_name", "psi", "js_divergence",
            "mean_shift", "std_ratio", "severity",
        ]
        assert len(out) == 2
        assert set(out["feature_name"]) == {"rf_a", "rf_b"}
        assert (out["cluster_id"] == "c0").all()

    def test_stable_features_classified_info(self):
        rng         = np.random.default_rng(9)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 5000),
            "rf_b": rng.normal(0, 1, 5000),
        })
        out = build_drift_table(baseline_df, cur, cluster_id="c0")
        assert (out["severity"] == SEVERITY_INFO).all()
        assert (out["psi"] < PSI_WARN_THRESHOLD).all()

    def test_shifted_features_classified_critical(self):
        rng         = np.random.default_rng(10)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(3, 1, 5000),
            "rf_b": rng.normal(3, 1, 5000),
        })
        out = build_drift_table(baseline_df, cur, cluster_id="c0")
        assert (out["severity"] == SEVERITY_CRITICAL).all()
        assert (out["mean_shift"] > 1.0).all()

    def test_missing_feature_in_current_emits_no_data_row(self):
        rng         = np.random.default_rng(11)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({"rf_a": rng.normal(0, 1, 1000)})  # rf_b missing
        out         = build_drift_table(baseline_df, cur, cluster_id="c0")
        b_row       = out.set_index("feature_name").loc["rf_b"]
        assert b_row["severity"] == SEVERITY_NO_DATA
        assert pd.isna(b_row["psi"])

    def test_all_nan_feature_in_current_emits_no_data_row(self):
        rng         = np.random.default_rng(12)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": np.full(1000, np.nan),
        })
        out   = build_drift_table(baseline_df, cur, cluster_id="c0")
        b_row = out.set_index("feature_name").loc["rf_b"]
        assert b_row["severity"] == SEVERITY_NO_DATA

    def test_extra_feature_in_current_ignored(self):
        rng         = np.random.default_rng(13)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(0, 1, 1000),
            "rf_z": rng.normal(0, 1, 1000),   # not in baseline
        })
        out = build_drift_table(baseline_df, cur, cluster_id="c0")
        assert "rf_z" not in out["feature_name"].values

    def test_empty_baseline_returns_empty_table_with_dtype_schema(self):
        out = build_drift_table(pd.DataFrame(), pd.DataFrame(), cluster_id="c0")
        assert list(out.columns) == [
            "cluster_id", "feature_name", "psi", "js_divergence",
            "mean_shift", "std_ratio", "severity",
        ]
        assert len(out) == 0

    def test_baseline_missing_required_columns_raises(self):
        bad = pd.DataFrame([{"feature_name": "x"}])
        with pytest.raises(ValueError, match="missing required columns"):
            build_drift_table(bad, pd.DataFrame({"x": [1, 2, 3]}), cluster_id="c0")

    def test_dtypes_canonical_for_concat(self):
        """Empty + populated tables share a dtype fingerprint."""
        empty = build_drift_table(pd.DataFrame(), pd.DataFrame(), cluster_id="c0")

        rng         = np.random.default_rng(14)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(0, 1, 1000),
        })
        populated = build_drift_table(baseline_df, cur, cluster_id="c1")
        assert empty.dtypes.to_dict() == populated.dtypes.to_dict()

        # pd.concat must work without dtype-promotion warnings
        combined = pd.concat([empty, populated], ignore_index=True)
        assert len(combined) == 2


# ═════════════════════════════════════════════════════════════════════
# build_portfolio_drift_summary
# ═════════════════════════════════════════════════════════════════════

class TestPortfolioSummary:
    @staticmethod
    def _drift(cid: str, severities: List[str], psis: List[float]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "cluster_id":    cid,
                "feature_name":  f"rf_{i}",
                "psi":           p,
                "js_divergence": 0.0,
                "mean_shift":    0.0,
                "std_ratio":     1.0,
                "severity":      s,
            }
            for i, (s, p) in enumerate(zip(severities, psis))
        ])

    def test_aggregates_across_clusters(self):
        t1  = self._drift("c0", [SEVERITY_INFO, SEVERITY_WARN], [0.02, 0.18])
        t2  = self._drift("c1", [SEVERITY_CRITICAL], [0.40])
        out = build_portfolio_drift_summary([t1, t2])
        assert out["n_clusters"]         == 2
        assert out["n_features_info"]    == 1
        assert out["n_features_warn"]    == 1
        assert out["n_features_crit"]    == 1
        assert out["max_psi"]            == pytest.approx(0.40, abs=1e-6)
        assert out["worst_cluster"]      == "c1"
        assert out["worst_feature"]      == "rf_0"

    def test_empty_input_returns_no_data(self):
        out = build_portfolio_drift_summary([])
        assert out["n_clusters"]   == 0
        assert out["severity"]     == SEVERITY_NO_DATA
        assert np.isnan(out["mean_psi"])

    def test_none_input_returns_no_data(self):
        out = build_portfolio_drift_summary(None)  # type: ignore[arg-type]
        assert out["severity"] == SEVERITY_NO_DATA

    def test_only_no_data_rows_returns_no_data(self):
        t   = self._drift("c0", [SEVERITY_NO_DATA, SEVERITY_NO_DATA], [np.nan, np.nan])
        out = build_portfolio_drift_summary([t])
        assert out["severity"]          == SEVERITY_NO_DATA
        assert out["n_features_nodata"] == 2
        assert out["worst_cluster"]     is None

    def test_severity_classification_from_mean_psi(self):
        # mean_psi == 0.15 → portfolio severity "warn"
        t   = self._drift("c0", [SEVERITY_WARN], [0.15])
        out = build_portfolio_drift_summary([t])
        assert out["severity"] == SEVERITY_WARN

    def test_skips_empty_drift_tables_silently(self):
        t1  = self._drift("c0", [SEVERITY_INFO], [0.05])
        t2  = pd.DataFrame()  # empty
        out = build_portfolio_drift_summary([t1, t2])
        assert out["n_clusters"] == 1


# ═════════════════════════════════════════════════════════════════════
# End-to-end smoke — writer (baselines.py) → loader → drift table
# ═════════════════════════════════════════════════════════════════════
# Lives here (rather than test_loaders.py) because the assertions are
# about drift outputs, not loader I/O.  Confirms the writer/reader/
# primitive trio compose correctly with zero adapter code in between.

class TestEndToEndSmoke:
    """Full chain: write a real baseline parquet, load it back, score drift."""

    def _write_and_load_baseline(self, tmp_path, features: pd.DataFrame) -> pd.DataFrame:
        # Local import so this module imports fine even when the
        # baselines writer is unavailable (e.g. running drift tests in
        # isolation in CI).
        from src.rade_ml_pt.monitoring.baselines import save_feature_baseline
        from src.rade_ml_pt.monitoring.loaders   import load_baseline

        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")
        return load_baseline(path)

    def test_stable_features_score_info(self, tmp_path):
        rng           = np.random.default_rng(20)
        training      = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 5000),
            "rf_b": rng.normal(5, 2, 5000),
        })
        baseline_df = self._write_and_load_baseline(tmp_path, training)
        current     = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(5, 2, 1000),
        })
        out = build_drift_table(baseline_df, current, cluster_id="c0")
        assert (out["severity"] == SEVERITY_INFO).all()

    def test_drift_pipeline_detects_real_shift(self, tmp_path):
        rng         = np.random.default_rng(21)
        training    = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 5000),
            "rf_b": rng.normal(0, 1, 5000),
        })
        baseline_df = self._write_and_load_baseline(tmp_path, training)
        current     = pd.DataFrame({
            "rf_a": rng.normal(2, 1, 1000),   # +2σ shift
            "rf_b": rng.normal(0, 1, 1000),
        })
        out         = build_drift_table(baseline_df, current, cluster_id="c0")
        by_feature  = out.set_index("feature_name")
        assert by_feature.loc["rf_a", "severity"] in (SEVERITY_WARN, SEVERITY_CRITICAL)
        assert by_feature.loc["rf_b", "severity"] == SEVERITY_INFO

    def test_portfolio_summary_compiles_from_real_baseline(self, tmp_path):
        rng       = np.random.default_rng(22)
        training  = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 5000),
            "rf_b": rng.normal(0, 1, 5000),
        })
        baseline_df = self._write_and_load_baseline(tmp_path, training)
        cur_stable  = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(0, 1, 1000),
        })
        cur_shifted = pd.DataFrame({
            "rf_a": rng.normal(3, 1, 1000),
            "rf_b": rng.normal(3, 1, 1000),
        })
        # Pretend we have two clusters — same baseline shape, different
        # current data, so the portfolio mixes a clean cluster with a
        # drifted one.
        t_stable  = build_drift_table(baseline_df, cur_stable,  cluster_id="c_stable")
        t_drifted = build_drift_table(baseline_df, cur_shifted, cluster_id="c_drifted")

        summary = build_portfolio_drift_summary([t_stable, t_drifted])
        assert summary["n_clusters"]      == 2
        assert summary["worst_cluster"]   == "c_drifted"
        assert summary["severity"]        in (SEVERITY_WARN, SEVERITY_CRITICAL)
        assert summary["n_features_crit"] >= 1
