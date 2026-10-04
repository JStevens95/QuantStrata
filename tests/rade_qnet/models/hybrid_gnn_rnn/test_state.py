"""
Tests for the flagship's fitted state.

Most of these defend against one failure mode: a state that loads
*partially* and produces a model that runs, trains, reports plausible
metrics, and is wrong. That was the original's behaviour with eleven loose
sidecar files, and it is the thing this module exists to make impossible.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from src.rade_qnet.core.runtime.errors import BundleError
from src.rade_qnet.models.hybrid_gnn_rnn.features.encoder import EntityEncoderState
from src.rade_qnet.models.hybrid_gnn_rnn.features.graph import build_graph
from src.rade_qnet.models.hybrid_gnn_rnn.spec import AttributeEncoderSpec, GraphSpec
from src.rade_qnet.models.hybrid_gnn_rnn.state import (
    HybridState,
    StandardScalerState,
    Universe,
)

#: Target P&L in currency, deliberately far from unit scale so that a
#: forgotten inverse transform cannot pass a tolerance by accident.
TARGET_SCALE = 50_000.0


@pytest.fixture
def attributes() -> Mapping[str, Sequence[Any]]:
    """Four instruments: two elementary, two target."""
    return {
        "trade_id": ["E1", "E2", "T1", "T2"],
        "moneyness": [0.95, 1.05, 0.98, 1.02],
        "yrs_to_maturity": [0.25, 1.00, 0.50, 1.50],
        "delta": [0.30, 0.55, 0.40, 0.60],
        "vega": [10.0, 15.0, 11.0, 18.0],
        "product_type": ["vanilla_option", "forward", "vanilla_option", "forward"],
        "product_subtype": ["european", "outright", "european", "outright"],
        "trade_type": ["elementary", "elementary", "target", "target"],
        "underlying_risk_factors": [["FX"], ["RATES"], ["FX", "RATES"], ["RATES"]],
    }


@pytest.fixture
def state(attributes: Mapping[str, Sequence[Any]]) -> HybridState:
    """Assemble a small but complete state."""
    generator = np.random.default_rng(0)
    encoder = EntityEncoderState.fit(attributes, spec=AttributeEncoderSpec())
    encoded = encoder.transform(attributes)
    return HybridState(
        feature_scaler=StandardScalerState.fit(generator.normal(size=(40, 2))),
        target_scaler=StandardScalerState.fit(generator.normal(scale=TARGET_SCALE, size=(40, 2))),
        selected_basis=("E2", "E1"),
        encoder=encoder,
        graph=build_graph(
            encoded,
            spec=GraphSpec(n_neighbours=2),
            is_target=np.array([False, False, True, True]),
        ),
        universe=Universe(elementary_ids=("E2", "E1"), target_ids=("T1", "T2")),
        elementary_indices=np.array([0, 1], dtype=np.int64),
        target_indices=np.array([2, 3], dtype=np.int64),
    )


class TestStandardScaler:
    """The hand-rolled replacement for a pickled scikit-learn scaler."""

    def test_the_statistics_are_held_in_float64(self) -> None:
        """
        A float32 mean over thousands of scenarios is measurably wrong.

        The network consumes float32, but the statistics it is standardised
        by should not be computed in it.
        """
        scaler = StandardScalerState.fit(np.ones((1000, 3), dtype=np.float32))
        assert scaler.centre.dtype == np.float64

    def test_the_transform_standardises(self) -> None:
        """Zero mean and unit scale, which is the stage's entire claim."""
        values = np.random.default_rng(0).normal(loc=7.0, scale=3.0, size=(200, 2))
        scaled = StandardScalerState.fit(values).transform(values)
        assert scaled.mean(axis=0) == pytest.approx(np.zeros(2), abs=1e-5)
        assert scaled.std(axis=0) == pytest.approx(np.ones(2), abs=1e-5)

    def test_the_inverse_round_trips(self) -> None:
        """
        Inverting a transform must return the input.

        The inverse is the only thing standing between a user and a metric
        in the wrong units, so it gets its own test rather than being
        covered incidentally.
        """
        values = np.random.default_rng(1).normal(loc=-4.0, scale=9.0, size=(50, 3))
        scaler = StandardScalerState.fit(values)
        assert scaler.inverse_transform(scaler.transform(values)) == pytest.approx(values, abs=1e-4)

    def test_a_constant_column_does_not_divide_by_zero(self) -> None:
        """
        A zero-variance column must not turn the feature matrix into NaN.

        It is ordinary: an instrument that did not trade, a flat risk
        factor.
        """
        values = np.column_stack([np.ones(20), np.arange(20.0)])
        scaled = StandardScalerState.fit(values).transform(values)
        assert np.isfinite(scaled).all()


class TestInverseTransformTargets:
    """The one method the base class refuses to give a default for."""

    def test_predictions_come_back_in_currency(self, state: HybridState) -> None:
        """
        The headline behaviour of the whole class.

        A prediction of zero in standardised space is the mean P&L, not
        zero P&L, and reporting it as the latter would misstate every error
        by the size of the book.
        """
        standardised = np.zeros((5, 2))
        recovered = state.inverse_transform_targets(standardised)
        assert np.abs(recovered).max() > 0.01 * TARGET_SCALE

    def test_it_round_trips_against_the_scaler(self, state: HybridState) -> None:
        """Inverting a transform must return the input."""
        original = np.random.default_rng(2).normal(scale=TARGET_SCALE, size=(10, 2))
        assert state.inverse_transform_targets(
            state.target_scaler.transform(original)
        ) == pytest.approx(original, rel=1e-5)

    def test_an_unscaled_target_passes_straight_through(self, state: HybridState) -> None:
        """
        An unscaled target must not be silently rescaled on the way out.

        Recorded on the state rather than inferred, so there is nothing to
        guess.
        """
        unscaled = dataclasses.replace(state, scale_targets=False)
        predictions = np.arange(10.0).reshape(5, 2)
        assert np.array_equal(unscaled.inverse_transform_targets(predictions), predictions)


class TestRoundTrip:
    """Saving and loading the whole state."""

    def test_everything_survives(self, state: HybridState, tmp_path) -> None:
        """Each component, checked individually so a failure names itself."""
        state.save(tmp_path)
        restored = HybridState.load(tmp_path)

        assert restored.selected_basis == state.selected_basis
        assert restored.universe == state.universe
        assert restored.scale_targets == state.scale_targets
        assert np.array_equal(restored.elementary_indices, state.elementary_indices)
        assert np.array_equal(restored.target_indices, state.target_indices)
        assert np.array_equal(restored.target_scaler.centre, state.target_scaler.centre)
        assert np.array_equal(restored.graph.indices, state.graph.indices)
        assert restored.encoder.numeric_names == state.encoder.numeric_names

    def test_the_basis_order_survives(self, state: HybridState, tmp_path) -> None:
        """
        ``selected_basis`` is a sequence that looks like a set.

        Its order is the column order of the feature matrix, so a load that
        sorted it -- or round-tripped it through a set -- would feed the
        saved weights their columns transposed. The fixture's basis is
        deliberately not in sorted order, so sorting would be visible here.
        """
        assert state.selected_basis != tuple(sorted(state.selected_basis))
        state.save(tmp_path)
        assert HybridState.load(tmp_path).selected_basis == state.selected_basis

    def test_a_reloaded_state_inverts_identically(self, state: HybridState, tmp_path) -> None:
        """The round trip is only worth anything if the predictions match."""
        state.save(tmp_path)
        predictions = np.random.default_rng(3).normal(size=(8, 2))
        assert np.array_equal(
            HybridState.load(tmp_path).inverse_transform_targets(predictions),
            state.inverse_transform_targets(predictions),
        )

    def test_nothing_is_pickled(self, state: HybridState, tmp_path) -> None:
        """
        No component may reintroduce a pickled third-party object.

        Six of the original's eleven artifacts were pickles, which pinned
        every saved model to the scikit-learn and SciPy versions that wrote
        it.
        """
        state.save(tmp_path)
        suffixes = {path.suffix for path in tmp_path.rglob("*") if path.is_file()}
        assert suffixes <= {".npy", ".npz", ".json"}

    @pytest.mark.parametrize("removed", ["scalers.npz", "universe.json", "encoder", "graph"])
    def test_a_partial_state_is_refused(self, state: HybridState, tmp_path, removed: str) -> None:
        """
        The whole point of the class, stated once per component.

        Loading ten of eleven files gave the original a model that ran and
        was wrong. Every component must therefore be load-bearing, and the
        parametrisation is what proves none of them is quietly optional.
        """
        state.save(tmp_path)
        target = tmp_path / removed
        if target.is_dir():
            for path in sorted(target.rglob("*"), reverse=True):
                path.unlink()
            target.rmdir()
        else:
            target.unlink()

        with pytest.raises(BundleError, match=removed.rstrip("/")):
            HybridState.load(tmp_path)


class TestDescribe:
    """The run-report summary."""

    def test_it_reports_what_a_reader_would_check(self, state: HybridState) -> None:
        """
        The summary carries what a reader would actually check.

        The basis-selection count is the headline: a run that kept every
        instrument has not reduced anything, and that is the first symptom
        of a misconfigured variance threshold.
        """
        summary = state.describe()
        assert summary["n_elementary_selected"] == 2
        assert summary["n_targets"] == 2
        assert summary["n_graph_edges"] > 0

    def test_it_is_json_encodable(self, state: HybridState) -> None:
        """
        The summary goes into the run manifest.

        A NumPy integer in there would fail the write at the very end of a
        long training run.
        """
        json.dumps(state.describe())
