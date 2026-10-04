"""
Guards on the committed golden fixture itself.

A fixture is data, not code, so its *values* are not asserted here -- their
correctness comes from the capture script having run against unmodified
``rade_ml_pt``, recorded in ``manifest.json`` with the source commit. What is
asserted is that the fixture is still capable of catching the things it was
built to catch.

That distinction matters because a fixture degrades silently. If someone
re-captures with a configuration where basis selection happens to keep every
instrument, every parity test still passes and the suite still looks green --
but two of its sharpest checks have quietly stopped testing anything. The
post-reduction index arrays would equal the pre-reduction ones, so a refactor
carrying the wrong indices forward would pass; and the selected basis would
be in input order, so a reordering would pass too.

These tests are the tripwire for that. They assert the fixture's *shape*: it
reduces, the surviving order is not the input order, and the artifacts the
five levels need are all present.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.rade_qnet.testkit.parity import load_golden

#: Size ceiling, as a definition-of-done item. A fixture that grows past this
#: stops being committed, and a parity suite nobody can run locally is a
#: parity suite nobody runs.
MAX_FIXTURE_BYTES = 10 * 1024 * 1024


@pytest.fixture(scope="module")
def golden():
    """
    Load the committed fixture, skipping the module if it is absent.

    Skipped rather than failed so a fresh clone that has not run the capture
    is not reported as broken code. The capture is a one-off step, and its
    absence is a setup state rather than a defect.

    Returns
    -------
    GoldenFixture
        The loaded fixture.
    """
    try:
        return load_golden("hybrid_gnn_rnn")
    except Exception as exc:
        pytest.skip(f"golden fixture not captured: {exc}")


class TestCompleteness:
    """Every level must have the artifacts it needs."""

    @pytest.mark.parametrize(
        "relative_path",
        [
            "level1_state/scaler_mean.npy",
            "level1_state/scaler_scale.npy",
            "level1_state/selected_basis.json",
            "level1_state/combined_features.npy",
            "level1_state/adjacency_indices.npy",
            "level1_state/adjacency_values.npy",
            "level1_state/adjacency_shape.npy",
            "level1_state/elementary_idx.npy",
            "level1_state/target_idx.npy",
            "level1_state/universe.json",
            "level2_tensors/split_indices.json",
            "level2_tensors/train_batch_000.npz",
            "level3_forward/state_dict.pt",
            "level3_forward/outputs.npy",
            "level4_training/curve.json",
        ],
    )
    def test_the_artifact_is_present(self, golden, relative_path):
        """
        Because a missing artifact makes its comparison a silent skip.

        The harness raises on a missing file rather than skipping, but only
        once something asks for it. This checks the whole set up front.
        """
        assert golden.has(relative_path), relative_path

    def test_the_input_is_stored_without_a_pickle(self, golden):
        """
        A checked-in pickle is code that executes on load.

        The capture materialises the pickles ``rade_ml_pt`` reads into a
        temporary directory instead, so the committed fixture stays
        reviewable in a diff and cannot carry an executable payload.
        """
        pickles = list(golden.directory.rglob("*.pkl"))
        assert not pickles, f"the fixture must not contain pickles: {pickles}"

    def test_the_fixture_is_small_enough_to_commit(self, golden):
        """
        Because a parity suite nobody can run locally is not a gate.

        The limit is generous; the real constraint is that the capture stays
        tiny enough to re-run in seconds.
        """
        total = sum(path.stat().st_size for path in golden.directory.rglob("*") if path.is_file())
        assert total < MAX_FIXTURE_BYTES


class TestProvenance:
    """A baseline without provenance cannot be a baseline."""

    @pytest.mark.parametrize(
        "key", ["captured_at", "source_commit", "seed", "versions", "data_config"]
    )
    def test_the_manifest_records_it(self, golden, key):
        """
        So a parity failure can be attributed.

        When a long-passing test starts failing, the first question is
        whether the baseline moved rather than the code, and only the
        manifest answers it.
        """
        assert key in golden.manifest

    def test_the_preserved_defects_are_named(self, golden):
        """
        Because reproducing a bug on purpose must be written down.

        Two defects change numerical output and are deliberately reproduced.
        Without the record, a later reader finds compatibility flags with no
        explanation and switches them off.
        """
        preserved = golden.manifest["preserved_defects"]
        assert "defect_9_basis_selection_leakage" in preserved
        assert "defect_3_shared_shuffle_flag" in preserved


class TestTheFixtureCanStillFail:
    """
    The tripwire against silent degradation.

    Each test here asserts that the fixture still exercises a trap. If one
    starts failing after a re-capture, the fixture has become weaker and the
    parity suite is passing for less reason than it appears to.
    """

    def test_basis_selection_actually_reduced(self, golden):
        """
        Otherwise the post-reduction index trap is not captured at all.

        With every instrument surviving, ``elementary_idx`` equals the
        pre-reduction indices, so a refactor carrying the wrong ones forward
        passes level 1. This is the specific way the earlier, two-per-group
        input was too weak: selection kept all sixteen.
        """
        basis = golden.json("level1_state/selected_basis.json")
        universe = json.loads(
            (golden.directory / "input" / "universe.json").read_text(encoding="utf-8")
        )
        assert len(basis) < len(universe["elementary_ids"])

    def test_the_index_arrays_are_post_reduction(self, golden):
        """
        They count the *survivors*, not the original columns.

        The original recomputes them as ``0..n_e`` and ``n_e..n_e+n_t``
        after selection. Asserted against the basis length rather than a
        literal, so the test survives a re-capture at a different size.
        """
        basis = golden.json("level1_state/selected_basis.json")
        elementary_idx = golden.array("level1_state/elementary_idx.npy")
        target_idx = golden.array("level1_state/target_idx.npy")

        assert elementary_idx.tolist() == list(range(len(basis)))
        assert target_idx[0] == len(basis)

    def test_the_selected_basis_is_not_in_input_order(self, golden):
        """
        Otherwise a reordering bug would pass the ordered comparison.

        Order fixes column positions in every array downstream. If the
        captured order happened to match the input order, comparing as a
        sequence and comparing as a set would be indistinguishable, and the
        trap level 1 exists for would be untested against real data.
        """
        basis = golden.json("level1_state/selected_basis.json")
        universe = json.loads(
            (golden.directory / "input" / "universe.json").read_text(encoding="utf-8")
        )
        input_order = [tid for tid in universe["elementary_ids"] if tid in set(basis)]
        assert basis != input_order

    def test_the_window_length_is_greater_than_one(self, golden):
        """
        Because at a length of one every boundary question disappears.

        Sequence-aware splitting, the boundary gap and the window-straddling
        half of defect 3 are only exercised when windows actually span
        several scenarios.
        """
        assert golden.manifest["shape"]["seq_length"] > 1

    def test_the_scaler_was_fitted_before_reduction(self, golden):
        """
        Pinning the stage order the refactor has to reproduce.

        The scaler is fitted over the *full* elementary book and the basis is
        selected afterwards, so the captured statistics have one entry per
        original instrument rather than per survivor. A refactor that
        reduces first would produce a shorter array -- caught here as a
        shape difference rather than as a mysterious value difference later.
        """
        basis = golden.json("level1_state/selected_basis.json")
        universe = json.loads(
            (golden.directory / "input" / "universe.json").read_text(encoding="utf-8")
        )
        scaler_mean = golden.array("level1_state/scaler_mean.npy")

        assert scaler_mean.shape == (len(universe["elementary_ids"]),)
        assert scaler_mean.shape[0] > len(basis)


class TestTheCapturedSplit:
    """Defect 3's compatibility mechanism: the exact indices, replayed."""

    def test_the_three_splits_are_recorded(self, golden):
        """So the refactor can replay them rather than reproduce a flag."""
        splits = golden.json("level2_tensors/split_indices.json")
        assert set(splits) == {"train", "validation", "test"}

    def test_the_splits_do_not_overlap(self, golden):
        """
        A captured split that overlapped would bake a leak into the gate.

        Parity would then require the refactor to reproduce the leak, which
        is the one thing the compatibility flags are designed to avoid.
        """
        splits = golden.json("level2_tensors/split_indices.json")
        train, validation, test = (set(splits[name]) for name in ("train", "validation", "test"))
        assert not train & validation
        assert not train & test
        assert not validation & test

    def test_the_split_is_chronological(self, golden):
        """
        Captured with ``shuffle=False``, which is what makes it replayable.

        The original couples the split to the batch order through one flag,
        so capturing with shuffling on would have produced a random split
        that no explicit strategy could reproduce.
        """
        splits = golden.json("level2_tensors/split_indices.json")
        assert max(splits["train"]) < min(splits["validation"])
        assert max(splits["validation"]) < min(splits["test"])


class TestTheCapturedTensors:
    """What the network actually received, which level 2 reproduces."""

    def test_static_inputs_arrive_unbatched(self, golden):
        """
        The claim that moving static inputs out of collation changed nothing.

        The original merged every static tensor into every sample and then
        compared across the batch to recover one copy. The captured
        adjacency has no batch dimension, which is what says the network
        already received exactly one copy -- so delivering it directly is a
        deletion of work, not a change of behaviour.
        """
        with np.load(golden.directory / "level2_tensors/train_batch_000.npz") as stored:
            batch_size = stored["pnl_history"].shape[0]
            assert stored["adjacency_indices"].shape[1] == 2
            assert stored["adjacency_indices"].shape[0] != batch_size
            assert stored["adjacency_dense_shape"].shape == (2,)

    def test_the_dynamic_input_is_windowed(self, golden):
        """
        Shaped (batch, window, instrument), with the window from the manifest.

        A refactor that produced (batch, instrument, window) would be a
        transpose that trains without error and learns nothing useful.
        """
        with np.load(golden.directory / "level2_tensors/train_batch_000.npz") as stored:
            basis = golden.json("level1_state/selected_basis.json")
            assert stored["pnl_history"].shape[1] == golden.manifest["shape"]["seq_length"]
            assert stored["pnl_history"].shape[2] == len(basis)

    def test_the_unused_key_is_present_in_the_baseline(self, golden):
        """
        Recording that the original *declares* ``elementary_indices``.

        Its forward pass never reads it. The refactored signature omits it,
        and this test documents what is being dropped -- so if parity later
        shifts, the first question is whether the key was load-bearing after
        all.
        """
        with np.load(golden.directory / "level2_tensors/train_batch_000.npz") as stored:
            assert "elementary_indices" in stored.files
