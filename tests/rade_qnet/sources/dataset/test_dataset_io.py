"""
Tests for reading tables and fingerprinting their contents.

Two decisions here have consequences well beyond this module.

**Columns are sorted, not taken in file order.** Two exports of the same data
with the columns in a different order would otherwise produce different
feature matrices, so column three means a different thing between two runs
over identical data -- and the cache key would differ too, so neither run
would reuse the other's work. Sorting makes the matrix a function of the
column *set*.

**The fingerprint hashes contents, not a path and a modification time.** A
path is not the data: two machines resolve the same path to different files. A
modification time changes when nothing did, so a cache keyed on it misses
after every checkout and the expensive build runs again for no reason.

The rest is about refusing where a guess would be worse. A missing target
column is an error naming the columns that are present, because the usual
cause is a typo and the alternative -- training against the first numeric
column -- produces a complete, plausible, meaningless run.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime

import numpy as np
import pytest

from src.rade_qnet.core.contract.data import DataLineage, SplitIndices
from src.rade_qnet.core.contract.signature import InputSignature, TensorSpec
from src.rade_qnet.core.runtime.errors import BundleError, SpecError
from src.rade_qnet.sources.dataset.io import (
    DatasetCache,
    PreparedDataset,
    fingerprint_source,
    read_table,
)
from src.rade_qnet.testkit.fixtures import StandardisingState


def write_csv(path, rows, columns=("a", "b", "target")):
    """
    Write a small CSV file.

    Parameters
    ----------
    path
        Destination.
    rows
        Row values, each matching ``columns``.
    columns
        Header names.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def prepared(*, static_inputs=None):
    """
    Build a minimal prepared dataset, optionally carrying static inputs.

    Four rows and two features, which is enough for a round trip and small
    enough that a mismatch is readable in a failure message.

    Parameters
    ----------
    static_inputs
        Static inputs to attach, or ``None`` for a dataset with none.

    Returns
    -------
    PreparedDataset
        The dataset.
    """
    features = np.arange(8, dtype=np.float64).reshape(4, 2)
    target = np.arange(4, dtype=np.float64)
    return PreparedDataset(
        features=features,
        target=target,
        splits=SplitIndices(
            train=np.array([0, 1], dtype=np.int64),
            validation=np.array([2], dtype=np.int64),
            test=np.array([3], dtype=np.int64),
        ),
        signature=InputSignature(
            dynamic={"features": TensorSpec(shape=(None, 2), dtype="float32")},
            target=TensorSpec(shape=(None,), dtype="float32"),
        ),
        state=StandardisingState.fit(target),
        lineage=DataLineage(
            source_fingerprint="0" * 8,
            spec_digest="1" * 8,
            split_indices={"train": [0, 1], "validation": [2], "test": [3]},
            n_scenarios=4,
            framework_version="0.0.0",
            created_at=datetime.now(UTC),
        ),
        static_inputs=static_inputs or {},
    )


@pytest.fixture
def table(tmp_path):
    """
    Provide a three-column CSV.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    return write_csv(tmp_path / "data.csv", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class TestReadingColumns:
    """What ends up in the feature matrix, and in what order."""

    def test_the_target_column_is_separated_from_the_features(self, table):
        """
        Because a target left among the features is a perfect predictor.

        The model learns the identity function, training loss goes to zero,
        and the held-out score is also excellent -- the leak is in every
        split, so nothing looks wrong until the model is served data that has
        no target column.
        """
        data = read_table(table)
        assert data.features.shape == (2, 2)
        assert data.target.tolist() == [3.0, 6.0]

    def test_the_remaining_columns_are_sorted_by_name(self, table):
        """
        Rather than taken in file order.

        Two exports of the same data with the columns swapped would otherwise
        produce different matrices, so column zero means a different thing
        between two runs over identical data -- and the cache key would differ
        too, so neither run reuses the other's work.
        """
        assert read_table(table).feature_names == ("a", "b")

    def test_explicitly_named_columns_keep_the_caller_s_order(self, tmp_path):
        """
        Because an explicit order is a decision, not an accident.

        A model whose first input must be the spot rate needs to be able to
        say so, and sorting would silently override it.
        """
        path = write_csv(tmp_path / "ordered.csv", [[1.0, 2.0, 3.0]])
        data = read_table(path, feature_columns=["b", "a"])
        assert data.feature_names == ("b", "a")
        assert data.features[0].tolist() == [2.0, 1.0]

    def test_a_named_target_column_is_honoured(self, tmp_path):
        """
        So the convention is a default rather than a requirement.

        A file whose target is called ``y`` should not need renaming before
        it can be read.
        """
        path = write_csv(tmp_path / "y.csv", [[1.0, 2.0, 9.0]], columns=("a", "b", "y"))
        assert read_table(path, target_column="y").target.tolist() == [9.0]

    def test_attribute_columns_are_kept_aside_as_strings(self, tmp_path):
        """
        Because an instrument identifier is not a number.

        Read as a feature it would be a meaningless magnitude, and the model
        would learn that instrument 7 is greater than instrument 3.
        """
        path = write_csv(
            tmp_path / "attrs.csv",
            [[1.0, 2.0, 3.0, "EURUSD"]],
            columns=("a", "b", "target", "pair"),
        )
        data = read_table(path, attribute_columns=["pair"])
        assert data.attributes["pair"] == ("EURUSD",)
        assert "pair" not in data.feature_names


class TestRefusals:
    """Where guessing would produce a complete, meaningless run."""

    def test_a_missing_file_is_reported(self, tmp_path):
        """
        Naming the path, because the cause is usually a working directory.

        An empty dataset would otherwise flow all the way to a split that
        reports zero rows, which is several stages from the actual problem.
        """
        with pytest.raises(BundleError, match="no such file"):
            read_table(tmp_path / "absent.csv")

    def test_a_missing_target_column_is_reported(self, table):
        """
        Rather than falling back to the first numeric column.

        A fallback produces a complete run against the wrong target: the loss
        falls, the metrics compute, the report renders, and the model
        predicts something nobody asked for.
        """
        with pytest.raises(SpecError):
            read_table(table, target_column="absent")

    def test_the_message_names_the_columns_that_are_present(self, table):
        """
        Because the fix is a one-character change to the spec.

        Which is not findable from a message that only says the column was
        missing.
        """
        with pytest.raises(SpecError, match="target"):
            read_table(table, target_column="targett")

    def test_an_unsupported_suffix_is_reported_with_the_alternatives(self, tmp_path):
        """
        So a user knows what to convert to.

        The suffix selects the reader, so an unrecognised one has no
        plausible default -- reading a JSON file as CSV would produce one
        column of unparseable text.
        """
        path = tmp_path / "data.xlsx"
        path.write_bytes(b"not really a spreadsheet")
        with pytest.raises(SpecError, match=r"\.csv"):
            read_table(path)

    def test_a_missing_feature_column_is_reported(self, table):
        """
        For the same reason as the target, applied to the inputs.

        Silently dropping it would train a model on fewer features than the
        spec asked for, and the spec is what a later run would be reproduced
        from.
        """
        with pytest.raises(SpecError):
            read_table(table, feature_columns=["a", "absent"])


class TestFingerprinting:
    """Contents, so a cache hit means the data really is the same."""

    def test_the_same_contents_give_the_same_digest(self, tmp_path):
        """
        Which is what makes the cache usable at all.

        A digest that varied between runs would mean every run missed, and
        the cache would be pure overhead.
        """
        first = write_csv(tmp_path / "first.csv", [[1.0, 2.0, 3.0]])
        second = write_csv(tmp_path / "second.csv", [[1.0, 2.0, 3.0]])
        assert fingerprint_source(first) == fingerprint_source(second)

    def test_different_contents_give_different_digests(self, tmp_path):
        """
        Which is what makes a cache hit trustworthy.

        A collision here means a run silently reuses another run's arrays,
        and every number afterwards describes the wrong data.
        """
        first = write_csv(tmp_path / "first.csv", [[1.0, 2.0, 3.0]])
        second = write_csv(tmp_path / "second.csv", [[1.0, 2.0, 4.0]])
        assert fingerprint_source(first) != fingerprint_source(second)

    def test_the_path_is_not_part_of_the_digest(self, tmp_path):
        """
        Because a path is not the data.

        Two machines resolve the same path to different files, and the same
        file sits at different paths on a developer's laptop and in
        production -- so a path-keyed cache never hits across the two.
        """
        directory = tmp_path / "nested"
        directory.mkdir()
        here = write_csv(tmp_path / "data.csv", [[1.0, 2.0, 3.0]])
        there = write_csv(directory / "other_name.csv", [[1.0, 2.0, 3.0]])
        assert fingerprint_source(here) == fingerprint_source(there)

    def test_extra_material_changes_the_digest(self, tmp_path):
        """
        So a build configured differently over the same file does not collide.

        Two runs reading one file with different feature columns produce
        different arrays, and a digest over the file alone would let the
        second reuse the first's cache.
        """
        path = write_csv(tmp_path / "data.csv", [[1.0, 2.0, 3.0]])
        assert fingerprint_source(path, extra={"columns": ["a"]}) != fingerprint_source(
            path, extra={"columns": ["a", "b"]}
        )

    def test_a_build_with_no_single_file_can_still_be_fingerprinted(self):
        """
        Because a model data module may assemble several sources.

        It supplies the upstream digests as ``extra`` instead, so the cache
        works for a multi-source build without the cache knowing anything
        about sources.
        """
        assert fingerprint_source(None, extra={"upstream": ["abc", "def"]})

    def test_the_digest_is_order_independent_for_equivalent_material(self):
        """
        So two equivalent configurations share a cache entry.

        A digest sensitive to mapping order would make the cache miss
        whenever an unrelated refactor changed the order keys were inserted
        in.
        """
        assert fingerprint_source(None, extra={"a": 1, "b": 2}) == fingerprint_source(
            None, extra={"b": 2, "a": 1}
        )


class TestCachedStaticInputs:
    """
    Static inputs survive a cache round trip.

    Everything else in a prepared dataset is reconstructible from the raw
    file, so an omission from the cache shows up as a rebuild. A static
    input is not: it is a *product* of the fitted state, and an entry that
    stored the state but not the arrays derived from it would load cleanly
    and then fail inside a forward pass, with nothing pointing back here.
    The second-run path is also the path nobody tests by hand, because the
    first run of any new configuration populates the cache and works.
    """

    def test_they_come_back(self, tmp_path):
        """
        By name and by value.

        The names are the ones the model's signature declares, so losing
        one is indistinguishable from the model declaring an input nobody
        supplies.
        """
        dataset = prepared(static_inputs={"adjacency": np.eye(3, dtype=np.float32)})
        cache = DatasetCache(tmp_path)
        cache.save("key", dataset)

        loaded = cache.load("key", state_type=StandardisingState)
        assert loaded is not None
        assert np.array_equal(loaded.static_inputs["adjacency"], np.eye(3, dtype=np.float32))

    def test_their_dtype_is_preserved(self, tmp_path):
        """
        Including the integer index arrays, which are not floats.

        A sparse adjacency travels as integer indices beside float values,
        and an index array promoted to float fails at the point it is used
        to index -- or worse, silently rounds.
        """
        indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
        cache = DatasetCache(tmp_path)
        cache.save("key", prepared(static_inputs={"indices": indices}))

        loaded = cache.load("key", state_type=StandardisingState)
        assert loaded is not None
        assert loaded.static_inputs["indices"].dtype == np.int64

    def test_a_dataset_with_none_round_trips_too(self, tmp_path):
        """
        The common case, which must not be made to fail by the feature.

        An empty ``savez`` is a valid archive with no members, so the read
        side needs no branch -- but that is worth asserting rather than
        assuming, because the alternative is that every tabular model
        breaks on its second run.
        """
        cache = DatasetCache(tmp_path)
        cache.save("key", prepared())

        loaded = cache.load("key", state_type=StandardisingState)
        assert loaded is not None
        assert loaded.static_inputs == {}

    def test_they_cannot_collide_with_the_feature_arrays(self, tmp_path):
        """
        Which is why they are written to a file of their own.

        Their names come from the model's signature, so a model with a
        static input called ``features`` or ``train`` is entirely legal --
        and in a single archive it would overwrite the real one, giving a
        cached dataset whose features are an adjacency matrix.
        """
        dataset = prepared(static_inputs={"features": np.full((2, 2), 7.0, dtype=np.float32)})
        cache = DatasetCache(tmp_path)
        cache.save("key", dataset)

        loaded = cache.load("key", state_type=StandardisingState)
        assert loaded is not None
        assert np.array_equal(loaded.features, dataset.features)
        assert np.array_equal(loaded.static_inputs["features"], np.full((2, 2), 7.0))


class TestParsedValues:
    """The numbers themselves, which everything downstream assumes about."""

    def test_the_features_are_float64(self, table):
        """
        So a dtype decision is made once, here, rather than per engine.

        An integer column read as integers would make a standardiser's
        division truncate, and the symptom is a feature that is mostly
        zeros.
        """
        assert read_table(table).features.dtype == np.float64

    def test_an_empty_cell_becomes_a_gap_rather_than_a_zero(self, tmp_path):
        """
        Because zero is a value and a gap is not.

        A missing price read as zero is a hundred per cent return, which
        dominates any statistic computed from the column.
        """
        path = tmp_path / "gappy.csv"
        path.write_text("a,b,target\n1.0,,3.0\n", encoding="utf-8")
        assert np.isnan(read_table(path).features).any()
