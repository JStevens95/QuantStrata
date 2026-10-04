"""
Tests for the dataset cache. Formerly part of the tests for reading tables and fingerprinting their contents.

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
from src.rade_qnet.sources.dataset.cache import DatasetCache, PreparedDataset
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
