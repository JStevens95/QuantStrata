# `tests/rade_qnet/sources/dataset`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 26 | 1204 | `aca25b3ce8fb419d` |
| 2 | `test_dataset_cache.py` | 197 | 7114 | `219df6062c0637eb` |
| 3 | `test_dataset_module.py` | 435 | 18100 | `6a34cddc7ef22200` |
| 4 | `test_dataset_splits.py` | 322 | 12446 | `34d44469c31c40c1` |
| 5 | `test_dataset_tables.py` | 344 | 12810 | `4645567f20904486` |

---

## 1. `tests/rade_qnet/sources/dataset/__init__.py`

1204 bytes · SHA-256 `aca25b3ce8fb419d`

```python
"""
Tests for ``rade_qnet.sources.dataset`` -- fixed training data.

This is the highest-value sub-tree in the suite, because the defects it guards
against do not announce themselves. A leaky split does not raise an error; it
produces an encouraging validation score and a model that fails in production.
The tests therefore assert leakage properties directly rather than inferring
them from metrics.

Planned modules
---------------
``test_dataset_module.py``
    ``DataModule`` stage ordering, and the ``DataBundle`` it produces.
    [Phase 2]
``test_dataset_splits.py``
    Each split strategy. Explicitly covered: splits are disjoint and cover the
    index exactly, chronological order is respected, no sequence window
    straddles a boundary, and the scenario split is unaffected by the batch
    shuffle flag -- which is the conflation that made splits random in the
    implementation this framework replaces.  [Phase 2]
``test_dataset_io.py``
    Readers and cache behaviour: a cache hit returns an identical bundle, and a
    changed source fingerprint invalidates the entry.  [Phase 2]
``test_dataset_transitions.py``
    Reading a stored transition table as a dataset source.  [Phase 7]
"""
```

---

## 2. `tests/rade_qnet/sources/dataset/test_dataset_cache.py`

7114 bytes · SHA-256 `219df6062c0637eb`

```python
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
```

---

## 3. `tests/rade_qnet/sources/dataset/test_dataset_module.py`

18100 bytes · SHA-256 `6a34cddc7ef22200`

```python
"""
Tests for the data-module base class.

``DataModule`` is the piece a user actually writes. It is a template method:
the base owns the *order* of the build and the user supplies the steps. That
split is the whole point -- the order is where the leaks live, and a user
reimplementing it would have to get the same four orderings right that this
framework exists to get right once.

The ordering tested hardest is that the split comes *before* the state is
fitted. The other way round, the scaler's mean and standard deviation are
computed over the whole history including the test period, and the leak is
invisible: the metrics look slightly too good in a way that is
indistinguishable from a slightly better model.

The second is that the sequence transform runs *after* the split, so a window
cannot be assembled from rows on both sides of a boundary.

Everything the base produces is also recorded. The lineage carries the split
indices, the source fingerprint and the spec digest, because a run whose data
cannot be identified cannot be compared against another run -- and comparison
is the only way anyone finds out that something changed.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.rade_qnet.core.contract.data import SPLIT_NAMES
from src.rade_qnet.core.spec.data import (
    ChronologicalSplitSpec,
    PurgedKFoldSplitSpec,
    ScalingSpec,
    TabularSourceSpec,
    TransformsSpec,
)
from src.rade_qnet.sources.dataset.module import DataModule
from src.rade_qnet.sources.dataset.tabular import TabularDataModule


@pytest.fixture
def csv_path(tmp_path):
    """
    Write a four-hundred-row CSV with a learnable target.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(loc=5.0, scale=2.0, size=(400, 3))
    target = features @ np.array([1.0, -2.0, 0.5])

    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "c", "target"])
        writer.writerows(np.column_stack([features, target]).tolist())
    return path


class TestTheBuildOrder:
    """The base owns the order, because the order is where the leaks live."""

    def test_the_state_is_fitted_after_the_split(self, csv_path):
        """
        The ordering that keeps a scaler out of the test period.

        Fitted first, the mean and standard deviation are computed over the
        whole history including held-out rows. The leak is invisible: metrics
        come out slightly too good in a way indistinguishable from a slightly
        better model. Asserted by comparing the fitted mean against the
        training rows' own mean.
        """
        spec = TabularSourceSpec(
            path=csv_path,
            transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
        )
        dataset = TabularDataModule().build(spec, seed=0)
        train = dataset.splits.train

        raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        expected = raw[train, :3].mean(axis=0)
        fitted = dataset.state.part("scaling")
        assert np.allclose(fitted.feature_centre, expected)

    def test_the_fitted_mean_is_not_the_whole_history_s_mean(self, csv_path):
        """
        The negative half of the same claim.

        Without it, a fixture whose training rows happened to have the same
        mean as the full history would pass the test above while the leak was
        present.
        """
        spec = TabularSourceSpec(
            path=csv_path,
            transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
        )
        dataset = TabularDataModule().build(spec, seed=0)
        raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        fitted = dataset.state.part("scaling")
        assert not np.allclose(fitted.feature_centre, raw[:, :3].mean(axis=0))

    def test_the_splits_do_not_overlap(self, csv_path):
        """
        Because an overlapping split is a leak with a clean-looking report.

        Every number computes, the split sizes are plausible, and the held-out
        score is partly a training score.
        """
        dataset = TabularDataModule().build(TabularSourceSpec(path=csv_path), seed=0)
        indices = [set(getattr(dataset.splits, name).tolist()) for name in SPLIT_NAMES]
        assert not (indices[0] & indices[1])
        assert not (indices[0] & indices[2])
        assert not (indices[1] & indices[2])

    def test_a_chronological_split_keeps_time_order(self, csv_path):
        """
        So the test period is after the training period.

        A shuffled split over a time series lets the model interpolate rather
        than forecast, and the resulting score is unachievable in production.
        """
        dataset = TabularDataModule().build(
            TabularSourceSpec(path=csv_path, split=ChronologicalSplitSpec()),
            seed=0,
        )
        assert dataset.splits.train.max() < dataset.splits.validation.min()
        assert dataset.splits.validation.max() < dataset.splits.test.min()

    def test_the_signature_stage_receives_the_fitted_state(self, csv_path):
        """
        Because for some modules the signature depends on what was fitted.

        The tabular module reads its feature count from the transformed
        matrix and needs nothing else, so this is not about the tabular
        module. It is about the hook's contract. A module whose static inputs
        are themselves fitted -- an encoded attribute matrix, a
        nearest-neighbour adjacency -- can only declare their shapes from the
        state, and a module whose index arrays are computed after a basis
        selection can only compute them from the selected basis.

        Given only ``features``, such a module would have to stash the state
        on ``self`` during ``fit_state`` and read it back here. That makes the
        module stateful across stages: a second build on one instance would
        reuse the first build's state, and a cached dataset would declare a
        signature fitted to data it was not built from. Both failures produce
        a signature that is the right shape and the wrong content.
        """
        received = {}

        class Recording(TabularDataModule):
            """A tabular module that records what its signature stage got."""

            def signature(self, spec, *, features, state):
                """Record the state, then defer to the real implementation."""
                received["state"] = state
                return super().signature(spec, features=features, state=state)

        spec = TabularSourceSpec(
            path=csv_path,
            transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
        )
        dataset = Recording().build(spec, seed=0)

        # Identity, not equality: the signature stage must see the very state
        # the dataset was built with. An equal-but-separate state would mean
        # something refitted in between.
        assert received["state"] is dataset.state


class TestStaticInputs:
    """
    The hook for inputs that do not vary by sample.

    A graph model's adjacency is the motivating case: it is the same object
    for every scenario, so collating it per sample multiplies its memory by
    the batch size and its transfer by the number of batches, for nothing.
    It therefore cannot travel with the rows -- it needs a route of its own
    from the module that built it to the engine that uploads it, and these
    tests cover that route's first leg.
    """

    def test_a_module_that_declares_none_gets_none(self, csv_path):
        """
        The default, and the case that must not pay for the feature.

        Most modules have no static inputs at all, so the hook returns an
        empty mapping and every later stage treats that as "nothing to do".
        """
        dataset = TabularDataModule().build(TabularSourceSpec(path=csv_path), seed=0)
        assert dataset.static_inputs == {}

    def test_what_the_hook_returns_reaches_the_dataset(self, csv_path):
        """
        The leg the end-to-end run was missing.

        Until this was wired, a module could build an adjacency and declare
        it in its signature, and the model would still be called without it
        -- which surfaces as ``forward() missing 5 required keyword-only
        arguments``, a long way from the hook that was never consulted.
        """
        adjacency = np.eye(3, dtype=np.float32)

        class WithGraph(TabularDataModule):
            """A tabular module that also carries a fixed adjacency."""

            def static_inputs(self, raw, state):
                """Return the adjacency, which does not vary by row."""
                del raw, state
                return {"adjacency": adjacency}

        dataset = WithGraph().build(TabularSourceSpec(path=csv_path), seed=0)
        assert np.array_equal(dataset.static_inputs["adjacency"], adjacency)

    def test_the_hook_sees_the_fitted_state(self, csv_path):
        """
        Because a static input is usually something that was fitted.

        An encoded attribute matrix and a nearest-neighbour adjacency are
        both products of ``fit_state``, so a hook that could not read the
        state would force the module to stash it on ``self`` -- the
        cross-stage statefulness the signature hook was given the state to
        avoid.
        """
        received = {}

        class Recording(TabularDataModule):
            """A tabular module that records what its static hook got."""

            def static_inputs(self, raw, state):
                """Record the state, then declare nothing."""
                del raw
                received["state"] = state
                return {}

        dataset = Recording().build(TabularSourceSpec(path=csv_path), seed=0)
        assert received["state"] is dataset.state


class TestTheProducedDataset:
    """What the build hands on, and what it records about itself."""

    def test_the_features_and_target_align_row_for_row(self, csv_path):
        """
        Because every metric pairs them by position.

        A misalignment produces a well-defined number from mismatched rows,
        and a correctly trained model then reports a negative r-squared on
        its own training split.
        """
        dataset = TabularDataModule().build(TabularSourceSpec(path=csv_path), seed=0)
        assert dataset.features.shape[0] == dataset.target.shape[0]

    def test_the_signature_matches_the_transformed_features(self, csv_path):
        """
        Declared from the data that will actually be served.

        Taken from the raw columns instead, a reduction would leave the
        signature claiming more inputs than the model receives -- and the
        mismatch appears inside a forward pass.
        """
        dataset = TabularDataModule().build(TabularSourceSpec(path=csv_path), seed=0)
        assert dataset.signature.dynamic["features"].shape[1] == dataset.features.shape[1]

    def test_the_lineage_records_the_split_indices(self, csv_path):
        """
        So a later run can be compared against this one exactly.

        The sizes alone would not distinguish two different seventy-per-cent
        splits of the same data, and those are different runs.
        """
        dataset = TabularDataModule().build(TabularSourceSpec(path=csv_path), seed=0)
        assert set(dataset.lineage.split_indices) == set(SPLIT_NAMES)

    def test_the_lineage_records_the_source_fingerprint(self, csv_path):
        """
        Because a run whose data cannot be identified cannot be compared.

        And comparison is the only way anyone finds out the data changed.
        """
        dataset = TabularDataModule().build(TabularSourceSpec(path=csv_path), seed=0)
        assert dataset.lineage.source_fingerprint

    def test_the_lineage_records_the_scenario_count(self, csv_path):
        """
        So a split share can be read as a number of rows.

        Seventy per cent of four hundred and of four are the same share and
        not comparable runs.
        """
        dataset = TabularDataModule().build(TabularSourceSpec(path=csv_path), seed=0)
        assert dataset.lineage.n_scenarios == 400

    def test_the_feature_names_survive_into_the_dataset(self, csv_path):
        """
        Because "column 2" is not something anyone can act on.

        They are what a missingness figure and a basis-selection report label
        their axes with.
        """
        dataset = TabularDataModule().build(TabularSourceSpec(path=csv_path), seed=0)
        assert dataset.feature_names == ("a", "b", "c")


class TestDeterminism:
    """Two runs of one configuration must be the same run."""

    def test_the_same_seed_gives_the_same_splits(self, csv_path):
        """
        Otherwise a reproduced run is not the same run.

        And a difference in its score cannot be attributed to anything,
        because the one thing that was supposed to be fixed was not.
        """
        spec = TabularSourceSpec(path=csv_path, split=PurgedKFoldSplitSpec())
        first = TabularDataModule().build(spec, seed=7)
        second = TabularDataModule().build(spec, seed=7)
        assert np.array_equal(first.splits.train, second.splits.train)

    def test_the_same_configuration_gives_the_same_fingerprint(self, csv_path):
        """
        Which is what lets the cache be keyed on it.

        A fingerprint that varied between identical runs would make every
        cache lookup miss, and the cache would be pure overhead.
        """
        spec = TabularSourceSpec(path=csv_path)
        first = TabularDataModule().build(spec, seed=0)
        second = TabularDataModule().build(spec, seed=0)
        assert first.lineage.source_fingerprint == second.lineage.source_fingerprint

    def test_a_purged_fold_does_not_depend_on_the_seed(self, csv_path):
        """
        Because the folds are contiguous blocks in time order.

        Randomising which scenarios land in a fold would scatter the held-out
        period through the training period, which is the leak the embargo
        exists to prevent -- so this splitter has nothing for a seed to
        influence, and two seeds must agree.
        """
        spec = TabularSourceSpec(path=csv_path, split=PurgedKFoldSplitSpec())
        first = TabularDataModule().build(spec, seed=1)
        second = TabularDataModule().build(spec, seed=2)
        assert np.array_equal(first.splits.train, second.splits.train)

    def test_a_purged_fold_embargoes_either_side_of_the_held_out_block(self, csv_path):
        """
        So no training scenario sits immediately beside a held-out one.

        Adjacent scenarios in a financial series are strongly autocorrelated,
        so a training row next to a test row makes the test score partly a
        measurement of that autocorrelation rather than of the model.
        """
        spec = TabularSourceSpec(path=csv_path, split=PurgedKFoldSplitSpec(embargo_scenarios=10))
        dataset = TabularDataModule().build(spec, seed=0)
        covered = (
            set(dataset.splits.train.tolist())
            | set(dataset.splits.validation.tolist())
            | set(dataset.splits.test.tolist())
        )
        assert len(covered) < dataset.lineage.n_scenarios


class TestBatchSources:
    """The bridge from a prepared dataset to something an engine can read."""

    def test_one_source_per_non_empty_split(self, csv_path):
        """
        So a caller never has to know which splits a run produced.

        A fixed three-source assumption would break on a run configured with
        no validation fraction.
        """
        module = TabularDataModule()
        spec = TabularSourceSpec(path=csv_path)
        sources = module.batch_sources(module.build(spec, seed=0), spec)
        assert set(sources) == {"train", "validation", "test"}

    def test_the_sources_see_the_transformed_features(self, csv_path):
        """
        Not the raw ones, which is what the model was declared against.

        A source over raw features would serve unscaled inputs to a model
        trained on scaled ones, and the predictions would be numbers computed
        from the wrong distribution.
        """
        module = TabularDataModule()
        spec = TabularSourceSpec(
            path=csv_path,
            transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
        )
        dataset = module.build(spec, seed=0)
        source = module.batch_sources(dataset, spec)["train"]
        batch = next(source.batches())
        # Standardised training features have a mean near zero; raw ones have
        # a mean near five.
        assert abs(float(np.mean(batch["features"]))) < 2.0


class TestTheTemplateIsAbstract:
    """The hooks a user must supply, declared rather than defaulted."""

    def test_a_module_must_say_how_to_load_its_data(self):
        """
        Because no default could be right.

        A base-class ``load`` returning an empty table would produce a run
        that completed over no data, and the first visible symptom would be a
        split reporting zero rows several stages later.
        """
        assert "load" in DataModule.__abstractmethods__

    def test_a_module_must_say_how_many_scenarios_it_has(self):
        """
        Because the splitter cannot infer it from an opaque raw object.

        That is the point of the generic parameter: the base never inspects
        the raw data, so it has to ask.
        """
        assert "n_scenarios" in DataModule.__abstractmethods__

    def test_the_build_order_is_not_a_hook(self):
        """
        Because the order is what the framework is for.

        A user who could override ``build`` would have to get the same four
        orderings right that this base exists to get right once -- and the
        failures are silent.
        """
        assert "build" not in DataModule.__abstractmethods__
```

---

## 4. `tests/rade_qnet/sources/dataset/test_dataset_splits.py`

12446 bytes · SHA-256 `34d44469c31c40c1`

```python
"""
Tests for the split strategies.

Splitting is where leakage is introduced, and leakage is the failure mode that
produces a *better* result than the truth. Nothing in a metric, a loss curve or
a report says a split leaked; the model simply scores well and keeps scoring
well until it meets real data.

So these tests are mostly about disjointness and ordering, asserted directly on
the indices rather than inferred from a downstream score. An assertion on a
metric cannot distinguish a model that generalises from a split that leaks.

The boundary gap gets the most attention. With a sequence length above one,
every sample is a *window* ending at its label, so a validation window whose
label sits one row after the training split's last row overlaps the training
data by ``length - 1`` rows. Dropping those rows is the whole purpose of the
gap, and it is invisible in the split fractions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.lifecycle.errors import SpecError
from src.rade_qnet.core.spec.data import (
    ChronologicalSplitSpec,
    ExplicitSplitSpec,
    GroupedSplitSpec,
    PurgedKFoldSplitSpec,
)
from src.rade_qnet.sources.dataset.splits import (
    boundary_gap,
    split_by_group,
    split_chronologically,
    split_explicitly,
    split_purged_kfold,
    split_scenarios,
)


class TestBoundaryGap:
    """The width that stops a window straddling two splits."""

    def test_a_non_sequential_model_needs_no_gap(self):
        """
        A length of one means each sample is a single row.

        Nothing overlaps, so a gap would discard data for no reason.
        """
        assert boundary_gap(1) == 0

    def test_the_gap_is_one_short_of_the_window(self):
        """
        A window of length ``n`` reaches ``n - 1`` rows back from its label.

        So dropping ``n - 1`` rows is exactly enough, and dropping ``n`` would
        waste one.
        """
        assert boundary_gap(20) == 19

    def test_an_explicit_gap_is_added_on_top(self):
        """
        For a target that looks forward as well as back.

        A five-day forward return observed at the split boundary is not known
        until five days into the next split, so the extra scenarios have to be
        dropped on top of the window.
        """
        assert boundary_gap(20, 5) == 24


class TestChronologicalSplit:
    """Time order preserved, which is the only honest split for a forecast."""

    def test_the_splits_are_in_time_order(self):
        """
        Training rows precede validation rows, which precede test rows.

        A random split of a time series trains on the future to predict the
        past, and scores spectacularly.
        """
        indices = split_chronologically(ChronologicalSplitSpec(), n_scenarios=1000)
        assert indices.train.max() < indices.validation.min()
        assert indices.validation.max() < indices.test.min()

    def test_the_splits_are_disjoint(self):
        """No scenario appears in two splits."""
        indices = split_chronologically(ChronologicalSplitSpec(), n_scenarios=1000)
        combined = np.concatenate([indices.train, indices.validation, indices.test])
        assert combined.size == np.unique(combined).size

    def test_every_scenario_is_used_when_no_gap_is_needed(self):
        """
        At a sequence length of one, nothing is discarded.

        A split that silently dropped rows would make the fractions a lie.
        """
        indices = split_chronologically(ChronologicalSplitSpec(), n_scenarios=1000)
        assert sum(indices.sizes.values()) == 1000

    def test_a_sequence_length_opens_a_gap_between_splits(self):
        """
        The leakage this module exists to prevent.

        With a window of twenty, a validation sample labelled at the first row
        after training would read nineteen training rows. The gap is what makes
        the splits disjoint in *windows* rather than merely in labels.
        """
        indices = split_chronologically(
            ChronologicalSplitSpec(), n_scenarios=1000, sequence_length=20
        )
        assert indices.validation.min() - indices.train.max() > 19
        assert indices.test.min() - indices.validation.max() > 19

    def test_the_gap_costs_scenarios(self):
        """
        Which is the price of the guarantee, and should be visible.

        A reader comparing a run with a sequence length against one without
        should see fewer usable scenarios, not the same number.
        """
        gapped = split_chronologically(
            ChronologicalSplitSpec(), n_scenarios=1000, sequence_length=20
        )
        assert sum(gapped.sizes.values()) < 1000

    def test_a_fraction_that_rounds_to_zero_is_refused(self):
        """
        Found in this implementation, by its own end-to-end test.

        Two groups at a fifteen percent validation fraction rounds to zero, so
        every row went to training and the run trained with no validation and
        no test -- silently, because an empty split is a legitimate
        configuration when the fraction is zero. The distinction is whether
        the caller asked for one.
        """
        # Three scenarios at a fifteen percent fraction: round(0.45) is zero.
        with pytest.raises(SpecError, match="rounds to zero"):
            split_chronologically(ChronologicalSplitSpec(), n_scenarios=3)

    def test_a_deliberately_empty_split_is_allowed(self):
        """
        The other half of the previous rule.

        A fraction of zero means the caller does not want the split, which is
        a different statement from a fraction that was too small to honour.
        """
        indices = split_chronologically(
            ChronologicalSplitSpec(validation_fraction=0.0, test_fraction=0.2),
            n_scenarios=100,
        )
        assert indices.validation.size == 0
        assert indices.train.size > 0

    def test_the_result_is_deterministic(self):
        """A chronological split has nothing to randomise, so it cannot vary."""
        first = split_chronologically(ChronologicalSplitSpec(), n_scenarios=500)
        second = split_chronologically(ChronologicalSplitSpec(), n_scenarios=500)
        assert np.array_equal(first.train, second.train)


class TestPurgedKFold:
    """A fold with its neighbours removed, for overlapping-label problems."""

    def test_the_folds_are_disjoint(self):
        """The basic requirement of any split."""
        indices = split_purged_kfold(PurgedKFoldSplitSpec(), n_scenarios=1000)
        combined = np.concatenate([indices.train, indices.validation, indices.test])
        assert combined.size == np.unique(combined).size

    def test_an_embargo_separates_training_from_the_held_out_fold(self):
        """
        What distinguishes this from plain k-fold.

        Without an embargo, a training label whose window overlaps a
        validation label leaks across the boundary -- and with overlapping
        labels that happens at every fold edge rather than only at the ends.
        """
        embargoed = split_purged_kfold(PurgedKFoldSplitSpec(embargo_scenarios=25), n_scenarios=1000)
        distances = np.abs(embargoed.train[:, None] - embargoed.validation[None, :])
        assert distances.min() > 25

    def test_each_fold_holds_out_a_different_slice(self):
        """
        One fold per run, so a sweep is a job set rather than a loop in here.

        A fold argument that was ignored would make every member of that sweep
        the same run, and the averaged result would look far more stable than
        it is.
        """
        first = split_purged_kfold(PurgedKFoldSplitSpec(), n_scenarios=1000, fold=0)
        second = split_purged_kfold(PurgedKFoldSplitSpec(), n_scenarios=1000, fold=1)
        assert not np.array_equal(first.validation, second.validation)


class TestGroupSplit:
    """Whole groups held out, so no group appears on both sides."""

    def test_no_group_spans_two_splits(self):
        """
        The property the strategy exists for.

        Splitting rows rather than groups puts the same entity in train and
        test, and the model learns that entity rather than the relationship.
        """
        labels = np.repeat(np.arange(20), 50)
        indices = split_by_group(
            GroupedSplitSpec(group_key="entity"), n_scenarios=labels.size, group_labels=labels
        )
        train_groups = set(labels[indices.train].tolist())
        test_groups = set(labels[indices.test].tolist())
        assert not train_groups & test_groups

    def test_a_group_fraction_that_rounds_to_zero_is_refused(self):
        """
        The case that produced the defect this guard was written for.

        Two groups at a fifteen percent fraction rounds to zero groups, so the
        run had no validation and no test and nothing said so.
        """
        labels = np.repeat(np.arange(2), 200)
        with pytest.raises(SpecError, match="rounds to zero"):
            split_by_group(
                GroupedSplitSpec(group_key="entity"), n_scenarios=labels.size, group_labels=labels
            )

    def test_the_assignment_is_reproducible_under_a_seed(self):
        """Group assignment is random, so a seed must pin it."""
        labels = np.repeat(np.arange(20), 50)
        first = split_by_group(
            GroupedSplitSpec(group_key="entity"),
            n_scenarios=labels.size,
            group_labels=labels,
            seed=3,
        )
        second = split_by_group(
            GroupedSplitSpec(group_key="entity"),
            n_scenarios=labels.size,
            group_labels=labels,
            seed=3,
        )
        assert np.array_equal(first.test, second.test)

    def test_a_different_seed_gives_a_different_assignment(self):
        """
        Otherwise the seed is decorative.

        A strategy that ignored its seed would make a repeated experiment look
        robust when it was the same experiment.
        """
        labels = np.repeat(np.arange(20), 50)
        first = split_by_group(
            GroupedSplitSpec(group_key="entity"),
            n_scenarios=labels.size,
            group_labels=labels,
            seed=1,
        )
        second = split_by_group(
            GroupedSplitSpec(group_key="entity"),
            n_scenarios=labels.size,
            group_labels=labels,
            seed=2,
        )
        assert not np.array_equal(first.test, second.test)


class TestExplicitSplit:
    """Caller-supplied indices, passed through and validated."""

    def test_the_supplied_indices_are_used_unchanged(self):
        """
        The point of the strategy: the caller knows something the spec cannot.

        A reproduction of a published result, or a split aligned to a market
        regime, cannot be expressed as a fraction.
        """
        indices = split_explicitly(
            ExplicitSplitSpec(train=(0, 1, 2, 3), validation=(5, 6), test=(8, 9)),
            n_scenarios=10,
        )
        assert indices.train.tolist() == [0, 1, 2, 3]
        assert indices.test.tolist() == [8, 9]

    def test_an_index_past_the_end_is_refused(self):
        """
        Out-of-range indices produce a silent wrap in numpy.

        A negative index reads from the other end of the array, so a mistake
        here trains on the wrong rows rather than failing.
        """
        with pytest.raises(SpecError):
            split_explicitly(
                ExplicitSplitSpec(train=(0, 1), validation=(2,), test=(99,)),
                n_scenarios=10,
            )


class TestTheDispatcher:
    """One entry point, so a caller never matches on the split kind."""

    @pytest.mark.parametrize(
        "spec",
        [
            ChronologicalSplitSpec(),
            PurgedKFoldSplitSpec(),
        ],
    )
    def test_each_strategy_is_reachable_through_the_dispatcher(self, spec):
        """So adding a strategy does not change any caller."""
        indices = split_scenarios(spec, n_scenarios=1000)
        assert indices.train.size > 0

    def test_a_group_split_needs_group_labels(self):
        """
        Refused rather than defaulted.

        Defaulting to one group per row would silently degrade a group split
        to a random one, which is exactly the leakage it exists to prevent.
        """
        with pytest.raises(SpecError):
            split_scenarios(GroupedSplitSpec(group_key="entity"), n_scenarios=100)
```

---

## 5. `tests/rade_qnet/sources/dataset/test_dataset_tables.py`

12810 bytes · SHA-256 `4645567f20904486`

```python
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
from src.rade_qnet.core.lifecycle.errors import BundleError, SpecError
from src.rade_qnet.sources.dataset.cache import PreparedDataset
from src.rade_qnet.sources.dataset.tables import fingerprint_source, read_table
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
```

