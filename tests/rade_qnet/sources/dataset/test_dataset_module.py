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
