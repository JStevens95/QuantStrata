# `tranql/models/rade/rade_qnet/tests/sources/batching`

3 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 27 | 1224 | `75d38714d92fee51` |
| 2 | `test_batching_dataset.py` | 444 | 16387 | `6dd0c4578f5be157` |
| 3 | `test_batching_rollout.py` | 378 | 14431 | `105e8d47ec66ea68` |

---

## 1. `tranql/models/rade/rade_qnet/tests/sources/batching/__init__.py`

1224 bytes · SHA-256 `75d38714d92fee51`

```python
"""
Tests for ``rade_qnet.sources.batching`` -- the ``BatchSource`` adapters.

Every module here is run through one shared contract suite, because the whole
value of the abstraction is that a training loop cannot tell the sources apart.
If a source satisfies the protocol only loosely, the loop acquires a special
case and the abstraction has failed.

Planned modules
---------------
``test_batching_contract.py``
    The shared suite applied to every source: batch shapes are as declared,
    iteration is reproducible under a fixed seed, and exhaustion is signalled
    rather than silently looping.  [Phase 2, extended in Phase 7]
``test_batching_dataset.py``
    Iterating a ``DataBundle`` split, and that static inputs are presented once
    per batch rather than once per sample.  [Phase 2]
``test_batching_rollout.py``
    On-policy collection against a toy environment.  [Phase 7]
``test_batching_replay.py``
    Buffer capacity, eviction order and sampling distribution.  [Phase 7]
``test_batching_offline.py``
    Serving batches from a stored transition table.  [Phase 7]
``test_batching_simulation.py``
    Path batches from a differentiable environment, with the computation graph
    preserved.  [Phase 7]
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/sources/batching/test_batching_dataset.py`

16387 bytes · SHA-256 `6dd0c4578f5be157`

```python
"""
Tests for the dataset batch source.

This module carries two of the ten defects and one bug found while building
Phase 2, and each has a test here that fails against the broken behaviour.

**Defect 3 -- one flag drove two decisions.** The previous implementation used
a single ``shuffle`` setting for both the train/validation/test *split* and the
*batch order*. Those are unrelated decisions: shuffling batch order is
standard and desirable, while shuffling a time-series split is leakage. The
test is that turning shuffling on changes the order of the batches and nothing
about which rows are in the split.

**Defect 4 -- static inputs were collated per sample.** A graph adjacency
matrix was merged into every sample and then compared against itself
``batch_size`` times per batch to confirm something true by construction. The
test is that static inputs are delivered once, outside the batch stream, and
that a batch carries only dynamic inputs.

**The alignment bug.** A training source reshuffles between passes, which is
correct for training and wrong for anything that needs two passes to line up.
``ordered()`` is the fix, and the tests assert both halves of its contract:
stable order, and the same samples.
"""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.contract.data import TARGET_KEY
from tranql.models.rade.rade_qnet.rade_qnet.core.contract.source import BatchSource, OrderedSource
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import (
    LoaderSpec,
    SequenceSpec,
    TabularSourceSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.sources.batching.dataset import (
    DatasetSource,
    sources_for,
)
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.tabular import TabularDataModule


@pytest.fixture
def prepared(tmp_path):
    """
    Build a prepared dataset from a small CSV file.

    Returns
    -------
    tuple
        The prepared dataset and the source spec that produced it.
    """
    rng = np.random.default_rng(3)
    features = rng.normal(size=(400, 3))
    target = features.sum(axis=1)

    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "c", "target"])
        writer.writerows(np.column_stack([features, target]).tolist())

    spec = TabularSourceSpec(path=path)
    return TabularDataModule().build(spec, seed=3), spec


def targets_of(source):
    """
    Drain one pass of a source's targets.

    Parameters
    ----------
    source
        A bounded source.

    Returns
    -------
    numpy.ndarray
        Flat target vector, in the order the source yielded it.
    """
    return np.concatenate([np.ravel(np.asarray(batch[TARGET_KEY])) for batch in source.batches()])


class TestTheProtocol:
    """Structural conformance, which is what lets a user's source substitute."""

    def test_a_dataset_source_satisfies_batch_source(self, prepared):
        """Without inheriting from it, which is the point of the protocol."""
        dataset, spec = prepared
        sources = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)
        assert isinstance(sources["train"], BatchSource)

    def test_a_dataset_source_is_also_iterable(self, prepared):
        """
        So one object can be both the source and the loader.

        Without it the pipeline would have to carry the sources alongside the
        data bundle, and the two could drift apart.
        """
        dataset, spec = prepared
        source = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        assert sum(1 for _ in source) == source.steps_per_epoch

    def test_the_batch_count_is_exact_not_estimated(self, prepared):
        """
        Callbacks, schedules and progress all compute from it.

        An estimate would make a cosine schedule anneal over the wrong period,
        which looks like a badly chosen learning rate.
        """
        dataset, spec = prepared
        source = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        assert sum(1 for _ in source.batches()) == source.steps_per_epoch


class TestDefectThreeSplitAndShuffleAreSeparate:
    """One flag drove two unrelated decisions; now it drives one."""

    def test_shuffling_changes_the_batch_order(self, prepared):
        """
        Which is the legitimate purpose of the flag.

        A source that ignored it would train on the same batch order every
        epoch, which correlates consecutive updates and slows convergence.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
            seed=1,
        )
        assert not np.array_equal(targets_of(source), targets_of(source))

    def test_shuffling_does_not_change_which_rows_are_in_the_split(self, prepared):
        """
        The half that was leakage: a shuffled *split*, not a shuffled order.

        The multiset of targets must be identical with shuffling on and off,
        because permuting a split cannot introduce a member of another one.
        """
        dataset = prepared[0]
        shuffled = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
            seed=1,
        )
        ordered = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=False),
            sequence=SequenceSpec(),
            seed=1,
        )
        assert np.allclose(np.sort(targets_of(shuffled)), np.sort(targets_of(ordered)))

    def test_the_held_out_splits_are_never_shuffled(self, prepared):
        """
        Shuffling a split that is only ever read once buys nothing.

        And it costs the alignment that scoring depends on, so the source does
        not do it regardless of the flag.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="test",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
            seed=1,
        )
        assert np.array_equal(targets_of(source), targets_of(source))

    def test_the_batch_order_is_reproducible_under_a_seed(self, prepared):
        """
        Two runs with the same seed must see the same orders in the same epochs.

        Otherwise a reproduced run is not the same run, and a difference in
        its score cannot be attributed.
        """
        dataset = prepared[0]

        def first_pass(seed):
            """Return the first pass's targets for a given seed."""
            source = DatasetSource(
                dataset,
                split="train",
                loader=LoaderSpec(shuffle=True),
                sequence=SequenceSpec(),
                seed=seed,
            )
            return targets_of(source)

        assert np.array_equal(first_pass(7), first_pass(7))
        assert not np.array_equal(first_pass(7), first_pass(8))


class TestDefectFourStaticInputsLeavePerSampleCollation:
    """Delivered once, outside the batch stream."""

    def test_static_inputs_are_exposed_separately(self, prepared):
        """
        So the engine uploads them to the device a single time.

        Empty for a tabular model, which is the common case and the one that
        must not pay for the feature.
        """
        dataset, spec = prepared
        source = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        assert source.static == {}

    def test_a_declared_static_input_is_served_from_the_dataset(self, prepared):
        """
        The last leg of the route from the data module to the engine.

        The empty default above is the common case, but it is also what a
        broken route looks like -- so a test that only ever saw an empty
        mapping would pass against a source that returned ``{}``
        unconditionally, which is exactly the bug this covers. Served by
        reference rather than copied, because the point of a static input
        is that one array is uploaded once.
        """
        dataset, spec = prepared
        adjacency = np.eye(3, dtype=np.float32)
        carrying = replace(dataset, static_inputs={"adjacency": adjacency})
        source = sources_for(carrying, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        assert source.static["adjacency"] is adjacency

    def test_every_split_sees_the_same_static_inputs(self, prepared):
        """
        Because they do not vary by scenario, let alone by split.

        A graph fitted on the training universe is the graph the model was
        trained with, and evaluating against a different one would measure
        a model that never existed.
        """
        dataset, spec = prepared
        carrying = replace(dataset, static_inputs={"adjacency": np.eye(3, dtype=np.float32)})
        sources = sources_for(carrying, loader=spec.loader, sequence=spec.transforms.sequence)
        statics = [source.static["adjacency"] for source in sources.values()]
        assert all(array is statics[0] for array in statics)

    def test_a_batch_carries_only_dynamic_inputs_and_the_target(self, prepared):
        """
        The deleted comparison, asserted as an absence.

        The old implementation merged static tensors into every sample and
        then ran ``torch.equal`` across the batch per static key per batch, to
        confirm what was true by construction.
        """
        dataset, spec = prepared
        source = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        batch = next(source.batches())
        assert set(batch) == {"features", TARGET_KEY}


class TestOrderedView:
    """The fix for the two-pass alignment bug."""

    def test_a_shuffling_source_offers_an_ordered_view(self, prepared):
        """
        Declared as a capability, so ``isinstance`` routes rather than guesses.

        Adding ``ordered`` to ``BatchSource`` itself would have silently
        dropped every existing structural implementation out of the protocol.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
        )
        assert isinstance(source, OrderedSource)

    def test_the_ordered_view_is_stable_across_passes(self, prepared):
        """
        The property the whole capability exists for.

        Without it, scoring pairs each prediction with another row's target;
        the counts match, every metric computes, and a correctly trained model
        reports a negative r-squared on its own training split.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
        ).ordered()
        assert np.array_equal(targets_of(source), targets_of(source))

    def test_the_ordered_view_has_the_same_samples(self, prepared):
        """
        So the fix for an alignment bug does not introduce a sampling bug.

        A stable order achieved by dropping rows would be harder to notice
        than the problem it replaced.
        """
        dataset = prepared[0]
        shuffled = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
        )
        assert np.allclose(np.sort(targets_of(shuffled.ordered())), np.sort(targets_of(shuffled)))

    def test_a_source_that_never_shuffles_returns_itself(self, prepared):
        """
        Nothing is copied when nothing needs to change.

        A held-out split is already stable, so an ordered view of it is the
        same object.
        """
        dataset = prepared[0]
        source = DatasetSource(dataset, split="test", loader=LoaderSpec(), sequence=SequenceSpec())
        assert source.ordered() is source

    def test_the_ordered_view_does_not_disturb_the_original(self, prepared):
        """
        Because a training loop may still be holding it.

        Flipping the flag in place would quietly change what the model is
        learning from, mid-run.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
        )
        source.ordered()
        assert not np.array_equal(targets_of(source), targets_of(source))


class TestSampleCounts:
    """Honest counts, including under ``drop_last``."""

    def test_dropping_the_last_batch_reduces_the_reported_samples(self, prepared):
        """
        Otherwise a metric denominator counts rows the loop never saw.

        The error would be small -- less than one batch in a split -- and so
        would never be large enough to notice.
        """
        dataset = prepared[0]
        kept = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(batch_size=32, drop_last=False),
            sequence=SequenceSpec(),
        )
        dropped = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(batch_size=32, drop_last=True),
            sequence=SequenceSpec(),
        )
        assert dropped.n_samples <= kept.n_samples
        assert dropped.n_samples == dropped.steps_per_epoch * 32

    def test_the_reported_count_matches_what_is_yielded(self, prepared):
        """The claim and the behaviour, checked against each other."""
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(batch_size=32, drop_last=True),
            sequence=SequenceSpec(),
        )
        assert targets_of(source).size == source.n_samples


class TestRefusals:
    """Loud failures where a quiet one would train on nothing."""

    def test_an_unknown_split_is_refused(self, prepared):
        """A typo should not produce an empty source."""
        dataset = prepared[0]
        with pytest.raises(ContractError):
            DatasetSource(
                dataset,
                split="validate",
                loader=LoaderSpec(),
                sequence=SequenceSpec(),
            )

    def test_a_window_wider_than_the_split_is_refused(self, prepared):
        """
        Refused loudly rather than yielding an empty source.

        The alternative is an epoch that completes instantly having trained
        on nothing, and a flat loss curve that reads as a learning-rate fault.
        """
        dataset = prepared[0]
        with pytest.raises(ContractError, match="no usable sample"):
            DatasetSource(
                dataset,
                split="test",
                loader=LoaderSpec(),
                sequence=SequenceSpec(length=10_000),
            )


class TestSourcesFor:
    """One call produces every non-empty split's source."""

    def test_one_source_per_non_empty_split(self, prepared):
        """So a caller never has to know which splits exist."""
        dataset, spec = prepared
        sources = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)
        assert set(sources) == {"train", "validation", "test"}

    def test_every_source_declares_the_dataset_signature(self, prepared):
        """
        A source does not get to invent its own interface.

        Two splits with different signatures would mean the model was served
        differently shaped data at training and scoring time.
        """
        dataset, spec = prepared
        sources = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)
        assert all(source.signature == dataset.signature for source in sources.values())
```

---

## 3. `tranql/models/rade/rade_qnet/tests/sources/batching/test_batching_rollout.py`

14431 bytes · SHA-256 `105e8d47ec66ea68`

```python
"""
Tests for the rollout batch source.

The claim this module has to earn is that an environment being acted in is
indistinguishable, from a loop's point of view, from a dataset. So the first
group tests that it satisfies ``BatchSource``, and the rest test the parts
where "just a source" is doing real work:

**Unboundedness is declared, not implied.** ``steps_per_epoch`` of ``None``
is the single value that selects the step driver over the epoch driver. A
source that reported its batch size there instead would be driven by the
wrong loop, and every metric denominator and schedule period would be wrong
with it -- silently, because both numbers are plausible.

**Episode boundaries land inside batches.** The hard part of a collector is
that an episode ends partway through a batch: the next transition belongs to
a fresh episode, the flags have to mark where, and the statistics have to
count the episode that finished rather than the batch that contained it.

**Termination and truncation stay apart.** A step limit is not a terminal
state. Conflating them biases every value estimate that bootstraps past the
boundary, which is a wrong answer rather than an error, so it is tested
directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.contract.data import TARGET_KEY
from tranql.models.rade.rade_qnet.rade_qnet.core.contract.signature import SpaceSpec
from tranql.models.rade.rade_qnet.rade_qnet.core.contract.source import BatchSource
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.sources.batching.rollout import (
    ACTION_KEY,
    NEXT_OBSERVATION_KEY,
    OBSERVATION_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    RolloutSource,
)
from tranql.models.rade.rade_qnet.rade_qnet.sources.environment import StepOutcome


class Counter:
    """
    An environment that counts up and terminates on a fixed step.

    The observation is the step index, so a test can read back exactly which
    transitions a batch contains and where an episode restarted.

    Parameters
    ----------
    episode_length
        The step on which the episode terminates.
    """

    observation_space = SpaceSpec(kind="box", shape=(1,), dtype="float32")
    action_space = SpaceSpec(kind="discrete", n=2)

    def __init__(self, episode_length: int = 3) -> None:
        self.episode_length = episode_length
        self.step_index = 0
        self.seeds: list[int | None] = []
        self.resets = 0

    def reset(self, *, seed: int | None = None):
        """
        Start a new episode, recording the seed it was given.

        Parameters
        ----------
        seed
            Recorded so a test can assert the seeding policy.

        Returns
        -------
        numpy.ndarray
            The zero observation.
        """
        self.seeds.append(seed)
        self.resets += 1
        self.step_index = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        """
        Advance one step.

        Parameters
        ----------
        action
            Recorded in the reward, so a test can tell which action produced
            which transition.

        Returns
        -------
        StepOutcome
            The next observation, with the reward carrying the action.
        """
        self.step_index += 1
        return StepOutcome(
            np.full(1, self.step_index, dtype=np.float32),
            reward=float(action),
            terminated=self.step_index >= self.episode_length,
        )


class Endless(Counter):
    """An environment with no terminal state, to exercise truncation alone."""

    def __init__(self) -> None:
        super().__init__(episode_length=10**9)


def always(action: int):
    """
    Build an action selector that always chooses one action.

    Parameters
    ----------
    action
        The action to return.

    Returns
    -------
    callable
        A selector ignoring its observation.
    """
    return lambda observation: np.int64(action)


class TestItIsJustASource:
    """The framework's central claim, tested rather than asserted."""

    def test_it_satisfies_batch_source(self):
        """A loop written against ``BatchSource`` can consume an environment."""
        source = RolloutSource(Counter(), act=always(1), batch_size=2)
        assert isinstance(source, BatchSource)

    def test_an_object_that_is_not_an_environment_is_refused_by_name(self):
        """
        The fault is named at construction, not at the first step.

        A collector that discovered this on its first ``step`` would report an
        ``AttributeError`` from inside a loop, which points at the loop rather
        than at the model's ``build_environment``.
        """
        with pytest.raises(ContractError, match="does not satisfy Environment"):
            RolloutSource(object(), act=always(0), batch_size=2)

    def test_the_signature_describes_the_batches_it_actually_yields(self):
        """
        Every declared key is present, with the declared shape.

        The signature is what a bundle records, so a signature that disagreed
        with the batches would misdescribe the run permanently.
        """
        source = RolloutSource(Counter(), act=always(1), batch_size=4)
        batch = source.collect()

        for name, spec in source.signature.dynamic.items():
            assert name in batch, f"{name} is declared but not produced"
            # The leading dimension is declared as varying, so it is the one
            # position compared loosely.
            assert np.asarray(batch[name]).shape[1:] == spec.shape[1:]

    def test_the_reward_is_the_target(self):
        """
        An interactive batch has the same shape as a supervised one.

        Keeping the reward under the shared target key is what makes the two
        paths comparable rather than merely adjacent.
        """
        source = RolloutSource(Counter(), act=always(1), batch_size=4)
        batch = source.collect()
        assert np.asarray(batch[TARGET_KEY]).tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_it_has_no_static_inputs(self):
        """An environment delivers everything through the stream."""
        assert RolloutSource(Counter(), act=always(0), batch_size=2).static == {}


class TestUnboundedness:
    """The one value that selects the loop driver."""

    def test_steps_per_epoch_is_none(self):
        """
        Declared unbounded, so the step driver is selected.

        Reporting the batch size here instead would be plausible and wrong:
        the epoch driver would accept the source and treat one batch as a
        complete pass.
        """
        assert RolloutSource(Counter(), act=always(0), batch_size=8).steps_per_epoch is None

    def test_n_samples_is_none(self):
        """There is no pass whose samples could be counted."""
        assert RolloutSource(Counter(), act=always(0), batch_size=8).n_samples is None

    def test_batches_never_run_out(self):
        """
        The generator keeps collecting, which is what "unbounded" means.

        Drained with an explicit bound rather than a ``for`` over the whole
        thing, because the whole thing does not end.
        """
        source = RolloutSource(Counter(), act=always(1), batch_size=2)
        stream = source.batches()
        collected = [next(stream) for _ in range(5)]
        assert len(collected) == 5
        assert all(np.asarray(batch[TARGET_KEY]).shape == (2,) for batch in collected)


class TestEpisodeBoundaries:
    """What happens when an episode ends partway through a batch."""

    def test_a_batch_spans_the_end_of_an_episode(self):
        """
        One batch holds the end of one episode and the start of the next.

        With a three-step episode and a five-transition batch, the
        observations acted on are steps 0, 1, 2 of the first episode and then
        0, 1 of the second -- so the counter restarts inside the batch.
        """
        environment = Counter(episode_length=3)
        source = RolloutSource(environment, act=always(1), batch_size=5)

        batch = source.collect()
        acted_on = np.asarray(batch[OBSERVATION_KEY]).reshape(-1).tolist()
        assert acted_on == [0.0, 1.0, 2.0, 0.0, 1.0]

    def test_the_terminal_flag_marks_the_boundary(self):
        """
        Exactly the terminating transition is flagged.

        This is what a bootstrapping learner reads, so a flag one position out
        would make it bootstrap past a terminal state.
        """
        source = RolloutSource(Counter(episode_length=3), act=always(1), batch_size=5)
        batch = source.collect()
        assert np.asarray(batch[TERMINATED_KEY]).tolist() == [False, False, True, False, False]

    def test_the_terminal_observation_is_kept(self):
        """
        The last ``next_observation`` of an episode is the terminal one.

        Not the reset observation of the next episode. A learner needs the
        state the agent arrived in, and overwriting it with a fresh reset
        would make every episode's final transition describe the wrong
        outcome.
        """
        source = RolloutSource(Counter(episode_length=3), act=always(1), batch_size=4)
        batch = source.collect()
        arrived_in = np.asarray(batch[NEXT_OBSERVATION_KEY]).reshape(-1).tolist()
        assert arrived_in == [1.0, 2.0, 3.0, 1.0]

    def test_episode_statistics_count_episodes_not_batches(self):
        """
        Two three-step episodes are reported, from one six-transition batch.

        The mean return is the number anybody actually asks for, and it is
        per episode -- a per-batch mean would change meaning whenever the
        batch size changed.
        """
        source = RolloutSource(Counter(episode_length=3), act=always(1), batch_size=6)
        source.collect()

        summary = source.episode_summary()
        assert summary == {"episode_return": 3.0, "episode_length": 3.0}
        assert source.describe()["episodes_finished"] == 2

    def test_no_statistics_before_the_first_episode_finishes(self):
        """
        Empty rather than zero.

        A run whose first episode is still going has no return, and reporting
        ``0.0`` would be indistinguishable from an agent earning nothing.
        """
        source = RolloutSource(Counter(episode_length=100), act=always(1), batch_size=4)
        source.collect()
        assert source.episode_summary() == {}


class TestTruncation:
    """A step limit is not a terminal state."""

    def test_the_step_limit_truncates_without_terminating(self):
        """
        The two flags disagree, which is the whole point.

        An environment with no terminal state, cut off at three steps: the
        transition is ``truncated`` and not ``terminated``, so a value
        estimate bootstraps past it rather than treating it as a failure.
        """
        source = RolloutSource(Endless(), act=always(1), batch_size=3, max_episode_steps=3)
        batch = source.collect()

        assert np.asarray(batch[TRUNCATED_KEY]).tolist() == [False, False, True]
        assert np.asarray(batch[TERMINATED_KEY]).tolist() == [False, False, False]

    def test_the_limit_restarts_the_episode(self):
        """A truncated episode is still over, so the next step is a fresh one."""
        environment = Endless()
        source = RolloutSource(environment, act=always(1), batch_size=4, max_episode_steps=2)
        batch = source.collect()

        acted_on = np.asarray(batch[OBSERVATION_KEY]).reshape(-1).tolist()
        assert acted_on == [0.0, 1.0, 0.0, 1.0]
        assert environment.resets == 2

    def test_a_truncated_episode_is_counted(self):
        """Its return is reported, or a run with no terminal state reports nothing."""
        source = RolloutSource(Endless(), act=always(1), batch_size=4, max_episode_steps=2)
        source.collect()
        assert source.episode_summary() == {"episode_return": 2.0, "episode_length": 2.0}


class TestSeeding:
    """Only the first reset is seeded."""

    def test_the_first_episode_gets_the_seed(self):
        """So a run is reproducible from its spec."""
        environment = Counter(episode_length=2)
        RolloutSource(environment, act=always(1), batch_size=1, seed=11).collect()
        assert environment.seeds == [11]

    def test_later_episodes_are_not_re_seeded(self):
        """
        Otherwise every episode would be the same episode.

        Re-seeding on each reset is an easy mistake that looks like extra
        rigour and destroys the variety the agent needs to learn from.
        """
        environment = Counter(episode_length=2)
        RolloutSource(environment, act=always(1), batch_size=5, seed=11).collect()
        assert environment.seeds == [11, None, None]

    def test_nothing_is_touched_until_the_first_collection(self):
        """
        Constructing a source does not reset the environment.

        A pipeline builds its sources before it decides what to drive them
        with, and a source that stepped at construction would make a failed
        run leave the environment half-used.
        """
        environment = Counter()
        RolloutSource(environment, act=always(1), batch_size=2)
        assert environment.resets == 0


class TestTheActionSelector:
    """The policy arrives as a callable, so that ``sources`` stays free of Torch."""

    def test_the_selector_sees_the_observation_it_acts_on(self):
        """
        Not the one that follows it.

        Off by one here would train every policy on the consequence of the
        action rather than on the state that prompted it.
        """
        seen: list[float] = []

        def record(observation):
            seen.append(float(np.asarray(observation).reshape(-1)[0]))
            return np.int64(1)

        source = RolloutSource(Counter(episode_length=100), act=record, batch_size=3)
        batch = source.collect()

        assert seen == np.asarray(batch[OBSERVATION_KEY]).reshape(-1).tolist()

    def test_the_chosen_action_is_the_one_recorded(self):
        """A batch's actions are what the selector returned, unmodified."""
        source = RolloutSource(Counter(episode_length=100), act=always(1), batch_size=3)
        batch = source.collect()
        assert np.asarray(batch[ACTION_KEY]).tolist() == [1, 1, 1]
```

