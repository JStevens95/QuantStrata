# `tranql/models/rade/rade_qnet/rade_qnet/sources/batching`

3 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 42 | 1770 | `705aee492f36f269` |
| 2 | `dataset.py` | 426 | 15290 | `e9bf1a479fecbcb8` |
| 3 | `rollout.py` | 494 | 17912 | `a23e2fda7fa4f1bb` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/sources/batching/__init__.py`

1770 bytes · SHA-256 `705aee492f36f269`

```python
"""
Adapters that present any source as a uniform stream of batches.

Each module here implements ``BatchSource``, the one protocol a training loop
consumes.  The loop therefore never branches on whether it is doing supervised
learning, on-policy reinforcement learning, off-policy reinforcement learning
or pathwise optimisation -- it asks for batches and applies the learner's
update rule.

Batches are NumPy, not tensors
------------------------------
``sources`` may import ``core`` and nothing else, so no module here can build
an engine's native tensor.  That is deliberate: batch ordering, window
arithmetic and split membership are identical for every engine, so they are
written once against NumPy and every engine inherits the same implementation.
Conversion to the native type belongs to the engine's loader.

Modules
-------
``dataset.py``
    ``DatasetSource`` -- iterates one split of a prepared dataset.  Supervised
    learning.  [Phase 2]
``rollout.py``
    ``RolloutSource`` -- collects a fresh batch of on-policy experience from an
    environment before each update.  Unbounded, which is what selects the
    ``fit_steps`` driver.  [Phase 7]

Planned modules
---------------
``replay.py``
    ``ReplaySource`` -- maintains a buffer and samples from it, uniformly or by
    priority.  Off-policy value-based methods.  [Phase 7]
``offline.py``
    ``OfflineSource`` -- serves batches from a stored transition table, with no
    live environment.  Offline reinforcement learning.  [Phase 7]
``simulation.py``
    ``SimulationSource`` -- draws batches of simulated paths from a
    differentiable environment, keeping the computation graph intact so the loss
    can be back-propagated through the dynamics.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `tranql/models/rade/rade_qnet/rade_qnet/sources/batching/dataset.py`

15290 bytes · SHA-256 `e9bf1a479fecbcb8`

```python
"""
``DatasetSource`` -- one split of a prepared dataset, as a ``BatchSource``.

The adapter that makes supervised learning a special case of the framework's
single training loop. It takes a :class:`~..dataset.cache.PreparedDataset`, a
split name and a :class:`~rade_qnet.core.spec.data.LoaderSpec`, and presents them
as the same protocol a live environment presents.

Why the batches are NumPy and not tensors
-----------------------------------------
``sources`` may import ``core`` and nothing else, so this module cannot
construct a ``torch.Tensor`` even if it wanted to. That restriction is doing
useful work rather than getting in the way: the split logic, the window
arithmetic and the batch ordering are identical for every engine, and writing
them once against NumPy means the XGBoost engine inherits a correct
implementation instead of a second one.

Converting to the engine's native type is
:mod:`rade_qnet.engines.torch.loaders`' job, and it is the only place that
conversion happens.

Shuffling here cannot move a scenario between splits
----------------------------------------------------
**Defect 3.** The implementation this framework replaces drove both the split
and the batch order from one ``shuffle`` flag, so ``shuffle=True`` -- the
natural choice for training throughput -- silently turned a chronological
split of a time series into a random one. Nothing raised, and the validation
score improved.

Here the split has already happened: this source is constructed *from* a
decided :class:`~rade_qnet.core.contract.data.SplitIndices`, and shuffling
permutes the labels within one split. There is no code path by which it could
do anything else, which is a stronger guarantee than a test -- though
``test_shuffle_cannot_change_split_membership`` asserts it anyway.

Shuffling applies to training only. Permuting a validation split changes
nothing about the metric and makes two runs' per-batch logs needlessly
incomparable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.data import SPLIT_NAMES, TARGET_KEY
from ...core.lifecycle.errors import ContractError
from ...core.provenance.logging import get_logger
from ...core.provenance.seeding import derive_seed
from ..dataset.transforms.sequence import extract_windows, usable_labels

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from numpy.typing import NDArray

    from ...core.contract.data import Batch, TensorLike
    from ...core.contract.signature import InputSignature
    from ...core.spec.data import LoaderSpec, SequenceSpec
    from ..dataset.cache import PreparedDataset

__all__ = ["TARGET_KEY", "DatasetSource", "sources_for"]

_LOGGER = get_logger(__name__)

#: The split whose batch order may be shuffled. See the module docstring.
_SHUFFLED_SPLIT = "train"


class DatasetSource:
    """
    A re-iterable stream of batches over one split.

    Satisfies :class:`~rade_qnet.core.contract.source.BatchSource` structurally,
    without inheriting from it -- which is the point of that protocol, and
    means this class is checked against the same definition a user's source is.

    Parameters
    ----------
    dataset
        The prepared dataset. Held whole rather than sliced to the split,
        because a sequence window ending at the split's first label needs
        rows from before it, and those rows belong to the discarded boundary
        gap rather than to any split.
    split
        Which split to serve.
    loader
        Batch size, order and worker settings.
    sequence
        Window length and stride.
    seed
        Base seed for the batch order.

    Raises
    ------
    ContractError
        If the split name is unrecognised, if the split is absent from the
        dataset, or if the split has no sample that can form a complete
        window. The last case is the one worth refusing loudly: it happens
        when a sequence length exceeds a split's width, and the alternative is
        a loop that completes every epoch instantly having trained on nothing.
    """

    def __init__(
        self,
        dataset: PreparedDataset,
        *,
        split: str,
        loader: LoaderSpec,
        sequence: SequenceSpec,
        seed: int = 0,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ContractError(f"unknown split {split!r}; expected one of {list(SPLIT_NAMES)}")

        self.dataset = dataset
        self.split = split
        self.loader = loader
        self.sequence = sequence
        self.seed = seed

        self._labels = usable_labels(
            dataset.splits[split],
            length=sequence.length,
            stride=sequence.stride,
            confine=sequence.confine_to_split,
        )
        if self._labels.size == 0:
            raise ContractError(
                f"the {split} split yields no usable sample with sequence length "
                f"{sequence.length} and stride {sequence.stride}: it has "
                f"{dataset.splits[split].size} scenario(s), and a window of length "
                f"{sequence.length} needs at least that many. Shorten the sequence "
                f"or widen the split"
            )

        self._input_name = self._single_dynamic_input(dataset.signature)
        # Counts how many passes have been served, so each epoch gets a
        # different-but-reproducible order.  See `batches`.
        self._pass_index = 0

    # -- BatchSource -------------------------------------------------------

    @property
    def signature(self) -> InputSignature:
        """
        The declared interface of the batches this source yields.

        Returns
        -------
        InputSignature
            The dataset's signature, unchanged. A source does not get to
            narrow it: the model declares what it consumes, and narrowing here
            would hide a genuine disagreement between the two.
        """
        return self.dataset.signature

    @property
    def static(self) -> Mapping[str, TensorLike]:
        """
        Inputs constant across every batch.

        Empty for a tabular dataset. Non-empty for a model whose data module
        produces a graph, which is what keeps those tensors out of per-sample
        collation -- defect 4.

        Returns
        -------
        Mapping
            Static inputs, keyed as the signature declares.
        """
        return self.dataset.static_inputs

    @property
    def steps_per_epoch(self) -> int:
        """
        Batches per pass.

        Never ``None``: a fixed dataset has a meaningful notion of a pass, and
        reporting a real count is what lets the engine drive this with
        ``fit_epochs`` rather than ``fit_steps``.

        Returns
        -------
        int
            Number of batches.
        """
        full, remainder = divmod(self._labels.size, self.loader.batch_size)
        return full if self.loader.drop_last else full + (1 if remainder else 0)

    @property
    def n_samples(self) -> int:
        """
        Samples actually served in one pass.

        With ``drop_last`` set this is *not* the split's size: the samples in
        the discarded short batch are never seen. Reporting the split size
        instead would make every mean metric divide by a denominator larger
        than the number of terms, understating the error by a few percent on a
        small split.

        Returns
        -------
        int
            Number of samples.
        """
        if not self.loader.drop_last:
            return int(self._labels.size)
        return self.steps_per_epoch * self.loader.batch_size

    def __iter__(self) -> Iterator[Batch]:
        """
        Yield one pass of batches, so this is also an ``Iterable[Batch]``.

        Satisfying both :class:`~rade_qnet.core.contract.source.BatchSource` and
        ``Iterable[Batch]`` is what lets one object be stored in
        :attr:`~rade_qnet.core.contract.data.TensorBatchData.loader` and handed
        straight to an engine's ``fit``. Without it the pipeline would have to
        keep the sources alongside the bundle and the two could drift apart.

        Returns
        -------
        Iterator
            A fresh iterator over one pass.
        """
        return self.batches()

    def batches(self) -> Iterator[Batch]:
        """
        Yield one pass of batches.

        Returns a fresh iterator on every call, which is what makes the source
        re-iterable: a training loop traverses it once per epoch, and a source
        that stored a generator would train in the first epoch and on nothing
        afterwards.

        The training split's order is permuted with a seed derived from the
        base seed, the split name and the pass index. Deriving it rather than
        reusing the base seed directly means two epochs see different orders
        -- which is the point of shuffling -- while the whole sequence of
        orders is still reproducible from the run's one seed.

        Yields
        ------
        Batch
            A mapping with the dynamic input and the target.
        """
        order = self._order()
        self._pass_index += 1

        batch_size = self.loader.batch_size
        for start in range(0, order.size, batch_size):
            labels = order[start : start + batch_size]
            if self.loader.drop_last and labels.size < batch_size:
                continue
            yield {
                self._input_name: self._features_for(labels),
                TARGET_KEY: self.dataset.target[labels].astype(np.float32, copy=False),
            }

    # -- OrderedSource -----------------------------------------------------

    def ordered(self) -> DatasetSource:
        """
        Return an equivalent source whose pass order is stable.

        Satisfies :class:`~rade_qnet.core.contract.source.OrderedSource`, which
        evaluation uses to pair predictions with targets across two passes.
        See that class for what goes wrong without it.

        Returns
        -------
        DatasetSource
            ``self`` when this source does not shuffle, and otherwise a new
            source over the same dataset and split with shuffling off. A new
            object rather than a flag flipped in place, because the training
            loop may still be holding this one -- turning its shuffling off
            underneath it would quietly change what the model is learning
            from.
        """
        if not (self.loader.shuffle and self.split == _SHUFFLED_SPLIT):
            return self
        return DatasetSource(
            self.dataset,
            split=self.split,
            loader=self.loader.model_copy(update={"shuffle": False}),
            sequence=self.sequence,
            seed=self.seed,
        )

    # -- Internals ---------------------------------------------------------

    def _order(self) -> NDArray[np.int64]:
        """
        Return the labels for one pass, in the order they will be served.

        Returns
        -------
        numpy.ndarray
            Permuted labels for the training split, ascending otherwise.
        """
        if not (self.loader.shuffle and self.split == _SHUFFLED_SPLIT):
            return self._labels

        seed = derive_seed(self.seed, "batch_order", self.split, str(self._pass_index))
        # Permuting the labels, never the splits: the labels came from one
        # split and a permutation cannot introduce a member of another.
        return np.random.default_rng(seed).permutation(self._labels)

    def _features_for(self, labels: NDArray[np.int64]) -> NDArray[np.float32]:
        """
        Return the feature block for a batch of labels.

        Parameters
        ----------
        labels
            Scenario indices, one per sample, each the *last* scenario of its
            window.

        Returns
        -------
        numpy.ndarray
            Samples by features for a non-sequential model, or samples by
            window by features for a sequential one.
        """
        if self.sequence.length == 1:
            # A length of one means a non-sequential model.  Windowing would
            # add an axis of size one that the signature does not declare, so
            # the rows are taken directly instead.
            block = self.dataset.features[labels]
        else:
            block = extract_windows(self.dataset.features, labels, length=self.sequence.length)
        return block.astype(np.float32, copy=False)

    @staticmethod
    def _single_dynamic_input(signature: InputSignature) -> str:
        """
        Return the name of the signature's one dynamic input.

        Parameters
        ----------
        signature
            The dataset's signature.

        Returns
        -------
        str
            The input name.

        Raises
        ------
        ContractError
            If the signature declares more than one dynamic input. This source
            serves a single feature block, so it cannot decide which of several
            declared inputs that block is. A model with several dynamic inputs
            needs its own batching, which is Phase 3's work.
        """
        names = sorted(signature.dynamic)
        if len(names) != 1:
            raise ContractError(
                f"DatasetSource serves one feature block, but the signature "
                f"declares {len(names)} dynamic input(s) {names}; a model with "
                f"several dynamic inputs needs its own BatchSource"
            )
        return names[0]

    def describe(self) -> dict[str, object]:
        """
        Return a summary for reports and logs.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {
            "split": self.split,
            "n_samples": self.n_samples,
            "steps_per_epoch": self.steps_per_epoch,
            "batch_size": self.loader.batch_size,
            "shuffled": self.loader.shuffle and self.split == _SHUFFLED_SPLIT,
            "sequence_length": self.sequence.length,
        }


def sources_for(
    dataset: PreparedDataset,
    *,
    loader: LoaderSpec,
    sequence: SequenceSpec,
    seed: int = 0,
) -> dict[str, DatasetSource]:
    """
    Build one source per non-empty split.

    The function the train pipeline calls, so that "which splits exist" is
    answered once. Empty splits are skipped rather than represented by an
    empty source, because an empty source is indistinguishable from a broken
    one at the point a loop consumes it.

    Parameters
    ----------
    dataset
        The prepared dataset.
    loader
        Batch size, order and worker settings.
    sequence
        Window length and stride.
    seed
        Base seed for the batch order.

    Returns
    -------
    dict
        Split name to source, in canonical split order.
    """
    sources: dict[str, DatasetSource] = {}
    for name in SPLIT_NAMES:
        if dataset.splits[name].size == 0:
            continue
        sources[name] = DatasetSource(
            dataset, split=name, loader=loader, sequence=sequence, seed=seed
        )
        _LOGGER.debug("built %s source: %s", name, sources[name].describe())
    return sources
```

---

## 3. `tranql/models/rade/rade_qnet/rade_qnet/sources/batching/rollout.py`

17912 bytes · SHA-256 `a23e2fda7fa4f1bb`

```python
"""
Fresh on-policy experience, presented as an ordinary batch source.

This is the module that makes the framework's central claim concrete. A
training loop consumes :class:`~rade_qnet.core.contract.source.BatchSource`
and nothing else; a dataset satisfies it, and so does an environment being
acted in. The loop therefore does not branch on whether it is doing
supervised or interactive learning -- it asks for batches.

Unbounded is the whole point
----------------------------
:attr:`RolloutSource.steps_per_epoch` is ``None``, and that single value is
what selects the loop driver: ``fit_epochs`` for a bounded dataset,
``fit_steps`` for this. The rule is stated in
:mod:`rade_qnet.core.contract.source` and it is the source that declares
which it is, so the engine never guesses. There is no meaningful notion of a
pass over an environment, and a source that invented one -- reporting, say,
the batch size as a step count -- would silently corrupt every callback,
progress figure and learning-rate schedule computed from it.

Why the policy arrives as a callable
------------------------------------
``sources`` may import ``core`` and nothing else, which means this module
cannot touch a policy: a policy is an engine-native object -- a
``torch.nn.Module`` here -- and importing one would break the layering that
lets a spec be parsed on a host with no training library installed.

So the caller passes ``act``, a plain callable from one observation to one
action. The engine owns it, because selecting an action is where the engine's
library genuinely appears: it is a forward pass, under ``no_grad``, with
whatever exploration the learner wants. What is left here is the part that is
identical for every engine -- stepping, resetting, episode bookkeeping and
stacking -- written once against NumPy, exactly as the package charter
requires.

Why the reward is the batch's target
------------------------------------
:class:`~rade_qnet.core.contract.signature.InputSignature` requires a target,
and for a rollout the reward is the honest answer: it is the quantity the
batch exists to explain. Naming it under
:data:`~rade_qnet.core.contract.data.TARGET_KEY` rather than inventing a
``"reward"`` key means an interactive batch has the same shape as a
supervised one, which is what keeps the two paths comparable rather than
merely adjacent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.data import TARGET_KEY
from ...core.contract.signature import InputSignature, PolicySignature, TensorSpec
from ...core.lifecycle.errors import ContractError
from ...core.provenance.logging import get_logger
from ..environment.protocol import Environment

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from ...core.contract.data import Batch, TensorLike
    from ...core.contract.signature import SpaceSpec

__all__ = [
    "ACTION_KEY",
    "NEXT_OBSERVATION_KEY",
    "OBSERVATION_KEY",
    "TERMINATED_KEY",
    "TRUNCATED_KEY",
    "RolloutSource",
]

_LOGGER = get_logger(__name__)

#: What the policy saw.
OBSERVATION_KEY = "observation"

#: What it did.
ACTION_KEY = "action"

#: What it saw next. Carried because a bootstrapping learner needs the value
#: of the state it arrived in, and recomputing it by shifting the observation
#: column is wrong at every episode boundary -- which is precisely where a
#: value estimate matters most.
NEXT_OBSERVATION_KEY = "next_observation"

#: Whether the task itself ended.
TERMINATED_KEY = "terminated"

#: Whether the episode was cut short from outside the task. Separate from
#: :data:`TERMINATED_KEY` for the reason given on
#: :class:`~rade_qnet.sources.environment.protocol.StepOutcome`.
TRUNCATED_KEY = "truncated"

#: Episode statistics a collector accumulates, reported through
#: :meth:`RolloutSource.episode_summary` so a run has something interpretable
#: to show. A mean loss says nothing about an agent; a mean episode return is
#: the number anybody actually asks for.
_RETURN_KEY = "episode_return"
_LENGTH_KEY = "episode_length"


class RolloutSource:
    """
    An unbounded batch source that collects experience by acting.

    Satisfies :class:`~rade_qnet.core.contract.source.BatchSource`.

    Parameters
    ----------
    environment
        The environment to act in.
    act
        Chooses an action for one observation. Supplied by the engine, for
        the layering reason in the module docstring.

        May be ``None`` at construction and supplied later through
        :meth:`bind`, because the two cannot always be built in that order: a
        learner is constructed from this source's own
        :attr:`policy_signature`, so a pipeline that had to pass a selector
        up front would need the learner before the source it is built from.
        A source with no selector refuses to collect rather than acting
        arbitrarily.
    batch_size
        Transitions per batch.
    seed
        Seeds the environment's first reset. Subsequent resets do not
        re-seed, so episodes differ from one another -- re-seeding every
        episode would make them all the same episode.
    max_episode_steps
        Truncate an episode after this many steps, or ``None`` for no limit.
        Worth having because an environment with no terminal state and no
        limit yields one infinite episode, from which no episode statistic is
        ever reported and no bootstrapping learner ever sees a boundary.

    Raises
    ------
    ContractError
        If ``environment`` does not satisfy
        :class:`~rade_qnet.sources.environment.protocol.Environment`, named
        here rather than failing on the first ``step``.
    """

    def __init__(
        self,
        environment: Environment,
        *,
        act: Callable[[TensorLike], TensorLike] | None = None,
        batch_size: int,
        seed: int = 0,
        max_episode_steps: int | None = None,
    ) -> None:
        self.environment = self._checked(environment)
        self.act = act
        self.batch_size = batch_size
        self.seed = seed
        self.max_episode_steps = max_episode_steps

        self._observation: TensorLike | None = None
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._returns: list[float] = []
        self._lengths: list[int] = []

    @staticmethod
    def _checked(environment: Environment) -> Environment:
        """
        Refuse an object that is not an environment.

        Parameters
        ----------
        environment
            The candidate.

        Returns
        -------
        Environment
            The same object.

        Raises
        ------
        ContractError
            If it lacks any member of the protocol.
        """
        if not isinstance(environment, Environment):
            raise ContractError(
                f"{type(environment).__name__} does not satisfy Environment; it "
                f"needs observation_space, action_space, reset() and step(). "
                f"A model's build_environment() must return one"
            )
        return environment

    @property
    def signature(self) -> InputSignature:
        """
        The declared interface of the batches this source yields.

        Returns
        -------
        InputSignature
            Observations, actions, next observations and the two episode-end
            flags as dynamic inputs, with the reward as the target.
        """
        observation = _spec_for(self.environment.observation_space)
        action = _spec_for(self.environment.action_space)
        flag = TensorSpec(shape=(None,), dtype="bool", description="episode-end flag")
        return InputSignature(
            dynamic={
                OBSERVATION_KEY: observation,
                ACTION_KEY: action,
                NEXT_OBSERVATION_KEY: observation,
                TERMINATED_KEY: flag,
                TRUNCATED_KEY: flag,
            },
            target=TensorSpec(shape=(None,), dtype="float32", description="step reward"),
        )

    def bind(self, act: Callable[[TensorLike], TensorLike]) -> None:
        """
        Supply the action selector, replacing any already set.

        Called by the engine once it has resolved the learner that acts.
        Separate from the constructor because of the ordering described under
        the ``act`` parameter, and a method rather than a bare attribute
        assignment so that the hand-off is a named, searchable event rather
        than something a reader has to notice.

        Parameters
        ----------
        act
            Chooses an action for one observation.
        """
        self.act = act
        _LOGGER.debug("bound an action selector to the rollout source")

    @property
    def policy_signature(self) -> PolicySignature:
        """
        The declared interface of the *policy*, as distinct from the batches.

        Two signatures, because the source sits between two things that need
        different descriptions. :attr:`signature` describes the experience it
        yields, in tensor terms, which is what a loop and a bundle read.
        This describes the spaces, which is what a learner needs to turn a
        network's output into an action the environment will accept -- and a
        tensor description cannot serve: it loses the number of discrete
        actions and the bounds of a continuous space.

        Returns
        -------
        PolicySignature
            The environment's two declared spaces.
        """
        return PolicySignature(
            observation=self.environment.observation_space,
            action=self.environment.action_space,
        )

    @property
    def static(self) -> Mapping[str, TensorLike]:
        """
        Inputs constant across every batch.

        Returns
        -------
        Mapping
            Always empty: an environment has no static inputs, and a policy
            that needs one gets it from its own construction rather than
            from the experience stream.
        """
        return {}

    @property
    def steps_per_epoch(self) -> None:
        """
        Always ``None``, because the source is unbounded.

        Returns
        -------
        None
            Declares this source as the kind ``fit_steps`` drives. See the
            module docstring.
        """
        return None

    @property
    def n_samples(self) -> None:
        """
        Always ``None``, because the source is unbounded.

        Returns
        -------
        None
            There is no pass whose samples could be counted.
        """
        return None

    def batches(self) -> Iterator[Batch]:
        """
        Yield batches of freshly collected experience, indefinitely.

        Re-iterable in the sense the protocol requires -- calling this again
        continues collecting -- but deliberately *not* reproducible batch for
        batch across calls, because the policy improves between them and
        re-collecting identical experience would mean the source had ignored
        it. The reproducibility that matters for an interactive run is the
        seeded environment plus the seeded policy, not a fixed batch
        sequence.

        Yields
        ------
        Batch
            ``batch_size`` transitions, stacked along the leading axis.
        """
        while True:
            yield self.collect()

    def collect(self) -> Batch:
        """
        Collect exactly one batch of transitions.

        Separate from :meth:`batches` so a driver that wants one update's
        worth of experience can ask for it directly, which is what
        ``fit_steps`` does. The generator is then a thin loop over this
        rather than the other way around.

        Returns
        -------
        Batch
            One batch, keyed as the signature declares.

        Raises
        ------
        ContractError
            If no action selector has been bound.
        """
        observations: list[TensorLike] = []
        actions: list[TensorLike] = []
        next_observations: list[TensorLike] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []

        if self.act is None:
            raise ContractError(
                "this rollout source has no action selector, so it cannot "
                "collect: something must call bind() with the learner that "
                "acts on the policy. An interactive engine does this when it "
                "resolves the learner"
            )

        for _ in range(self.batch_size):
            observation = self._current_observation()
            action = self.act(observation)
            outcome = self.environment.step(action)

            self._episode_steps += 1
            self._episode_reward += float(outcome.reward)
            cut_short = outcome.truncated or self._reached_the_step_limit()

            observations.append(observation)
            actions.append(action)
            next_observations.append(outcome.observation)
            rewards.append(float(outcome.reward))
            terminated.append(bool(outcome.terminated))
            truncated.append(bool(cut_short))

            if outcome.terminated or cut_short:
                self._close_episode()
            else:
                self._observation = outcome.observation

        return {
            OBSERVATION_KEY: np.stack([np.asarray(value) for value in observations]),
            ACTION_KEY: np.stack([np.asarray(value) for value in actions]),
            NEXT_OBSERVATION_KEY: np.stack([np.asarray(value) for value in next_observations]),
            TARGET_KEY: np.asarray(rewards, dtype=np.float32),
            TERMINATED_KEY: np.asarray(terminated, dtype=np.bool_),
            TRUNCATED_KEY: np.asarray(truncated, dtype=np.bool_),
        }

    def episode_summary(self) -> dict[str, float]:
        """
        Return the mean return and length of the episodes finished so far.

        The interpretable half of an interactive run. A learner's loss is not
        comparable between algorithms and barely comparable between runs; a
        mean episode return is the number that answers "is this agent any
        good". Reported per update by the driver and folded into the run's
        metrics.

        Returns
        -------
        dict
            Mean return and mean length, or empty when no episode has
            finished yet -- empty rather than zero, because a run whose first
            episode is still going has no return, and reporting ``0.0`` would
            be indistinguishable from an agent earning nothing.
        """
        if not self._returns:
            return {}
        return {
            _RETURN_KEY: float(np.mean(self._returns)),
            _LENGTH_KEY: float(np.mean(self._lengths)),
        }

    def _current_observation(self) -> TensorLike:
        """
        Return the observation to act on, resetting if no episode is running.

        Returns
        -------
        TensorLike
            The current observation.
        """
        if self._observation is None:
            # Seeded only on the very first reset. See the ``seed`` parameter.
            first = self.seed if not self._returns and self._episode_steps == 0 else None
            self._observation = self.environment.reset(seed=first)
        return self._observation

    def _reached_the_step_limit(self) -> bool:
        """
        Whether the running episode has hit its step limit.

        Returns
        -------
        bool
            True when a limit is configured and has been reached.
        """
        return self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps

    def _close_episode(self) -> None:
        """Record the finished episode's statistics and arm the next reset."""
        self._returns.append(self._episode_reward)
        self._lengths.append(self._episode_steps)
        _LOGGER.debug(
            "episode finished: return=%.4f length=%d", self._episode_reward, self._episode_steps
        )
        self._episode_reward = 0.0
        self._episode_steps = 0
        # Cleared rather than reset here, so the next collect() resets lazily
        # and a source that is never iterated never touches the environment.
        self._observation = None

    def describe(self) -> dict[str, object]:
        """
        Return a short description, for logs and reports.

        Returns
        -------
        dict
            Batch size, the step limit, and how many episodes have finished.
        """
        return {
            "kind": "rollout",
            "batch_size": self.batch_size,
            "max_episode_steps": self.max_episode_steps,
            "episodes_finished": len(self._returns),
        }


def _spec_for(space: SpaceSpec) -> TensorSpec:
    """
    Describe one space as the tensor a batch of it will be.

    Parameters
    ----------
    space
        An observation or action space.

    Returns
    -------
    TensorSpec
        The batched shape, with a leading ``None`` for the batch dimension.

    Raises
    ------
    ContractError
        If the space names a kind this function has not been taught, which
        would otherwise produce a signature that silently misdescribes the
        batches.
    """
    if space.kind == "box":
        return TensorSpec(shape=(None, *space.shape), dtype=space.dtype)
    if space.kind == "discrete":
        # One index per sample, not a one-hot row: an index is what an
        # environment accepts and what a discrete policy emits, and widening
        # it here would make every learner undo the widening.
        return TensorSpec(shape=(None,), dtype="int64")
    raise ContractError(
        f"space kind {space.kind!r} has no tensor description; expected 'box' or 'discrete'"
    )
```

