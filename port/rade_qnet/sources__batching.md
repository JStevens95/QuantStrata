# `src/rade_qnet/sources/batching`

2 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 41 | 1732 | `8e7381b320b6c155` |
| 2 | `dataset.py` | 426 | 15276 | `40393e1e40fc3e40` |

---

## 1. `src/rade_qnet/sources/batching/__init__.py`

1732 bytes · SHA-256 `8e7381b320b6c155`

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

Planned modules
---------------
``rollout.py``
    ``RolloutSource`` -- collects a fresh batch of on-policy experience from an
    environment before each update.  Policy-gradient methods.  [Phase 7]
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

## 2. `src/rade_qnet/sources/batching/dataset.py`

15276 bytes · SHA-256 `40393e1e40fc3e40`

```python
"""
``DatasetSource`` -- one split of a prepared dataset, as a ``BatchSource``.

The adapter that makes supervised learning a special case of the framework's
single training loop. It takes a :class:`~..dataset.io.PreparedDataset`, a
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
from ...core.runtime.errors import ContractError
from ...core.runtime.logging import get_logger
from ...core.runtime.seeding import derive_seed
from ..dataset.transforms.sequence import extract_windows, usable_labels

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from numpy.typing import NDArray

    from ...core.contract.data import Batch, TensorLike
    from ...core.contract.signature import InputSignature
    from ...core.spec.data import LoaderSpec, SequenceSpec
    from ..dataset.io import PreparedDataset

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

