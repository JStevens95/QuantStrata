"""
``BatchSource`` -- the one protocol every training loop consumes.

This is the framework's unification point, and the reason there is one training
loop rather than two frameworks sharing a repository.

Supervised learning and reinforcement learning differ far less than their
tooling suggests. Both consume a stream of batches. They disagree only about
where the batches come from: a fixed dataset, a freshly collected rollout, a
replay buffer, a stored transition table, or a differentiable simulator. Wrap
each of those as a ``BatchSource`` and the loop stops needing to know.

Why a protocol rather than a base class
---------------------------------------
A source satisfies this by having the right members. It need not inherit from
anything of ours and need not import this module at all, which is what lets a
source live outside this repository. It also means the five framework sources
are checked against the same definition a user's source is.

Bounded and unbounded sources
-----------------------------
:attr:`BatchSource.steps_per_epoch` returning ``None`` means "unbounded", and
that single value is what selects the loop driver. A fixed dataset has a
meaningful notion of a pass over the data, so it is driven by ``fit_epochs``.
An environment does not, so it is driven by ``fit_steps``. The source declares
which it is; the engine does not guess.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol, runtime_checkable

from .data import Batch, TensorLike
from .signature import InputSignature

__all__ = ["BatchSource", "OrderedSource"]


@runtime_checkable
class BatchSource(Protocol):
    """
    A re-iterable stream of training batches.

    Implementations must satisfy two properties that are easy to get wrong and
    both tested by the shared contract suite:

    **Re-iterable.** :meth:`batches` may be called more than once, and a
    bounded source must yield the same batches each time under a fixed seed. A
    training loop traverses a dataset once per epoch; a source backed by a bare
    generator silently yields nothing on the second epoch.

    **Honest about size.** :attr:`steps_per_epoch` is either the true number of
    batches per pass, or ``None`` for an unbounded source. It is not an
    estimate: callbacks, progress reporting and learning-rate schedules are
    computed from it.

    Note that "the same batches" constrains the *contents* of a pass, not its
    order. A training source is expected to reshuffle between epochs, which is
    the whole purpose of shuffling. Any consumer that needs two passes to line
    up row for row must say so -- see :class:`OrderedSource`.
    """

    @property
    def signature(self) -> InputSignature:
        """
        The declared interface of the batches this source yields.

        Returns
        -------
        InputSignature
            Static inputs, dynamic inputs and the target.
        """
        ...

    @property
    def static(self) -> Mapping[str, TensorLike]:
        """
        Inputs constant across every batch.

        Delivered once, outside the batch stream, so the engine can upload
        them to the device a single time. Empty for most sources.

        Returns
        -------
        Mapping
            Static inputs, keyed as declared in the signature.
        """
        ...

    @property
    def steps_per_epoch(self) -> int | None:
        """
        Batches per pass, or ``None`` if the source is unbounded.

        Returns
        -------
        int or None
            The batch count, or ``None``.
        """
        ...

    @property
    def n_samples(self) -> int | None:
        """
        Samples per pass, or ``None`` if unbounded.

        Used for metric denominators and for reports. Distinct from
        :attr:`steps_per_epoch` because a final short batch means the two are
        not related by a constant.

        Returns
        -------
        int or None
            The sample count, or ``None``.
        """
        ...

    def batches(self) -> Iterator[Batch]:
        """
        Yield batches for one pass.

        For an unbounded source this yields indefinitely, and the loop driver
        decides when to stop.

        Yields
        ------
        Batch
            A mapping from input name to tensor, carrying every dynamic input
            declared in the signature.
        """
        ...


@runtime_checkable
class OrderedSource(Protocol):
    """
    A source that can produce a stable-order view of itself.

    Why this is a separate capability
    ---------------------------------
    Evaluation needs two passes over the same split: one to run the forward
    pass and one to collect the targets to score it against. Over a training
    source those two passes are in *different orders*, because a training
    source reshuffles between epochs by design. Pairing them row for row then
    compares every prediction against some other row's target.

    The failure is specific and nasty. The sample count matches, so no shape
    check fires. Every metric computes cleanly. The numbers are simply those of
    a model predicting at random, so a correctly trained model reports a
    negative r-squared on its training split and a good one on its held-out
    splits -- a pattern that reads as a bizarre inversion of overfitting rather
    than as an alignment bug.

    Keeping this separate from :class:`BatchSource` rather than adding
    ``ordered`` to it is what stops the addition being a breaking change: a
    source satisfies ``BatchSource`` structurally, so a new required member
    would silently drop every existing implementation out of
    ``isinstance``. Declared as its own capability, ``isinstance`` *routes* --
    a source that shuffles implements it, and one that never shuffles does not
    need to.
    """

    def ordered(self) -> BatchSource:
        """
        Return a view of this source whose pass order is stable.

        Same samples, same batch boundaries, repeatable order. May return
        ``self`` when the source never shuffles.

        Returns
        -------
        BatchSource
            A source safe to traverse more than once with aligned results.
        """
        ...
