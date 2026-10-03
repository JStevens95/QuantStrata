"""
Turning a ``BatchSource``'s NumPy batches into device-resident tensors.

The one place in the framework where a NumPy array becomes a
``torch.Tensor``. Everything upstream -- splitting, windowing, batch ordering
-- is engine-neutral, so every engine inherits one implementation of the parts
that are easy to get subtly wrong.

Static inputs leave per-sample collation
----------------------------------------
**Defect 4.** The implementation this framework replaces merged every static
tensor into every sample, and its collation function then ran ``torch.equal``
across the batch for each static key and returned ``values[0]``.

So for every batch of every epoch, a graph adjacency matrix was compared
against itself ``batch_size`` times to confirm something true by
construction. On a batch size of 64 and a 400-node graph that is 64
comparisons of a 160,000-element tensor, per batch, per epoch, to learn
nothing.

:class:`StaticInputs` replaces all of it. The static tensors are uploaded once
at the start of a fit and passed to every forward call by reference. This is
not a behavioural change and is worth being precise about why: the old
collation *already* returned exactly one copy, and the network *already*
received exactly one tensor. The same object reaches the same place; only the
comparison is gone.

Workers, and why the default is none
------------------------------------
``LoaderSpec.num_workers`` defaults to zero, meaning load in the main process.
That is the right default here and not merely a conservative one: the
framework fans a job set out across a process pool, and a worker process that
spawns its own loader workers oversubscribes the machine -- N jobs each
claiming W workers on a C-core box is NW processes competing for C cores, and
throughput collapses rather than degrading.

Because a ``BatchSource`` already yields whole batches, this module does not
construct a ``torch.utils.data.DataLoader`` at all. Prefetching across
processes would mean pickling the source, and the win for an in-memory array
is not worth the failure modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...core.contract.data import TARGET_KEY
from ...core.runtime.errors import ContractError
from ...core.runtime.logging import get_logger
from .materialise import torch_dtype

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ...core.contract.signature import InputSignature
    from ...core.contract.source import BatchSource

__all__ = ["TARGET_KEY", "StaticInputs", "to_device_batches", "to_tensor"]

_LOGGER = get_logger(__name__)


def to_tensor(
    value: object,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert one batch entry to a tensor on a device.

    Accepts a tensor as well as an array, and returns an existing tensor
    unchanged when it already has the right dtype and device. That matters for
    a static input: re-uploading a 400-node adjacency matrix on every forward
    call would reintroduce defect 4's cost by a different route.

    Parameters
    ----------
    value
        A NumPy array, a Torch tensor, or anything ``torch.as_tensor``
        accepts.
    dtype
        Target dtype. ``None`` keeps whatever the input has.
    device
        Target device. ``None`` keeps the input where it is.

    Returns
    -------
    torch.Tensor
        The converted tensor.
    """
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None and tensor.device != device:
        # `non_blocking` is deliberately not set.  It is only an advantage
        # from pinned host memory, and setting it without pinning silently
        # does nothing on some backends and returns an unready tensor on
        # others.
        tensor = tensor.to(device=device)
    return tensor


class StaticInputs:
    """
    The inputs that are the same for every batch, uploaded once.

    Parameters
    ----------
    tensors
        Static inputs, already on the target device.

    Notes
    -----
    Held in a small class rather than a bare dict so that the single upload is
    a visible event with a place to log it, and so the set can report itself in
    a run summary. An absent static set is an empty instance rather than
    ``None``, which keeps the forward call free of a branch.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor] | None = None) -> None:
        self.tensors: dict[str, torch.Tensor] = dict(tensors or {})

    @classmethod
    def from_source(
        cls,
        source: BatchSource,
        *,
        signature: InputSignature,
        device: torch.device,
    ) -> StaticInputs:
        """
        Upload a source's static inputs to a device, once.

        Parameters
        ----------
        source
            The batch source.
        signature
            The declared interface, used for the dtype of each static input.
        device
            Where to upload.

        Returns
        -------
        StaticInputs
            The uploaded set, empty when the source has none.

        Raises
        ------
        ContractError
            If the source supplies a static input the signature does not
            declare. A tensor nothing declared will be passed to ``forward``
            as a keyword argument and rejected there, with a message naming
            neither side.
        """
        supplied = dict(source.static)
        if not supplied:
            return cls()

        undeclared = sorted(set(supplied) - set(signature.static))
        if undeclared:
            raise ContractError(
                f"the source supplies static input(s) {undeclared} that the "
                f"signature does not declare; it declares "
                f"{sorted(signature.static)}"
            )

        tensors = {
            name: to_tensor(value, dtype=torch_dtype(signature.static[name].dtype), device=device)
            for name, value in supplied.items()
        }
        _LOGGER.info(
            "uploaded %d static input(s) to %s once: %s",
            len(tensors),
            device,
            {name: tuple(tensor.shape) for name, tensor in sorted(tensors.items())},
        )
        return cls(tensors)

    def __bool__(self) -> bool:
        """Return whether any static input is present."""
        return bool(self.tensors)

    def __len__(self) -> int:
        """Return how many static inputs are present."""
        return len(self.tensors)

    def merge_into(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Add the static inputs to a batch, by reference.

        Parameters
        ----------
        batch
            A batch of dynamic inputs. Mutated and returned, because this runs
            once per batch per epoch and allocating a fresh dict each time is
            measurable at small batch sizes.

        Returns
        -------
        dict
            The same dict, with the static inputs added.
        """
        batch.update(self.tensors)
        return batch

    def describe(self) -> dict[str, list[int]]:
        """
        Return the shape of each static input, for reports and logs.

        Returns
        -------
        dict
            Input name to shape.
        """
        return {name: list(tensor.shape) for name, tensor in sorted(self.tensors.items())}


def to_device_batches(
    source: BatchSource,
    *,
    signature: InputSignature,
    device: torch.device,
    static: StaticInputs | None = None,
    validate_first: bool = True,
) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """
    Convert one pass of a source into device-resident tensor batches.

    Yields the inputs and the target separately rather than as one mapping,
    because the learner needs them apart -- the inputs go to ``forward`` as
    keyword arguments and the target goes to the loss -- and splitting them
    here means the learner never has to know which key the target uses.

    Parameters
    ----------
    source
        The batch source.
    signature
        The declared interface, supplying each input's dtype.
    device
        Where to place the tensors.
    static
        Already-uploaded static inputs. ``None`` means none; pass the result
        of :meth:`StaticInputs.from_source` to avoid re-uploading per epoch.
    validate_first
        Whether to key-check the first batch against the signature. Only the
        first: the check is cheap but not free, and a source that produced a
        correctly keyed first batch and a differently keyed tenth one is a
        failure mode worth catching in the conformance suite rather than on
        every batch of every epoch.

    Yields
    ------
    tuple
        The inputs as a mapping, and the target.

    Raises
    ------
    ContractError
        If a batch carries no target, or -- on the first batch -- disagrees
        with the signature's declared keys.
    """
    statics = static if static is not None else StaticInputs()
    dtypes = {name: torch_dtype(spec.dtype) for name, spec in signature.dynamic.items()}
    target_dtype = torch_dtype(signature.target.dtype)

    for index, batch in enumerate(source.batches()):
        if index == 0 and validate_first:
            signature.validate_batch_keys(batch, where="to_device_batches", target_key=TARGET_KEY)
        if TARGET_KEY not in batch:
            raise ContractError(
                f"batch {index} carries no {TARGET_KEY!r}; a training batch must "
                f"carry the target. An inference stream without one belongs in "
                f"predict(), not here"
            )

        inputs = {
            name: to_tensor(value, dtype=dtypes.get(name), device=device)
            for name, value in batch.items()
            if name != TARGET_KEY
        }
        target = to_tensor(batch[TARGET_KEY], dtype=target_dtype, device=device)
        yield statics.merge_into(inputs), target
