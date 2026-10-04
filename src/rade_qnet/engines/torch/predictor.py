"""
Batched inference, including the precompute path.

Training and inference ask a model for the same thing -- a forward pass --
under opposite constraints. Training updates parameters, so nothing computed
from them survives a step. Inference does not, so anything that does not vary
across batches can be computed once.

For the flagship that distinction is most of the runtime. Its graph encoder
turns a static adjacency into node embeddings, and during training that work
is unavoidable because the encoder's weights are moving. During a prediction
pass the weights are frozen and the graph does not change, so recomputing the
embeddings for every batch repeats identical arithmetic -- which on a long
evaluation pass can dominate everything else.

:class:`~rade_qnet.core.capability.protocols.Precomputable` is how a model says
so. This module is where the framework acts on the declaration, and the
critical property is that it must not change any number. A performance path
that quietly perturbs results is worse than no performance path at all,
because the discrepancy shows up as a model that scores differently in
evaluation than it did in training, with nothing pointing at the cause.

So the two routes are kept as close as possible: the same device placement,
the same static inputs, the same batch order, the same dtype on the way out.
The only difference is which method the module is called through, and
``test_precompute_matches_plain_path`` asserts the outputs are identical
rather than merely close.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...core.capability.protocols import Precomputable
from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger
from .loaders import StaticInputs, to_device_batches

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from ...core.contract.source import BatchSource
    from .hardware import ResolvedHardware

__all__ = ["predict_batches"]

_LOGGER = get_logger(__name__)


def predict_batches(
    module: torch.nn.Module,
    source: BatchSource,
    *,
    resolved: ResolvedHardware,
    allow_precompute: bool = True,
) -> NDArray[np.floating]:
    """
    Run a forward pass over a bounded source and return the raw output.

    Returns the model's own output space. Inverting the target transform is
    the pipeline's job, using the fitted state -- an engine that inverted it
    here would invert it twice, once for the pipeline and once for itself.

    Parameters
    ----------
    module
        The model, already on the right device. Put into evaluation mode
        here rather than by the caller, because a forward pass left in
        training mode would apply dropout and update batch-norm statistics,
        and the resulting predictions would be both wrong and irreproducible.
    source
        Batches to predict over. Must be bounded: an unbounded source has no
        last batch, so there is nothing to return.
    resolved
        The hardware placement, supplying the device and the autocast
        context.
    allow_precompute
        Whether to use the precompute path when the model declares it.
        Defaults to true. The escape hatch exists so a test can run both
        routes over one model and compare them, which is the only way to
        know the optimisation is safe.

    Returns
    -------
    numpy.ndarray
        Predictions in source order, one row per sample.

    Raises
    ------
    EngineError
        If the source is unbounded, or yields no batches at all.
    """
    if source.steps_per_epoch is None:
        raise EngineError(
            "predict needs a bounded source; this one reports "
            "steps_per_epoch=None, so it has no last batch and there is "
            "nothing to return"
        )

    signature = source.signature
    static = StaticInputs.from_source(source, signature=signature, device=resolved.device)

    module.eval()
    blocks: list[NDArray[np.floating]] = []

    with torch.no_grad(), resolved.autocast():
        precomputed = _precompute(module, static, enabled=allow_precompute)

        for inputs, _ in to_device_batches(
            source, signature=signature, device=resolved.device, static=static
        ):
            if precomputed is None:
                output = module(**inputs)
            else:
                output = module.forward_with_precomputed(inputs, precomputed)
            # Cast to float32 before leaving Torch: NumPy has no bfloat16, so
            # a reduced-precision tensor would fail to convert, and a float16
            # one would silently lose decimal digits of a P&L figure.
            blocks.append(output.detach().to(dtype=torch.float32).cpu().numpy())

    if not blocks:
        raise EngineError(
            f"the source yielded no batches, so there is nothing to predict "
            f"(it reports steps_per_epoch={source.steps_per_epoch})"
        )
    return np.concatenate(blocks, axis=0)


def _precompute(
    module: torch.nn.Module,
    static: StaticInputs,
    *,
    enabled: bool,
) -> Mapping[str, torch.Tensor] | None:
    """
    Compute the reusable static encoding, when the model offers one.

    Parameters
    ----------
    module
        The model, already in evaluation mode and inside ``no_grad``. Both
        matter: the encoding is computed once and reused, so it must be
        produced under exactly the conditions the batches will be.
    static
        The static inputs, already on the device.
    enabled
        Whether to try at all.

    Returns
    -------
    Mapping or None
        The precomputed tensors, or ``None`` to take the ordinary forward
        path -- which is also what a model with no static inputs gets, since
        there is nothing for it to encode.
    """
    if not enabled or not isinstance(module, Precomputable):
        return None

    tensors = static.tensors
    if not tensors:
        _LOGGER.debug(
            "%s declares precompute() but the source has no static inputs; "
            "using the ordinary forward path",
            type(module).__name__,
        )
        return None

    precomputed = module.precompute(tensors)
    _LOGGER.debug(
        "precomputed %d static tensor(s) for %s, reused across the pass",
        len(precomputed),
        type(module).__name__,
    )
    return precomputed
