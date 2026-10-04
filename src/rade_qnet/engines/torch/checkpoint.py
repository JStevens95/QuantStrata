"""
Checkpoints as tensors, never as pickled modules.

**Defect 10.** The implementation this framework replaces saved whole modules
and loaded them with ``weights_only=False``. Two separate problems come with
that, and only one of them is a security problem.

**It executes arbitrary code.** A pickle names the classes to construct and
``torch.load`` constructs them. A checkpoint is therefore executable, and a
checkpoint directory is an ordinary writable path -- a shared scratch mount, an
artifact bucket, a CI cache. Loading one is running whatever it contains with
the privileges of whoever loaded it.

**It welds the weights to the code.** A pickled module records its class by
import path, so renaming the class, moving the module or reorganising the
package breaks every checkpoint written before the change. The failure arrives
as an ``AttributeError`` during unpickling, months later, from a model that
trained fine. There is no migration path: the numbers are there and
unreachable.

A ``state_dict`` has neither problem. It is a flat mapping of names to
tensors, it loads under ``weights_only=True``, and it is reattached to a model
the caller constructs from the spec and the signature. That the caller must
supply a model is the feature: it is what guarantees the model can be rebuilt
from its specification rather than resurrected from a blob.

Saving the unwrapped model
--------------------------
Distributed and compiled wrappers prefix every parameter name --
``module.layer.weight``, ``_orig_mod.layer.weight``. A checkpoint written
through the wrapper therefore loads only into an identically wrapped model, so
a single-GPU evaluation of a model trained on four GPUs fails on every key.

:func:`save_weights` takes the handle rather than the model and writes
:attr:`~rade_qnet.engines.base.ModelHandle.unwrapped`, which is why that field
exists.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from ...core.runtime.errors import BundleError, EngineError
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ..base import ModelHandle

__all__ = [
    "WEIGHTS_FILENAME",
    "load_weights",
    "restore_state_dict",
    "save_weights",
    "snapshot_state_dict",
]

_LOGGER = get_logger(__name__)

#: On-disk filename for the tensors. Part of the format.
WEIGHTS_FILENAME = "weights.pt"

#: Sidecar describing what the weights belong to. Written for the person
#: holding an unlabelled checkpoint in six months, not for the loader -- which
#: is why a missing or stale sidecar is never fatal.
_MANIFEST_FILENAME = "weights.json"

#: Prefixes a wrapper adds to every parameter name.
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def _strip_wrapper_prefixes(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Remove a distributed or compile wrapper's prefix from every key.

    A belt-and-braces measure. :func:`save_weights` writes the unwrapped
    model, so these prefixes should never reach disk -- but a checkpoint
    written by an older version, or by a caller that bypassed the handle,
    would otherwise fail to load with an unhelpful list of several hundred
    missing keys.

    Parameters
    ----------
    state
        A state dictionary, possibly prefixed.

    Returns
    -------
    dict
        The same tensors under unprefixed names.
    """
    stripped: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        name = key
        # Looped rather than applied once: a compiled model inside a
        # distributed wrapper carries both prefixes.
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
        stripped[name] = tensor
    return stripped


def snapshot_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Return a detached CPU copy of a model's tensors.

    Copied rather than referenced, and that is the whole point of the
    function. ``model.state_dict()`` returns views of the live parameters, so
    a "best epoch" snapshot taken that way tracks the model and ends up
    holding the *last* epoch's weights -- which is how a run reports the best
    epoch's metrics beside the final epoch's weights and nothing notices.

    Moved to the CPU so that a snapshot held across an epoch does not occupy
    accelerator memory that the next epoch needs.

    Parameters
    ----------
    model
        The model to snapshot. Pass the unwrapped model, for the reason in the
        module docstring.

    Returns
    -------
    dict
        Parameter name to a detached CPU tensor.
    """
    return {
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }


def restore_state_dict(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """
    Load a snapshot back into a model, in place.

    Parameters
    ----------
    model
        The model to restore into.
    state
        A snapshot from :func:`snapshot_state_dict`.

    Raises
    ------
    EngineError
        If any key is missing or unexpected. Strict by design: Torch's
        non-strict mode would leave some layers at their initial values and
        report success, so a model would evaluate as though it had trained
        while part of it never did.
    """
    stripped = _strip_wrapper_prefixes(state)
    try:
        model.load_state_dict(stripped, strict=True)
    except RuntimeError as error:
        raise EngineError(
            f"could not restore weights into {type(model).__name__}: {error}. "
            f"The checkpoint and the model disagree on their parameters, which "
            f"usually means the model was constructed from a different "
            f"specification or signature than the one that trained it"
        ) from error


def save_weights(
    handle: ModelHandle,
    path: Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """
    Write a handle's weights to a file.

    Writes :attr:`~rade_qnet.engines.base.ModelHandle.unwrapped`, so the
    checkpoint loads into a plain model regardless of how training was
    parallelised.

    Parameters
    ----------
    handle
        The prepared model.
    path
        Destination file. Its parent is created if absent.
    metadata
        Extra annotations for the sidecar: the epoch, the monitored value.

    Raises
    ------
    EngineError
        If the unwrapped model is not a Torch module, which would mean a
        handle from another engine reached this one.
    """
    model = handle.unwrapped
    if not isinstance(model, torch.nn.Module):
        raise EngineError(
            f"the Torch engine can only checkpoint a torch.nn.Module, received "
            f"{type(model).__name__}; this handle was prepared by a different engine"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    state = snapshot_state_dict(model)
    torch.save(state, path)

    sidecar = {
        "model_class": type(model).__name__,
        "n_tensors": len(state),
        "n_elements": sum(tensor.numel() for tensor in state.values()),
        "device_trained_on": handle.device,
        "precision": handle.precision,
        "was_distributed": handle.is_distributed,
        **dict(metadata or {}),
    }
    (path.parent / _MANIFEST_FILENAME).write_text(
        json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
    )
    _LOGGER.info(
        "saved %d tensor(s), %d element(s) to %s",
        sidecar["n_tensors"],
        sidecar["n_elements"],
        path,
    )


def load_weights(model: object, path: Path) -> torch.nn.Module:
    """
    Load weights from a file into a model the caller constructed.

    The model is a parameter rather than something this function builds, and
    that asymmetry is the design: a checkpoint holds numbers, a specification
    holds the architecture, and keeping them apart is what makes a six-month-
    old bundle loadable after the code has moved on.

    Parameters
    ----------
    model
        A model built from the same spec and signature that trained it.
        Annotated ``object`` to match the engine protocol, and narrowed here.
    path
        The checkpoint file.

    Returns
    -------
    torch.nn.Module
        The same model, with the saved weights loaded.

    Raises
    ------
    BundleError
        If the file does not exist.
    EngineError
        If the model is not a Torch module, or the weights do not fit it.
    """
    if not isinstance(model, torch.nn.Module):
        raise EngineError(
            f"the Torch engine can only load into a torch.nn.Module, received "
            f"{type(model).__name__}"
        )
    if not path.is_file():
        raise BundleError(
            f"no checkpoint at {path}; the bundle may be incomplete, or the run "
            f"may have been configured with checkpoint.enabled=False"
        )

    # `weights_only=True` is the whole point of this module: it restricts
    # unpickling to tensors and plain containers, so a checkpoint cannot
    # execute code.  See the module docstring for the two reasons.
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        # Torch's own message for a rejected pickle explains how to disable
        # the safety check, which is the opposite of the advice a reader
        # should act on.  Replaced rather than chained through, so the
        # suggested fix is to regenerate the checkpoint.
        raise EngineError(
            f"the checkpoint at {path} could not be loaded as tensors "
            f"({type(error).__name__}). It appears to be a pickled object rather "
            f"than a state dictionary, which this engine will not load because "
            f"unpickling it would execute whatever the file contains. Regenerate "
            f"it by re-running the fit, or convert it on a trusted machine by "
            f"loading the module once and saving model.state_dict()"
        ) from error

    if not isinstance(state, dict):
        raise EngineError(
            f"the checkpoint at {path} holds a {type(state).__name__} rather than "
            f"a state dictionary; it must be regenerated"
        )

    restore_state_dict(model, state)
    _LOGGER.info("loaded %d tensor(s) from %s", len(state), path)
    return model
