"""
Giving a lazily shaped model its parameters, before anything else touches it.

**Defect 6.** A module built from ``LazyLinear`` and friends has no parameters
until it has seen one input: the shapes are inferred from the first forward
pass. The implementation this framework replaces wrapped such a model for
distributed training before that point, which hands the wrapper an empty
parameter group to synchronise. The outcomes are a crash deep inside the
distributed library, or -- considerably worse -- a wrapper that synchronises
nothing, so every rank trains its own private copy and the run reports a
plausible loss curve for a model that was never actually distributed.

The same ordering trap catches three other things, which is why this runs
first and not just before the distributed wrapper:

* an **optimiser** constructed over an empty parameter list tracks nothing,
  and ``step()`` is then a no-op that raises no error;
* a **checkpoint** of an unmaterialised module has no tensors in it, and
  loading it back succeeds while restoring nothing;
* ``torch.compile`` traces a graph whose shapes are not yet known.

So the order is fixed: materialise, then hardware, then distribute, then
optimiser. ``ARCHITECTURE.md`` §5 makes ``materialise`` its own pipeline stage
rather than a detail inside the engine, precisely so that this ordering is
visible where runs are defined and cannot be rearranged by accident.

Why a signature is enough
-------------------------
The dummy forward needs exact shapes and dtypes with no data present, and
that is the whole reason :class:`~rade_qnet.core.contract.signature.InputSignature`
exists. Synthesising the batch from the signature rather than borrowing a real
one keeps materialisation independent of the data build, which is what makes a
six-month-old bundle reconstructible: saved signature plus saved weights, no
dataset required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...core.contract.signature import TensorSpec
from ...core.lifecycle.errors import EngineError
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.signature import InputSignature, PolicySignature

__all__ = [
    "count_parameters",
    "dummy_batch",
    "has_lazy_parameters",
    "materialise",
    "materialise_policy",
    "torch_dtype",
]

_LOGGER = get_logger(__name__)

#: Batch size for the dummy forward. Two rather than one, so that a module
#: which collapses a singleton batch dimension -- or a batch-norm layer, which
#: refuses a batch of one in training mode -- is exercised honestly.
_DUMMY_BATCH_SIZE = 2

#: Spec dtype names mapped to Torch dtypes. A fixed table rather than
#: ``getattr(torch, name)``, so an unrecognised name produces a message naming
#: the supported set instead of an ``AttributeError``.
_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

#: Dtypes for which a plain ``ones`` fill is the right dummy value. Integer
#: inputs are usually indices, and zero is the one index guaranteed to be
#: inside any embedding table -- ones would be out of range for a table of
#: size one, which is exactly what a minimal test fixture has.
_INTEGER_DTYPES = frozenset({torch.int8, torch.int16, torch.int32, torch.int64, torch.bool})


def torch_dtype(name: str) -> torch.dtype:
    """
    Return the Torch dtype for a signature's dtype name.

    Parameters
    ----------
    name
        A library-agnostic dtype name, as carried by
        :class:`~rade_qnet.core.contract.signature.TensorSpec`.

    Returns
    -------
    torch.dtype
        The corresponding Torch dtype.

    Raises
    ------
    EngineError
        If the name is not recognised, listing the supported names.
    """
    try:
        return _DTYPES[name]
    except KeyError:
        raise EngineError(
            f"dtype {name!r} is not supported by the Torch engine; supported "
            f"dtypes are {sorted(_DTYPES)}"
        ) from None


def _dummy_tensor(spec: TensorSpec, *, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Build one synthetic tensor from a spec.

    Parameters
    ----------
    spec
        The declared shape and dtype.
    batch_size
        Size to substitute for each wildcard dimension.
    device
        Where to allocate.

    Returns
    -------
    torch.Tensor
        A tensor matching the spec.
    """
    shape = spec.concrete_shape(batch_size)
    dtype = torch_dtype(spec.dtype)
    # Zeros for integers and booleans, which are indices or masks: see the
    # note on `_INTEGER_DTYPES`.  Ones for floats, because a zero-filled float
    # input makes a multiplicative layer's output independent of its weights,
    # so a shape error in the weight initialisation would not show up.
    if dtype in _INTEGER_DTYPES:
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.ones(shape, dtype=dtype, device=device)


def dummy_batch(
    signature: InputSignature,
    *,
    batch_size: int = _DUMMY_BATCH_SIZE,
    device: torch.device | None = None,
    include_static: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Synthesise a batch from a signature, with no data present.

    Parameters
    ----------
    signature
        The declared interface.
    batch_size
        Size for each wildcard dimension.
    device
        Where to allocate. ``None`` means the CPU.
    include_static
        Whether to include the static inputs. False when the caller already
        holds the real static tensors and wants only the dynamic part
        synthesised -- which is the better path when they are available,
        because a graph adjacency matrix of ones is a complete graph and some
        models will not run on one.

    Returns
    -------
    dict
        Input name to synthetic tensor.
    """
    target_device = device or torch.device("cpu")
    batch = {
        name: _dummy_tensor(spec, batch_size=batch_size, device=target_device)
        for name, spec in signature.dynamic.items()
    }
    if include_static:
        batch.update(
            {
                name: _dummy_tensor(spec, batch_size=batch_size, device=target_device)
                for name, spec in signature.static.items()
            }
        )
    return batch


def has_lazy_parameters(model: torch.nn.Module) -> bool:
    """
    Return whether any parameter is still waiting for its shape.

    Parameters
    ----------
    model
        The model to inspect.

    Returns
    -------
    bool
        True if the model contains an uninitialised parameter or buffer.
    """
    uninitialised = (
        torch.nn.parameter.UninitializedParameter,
        torch.nn.parameter.UninitializedBuffer,
    )
    return any(
        isinstance(tensor, uninitialised)
        for tensor in (*model.parameters(recurse=True), *model.buffers(recurse=True))
    )


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    """
    Return the number of parameter elements.

    Zero for an unmaterialised model, which is the signal that makes defect 6
    detectable: an optimiser built over zero parameters is a silent no-op, so
    the count is asserted after materialisation rather than trusted.

    Parameters
    ----------
    model
        The model to inspect.
    trainable_only
        Count only parameters that require a gradient.

    Returns
    -------
    int
        Number of elements.
    """
    if has_lazy_parameters(model):
        return 0
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def materialise(
    model: torch.nn.Module,
    signature: InputSignature,
    *,
    static: Mapping[str, torch.Tensor] | None = None,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Run one dummy forward pass so lazy parameters acquire their shapes.

    Safe to call on an already-materialised model, in which case it returns
    immediately. That makes it callable unconditionally from the pipeline,
    which is what keeps the ordering guarantee from depending on a model
    author remembering to declare that their model is lazy.

    Parameters
    ----------
    model
        The constructed model, possibly unmaterialised.
    signature
        The declared interface, from which the dummy batch is built.
    static
        Real static tensors, when the caller has them. Preferred over
        synthetic ones: a synthetic adjacency matrix of ones is a complete
        graph, and a model that normalises by node degree may not accept one.
    device
        Where to run the pass.

    Returns
    -------
    torch.nn.Module
        The same model, materialised.

    Raises
    ------
    EngineError
        If the dummy pass fails, or if it completes without materialising the
        parameters. The second case is the quiet one: it means the model's
        forward does not route through its lazy submodules, so the run would
        proceed with an optimiser tracking nothing.
    """
    if not has_lazy_parameters(model):
        _LOGGER.debug("model has no lazy parameters; nothing to materialise")
        return model

    batch = dummy_batch(signature, device=device, include_static=not static)
    if static:
        batch.update(static)

    _LOGGER.info(
        "materialising lazy parameters with a dummy batch: %s",
        {name: tuple(tensor.shape) for name, tensor in sorted(batch.items())},
    )

    was_training = model.training
    model.eval()
    try:
        # `no_grad` because this pass exists only to fix shapes.  Without it
        # the dummy batch would build a graph and the first real backward pass
        # could be computed against synthetic activations.
        with torch.no_grad():
            model(**batch)
    except Exception as error:
        raise EngineError(
            f"the dummy forward pass used to materialise lazy parameters failed: "
            f"{type(error).__name__}: {error}. The batch was built from the input "
            f"signature:\n{signature.describe()}\nEither the signature disagrees "
            f"with what forward() accepts, or forward() needs real values rather "
            f"than synthetic ones -- in which case pass the real static inputs"
        ) from error
    finally:
        # Restored rather than left in eval mode: a model that arrived in
        # training mode must leave in training mode, or dropout and batch-norm
        # would behave differently for reasons invisible at the call site.
        model.train(was_training)

    if has_lazy_parameters(model):
        raise EngineError(
            "the dummy forward pass completed but the model still has "
            "uninitialised parameters; its forward() does not route through "
            "every lazy submodule. An optimiser built now would track nothing "
            "and training would appear to run while changing no weights"
        )

    _LOGGER.info("materialised %d parameter element(s)", count_parameters(model))
    return model


def materialise_policy(
    policy: torch.nn.Module,
    signature: PolicySignature,
    *,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Run one dummy forward pass so a lazy policy acquires its shapes.

    The interactive counterpart of :func:`materialise`, and it exists for the
    same defect: an optimiser built over unmaterialised parameters tracks
    nothing, so the run appears to train while changing no weights.

    Separate from :func:`materialise` rather than folded into it, because the
    dummy input is built from a different description. A supervised dummy
    batch is a mapping of named inputs plus a target, synthesised from an
    :class:`~rade_qnet.core.contract.signature.InputSignature`. A policy
    takes one thing -- an observation -- and its signature has no target to
    synthesise. A single function would have had to branch on which
    signature it received, which is the branch the two call sites remove.

    Parameters
    ----------
    policy
        The constructed policy, possibly unmaterialised.
    signature
        The observation and action spaces. Only the observation space is
        read: the action space constrains what the policy's *output* means,
        which a forward pass does not need to know.
    device
        Where to run the pass.

    Returns
    -------
    torch.nn.Module
        The same policy, materialised.

    Raises
    ------
    EngineError
        If the dummy pass fails, or if it completes without materialising the
        parameters.
    """
    if not has_lazy_parameters(policy):
        _LOGGER.debug("policy has no lazy parameters; nothing to materialise")
        return policy

    observation = _dummy_observation(signature, device=device)
    _LOGGER.info(
        "materialising lazy policy parameters with an observation of %s",
        tuple(observation.shape),
    )

    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            # Passed by keyword, matching the convention every model in this
            # framework is called with: a policy declares what it consumes in
            # its own signature rather than depending on positional order.
            policy(observation=observation)
    except Exception as error:
        raise EngineError(
            f"the dummy forward pass used to materialise a lazy policy failed: "
            f"{type(error).__name__}: {error}. The observation was built from the "
            f"policy signature's observation space "
            f"({signature.observation.kind}, shape {signature.observation.shape}). "
            f"Either the space disagrees with what forward() accepts, or forward() "
            f"does not take an 'observation' keyword"
        ) from error
    finally:
        policy.train(was_training)

    if has_lazy_parameters(policy):
        raise EngineError(
            "the dummy forward pass completed but the policy still has "
            "uninitialised parameters; its forward() does not route through "
            "every lazy submodule. An optimiser built now would track nothing "
            "and the run would appear to train while changing no weights"
        )

    _LOGGER.info("materialised %d parameter element(s)", count_parameters(policy))
    return policy


def _dummy_observation(
    signature: PolicySignature, *, device: torch.device | None = None
) -> torch.Tensor:
    """
    Synthesise one batch of observations from a policy signature.

    A batch of one, because the shapes a lazy module needs are fixed by the
    trailing dimensions and a larger batch would only cost more.

    Parameters
    ----------
    signature
        The policy signature, whose observation space gives shape and dtype.
    device
        Where to place the tensor.

    Returns
    -------
    torch.Tensor
        A synthetic observation with a leading batch dimension.

    Raises
    ------
    EngineError
        If the observation space names a kind this function cannot
        synthesise.
    """
    space = signature.observation
    if space.kind == "box":
        spec = TensorSpec(shape=(None, *space.shape), dtype=space.dtype)
    elif space.kind == "discrete":
        # One index per sample, not a one-hot row: a discrete observation
        # reaches a network as something to look up, and widening it here
        # would make every policy undo the widening.
        spec = TensorSpec(shape=(None,), dtype="int64")
    else:
        raise EngineError(
            f"observation space kind {space.kind!r} cannot be synthesised; "
            f"expected 'box' or 'discrete'"
        )

    # Built through the same helper the supervised dummy batch uses, so the
    # choice between zeros and ones is made in one place -- and it matters:
    # a zero-filled float input makes a multiplicative layer's output
    # independent of its weights, hiding a shape error in the initialisation.
    return _dummy_tensor(spec, batch_size=1, device=device or torch.device("cpu"))
