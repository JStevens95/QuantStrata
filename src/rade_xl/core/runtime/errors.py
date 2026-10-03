"""
The framework's error hierarchy.

Every error the framework raises derives from :class:`RadeXLError`, and the
subclass carries information a bare ``ValueError`` cannot: **whose fault it
is**.

That distinction is not pedantry. A :class:`SpecError` means the user wrote
something invalid and can fix it by editing their configuration. A
:class:`ContractError` means the framework or a model violated an internal
promise, and no amount of configuration will help. Collapsing both into
``ValueError`` throws that information away at exactly the moment someone is
trying to work out what to do next.
"""

from __future__ import annotations

__all__ = [
    "BundleError",
    "CapabilityError",
    "ComponentError",
    "ContractError",
    "EngineError",
    "RadeXLError",
    "SpecError",
    "StageError",
]


class RadeXLError(Exception):
    """
    Base class for every error the framework raises.

    Catching this catches everything the framework considers its own, and
    nothing else. A ``KeyError`` escaping from inside a stage is therefore
    distinguishable from a deliberate framework error -- it is a bug.
    """


class SpecError(RadeXLError, ValueError):
    """
    A specification is invalid.

    Actionable by the user: a key is misspelled, a value is out of range, or
    two settings contradict each other. Raised during validation, before any
    expensive work begins.

    Notes
    -----
    This is the one error that also derives from ``ValueError``, and the reason
    is pydantic. A cross-field validator raising a plain ``Exception`` escapes
    pydantic untouched, so it arrives without the dotted path to the offending
    field -- while a field-level failure arrives with one. The result is two
    classes of configuration error reported in two different shapes.

    Deriving from ``ValueError`` makes pydantic collect these alongside
    field-level failures, so ``parse_run_spec`` can report every problem in one
    message with every path. The dual inheritance is therefore not convenience:
    it is what makes configuration errors uniform.
    """


class ContractError(RadeXLError):
    """
    A stage contract was violated.

    Not actionable by the user -- this is a framework or model bug. Raised
    when a payload arrives with the wrong shape, a required field is absent,
    or a declared signature does not match the data produced.
    """


class CapabilityError(RadeXLError):
    """
    A model was asked for a capability it does not implement.

    Raised in preference to returning a plausible-looking result. A model with
    no inductive capability asked to predict for an unseen entity must fail,
    not guess.
    """


class ComponentError(RadeXLError):
    """
    A component could not be registered or resolved by name.

    Covers both directions: two components claiming the same name, and a
    specification naming a component that was never registered.
    """


class EngineError(RadeXLError):
    """
    An engine could not do what it was asked.

    Covers the engine's own failures and the mismatches it is the first to
    notice: a training spec belonging to a different engine, a model whose
    shapes disagree with the signature it was built from, a checkpoint whose
    parameter names do not match the model loading it.

    Distinct from :class:`ContractError` because the two have different
    audiences. A contract violation is a framework or model bug; an engine
    error is frequently a *configuration* problem the user can fix -- asking
    for a graph network on the XGBoost engine, or for a precision the device
    cannot provide.

    Notes
    -----
    An *absent* accelerator is deliberately not one of these. A run that asked
    for CUDA on a machine without it warns and continues on the CPU, because
    failing would discard a run that would otherwise have succeeded. What is
    reported here is a request that can never be honoured, not one that cannot
    be honoured here.
    """


class BundleError(RadeXLError):
    """
    A bundle could not be written, read or verified.

    Includes checksum mismatches, which is how silent corruption is turned
    into a loud failure at load time rather than into inexplicable
    predictions later.
    """


class StageError(RadeXLError):
    """
    A pipeline stage failed.

    Always names the stage, so a traceback from a forty-job run identifies
    *where* in the lifecycle the failure happened without needing to be read.
    The original exception is retained on :attr:`cause` and chained with
    ``raise ... from``, so nothing is lost by wrapping it.

    Parameters
    ----------
    stage
        Name of the stage that failed, as passed to ``Pipeline.step``.
    cause
        The exception the stage raised.
    run_id
        Identifier of the run, included in the message when available.
    """

    def __init__(self, stage: str, cause: BaseException, *, run_id: str | None = None) -> None:
        self.stage = stage
        self.cause = cause
        self.run_id = run_id
        # The run identifier is the first thing needed to find the artifacts of
        # a failed job, so it leads the message when it is known.
        prefix = f"run '{run_id}': " if run_id is not None else ""
        super().__init__(f"{prefix}stage '{stage}' failed [{type(cause).__name__}] {cause}")
