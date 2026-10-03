"""
Pick the pipeline class a model wants for a given lifecycle stage.

Why this module exists
----------------------
A model declares its overrides on its framework definition::

    class HybridGnnRnnModel:
        pipelines = MappingProxyType({"train": HybridTrainPipeline, ...})

That declaration had, until this module, no reader. Every override was
reachable only by importing the subclass and instantiating it by hand,
which is what the phase examples did -- so the overrides worked, were
tested, and were nevertheless unreachable through ``api.train`` and every
other documented entry point. A user following the documentation got the
framework's base pipeline and a flagship model quietly missing its graph
diagnostics.

That is the worst shape a defect can take: the feature exists, its tests
pass, and the only thing wrong is that nothing connects it. This module is
the connection, and it is deliberately one function so there is exactly one
place where the lookup can be got wrong.

Why the override must be a subclass
------------------------------------
The caller has already decided which lifecycle it is running, and has a
contract with that pipeline's return type: ``api.evaluate`` promises an
:class:`~rade_xl.core.contract.result.EvaluationResult`, and a model that
returned something else would break a caller that never asked for a custom
pipeline in the first place. Requiring a subclass is what makes "the model
customises training" different from "the model replaces training".

The check also catches the likeliest mistake by far, which is a model
listing a class under the wrong key -- a tuning pipeline under ``"eval"``
reads perfectly well in a dictionary literal and is nonsense at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.runtime.errors import ComponentError
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["pipeline_for"]

_LOGGER = get_logger(__name__)

#: The lifecycle keys a model may declare. Checked so that a typo -- a
#: model declaring ``"evaluate"`` where the framework reads ``"eval"`` --
#: is reported rather than silently ignored, which would present as the
#: override simply not running.
LIFECYCLES: frozenset[str] = frozenset({"train", "eval", "infer", "tune"})


def pipeline_for[PipelineT: type](
    definition: object, lifecycle: str, default: PipelineT
) -> PipelineT:
    """
    Return the pipeline class to run, honouring the model's override.

    Parameters
    ----------
    definition
        The model's framework definition, or ``None`` when the caller has
        none -- a search over a model that is resolved per trial, for
        instance. A definition that declares no overrides is as ordinary
        as one that declares some.
    lifecycle
        Which pipeline is wanted: ``train``, ``eval``, ``infer`` or
        ``tune``.
    default
        The framework's own pipeline for that lifecycle, returned when the
        model declares no override.

    Returns
    -------
    type
        The override if the model declares one, otherwise ``default``.

    Raises
    ------
    ComponentError
        If the model declares an override under an unknown lifecycle key,
        or one that is not a subclass of the framework's pipeline.
    """
    if lifecycle not in LIFECYCLES:
        raise ComponentError(
            f"{lifecycle!r} is not a lifecycle; expected one of {sorted(LIFECYCLES)}"
        )

    overrides: Mapping[str, type] = getattr(definition, "pipelines", None) or {}
    unknown = set(overrides) - LIFECYCLES
    if unknown:
        raise ComponentError(
            f"{type(definition).__name__} declares pipeline override(s) under "
            f"{sorted(unknown)}, which no lifecycle reads. The override would "
            f"never run, and the model would appear to work while silently "
            f"using the framework's pipeline. Expected keys: {sorted(LIFECYCLES)}"
        )

    override = overrides.get(lifecycle)
    if override is None:
        return default

    if not (isinstance(override, type) and issubclass(override, default)):
        raise ComponentError(
            f"{type(definition).__name__} declares {override!r} for the "
            f"{lifecycle!r} lifecycle, but it is not a subclass of "
            f"{default.__name__}. A pipeline that does not extend the "
            f"framework's own has no obligation to return what the caller was "
            f"promised, which turns a model's customisation into a broken "
            f"contract for every caller that never asked for one"
        )

    _LOGGER.debug(
        "using %s's %r pipeline override %s",
        type(definition).__name__,
        lifecycle,
        override.__name__,
    )
    return override
