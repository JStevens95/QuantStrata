"""
Proposing points in a search space.

Separated from the pipeline that consumes them because proposing and
evaluating are independent concerns, and keeping them apart means a sampler
can be tested exhaustively without training anything -- which matters, since
the properties worth asserting about a sampler (it is reproducible, it stays
in bounds, it does not repeat a grid point) are all cheap to check and all
expensive to discover from a search that merely finished.

Reproducibility
---------------
Every sampler draws from a seeded generator created once per search, not per
trial. A per-trial generator seeded by the trial number would also be
reproducible, and would be worse: the sequences for adjacent trials would be
correlated in whatever way the underlying algorithm correlates adjacent
seeds, and a search would explore less of the space than its budget suggests
while looking exactly like one that explored more.

Dotted paths
------------
A dimension names ``training.learning_rate``; a run specification nests. The
expansion happens here, once, in :func:`expand`, and the result goes through
the same validated merge a job set's overrides do -- so a path that does not
exist fails at trial construction rather than being explored as though it
were a real knob. That is the structural cure for defect 7.
"""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

from ...core.lifecycle.errors import SpecError
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from ...core.spec.tune import Dimension, TuneSpec

__all__ = ["expand", "propose"]

_LOGGER = get_logger(__name__)


def propose(spec: TuneSpec) -> list[dict[str, Any]]:
    """
    Produce the flat proposals for a whole search, in trial order.

    Produced up front rather than one at a time. A search that drew its next
    point only when the previous one finished would be a prerequisite for
    adaptive sampling, which this does not do -- and producing them all now
    buys two things that matter more today: the proposals can be logged and
    compared before anything trains, and a grid search can say honestly how
    many distinct points it actually has.

    Parameters
    ----------
    spec
        The search specification.

    Returns
    -------
    list of dict
        One flat mapping of dotted path to value per trial.
    """
    if spec.sampler == "grid":
        return _grid(spec)
    return _random(spec)


def expand(flat: Mapping[str, Any]) -> dict[str, Any]:
    """
    Turn a flat mapping of dotted paths into the nested shape a spec expects.

    Parameters
    ----------
    flat
        Dotted path to value, such as ``{"training.learning_rate": 0.01}``.

    Returns
    -------
    dict
        The nested equivalent, such as
        ``{"training": {"learning_rate": 0.01}}``.

    Raises
    ------
    SpecError
        If two paths disagree about whether a segment is a mapping --
        ``training`` and ``training.learning_rate`` in the same space, say.
        One of the two would overwrite the other depending on iteration
        order, so the search would explore a different space on a different
        day.
    """
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        segments = path.split(".")
        cursor = nested
        for segment in segments[:-1]:
            existing = cursor.setdefault(segment, {})
            if not isinstance(existing, dict):
                raise SpecError(
                    f"the search path {path!r} needs {segment!r} to be a mapping, "
                    f"but another path in the same space sets it to a value. "
                    f"Vary one or the other, not both"
                )
            cursor = existing
        leaf = segments[-1]
        if isinstance(cursor.get(leaf), dict):
            raise SpecError(
                f"the search path {path!r} sets a value, but another path in the "
                f"same space treats it as a mapping. Vary one or the other"
            )
        cursor[leaf] = value
    return nested


def _random(spec: TuneSpec) -> list[dict[str, Any]]:
    """
    Draw independent points from the space.

    Parameters
    ----------
    spec
        The search specification.

    Returns
    -------
    list of dict
        One proposal per trial.
    """
    # One generator for the whole search, for the reason in the module
    # docstring.
    generator = np.random.default_rng(spec.seed)
    proposals = [
        {dimension.path: _draw(dimension, generator) for dimension in spec.space.dimensions}
        for _ in range(spec.trials)
    ]
    _LOGGER.info(
        "proposed %d random trial(s) over %d dimension(s): %s",
        len(proposals),
        len(spec.space.dimensions),
        list(spec.space.paths),
    )
    return proposals


def _grid(spec: TuneSpec) -> list[dict[str, Any]]:
    """
    Enumerate the product of the axes, truncated to the trial budget.

    Parameters
    ----------
    spec
        The search specification.

    Returns
    -------
    list of dict
        One proposal per grid point, at most ``spec.trials`` of them.
    """
    points: Iterator[tuple[Any, ...]] = product(
        *(dimension.values for dimension in spec.space.dimensions)
    )
    proposals = [
        dict(zip(spec.space.paths, values, strict=True))
        for _, values in zip(range(spec.trials), points, strict=False)
    ]

    total = spec.grid_size
    if total > spec.trials:
        # Said rather than silently truncated. A grid that ran two thirds of
        # its points and reported a best trial reads exactly like a complete
        # one, and the conclusion drawn from it would be wrong in a way
        # nothing in the output hints at.
        _LOGGER.warning(
            "the grid holds %d point(s) but the budget is %d trial(s); the last "
            "%d point(s) will not be explored, so the reported best trial is the "
            "best of a partial grid",
            total,
            spec.trials,
            total - spec.trials,
        )
    else:
        _LOGGER.info("proposed the complete grid of %d point(s)", len(proposals))
    return proposals


def _draw(dimension: Dimension, generator: np.random.Generator) -> object:
    """
    Draw one value from one axis.

    Parameters
    ----------
    dimension
        The axis.
    generator
        The search's generator.

    Returns
    -------
    object
        A value from the enumeration, or a float from the range.
    """
    if not dimension.is_continuous:
        return _choice(dimension.values, generator)

    low, high = float(dimension.low or 0.0), float(dimension.high or 0.0)
    if dimension.log:
        # Uniform in the exponent, so each decade gets an equal share of the
        # budget. Sampling uniformly in the value instead would put nine
        # tenths of the trials in the top decade of a range like 1e-4 to
        # 1e-1, which is almost never what the range was meant to express.
        return float(math.exp(generator.uniform(math.log(low), math.log(high))))
    return float(generator.uniform(low, high))


def _choice(values: Sequence[object], generator: np.random.Generator) -> object:
    """
    Pick one of an enumeration's values.

    Indexed rather than passed to ``Generator.choice`` directly, because that
    coerces its input to an array -- which would turn an integer axis into
    ``numpy.int64`` and a mixed axis into strings. Either would then be
    merged into a specification and fail validation for a reason that has
    nothing to do with what the user wrote.

    Parameters
    ----------
    values
        The enumeration.
    generator
        The search's generator.

    Returns
    -------
    object
        One value, with its original Python type.
    """
    return values[int(generator.integers(len(values)))]
