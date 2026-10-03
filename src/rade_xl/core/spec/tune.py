"""
Declaring a hyper-parameter search.

A search is a base run specification plus a space to vary, a budget, and a
rule for deciding which trial won. None of those fit in a
:data:`~.run.RunSpec`, which describes *one* run -- so this is a separate
document, and a trial is the result of folding one point in the space into
the base.

Why the overrides stay raw until the merge
-------------------------------------------
Exactly as in :mod:`~.jobs`: a search dimension names a dotted path into a
run specification, and the value it proposes is merged in before anything is
validated. Validating a fragment on its own is not possible -- half a
specification is not a specification -- and the merged document is validated
in full, so nothing escapes.

The difference from a job set is who writes the values. A job set's overrides
are typed by a person; a search's are generated. That makes the validation
*more* important rather than less, because a generated value that lands on a
path nobody spelled correctly would otherwise be explored, measured, and
reported as making no difference.

Defect 7
--------
``rade_ml_pt``'s ``TunePipeline._resolve_trial_training_config`` passed
``learning_rate`` and ``batch_size`` straight into
``dataclasses.replace(TrainingConfig, ...)``, which raises ``TypeError``
because neither is a field of that class. It never fired in production only
because the hybrid model overrode the method -- so the framework's own tuning
path was broken for every model that did not.

The cure is structural rather than a fix: a trial's overrides go through the
same validated merge a job's do, so an unknown path fails at trial
construction, before a single epoch runs, with a message naming the path.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from .base import Spec
from .merge import deep_merge
from .run import RunSpec, parse_run_spec

__all__ = [
    "Dimension",
    "SearchSpace",
    "TuneSpec",
    "load_tune_spec",
    "parse_tune_spec",
]

#: Directions an objective can be optimised in. Spelled out rather than
#: inferred from the metric's name: ``r2`` is maximised and ``mae`` minimised,
#: and a framework that guessed from a substring would eventually guess wrong
#: on a custom metric and silently select the worst trial in the search.
Direction = Literal["minimise", "maximise"]


class Dimension(Spec):
    """
    One axis of a search, and the values it may take.

    Parameters
    ----------
    path
        A dotted path into a run specification, such as
        ``training.learning_rate`` or ``model.params.hidden_size``. Dotted
        rather than nested so that a space reads as a flat list of knobs,
        which is how anyone thinks about one.
    values
        The values to choose among, for a categorical or discrete axis.
    low
        Lower bound, for a continuous axis.
    high
        Upper bound, for a continuous axis.
    log
        Whether to sample a continuous axis on a logarithmic scale. Almost
        always correct for a learning rate or a regularisation strength,
        where the interesting range spans orders of magnitude and a uniform
        sample would spend nine tenths of its budget in the top decade.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if the axis is neither a clean enumeration
        nor a clean range.
    """

    path: str = Field(min_length=1)
    values: tuple[Any, ...] = ()
    low: float | None = None
    high: float | None = None
    log: bool = False

    @model_validator(mode="after")
    def _reject_an_ambiguous_axis(self) -> Dimension:
        """
        Require exactly one of an enumeration or a range, and a sane range.

        Returns
        -------
        Dimension
            The validated dimension.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError``. An axis with both would leave the
            sampler to decide which to honour, and an axis with neither has
            nothing to propose -- in both cases the search would run to
            completion and report a result, which is why this is rejected
            at parse time rather than handled at sample time.
        """
        low, high = self.low, self.high
        bounded = low is not None and high is not None

        if bool(self.values) == bounded:
            described = "both an enumeration and a range" if self.values else "neither"
            raise SpecError(
                f"dimension {self.path!r} gives {described}; it must give either "
                f"'values' or both 'low' and 'high'"
            )

        if not bounded:
            if self.log:
                raise SpecError(
                    f"dimension {self.path!r} sets log=True but enumerates its "
                    f"values; a scale only applies to a range"
                )
            return self

        # Narrowed by `bounded`, and re-read from the locals so the narrowing
        # is visible to a reader and to a type checker alike.
        assert low is not None and high is not None
        if low >= high:
            raise SpecError(
                f"dimension {self.path!r} has low={low} and high={high}; the "
                f"range is empty, so every trial would propose the same value"
            )
        if self.log and low <= 0:
            raise SpecError(
                f"dimension {self.path!r} is sampled logarithmically but has "
                f"low={low}; a log scale is undefined at or below zero"
            )
        return self

    @property
    def is_continuous(self) -> bool:
        """
        Return whether this axis is a range rather than an enumeration.

        Returns
        -------
        bool
            True for a range.
        """
        return self.low is not None


class SearchSpace(Spec):
    """
    The axes a search varies, and nothing else.

    A type of its own rather than a bare list, so that the checks which
    belong to a space as a whole -- no path proposed twice -- have somewhere
    to live.

    Parameters
    ----------
    dimensions
        The axes. At least one: a search over nothing is a single run, and
        should be written as one.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if a path appears more than once.
    """

    dimensions: tuple[Dimension, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_duplicate_paths(self) -> SearchSpace:
        """
        Reject a space that varies the same path twice.

        Returns
        -------
        SearchSpace
            The validated space.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError``. Two axes over one path means the
            second silently wins every merge, so half the search budget is
            spent exploring a dimension that has no effect -- and the
            report would show it having none, which reads as a finding
            rather than a bug.
        """
        counts = Counter(dimension.path for dimension in self.dimensions)
        repeated = sorted(path for path, count in counts.items() if count > 1)
        if repeated:
            raise SpecError(
                f"a search space must not vary the same path twice; repeated: "
                f"{repeated}. The later axis would win every merge, so the "
                f"earlier one would be explored and have no effect"
            )
        return self

    @property
    def paths(self) -> tuple[str, ...]:
        """
        Return every path this space varies, in declaration order.

        Returns
        -------
        tuple of str
            The dotted paths.
        """
        return tuple(dimension.path for dimension in self.dimensions)


class TuneSpec(Spec):
    """
    A base run, a space to search over it, and how the winner is chosen.

    Parameters
    ----------
    base
        A run-specification fragment every trial starts from. Unvalidated on
        its own, for the reason in the module docstring: a base that omits
        whatever the space supplies is a legitimate base, and validating it
        alone would reject it.
    space
        The axes to vary.
    trials
        How many to run.
    objective
        The metric to optimise.
    direction
        Whether that metric is better high or low.
    objective_split
        Which split the objective is read from. Validation by default, which
        is the only defensible choice: selecting on test turns the held-out
        split into part of the training procedure, and the reported test
        metric then overstates the model.
    sampler
        How points are proposed. ``random`` samples independently; ``grid``
        enumerates the product of the axes, which requires every axis to be
        an enumeration.
    seed
        Seed for the proposal sequence, so a search is reproducible.
    refit
        Whether to retrain the winner on train and validation combined once
        the search is over. Off by default, because it produces a model
        whose reported objective was measured on data it has now seen, and
        that should be an explicit choice.
    name
        A label for the search.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if a grid search is asked to enumerate a
        continuous axis.
    """

    base: Mapping[str, Any] = Field(default_factory=dict)
    space: SearchSpace
    trials: int = Field(default=20, ge=1)
    objective: str = "mae"
    direction: Direction = "minimise"
    objective_split: str = "validation"
    sampler: Literal["random", "grid"] = "random"
    seed: int = Field(default=0, ge=0)
    refit: bool = False
    name: str | None = None

    @model_validator(mode="after")
    def _reject_a_grid_over_a_range(self) -> TuneSpec:
        """
        Reject a grid over an axis that is a range rather than a list.

        Returns
        -------
        TuneSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError``. The alternative -- discretising a
            range into some number of steps the user did not choose --
            would make the search silently coarser than it reads, and the
            number of steps would be a framework decision affecting the
            result.
        """
        if self.sampler != "grid":
            return self
        continuous = sorted(
            dimension.path for dimension in self.space.dimensions if dimension.is_continuous
        )
        if continuous:
            raise SpecError(
                f"a grid search cannot enumerate the continuous axis(es) {continuous}. "
                f"Either list their values explicitly or use the random sampler; "
                f"discretising a range into steps you did not choose would make "
                f"the search coarser than it reads"
            )
        return self

    @property
    def grid_size(self) -> int:
        """
        Return how many distinct points a grid over this space contains.

        Returns
        -------
        int
            The product of the axis lengths, or zero for a space with a
            continuous axis, which has no finite grid.
        """
        if any(dimension.is_continuous for dimension in self.space.dimensions):
            return 0
        size = 1
        for dimension in self.space.dimensions:
            size *= len(dimension.values)
        return size

    def run_spec_for(self, overrides: Mapping[str, Any], *, trial: int) -> RunSpec:
        """
        Fold one proposal into the base and validate the result.

        The cure for defect 7, and the only place a trial becomes a real run
        specification -- so the merge rules, the validation and the error
        message live in one spot, as they do for a job set.

        Parameters
        ----------
        overrides
            A nested mapping, already expanded from the dotted paths.
        trial
            The trial number, named in errors. A validation failure against
            a merged document is otherwise very hard to trace back to the
            proposal that caused it, and in a search nobody typed the
            proposal.

        Returns
        -------
        RunSpec
            The validated specification for that trial.

        Raises
        ------
        SpecError
            If the merged specification is invalid.
        """
        merged = deep_merge(self.base, overrides)
        return parse_run_spec(merged, origin=f"trial {trial}")

    def is_better(self, candidate: float, incumbent: float) -> bool:
        """
        Return whether one objective value beats another.

        One implementation of the comparison, so that the direction is
        honoured identically wherever a winner is picked. A search that
        compared with ``<`` in one place and ``>`` in another would select
        correctly for half its configurations.

        Parameters
        ----------
        candidate
            The new value.
        incumbent
            The best so far.

        Returns
        -------
        bool
            True if the candidate is better. Ties go to the incumbent, which
            makes the winner the *earliest* best trial and therefore stable
            under a re-run.
        """
        if self.direction == "maximise":
            return candidate > incumbent
        return candidate < incumbent


def parse_tune_spec(payload: Mapping[str, Any], *, origin: str = "<mapping>") -> TuneSpec:
    """
    Validate a mapping into a tuning specification.

    Accepts two conveniences the schema does not, both because they are how
    a person writes a search rather than how it is structured:

    - ``model`` at the top level, folded into ``base``. A search is "this
      model, over these knobs", and burying the model inside ``base`` reads
      as though it were incidental.
    - ``space`` as a plain mapping of path to axis, so a space can be
      written as a dictionary of knobs rather than a list of objects with a
      repeated ``path`` key.

    Parameters
    ----------
    payload
        The raw mapping, normally from YAML.
    origin
        Where it came from, named in errors.

    Returns
    -------
    TuneSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the mapping is not a valid tuning specification.
    """
    fields = dict(payload)

    model = fields.pop("model", None)
    if model is not None:
        base = dict(fields.get("base") or {})
        base["model"] = {"name": model} if isinstance(model, str) else model
        fields["base"] = base

    space = fields.get("space")
    if isinstance(space, Mapping) and "dimensions" not in space:
        fields["space"] = {
            "dimensions": [
                {"path": path, **_axis(path, axis)} for path, axis in space.items()
            ]
        }

    try:
        return TuneSpec.model_validate(fields)
    except SpecError:
        raise
    except ValueError as error:
        raise SpecError(f"{origin} is not a valid tuning specification: {error}") from error


def load_tune_spec(path: Path | str) -> TuneSpec:
    """
    Load and validate a tuning specification from a YAML or JSON file.

    Parameters
    ----------
    path
        Path to a ``.yaml``, ``.yml`` or ``.json`` file.

    Returns
    -------
    TuneSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the file is missing, unparseable, not a mapping, or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"tuning specification file not found: {path}")

    try:
        # YAML is a superset of JSON, so one parser handles both suffixes and
        # a mislabelled file still loads.
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"could not parse {path} as YAML or JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise SpecError(
            f"{path} holds a {type(payload).__name__} at the top level; a tuning "
            f"specification must be a mapping"
        )
    return parse_tune_spec(payload, origin=str(path))


def _axis(path: str, axis: object) -> Mapping[str, Any]:
    """
    Normalise one entry of the shorthand space mapping.

    Parameters
    ----------
    path
        The dotted path, named in errors.
    axis
        Either a mapping of axis fields, or a bare sequence of values --
        which is the shortest honest way to write a categorical knob and is
        what a user reaches for first.

    Returns
    -------
    Mapping
        Axis fields, without the path.

    Raises
    ------
    SpecError
        If the entry is neither.
    """
    if isinstance(axis, Mapping):
        return {key: value for key, value in axis.items() if key != "path"}
    if isinstance(axis, Sequence) and not isinstance(axis, str | bytes):
        return {"values": tuple(axis)}
    raise SpecError(
        f"the search axis for {path!r} is a {type(axis).__name__}; it must be "
        f"a list of values or a mapping with 'values' or 'low' and 'high'"
    )
