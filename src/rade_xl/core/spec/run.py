"""
The top-level specification: everything one run is.

:data:`RunSpec` is the single object a pipeline is constructed from, and it is
discriminated on ``task`` into a supervised run and an interactive one. The two
differ in exactly one respect -- where the training signal comes from -- which
is why they share a base holding everything else.

This module also owns loading. A specification file is the user's interface to
the framework, so parsing it, validating it and reporting a useful error when
it is wrong is a core concern rather than something each entry point
reimplements.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from ..runtime.errors import SpecError
from .base import Spec
from .data import ModelSourceSpec, SourceSpec
from .hardware import HardwareSpec
from .reports import ReportsSpec
from .training import RlTrainingSpec, TorchTrainingSpec, TrainingSpec

__all__ = [
    "ComponentRef",
    "ReinforcementRunSpec",
    "RunSpec",
    "SupervisedRunSpec",
    "dump_run_spec",
    "load_run_spec",
    "parse_run_spec",
]

#: Upper bound on a seed, matching the numerical libraries the framework wraps.
_SEED_MODULUS = 2**32


class ComponentRef(Spec):
    """
    A reference to a registered component, plus its parameters.

    Two spellings are accepted, because both are natural in YAML::

        model: ridge  # no parameters
        model: {name: ridge, alpha: 1.0}  # with parameters

    The second form keeps parameters at the top level rather than nesting them
    under ``params``, which is how a user would expect to write them. The
    ``before`` validator below normalises both spellings into this type.

    Parameters
    ----------
    name
        Registered component name, resolved through the component registry.
    params
        Parameters for the component, validated by that component's own spec
        class rather than here. This framework cannot know a model's
        hyper-parameters, so the open-endedness is confined to this one named
        field instead of being granted to the whole spec.
    """

    name: str
    params: Mapping[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, value: object) -> object:
        """
        Normalise the accepted spellings into ``{name, params}``.

        Parameters
        ----------
        value
            The raw value from the specification file.

        Returns
        -------
        object
            Either the value unchanged, or a normalised mapping.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if a mapping is given with no ``name``.
        """
        if isinstance(value, str):
            return {"name": value, "params": {}}
        if isinstance(value, Mapping) and "params" not in value:
            remaining = dict(value)
            name = remaining.pop("name", None)
            if name is None:
                raise SpecError(
                    f"a component reference needs a 'name'; received keys {sorted(remaining)}"
                )
            return {"name": name, "params": remaining}
        return value

    def describe(self) -> str:
        """
        Return a compact one-line form, for logs and reports.

        Provided so a caller never has to fall back on the pydantic repr,
        which renders as ``name='ridge' params={}`` and reads as a debugging
        artefact when it lands in a report a human is meant to read.

        Returns
        -------
        str
            For example ``ridge`` or ``ridge(alpha=1.0)``.
        """
        if not self.params:
            return self.name
        rendered = ", ".join(f"{key}={self.params[key]!r}" for key in sorted(self.params))
        return f"{self.name}({rendered})"


class _RunSpecBase(Spec):
    """
    Everything a run needs regardless of where its training signal comes from.

    Not part of the public union -- see :data:`RunSpec`.
    """

    model: ComponentRef
    name: str | None = None
    seed: int = Field(default=0, ge=0, lt=_SEED_MODULUS)
    output_root: Path = Path("artifacts/rade_xl")
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    reports: ReportsSpec = Field(default_factory=ReportsSpec)
    tags: tuple[str, ...] = ()


class SupervisedRunSpec(_RunSpecBase):
    """
    A run that learns from a fixed dataset.

    Parameters
    ----------
    task
        Discriminator.
    source
        Where the data comes from. Defaults to the model's own data module,
        which is the case for any model complex enough to need one.
    training
        Engine-specific training settings.
    """

    task: Literal["supervised"] = "supervised"
    source: SourceSpec = Field(default_factory=ModelSourceSpec)
    training: TrainingSpec = Field(default_factory=TorchTrainingSpec)

    @model_validator(mode="after")
    def _check_sequence_split_compatibility(self) -> SupervisedRunSpec:
        """
        Reject a sequential model with a split that cannot respect windows.

        Returns
        -------
        SupervisedRunSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if a sequence length above one is
            combined with an explicit split, where the framework cannot
            insert the boundary gap that
            stops a window from straddling two splits. An explicit split is
            caller-supplied, so the caller must supply indices that already
            account for the window.
        """
        sequence_length = self.source.transforms.sequence.length
        if sequence_length > 1 and self.source.split.kind == "explicit":
            raise SpecError(
                f"sequence length {sequence_length} with an explicit split: the "
                f"framework cannot insert a boundary gap into caller-supplied "
                f"indices, so a window could straddle two splits. Either supply "
                f"indices that already leave a gap of {sequence_length - 1} "
                f"scenarios, or use a chronological split"
            )
        return self


class ReinforcementRunSpec(_RunSpecBase):
    """
    A run that learns by interacting with an environment.

    Parameters
    ----------
    task
        Discriminator.
    environment
        The environment to learn in, referenced by registered name. Phase 7
        replaces this with the richer ``EnvSpec`` in
        ``rade_xl.core.spec.environment``; the shape of the reference does not
        change, so specifications written now remain valid.
    training
        Interactive training settings.
    """

    task: Literal["reinforcement"] = "reinforcement"
    environment: ComponentRef
    training: RlTrainingSpec = Field(default_factory=RlTrainingSpec)


#: A complete run, discriminated on ``task``.
RunSpec = Annotated[
    SupervisedRunSpec | ReinforcementRunSpec,
    Field(discriminator="task"),
]

#: Built once at import. Constructing a ``TypeAdapter`` compiles a validator,
#: which is expensive enough that doing it per call would be visible in a
#: forty-job run.
_RUN_SPEC_ADAPTER: TypeAdapter[SupervisedRunSpec | ReinforcementRunSpec] = TypeAdapter(RunSpec)


def _format_validation_error(error: ValidationError, *, origin: str) -> str:
    """
    Render a pydantic validation error as an actionable message.

    Pydantic's default rendering is accurate but hard to act on. This version
    leads with the dotted path to the offending field, because that is the
    thing the user has to go and edit.

    Parameters
    ----------
    error
        The validation error.
    origin
        Where the specification came from, for the message header.

    Returns
    -------
    str
        A multi-line message, one line per problem.
    """
    lines = [f"invalid specification from {origin}:"]
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "<root>"
        lines.append(f"  {location}: {detail['msg']}")
    return "\n".join(lines)


def parse_run_spec(payload: Mapping[str, Any], *, origin: str = "<mapping>") -> RunSpec:
    """
    Validate a mapping into a run specification.

    ``task`` may be omitted, in which case ``supervised`` is assumed. The
    default has to be applied here rather than declared on
    :class:`SupervisedRunSpec`, because a discriminated union reads its tag
    from the *input* before any member's defaults exist -- so a field default
    cannot serve as the union's default tag.

    Defaulting at this single entry point, rather than inside the schema, keeps
    the rule in one documented place. An unrecognised ``task`` still fails with
    pydantic's message listing the permitted values.

    Parameters
    ----------
    payload
        The raw mapping, typically parsed from YAML or JSON.
    origin
        Description of where the payload came from, used in error messages.

    Returns
    -------
    RunSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the payload is not a mapping, or if validation fails. The pydantic
        error is chained, so the full detail remains available, but the
        message is the actionable summary.
    """
    # Checked before anything else, because a YAML file whose top level is a
    # list is a common mistake and indexing it would raise a bare TypeError
    # from inside the defaulting below -- which says nothing a user can act on.
    if not isinstance(payload, Mapping):
        raise SpecError(
            f"{origin}: a run specification must be a mapping of fields, "
            f"received {type(payload).__name__}. A file whose top level is a "
            f"list is the usual cause -- remove the leading '- '."
        )
    if "task" not in payload:
        payload = {**payload, "task": "supervised"}
    try:
        return _RUN_SPEC_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise SpecError(_format_validation_error(exc, origin=origin)) from exc


def load_run_spec(path: Path | str) -> RunSpec:
    """
    Load and validate a run specification from a YAML or JSON file.

    Parameters
    ----------
    path
        Path to a ``.yaml``, ``.yml`` or ``.json`` file.

    Returns
    -------
    RunSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the file is missing, unreadable, not a mapping, or invalid. Every
        failure mode of "the user pointed us at the wrong file" is reported
        as a spec error, because in every case the fix is the user's.
    """
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"specification file not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        # YAML is a superset of JSON, so one parser handles both suffixes and
        # a mislabelled file still loads.
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"could not parse {path} as YAML or JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        received = type(payload).__name__
        raise SpecError(f"{path} must contain a mapping at the top level, found {received}")
    return parse_run_spec(payload, origin=str(path))


def dump_run_spec(spec: RunSpec, path: Path | str) -> Path:
    """
    Write a run specification to a YAML or JSON file.

    The output round-trips exactly: loading it returns a specification equal to
    the one written. That is what makes a persisted spec a faithful record of
    how a model was produced, and it is asserted by test rather than assumed.

    Parameters
    ----------
    spec
        The specification to write.
    path
        Destination. The suffix selects the format.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    SpecError
        If the suffix is not a recognised format.
    """
    path = Path(path)
    # `mode="json"` reduces Path, Enum and similar fields to their serialised
    # form, which is what makes the round trip exact rather than approximate.
    payload = spec.model_dump(mode="json")

    if path.suffix in {".yaml", ".yml"}:
        text = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    elif path.suffix == ".json":
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        raise SpecError(
            f"unsupported specification format {path.suffix!r}; use .yaml, .yml or .json"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
