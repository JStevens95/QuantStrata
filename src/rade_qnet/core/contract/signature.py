"""
Declared shapes and dtypes of a model's inputs and target.

:class:`InputSignature` is the most load-bearing contract in the framework,
because three unrelated mechanisms depend on it:

**Static input handling.** The signature says which inputs are constant across
batches, so the engine can upload them to the device once instead of collating
them into every sample of every batch.

**Lazy parameter materialisation.** A model whose shapes are known only after
the data build has no parameters until it has seen one batch. The signature
carries exactly enough information to synthesise that batch, so the model can
be materialised before an optimiser, a checkpoint or a distributed wrapper
touches it.

**Rebuilding from a bundle.** Given a saved signature and saved weights, a
model can be reconstructed months later without re-running the data build.

The static/dynamic split
------------------------
A *static* input is the same for every sample: an adjacency matrix, an entity
feature table, an index array. A *dynamic* input varies per sample: a history
window, a feature row. Keeping the two apart in the type is what allows the
engine to treat them differently without inspecting the data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_validator

from ..lifecycle.errors import ContractError, SpecError
from .base import ContractModel

__all__ = ["InputSignature", "PolicySignature", "SpaceSpec", "TensorSpec"]


class TensorSpec(ContractModel):
    """
    The shape and dtype of one tensor, named independently of any library.

    Parameters
    ----------
    shape
        Dimensions, with ``None`` marking a dimension that varies -- normally
        the batch dimension. A fully concrete shape is permitted and is what
        a static input usually has.
    dtype
        Element type, as a library-agnostic string such as ``float32`` or
        ``int64``. A string rather than a library's dtype object, because
        ``core`` may not import a training library and because a signature
        written by the Torch engine must be readable by the XGBoost one.
    description
        Optional human-readable note, surfaced in error messages and reports.
    """

    shape: tuple[int | None, ...]
    dtype: str
    description: str | None = None

    @model_validator(mode="after")
    def _check_dimensions_are_positive(self) -> TensorSpec:
        """
        Reject a shape with a non-positive concrete dimension.

        Returns
        -------
        TensorSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any concrete dimension is zero or
            negative. A zero dimension
            produces an empty tensor that trains without error and learns
            nothing, which is far harder to diagnose later than a rejection
            here.
        """
        for position, dimension in enumerate(self.shape):
            if dimension is not None and dimension <= 0:
                raise SpecError(
                    f"dimension {position} of shape {self.shape} is {dimension}; "
                    f"concrete dimensions must be positive"
                )
        return self

    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def is_compatible_with(self, other: TensorSpec) -> bool:
        """
        Return whether another spec could describe the same tensor.

        Compatibility is deliberately weaker than equality: a ``None``
        dimension matches any size. That is what lets a signature declared
        with an unknown batch size be checked against a concrete batch.

        Parameters
        ----------
        other
            The spec to compare against.

        Returns
        -------
        bool
            True if the dtypes match and every dimension pair is either equal
            or wildcarded on at least one side.
        """
        if self.dtype != other.dtype or self.rank != other.rank:
            return False
        return all(
            mine is None or theirs is None or mine == theirs
            for mine, theirs in zip(self.shape, other.shape, strict=True)
        )

    def concrete_shape(self, batch_size: int) -> tuple[int, ...]:
        """
        Resolve wildcard dimensions to a concrete shape.

        Used to synthesise the dummy batch that materialises a model's lazy
        parameters.

        Parameters
        ----------
        batch_size
            Size to substitute for every ``None`` dimension.

        Returns
        -------
        tuple of int
            A fully concrete shape.

        Raises
        ------
        SpecError
            If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise SpecError(f"batch_size must be positive, received {batch_size}")
        return tuple(batch_size if dimension is None else dimension for dimension in self.shape)

    def describe(self) -> str:
        """
        Return a compact one-line description, for error messages and reports.

        Returns
        -------
        str
            For example ``float32[?, 20, 64]``.
        """
        dimensions = ", ".join("?" if size is None else str(size) for size in self.shape)
        return f"{self.dtype}[{dimensions}]"


class InputSignature(ContractModel):
    """
    The complete declared interface between a data source and a model.

    Parameters
    ----------
    dynamic
        Inputs that vary per sample. At least one is required -- a model with
        no varying input cannot learn from data.
    static
        Inputs constant across every batch. Usually empty; non-empty for
        models that consume a graph or an entity table.
    target
        The quantity being predicted.
    """

    dynamic: Mapping[str, TensorSpec]
    static: Mapping[str, TensorSpec] = Field(default_factory=dict)
    target: TensorSpec

    @model_validator(mode="after")
    def _check_names_are_usable(self) -> InputSignature:
        """
        Reject an empty dynamic set, or a name used in both groups.

        Returns
        -------
        InputSignature
            The validated signature.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if ``dynamic`` is empty, or a name appears
            in both ``dynamic`` and ``static``. A duplicated name is unresolvable: the engine would
            have to guess whether to collate it or upload it once, and either
            choice is silently wrong half the time.
        """
        if not self.dynamic:
            raise SpecError(
                "an input signature needs at least one dynamic input; "
                "a model with only static inputs cannot vary with the data"
            )
        shared = sorted(set(self.dynamic) & set(self.static))
        if shared:
            raise SpecError(
                f"input name(s) {shared} appear in both dynamic and static; "
                f"each input must be one or the other"
            )
        return self

    @property
    def input_names(self) -> tuple[str, ...]:
        """Every input name, static and dynamic, sorted."""
        return tuple(sorted({*self.dynamic, *self.static}))

    def spec_for(self, name: str) -> TensorSpec:
        """
        Return the spec for one input, from either group.

        Parameters
        ----------
        name
            An input name.

        Returns
        -------
        TensorSpec
            The input's spec.

        Raises
        ------
        ContractError
            If the name is not declared. Listing the declared names makes the
            usual cause -- a renamed key in a data module -- immediately
            obvious.
        """
        if name in self.dynamic:
            return self.dynamic[name]
        if name in self.static:
            return self.static[name]
        raise ContractError(
            f"input {name!r} is not declared in the signature; "
            f"declared inputs are {list(self.input_names)}"
        )

    def validate_batch_keys(
        self,
        batch: Mapping[str, object],
        *,
        where: str,
        target_key: str = "target",
    ) -> None:
        """
        Check that a batch carries every dynamic input and no unknown key.

        Only keys are checked, not shapes: ``core`` cannot inspect an engine's
        tensors. Shape checking belongs to the engine, which knows what a
        tensor is. Key checking here still catches the most common failure --
        a data module and a model disagreeing on a name -- and catches it with
        a message naming both sides rather than as a dimension mismatch six
        frames into a forward pass.

        Parameters
        ----------
        batch
            A batch produced by a source.
        where
            Description of the caller, used in the error message so the
            failure names the stage that produced the bad batch.
        target_key
            Key the target is carried under, permitted but not required -- an
            inference batch legitimately has no target.

        Raises
        ------
        ContractError
            If a dynamic input is absent, or the batch carries a key that is
            neither a declared input nor the target.
        """
        keys = set(batch)
        missing = sorted(set(self.dynamic) - keys)
        if missing:
            raise ContractError(
                f"{where}: batch is missing declared dynamic input(s) {missing}; "
                f"batch provided {sorted(keys)}"
            )
        # Static inputs are permitted in a batch but not required: a source may
        # deliver them separately, which is the path that avoids collating them
        # per sample.
        permitted = {*self.dynamic, *self.static, target_key}
        unexpected = sorted(keys - permitted)
        if unexpected:
            raise ContractError(
                f"{where}: batch carries undeclared key(s) {unexpected}; "
                f"the signature declares {list(self.input_names)} and target "
                f"{target_key!r}"
            )

    def describe(self) -> str:
        """
        Return a multi-line description of the signature.

        Returns
        -------
        str
            One line per input, plus the target.
        """
        lines = [f"target: {self.target.describe()}"]
        for group, specs in (("dynamic", self.dynamic), ("static", self.static)):
            for name in sorted(specs):
                lines.append(f"{group}: {name} = {specs[name].describe()}")
        return "\n".join(lines)


class SpaceSpec(ContractModel):
    """
    An observation or action space.

    Minimal in Phase 1 and expanded by the reinforcement-learning phase. The
    shape is settled now because :class:`PolicySignature` is referenced by the
    run spec, and a contract that changes shape later invalidates every bundle
    written against it.

    Parameters
    ----------
    kind
        ``box`` for a continuous space, ``discrete`` for a finite one.
    shape
        Dimensions, for a ``box`` space.
    dtype
        Element type, for a ``box`` space.
    low, high
        Inclusive bounds, for a ``box`` space. ``None`` means unbounded.
    n
        Number of actions, for a ``discrete`` space.
    """

    kind: Literal["box", "discrete"]
    shape: tuple[int, ...] = ()
    dtype: str = "float32"
    low: float | None = None
    high: float | None = None
    n: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_fields_match_kind(self) -> SpaceSpec:
        """
        Reject fields that do not apply to the chosen kind.

        Returns
        -------
        SpaceSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if a ``discrete`` space has no ``n``, or a
            ``box`` space has one.
        """
        if self.kind == "discrete" and self.n is None:
            raise SpecError("a discrete space requires 'n', the number of actions")
        if self.kind == "box" and self.n is not None:
            raise SpecError("'n' applies to a discrete space, not a box space")
        return self


class PolicySignature(ContractModel):
    """
    The declared interface between an environment and a policy.

    The interactive counterpart of :class:`InputSignature`, serving the same
    three purposes: constructing a policy, materialising it, and rebuilding it
    from a saved bundle.

    Parameters
    ----------
    observation
        What the policy sees.
    action
        What the policy produces.
    """

    observation: SpaceSpec
    action: SpaceSpec
