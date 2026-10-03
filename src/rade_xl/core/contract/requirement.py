"""
What a model consumes, declared by the model rather than inferred from data.

The gap this closes
-------------------
:class:`~rade_xl.core.contract.signature.InputSignature` describes what a
data build *produced*. Until this module existed there was nothing
describing what a model *requires*, so the information only ever flowed one
way -- data to model -- and the model had to cope at runtime with whatever
arrived.

Coping looks reasonable and fails quietly. The recurrent baseline used to
read its window like this::

    for value in inputs.values():
        if isinstance(value, Tensor):
            return value

Handed two feature blocks, that trains on whichever one the data build's
dictionary happened to yield first. Reordering the build changes the model
and nothing reports it: no exception, no warning, a plausible loss curve
and a different model. The same shape of defect produced the
``torch.atleast_3d`` bug, where a wrongly placed time axis gave a tensor
the recurrence accepted happily and learned nonsense from.

Both are the same root cause. The model knew what it needed and had no way
to say so, so the framework could not check it.

What a requirement constrains
------------------------------
Names and ranks, which are the two properties a data build can get wrong
while still producing something that runs. Dtype is optional, and exact
shapes are deliberately not expressible: a model that works at any feature
width should not have to restate its width, and pinning one here would
duplicate a number the signature already carries and make the two able to
disagree.

Unnamed requirements
---------------------
A model like ``ridge`` or ``lstm_tabular`` is generic over what its input
is *called* -- the name is the source's choice. Such a model declares an
unnamed requirement, which matches any one input that no named requirement
claimed. One per group, because two anonymous requirements would be
indistinguishable and the matching would depend on iteration order, which
is the defect this module exists to prevent.

Examples
--------
A model that accepts whatever it is given::

    REQUIRES = InputRequirement.unconstrained()

A model needing exactly one dynamic input of rank 2 or 3, under any name::

    REQUIRES = InputRequirement(
        dynamic=(RequiredInput(rank=(2, 3), description="the feature window"),),
    )

A model needing specific inputs by name::

    REQUIRES = InputRequirement(
        dynamic=(RequiredInput(name="pnl_history", rank=3),),
        static=(RequiredInput(name="adjacency_values", rank=1),),
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import field_validator, model_validator

from ..runtime.errors import ContractError, SpecError
from .base import ContractModel
from .signature import TensorSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .signature import InputSignature

__all__ = ["InputRequirement", "RequiredInput"]


class RequiredInput(ContractModel):
    """
    One input a model consumes.

    Parameters
    ----------
    rank
        Acceptable numbers of axes, counting the batch axis. A tuple rather
        than a single integer because accepting more than one rank is a
        real and reasonable position -- the recurrent baseline treats a
        rank-2 input as a window of length one -- and because writing that
        down is better than discovering it from a reshape buried in a
        forward pass. A bare integer is accepted and widened.
    name
        The batch key. ``None`` means the model does not care what the
        input is called, which is the right declaration for a model that
        should run against whatever a user already has.
    dtype
        Required element type, as a library-agnostic string. ``None``
        leaves it unconstrained, which is the common case: an engine that
        casts on the way to the device makes the data build's choice
        irrelevant.
    shape
        Optional refinement of ``rank``, with ``None`` for any axis whose
        size the model does not care about. ``(None, None, 2)`` says "rank
        3, and the last axis is a pair" -- which is the useful case: a
        model that reads edge endpoints needs exactly two of them, and that
        is a genuine requirement rather than a restatement of the data.

        Most models should leave this unset. Pinning a feature width here
        duplicates a number the signature already carries, and two copies
        of a number can disagree.
    description
        What the model does with this input. Surfaced in the error message
        when the requirement is not met, where it is usually the thing that
        tells a user which of their columns belongs here.
    """

    rank: tuple[int, ...]
    name: str | None = None
    dtype: str | None = None
    shape: tuple[int | None, ...] | None = None
    description: str | None = None

    @field_validator("rank", mode="before")
    @classmethod
    def _widen_a_bare_rank(cls, value: object) -> object:
        """
        Accept ``rank=3`` as shorthand for ``rank=(3,)``.

        Parameters
        ----------
        value
            Whatever was supplied.

        Returns
        -------
        object
            A tuple if the input was a bare integer, otherwise unchanged.
        """
        return (value,) if isinstance(value, int) else value

    @model_validator(mode="after")
    def _check_ranks_are_usable(self) -> RequiredInput:
        """
        Reject an empty or non-positive rank set.

        Returns
        -------
        RequiredInput
            The validated requirement.

        Raises
        ------
        ValidationError
            Wrapping a :class:`SpecError`. An empty tuple would be a
            requirement nothing can satisfy, and a rank of zero would
            describe a scalar with no batch axis -- both are far more
            likely to be a typo than an intention.
        """
        if not self.rank:
            raise SpecError("a required input must accept at least one rank")
        for value in self.rank:
            if value < 1:
                raise SpecError(
                    f"rank {value} is not usable; every input has at least a "
                    f"batch axis, so the minimum rank is 1"
                )
        # A shape whose length is not an acceptable rank is a requirement
        # nothing can satisfy. Caught here rather than at check time, so the
        # author of the model hears about it instead of their user.
        if self.shape is not None and len(self.shape) not in self.rank:
            raise SpecError(
                f"shape {self.shape} has rank {len(self.shape)}, which is not "
                f"among the accepted ranks {self.rank}"
            )
        return self

    def describe(self) -> str:
        """
        Return a compact one-line description for error messages.

        Returns
        -------
        str
            For example ``'pnl_history' rank 3, float32 -- the window``.
        """
        name = f"{self.name!r}" if self.name is not None else "<any name>"
        ranks = " or ".join(str(value) for value in self.rank)
        parts = [f"{name} rank {ranks}"]
        if self.dtype is not None:
            parts.append(self.dtype)
        if self.description is not None:
            parts.append(f"-- {self.description}")
        return " ".join(parts)

    def accepts(self, spec: TensorSpec) -> str | None:
        """
        Return why this requirement rejects a tensor spec, or ``None``.

        A reason rather than a boolean, because the caller's job is to
        produce an error a user can act on, and "rank 2, expected 3" is
        actionable in a way that "incompatible" is not.

        Parameters
        ----------
        spec
            What the data build declared for this input.

        Returns
        -------
        str or None
            ``None`` if the spec satisfies the requirement.
        """
        if spec.rank not in self.rank:
            ranks = " or ".join(str(value) for value in self.rank)
            return f"has rank {spec.rank}, but rank {ranks} is required"
        if self.dtype is not None and spec.dtype != self.dtype:
            return f"has dtype {spec.dtype}, but {self.dtype} is required"
        if self.shape is not None:
            # Delegated to TensorSpec, which already implements exactly this
            # comparison: equal dtypes, equal ranks, and every dimension pair
            # either equal or wildcarded on at least one side. The dtype is
            # borrowed from the actual spec when the requirement does not
            # constrain it, so that the shape check stays a shape check.
            declared = TensorSpec(shape=self.shape, dtype=self.dtype or spec.dtype)
            if not declared.is_compatible_with(spec):
                return (
                    f"has shape {spec.describe()}, but {declared.describe()} "
                    f"is required"
                )
        return None


class InputRequirement(ContractModel):
    """
    The complete set of inputs a model consumes.

    The dual of :class:`~rade_xl.core.contract.signature.InputSignature`:
    one says what the data produced, this says what the model needs, and
    :meth:`check` is where the two meet.

    Parameters
    ----------
    dynamic
        Requirements on inputs that vary per sample.
    static
        Requirements on inputs constant across every batch -- a graph, an
        entity table.
    exact
        Whether an input the model did not ask for is an error. Default
        ``True``, which is the position worth defending: a data build
        producing something the model never reads is either wasted work or
        a sign that the user believes the model is using information it
        cannot see. The second is the dangerous one, because it is
        invisible in every metric.

        Set ``False`` for a model that is genuinely generic over its
        inputs, such as one that flattens everything it is handed.
    """

    dynamic: tuple[RequiredInput, ...] = ()
    static: tuple[RequiredInput, ...] = ()
    exact: bool = True

    @model_validator(mode="after")
    def _check_the_declaration_is_unambiguous(self) -> InputRequirement:
        """
        Reject duplicate names, or more than one unnamed entry per group.

        Returns
        -------
        InputRequirement
            The validated requirement.

        Raises
        ------
        ValidationError
            Wrapping a :class:`SpecError`. Two unnamed requirements in one
            group cannot be told apart, so which actual input satisfied
            which would depend on iteration order -- reintroducing exactly
            the order-dependence this type exists to remove.
        """
        for group, entries in (("dynamic", self.dynamic), ("static", self.static)):
            names = [entry.name for entry in entries if entry.name is not None]
            if len(names) != len(set(names)):
                raise SpecError(
                    f"the {group} requirements name an input twice: {sorted(names)}"
                )
            if sum(1 for entry in entries if entry.name is None) > 1:
                raise SpecError(
                    f"the {group} requirements contain more than one unnamed "
                    f"entry; unnamed requirements are indistinguishable, so "
                    f"which input satisfied which would depend on ordering"
                )
        return self

    @classmethod
    def unconstrained(cls) -> InputRequirement:
        """
        Return a requirement that accepts any signature.

        The honest declaration for a model that reads whatever it is given
        -- a linear model or a tree over a flattened matrix genuinely does
        not care how many blocks the data build emitted or what they are
        called.

        Preferred over leaving the requirement unset, because "this model
        has no constraints" and "nobody has written the constraints down"
        look identical from the outside and mean very different things.

        Returns
        -------
        InputRequirement
            Empty, and not exact.
        """
        return cls(exact=False)

    def check(self, signature: InputSignature, *, model: str) -> None:
        """
        Verify a data build produced what this model consumes.

        Called by the pipeline between building the data and building the
        model, which is the last moment at which a mismatch can be reported
        before it becomes a tensor of the wrong shape inside a forward pass.

        Parameters
        ----------
        signature
            What the data build declared.
        model
            The registered model name, for the error message.

        Raises
        ------
        ContractError
            If any requirement is unmet, listing every problem rather than
            the first. A user fixing a data build wants the whole list:
            reporting one mismatch per run turns a five-minute correction
            into five runs.
        """
        problems = [
            *self._problems(self.dynamic, signature.dynamic, group="dynamic"),
            *self._problems(self.static, signature.static, group="static"),
        ]
        if not problems:
            return
        raise ContractError(
            f"the data build does not match what {model!r} consumes:\n  "
            + "\n  ".join(problems)
            + "\n\nthe model declares:\n  "
            + "\n  ".join(self.describe())
        )

    def describe(self) -> tuple[str, ...]:
        """
        Return one line per declared input, for error messages and reports.

        Returns
        -------
        tuple of str
            Empty if the requirement is unconstrained.
        """
        return tuple(
            f"{group}: {entry.describe()}"
            for group, entries in (("dynamic", self.dynamic), ("static", self.static))
            for entry in entries
        )

    def _problems(
        self,
        required: tuple[RequiredInput, ...],
        actual: Mapping[str, TensorSpec],
        *,
        group: str,
    ) -> list[str]:
        """
        Return every way one group of inputs fails its requirements.

        Named requirements are matched first and by name, so that an
        unnamed requirement never consumes an input that something else
        asked for. Whatever is left over is then matched against the single
        unnamed requirement, if there is one.

        Parameters
        ----------
        required
            The declared requirements for this group.
        actual
            What the data build produced for this group.
        group
            ``"dynamic"`` or ``"static"``, for the message.

        Returns
        -------
        list of str
            One entry per problem, in a stable order.
        """
        problems: list[str] = []
        unclaimed = dict(actual)

        for entry in required:
            if entry.name is None:
                continue
            spec = unclaimed.pop(entry.name, None)
            if spec is None:
                problems.append(
                    f"{group} input {entry.name!r} is required but was not "
                    f"produced; the build produced {sorted(actual)}"
                )
                continue
            reason = entry.accepts(spec)
            if reason is not None:
                problems.append(f"{group} input {entry.name!r} {reason}")

        anonymous = next((e for e in required if e.name is None), None)
        if anonymous is not None:
            if len(unclaimed) != 1:
                problems.append(
                    f"exactly one unnamed {group} input is required "
                    f"({anonymous.describe()}), but {len(unclaimed)} were "
                    f"available: {sorted(unclaimed)}"
                )
                # Cleared so the exactness check below stays quiet: the
                # leftovers have already been reported, and saying the same
                # thing twice in different words makes a user wonder which
                # of the two problems they have.
                unclaimed.clear()
            else:
                name, spec = next(iter(unclaimed.items()))
                reason = anonymous.accepts(spec)
                if reason is not None:
                    problems.append(f"{group} input {name!r} {reason}")
                unclaimed.pop(name)

        if self.exact and unclaimed:
            problems.append(
                f"the build produced {group} input(s) {sorted(unclaimed)} that "
                f"the model does not consume; either the model is not reading "
                f"data the user believes it is reading, or the build is doing "
                f"work nothing uses"
            )
        return problems
