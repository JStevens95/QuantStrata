"""
The base class for contract metadata.

Contracts come in two shapes, and the split is principled rather than
historical:

**Metadata contracts** are pure description -- shapes, dtypes, digests,
metrics, file lists. They are written to disk as JSON and read back months
later, so they need validation, exact round-tripping and immutability. Those
derive from :class:`ContractModel`.

**Payload contracts** carry live data -- arrays, loaders, a built model. They
are never serialised as a whole, they are constructed and consumed within one
process, and routing a loader through a validation layer would be pointless
overhead. Those are frozen dataclasses.

The rule is therefore: *pydantic where it must round-trip through JSON, a
dataclass where it holds live data.*

How validation failures surface
-------------------------------
The two shapes report failures differently, and the ``Raises`` sections in
this package reflect that rather than papering over it.

A **dataclass** validates in ``__post_init__`` and its exception propagates
unchanged, so it raises :class:`~rade_qnet.core.runtime.errors.ContractError`
(or ``SpecError`` for :class:`~rade_qnet.core.contract.data.SplitIndices`, where
the fault is a split specification rather than a payload) exactly as
documented.

A **contract model** validates through pydantic, which *collects* a validator
exception instead of letting it propagate -- so the caller sees
``ValidationError`` carrying the ``SpecError`` message and the dotted path of
the offending field. That is the right outcome here: unlike a run spec, a
contract model is constructed by framework code rather than parsed from user
input, so an invalid one is a bug and pydantic's report locates it precisely.
See :mod:`rade_qnet.core.spec.base` for the matching policy on specifications.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["ContractModel"]


class ContractModel(BaseModel):
    """
    Base class for contract metadata that is persisted as JSON.

    Carries the same immutability and strictness as a specification, for
    similar reasons: a stage must not be able to mutate a payload another
    stage is holding, and a manifest read from a six-month-old bundle must
    either validate or fail loudly rather than load with fields missing.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        # Contract metadata is description, never a live object. Keeping this
        # false makes "a signature cannot smuggle a tensor" structural rather
        # than a convention.
        arbitrary_types_allowed=False,
    )
