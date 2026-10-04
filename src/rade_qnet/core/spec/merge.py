"""
Combining shared defaults with per-job overrides.

A job set states most of its configuration once and overrides a little of it
per job. Doing that correctly is a surprisingly narrow target, and the
failure mode is silent, so the rules are written down here and tested at
every nesting depth rather than inferred from the implementation.

Why this merges raw mappings rather than validated specs
---------------------------------------------------------
This module deliberately operates on plain mappings, *before* anything is
validated, and the merged result is validated once at the end.

Merging validated specs instead cannot work, and fails in the worst possible
way -- plausibly. A validated spec has no notion of which of its fields the
user set and which came from a default; both are simply fields with values.
So given shared defaults of ``model: {name: hybrid_gnn_rnn, units: 256}`` and
a job override of ``model: {gnn_layers: 1}``, validating the override on its
own produces a spec carrying ``units`` at *its default*, which then overwrites
the shared ``256``. The job trains at the wrong width, nothing raises, and
the only symptom is a model that underperforms for no visible reason.

Merging raw mappings has no such failure mode. A key the override does not
mention is simply not present in the override, so the default survives.

The rules
---------
========================== ============================================
Case                       Rule
========================== ============================================
Both sides are mappings    Recurse, key by key
Override is not a mapping  Replaces, wholesale
Either side is a sequence  Replaces, wholesale
Override value is ``None`` Replaces, with ``None``
Key only in the override   Added
Key only in the base       Kept
========================== ============================================

Two of those deserve their reasoning stated.

**Sequences replace rather than concatenate.** There is no identity to merge
list elements on -- two lists of reports, or of tags, or of column names have
no key by which an element of one corresponds to an element of the other. And
the case that matters most argues the same way: a job naming
``reports: [summary]`` means *those* reports, not those plus whatever the
defaults asked for.

**``None`` is a value, not an absence.** Many optional settings use ``None``
to mean "let the framework decide" -- ``device_index``, ``threads_per_worker``,
``workers``. A job that explicitly sets one back to ``None`` is asking for
that behaviour, and a merge that treated ``None`` as "no opinion" would make
it impossible to ask. Absence is expressed by omitting the key, which is
exactly what a configuration file makes easy.

"Deep merge" means at least three different things in common usage, which is
why none of this is left implicit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["deep_merge", "merge_all"]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge ``override`` onto ``base``, recursing into nested mappings.

    Neither argument is modified, and the result shares no mutable mapping
    with either of them -- a nested dictionary in the output is a fresh one.
    That matters here more than it usually would: the base is a job set's
    shared defaults, merged once per job, and a result that aliased it would
    let one job's later mutation reach every other job's configuration.

    Parameters
    ----------
    base
        The shared defaults. Values survive unless the override mentions
        their key.
    override
        The per-job overrides. Wins at every depth.

    Returns
    -------
    dict
        A new mapping. Nested mappings are new too.

    Examples
    --------
    A nested override keeps its siblings, which is the entire point::

        >>> deep_merge(
        ...     {"model": {"units": 256, "layers": 3}},
        ...     {"model": {"layers": 1}},
        ... )
        {'model': {'units': 256, 'layers': 1}}

    A sequence replaces rather than extends::

        >>> deep_merge({"reports": ["summary", "curves"]}, {"reports": ["summary"]})
        {'reports': ['summary']}
    """
    merged: dict[str, Any] = {key: _copied(value) for key, value in base.items()}

    for key, value in override.items():
        existing = merged.get(key)
        # Both sides mappings is the only case that recurses. Everything else
        # replaces, including the case where one side is a mapping and the
        # other is not -- there is no sensible way to merge a mapping with a
        # scalar, and guessing one would hide a configuration mistake.
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = _copied(value)

    return merged


def merge_all(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge a sequence of layers left to right, each overriding the last.

    Provided so a caller with three layers -- framework defaults, job-set
    defaults, per-job overrides -- expresses that as one call rather than as
    a fold that a reader has to evaluate to understand the precedence.

    Parameters
    ----------
    *layers
        Mappings in increasing order of precedence. The rightmost wins.

    Returns
    -------
    dict
        A new mapping. Empty if no layers were given.

    Examples
    --------
    Precedence reads left to right, like the argument list::

        >>> merge_all({"a": 1, "b": 1}, {"b": 2, "c": 2}, {"c": 3})
        {'a': 1, 'b': 2, 'c': 3}
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        merged = deep_merge(merged, layer)
    return merged


def _copied(value: Any) -> Any:  # noqa: ANN401 -- merges arbitrary YAML payloads
    """
    Return a value detached from the mapping it came from.

    Only containers are copied, and only one level at a time: a nested
    mapping is copied recursively through this function, a sequence is
    rebuilt as a list, and everything else is returned as it is.

    Scalars are returned unchanged because they are immutable, and the
    arbitrary objects a ``params`` mapping may legitimately hold are
    returned unchanged because copying them is not this module's business --
    a deep copy here would silently duplicate whatever a user put in a
    specification, which could be large and need not be copyable at all.

    Parameters
    ----------
    value
        Any value from a specification mapping.

    Returns
    -------
    object
        The value, detached if it is a container.
    """
    if isinstance(value, Mapping):
        return {key: _copied(item) for key, item in value.items()}
    # Strings and bytes are sequences, and rebuilding them as lists would turn
    # a model name into a list of characters.
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_copied(item) for item in value]
    return value
