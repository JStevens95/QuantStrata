"""
Turning a set of data groups into the jobs a job set fans out over.

This is the bridge between "a directory holding four groups" and "a job set
with four jobs". It produces a :class:`~rade_qnet.core.spec.jobs.JobSetSpec`:
shared defaults, one job per group, each job overriding the directory it
reads and anything the caller wants to vary.

Why expansion is separate from the runner
-----------------------------------------
:mod:`rade_qnet.orchestration.jobs.set` knows how to run *jobs* and nothing
else. By the time it sees anything, there are only jobs -- a caller wrote the
set by hand, or this module expanded a group set into one, and the runner
cannot tell the difference. That is exactly the property that makes the
runner reusable: the moment it knew what a group was, it would stop being a
job runner and start being a group runner.

Settings per group
------------------
The reason a group set is a job set rather than a loop is that groups are not
alike. One with abundant history supports a wider, deeper model than a thin
one, and forcing both to the same configuration means either underfitting the
first or overfitting the second. :func:`job_set_for_groups` therefore takes a
per-group override hook, which is the mechanism behind that whole
proposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.provenance.logging import get_logger
from ...core.spec.jobs import JobSetSpec, parse_job_set_spec
from ...core.spec.merge import deep_merge

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from .groups import DataGroup, GroupSet

__all__ = ["FINGERPRINT_KEY", "fingerprint_tag", "group_overrides", "job_set_for_groups"]

_LOGGER = get_logger(__name__)

#: Prefix the snapshot digest is tagged with, so every bundle in the set can
#: be traced back to the data it was trained on.
FINGERPRINT_KEY = "data_fingerprint"


def fingerprint_tag(fingerprint: str) -> str:
    """
    Encode a snapshot digest as a set tag.

    A set's tags are free-form labels rather than a mapping, so a key-value
    fact has to be encoded into one. Done here, in one place, because the
    writer and every future reader have to agree on the encoding, and a
    convention applied at two call sites is a convention that will eventually
    be applied at one.

    Parameters
    ----------
    fingerprint
        The group set's snapshot digest.

    Returns
    -------
    str
        For example ``data_fingerprint=7f3a9c21...``.
    """
    return f"{FINGERPRINT_KEY}={fingerprint}"


def group_overrides(group: DataGroup) -> dict[str, Any]:
    """
    Return the overrides every group job needs, whatever the model.

    Only the source directory. Deliberately minimal: anything else would be
    this module deciding how a model should be configured, which it is in no
    position to know -- the whole point of the framework is that the model
    declares that.

    Parameters
    ----------
    group
        The group.

    Returns
    -------
    dict
        An override fragment, merged over the set's defaults.
    """
    return {"source": {"params": {"directory": str(group.directory)}}}


def job_set_for_groups(
    groups: GroupSet,
    *,
    defaults: Mapping[str, Any],
    output_root: Path,
    name: str | None = None,
    placement: Mapping[str, Any] | None = None,
    overrides_for: Callable[[DataGroup], Mapping[str, Any]] | None = None,
) -> JobSetSpec:
    """
    Expand a group set into a job set, one job per group.

    Parameters
    ----------
    groups
        The groups to train over.
    defaults
        The run-specification fragment every job shares: the model, the
        source kind, the training settings. Merged *under* each job's
        overrides, so a per-group setting wins and everything it does not
        mention survives.
    output_root
        Where the set writes.
    name
        The set's label, used in its run identifier and so in its directory
        name. Defaults to ``groups``.
    placement
        Where jobs run. ``None`` leaves it to the placement policy, which is
        the normal path.
    overrides_for
        Per-group overrides, merged over :func:`group_overrides`. This is the
        hook that lets a data-rich group get a wider model than a thin one,
        which is the reason a group set is a job set at all.

    Returns
    -------
    JobSetSpec
        A validated job set. Validated here rather than at dispatch, so a
        set that expands into something misconfigured fails at expansion --
        where the error can say which group -- rather than after some of its
        jobs have already trained.
    """
    jobs = []
    for group in groups:
        overrides: dict[str, Any] = dict(group_overrides(group))
        if overrides_for is not None:
            # Merged by the job-set loader's own rules rather than ad hoc
            # here, so a per-group fragment follows exactly the same merge
            # semantics as one written by hand in a file. Two merge
            # implementations would eventually disagree, and the
            # disagreement would look like a model behaving differently
            # depending on how it was launched.
            overrides = deep_merge(overrides, overrides_for(group))
        jobs.append({"id": group.name, "overrides": overrides, "description": group.describe()})

    spec = parse_job_set_spec(
        {
            "name": name or "groups",
            "output_root": str(output_root),
            "defaults": dict(defaults),
            "jobs": jobs,
            **({"placement": dict(placement)} if placement is not None else {}),
            # Carried as a tag so it reaches every bundle. A result that
            # looks surprising six months from now is asked one question
            # first: was this trained on the data I think it was.
            "tags": [fingerprint_tag(groups.fingerprint)],
        }
    )
    _LOGGER.info("expanded %s into %d job(s)", groups.describe(), len(spec.jobs))
    return spec
