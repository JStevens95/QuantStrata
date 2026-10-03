"""
Turning a portfolio into the jobs a set fans out over.

This is the bridge between "a book of four clusters" and "a job set with four
jobs". It produces a :class:`~rade_xl.core.spec.jobs.JobSetSpec`: shared
defaults, one job per cluster, each job overriding the directory it reads and
anything the caller wants to vary.

Why expansion happens here and not in the runner
------------------------------------------------
``orchestration`` may not import ``domains`` or ``models``. That is the
layering rule, and it is not bureaucratic: the job-set runner is a general
piece of machinery that knows how to run *jobs*, and the moment it knows what
a cluster is, it becomes a P&L tool that happens to run jobs. Every other
domain would then need its own runner.

So the expansion is pushed up. By the time the runner sees anything, there
are only jobs. A caller writes the job set by hand, or a domain writes it,
or ``api`` does -- and the runner cannot tell the difference, which is
exactly the property that makes it reusable.

Complexity per cluster
----------------------
The reason a portfolio is a job set rather than a loop is that clusters are
not alike. A liquid cluster with abundant history supports a wider, deeper
model than a thin one, and forcing both to the same configuration means
either underfitting the first or overfitting the second.
:func:`job_set_for` therefore takes a per-cluster override hook, which is the
mechanism behind that whole proposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.runtime.logging import get_logger
from ...core.spec.jobs import JobSetSpec, parse_job_set_spec
from ...core.spec.merge import deep_merge

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from .portfolio import Cluster, Portfolio

__all__ = ["FINGERPRINT_KEY", "cluster_overrides", "fingerprint_tag", "job_set_for"]

_LOGGER = get_logger(__name__)

#: Prefix the portfolio's snapshot digest is tagged with, so every bundle in
#: the set can be traced back to the data it was trained on.
FINGERPRINT_KEY = "portfolio_fingerprint"


def fingerprint_tag(fingerprint: str) -> str:
    """
    Encode a snapshot digest as a set tag.

    A set's tags are free-form labels rather than a mapping, so a key-value
    fact has to be encoded into one. Done here, in one place, because the
    writer and every future reader have to agree on the encoding and a
    convention applied at two call sites is a convention that will
    eventually be applied at one.

    Parameters
    ----------
    fingerprint
        The portfolio's snapshot digest.

    Returns
    -------
    str
        For example ``portfolio_fingerprint=7f3a9c21...``.
    """
    return f"{FINGERPRINT_KEY}={fingerprint}"


def cluster_overrides(cluster: Cluster) -> dict[str, Any]:
    """
    Return the overrides every cluster job needs, whatever the model.

    Only the source directory. Deliberately minimal: anything else would be
    this module deciding how a model should be configured, which it is in no
    position to know -- the whole point of the framework is that the model
    declares that.

    Parameters
    ----------
    cluster
        The cluster.

    Returns
    -------
    dict
        An override fragment, merged over the set's defaults.
    """
    return {"source": {"params": {"directory": str(cluster.directory)}}}


def job_set_for(
    portfolio: Portfolio,
    *,
    defaults: Mapping[str, Any],
    output_root: Path,
    name: str | None = None,
    placement: Mapping[str, Any] | None = None,
    overrides_for: Callable[[Cluster], Mapping[str, Any]] | None = None,
) -> JobSetSpec:
    """
    Expand a portfolio into a job set, one job per cluster.

    Parameters
    ----------
    portfolio
        The clusters to train over.
    defaults
        The run-specification fragment every job shares: the model, the
        source kind, the training settings. Merged *under* each job's
        overrides, so a per-cluster setting wins and everything it does not
        mention survives.
    output_root
        Where the set writes.
    name
        The set's label, used in its run identifier and so in its directory
        name. Defaults to ``portfolio``.
    placement
        Where jobs run. ``None`` leaves it to the placement policy, which is
        the normal path.
    overrides_for
        Per-cluster overrides, merged over :func:`cluster_overrides`. This is
        the hook that lets a liquid cluster get a wider model than a thin
        one, which is the reason a portfolio is a job set at all.

    Returns
    -------
    JobSetSpec
        A validated job set. Validated here rather than at dispatch, so a
        portfolio that expands into something misconfigured fails at
        expansion -- where the error can say which cluster -- rather than
        after some of its jobs have already trained.
    """
    jobs = []
    for cluster in portfolio:
        overrides: dict[str, Any] = dict(cluster_overrides(cluster))
        if overrides_for is not None:
            # Merged by the job-set loader rather than here, so that a
            # per-cluster fragment follows exactly the same merge rules as
            # one written by hand in a file. Two merge implementations would
            # eventually disagree, and the disagreement would look like a
            # model behaving differently depending on how it was launched.
            overrides = _merged(overrides, overrides_for(cluster))
        jobs.append({"id": cluster.name, "overrides": overrides, "description": cluster.describe()})

    spec = parse_job_set_spec(
        {
            "name": name or "portfolio",
            "output_root": str(output_root),
            "defaults": dict(defaults),
            "jobs": jobs,
            **({"placement": dict(placement)} if placement is not None else {}),
            # Carried as a tag so it reaches every bundle. A result that
            # looks surprising six months from now is asked one question
            # first: was this trained on the data I think it was.
            "tags": [fingerprint_tag(portfolio.fingerprint)],
        }
    )
    _LOGGER.info("expanded %s into %d job(s)", portfolio.describe(), len(spec.jobs))
    return spec


def _merged(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """
    Deep-merge two override fragments.

    Parameters
    ----------
    base
        The fragment this module produced.
    extra
        The caller's fragment, which wins where the two overlap.

    Returns
    -------
    dict
        The merged fragment.
    """
    return deep_merge(base, extra)
