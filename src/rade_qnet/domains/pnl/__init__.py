"""
The P&L replication domain.

Holds everything that is true about the replication problem and nothing that
is true about a particular model. The flagship network reads its data through
these adapters, but so could a ridge regression -- the domain defines the
problem, not the solution.

Where this package sits
-----------------------
``models`` may import ``domains``; ``domains`` may not import ``models``, and
``orchestration`` may import neither. That one-way flow is what keeps a
second model from having to depend on the flagship in order to describe the
same book, and what keeps the job-set runner from becoming a P&L tool.

The practical consequence is that a portfolio is expanded into jobs *here*,
before the runner sees anything. By the time work is dispatched there are
only jobs, which is exactly why the same runner serves any domain.

Modules
-------
``universe.py``
    Which instrument each column refers to. The contract between a matrix
    position and a real position, and the thing that keeps a prediction
    attributable.
``portfolio.py``
    Reads a portfolio's clusters and fingerprints the snapshot, so a bundle
    records which version of the book it was trained on.
``clusters.py``
    Expands a portfolio into the job set a run fans out over, including the
    per-cluster complexity overrides that are the reason it is a set.

Planned
-------
``metrics.py``
    Replication quality as the desk understands it: unexplained P&L, tail
    replication error and hedge-ratio stability. Deferred to Phase 5, which
    is where the evaluation surface they belong to is built.
"""

from .clusters import cluster_overrides, job_set_for
from .portfolio import Cluster, Portfolio, read_portfolio
from .universe import Universe

__all__ = [
    "Cluster",
    "Portfolio",
    "Universe",
    "cluster_overrides",
    "job_set_for",
    "read_portfolio",
]
