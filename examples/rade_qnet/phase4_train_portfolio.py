"""
Train the flagship across a portfolio of clusters, in parallel.

This is the Phase 4 deliverable in executable form. Phase 3 trained one
cluster; this trains several at once, each with its own model complexity,
across however many processes the placement policy decides the machine can
support -- and it does so in one call.

What it demonstrates
--------------------
Three things that only exist once a run can fan out.

**Per-cluster complexity.** A liquid cluster with abundant history supports
a wider model than a thin one. The ``complexity_for`` hook below sizes each
model from the cluster it will train on, which is the whole reason a
portfolio is a job set rather than a loop over clusters.

**Partial failure.** One cluster failing does not cost the others. The set
below deliberately includes a cluster pointing at a directory that is not
there, and the run completes with that job recorded as a failure beside the
ones that worked.

**Placement as an operational choice.** Passing ``--sequential`` changes
where the jobs run and nothing else: with the thread budget pinned in the
specification, the metrics are bit-identical either way. That property is
the Phase 4 gate, and it is checked by
``tests/rade_qnet/orchestration/jobs/test_jobs_parity.py``.

Running it
----------
::

    python examples/rade_qnet/phase4_train_portfolio.py
    python examples/rade_qnet/phase4_train_portfolio.py --sequential

It builds its portfolio from the golden fixture captured in Phase 0, so it
needs no external data. Artifacts land under ``artifacts/rade_qnet/phase4``.

The ``if __name__ == "__main__"`` guard at the bottom is **required**, not
decorative: a process pool using the spawn start method re-imports the main
module in every worker, and without the guard each worker would launch the
whole portfolio again.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from rade_qnet import api
from rade_qnet.domains.pnl.portfolio import MANIFEST_FILENAME

# Imported for its registration side effect, which is what makes
# `hybrid_gnn_rnn` resolvable by name. The workers do not rely on this
# import: the job payload names the module and each worker replays it, so
# the set would run identically from a CLI that never imported the model.
import rade_qnet.models.hybrid_gnn_rnn.register  # noqa: F401  (isort: skip)

#: The Phase 0 fixture, copied into each cluster directory below.
FIXTURE = Path(__file__).resolve().parents[2] / (
    "tests/fixtures/rade_qnet/golden/hybrid_gnn_rnn/input"
)

#: Where this example writes.
OUTPUT_ROOT = Path("artifacts/rade_qnet/phase4")

#: The clusters to build, and how many elementary instruments to claim for
#: each. The counts drive the complexity hook; the data is the same fixture
#: in every case, because this example is about fan-out rather than about
#: any particular book.
CLUSTERS: tuple[tuple[str, int], ...] = (
    ("FX__G10", 24),
    ("FX__EM", 11),
    ("RATES__USD", 18),
)

#: A cluster declared in the manifest whose directory is deliberately absent,
#: so the run demonstrates that one failure does not cost the others.
BROKEN_CLUSTER = "RATES__GBP"


def build_portfolio(root: Path) -> Path:
    """
    Lay out a portfolio directory from the golden fixture.

    A real portfolio is produced upstream by whatever reads the firm's
    books. Building one here keeps the example self-contained, and the
    layout is the real one: a manifest listing clusters, and a directory
    per cluster.

    Parameters
    ----------
    root
        Where to build it. Rebuilt from scratch each run, so a changed
        example never reads a stale directory.

    Returns
    -------
    pathlib.Path
        The portfolio directory.
    """
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    entries = []
    for name, n_elementary in CLUSTERS:
        shutil.copytree(FIXTURE, root / name)
        entries.append(
            {
                "name": name,
                "elementary_ids": [f"{name}_E{index}" for index in range(n_elementary)],
                "target_ids": [f"{name}_T0"],
                "asset_class": name.split("__")[0],
            }
        )

    # Declared but never created: the partial-failure demonstration.
    entries.append({"name": BROKEN_CLUSTER, "elementary_ids": [], "target_ids": []})
    (root / BROKEN_CLUSTER).mkdir()

    (root / MANIFEST_FILENAME).write_text(json.dumps({"clusters": entries}, indent=2))
    return root


def complexity_for(cluster) -> dict:  # noqa: ANN001  - the hook's own type
    """
    Size a cluster's model from the cluster itself.

    The hook that makes a portfolio a job set. Forcing one configuration on
    every cluster means either underfitting the liquid ones or overfitting
    the thin ones, and a rule written against the cluster beats a
    hand-maintained lookup that goes stale the first time the book changes.

    Parameters
    ----------
    cluster
        The cluster about to be trained.

    Returns
    -------
    dict
        An override fragment merged over the shared defaults.
    """
    # Wider for clusters with more to learn from, bounded at both ends so a
    # single outsized cluster cannot produce a model that will not fit in
    # memory alongside the others running beside it.
    units = max(8, min(32, cluster.universe.n_elementary))
    return {"model": {"params": {"units": units}}}


def main() -> None:
    """Build a portfolio, train every cluster, and report the set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="run jobs one after another instead of across processes",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

    portfolio_root = build_portfolio(OUTPUT_ROOT / "portfolio")

    manifest = api.train_portfolio(
        portfolio_root,
        defaults={
            "model": {"name": "hybrid_gnn_rnn"},
            "source": {"kind": "model", "transforms": {"sequence": {"length": 4}}},
            "training": {"engine": "torch", "epochs": 3},
            # Pinned, so placement cannot change the numbers. See
            # `PHASE_4_JOB_SETS.md` §8.1: the thread budget fixes the order
            # a reduction accumulates in, and therefore the last few
            # significant figures.
            "hardware": {"device": "cpu", "determinism": "strict", "threads_per_worker": 1},
            "reports": {"enabled": []},
        },
        output_root=OUTPUT_ROOT / "runs",
        name="portfolio",
        placement={"executor": "local"} if arguments.sequential else None,
        overrides_for=complexity_for,
    )

    print(f"\n{manifest.summary()}")
    print(f"placement: {manifest.placement} ({manifest.placement_reason})\n")

    width = max(len(record.job_id) for record in manifest.jobs)
    for record in manifest.jobs:
        if record.succeeded:
            r2 = record.metric("test", "r2")
            print(f"  {record.job_id:<{width}}  r2={r2: .6f}  {record.wall_seconds:5.1f}s")
        else:
            print(f"  {record.job_id:<{width}}  FAILED: {record.failure_kind}")

    print(f"\nmanifest: {OUTPUT_ROOT / 'runs' / manifest.run_id}")


if __name__ == "__main__":
    # Required. A spawn-based pool re-imports this module in every worker,
    # and without the guard each worker would launch the whole portfolio.
    main()
