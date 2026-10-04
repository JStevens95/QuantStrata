"""
Train the hybrid graph-temporal network on one cluster, through the framework.

This is the Phase 3 deliverable in executable form: the flagship model run
end to end with nothing hand-wired. The script builds a run specification
and hands it to the framework's training pipeline, which does everything
else -- splits, scalers, basis selection, the graph, the training loop,
scoring, the bundle and the reports.

What it demonstrates
--------------------
Compare it to what the same run took in ``rade_ml_pt``: a build-dataset
call, a model-config dictionary, a trainer, a callback list, and the
bookkeeping to connect them. Here the model package supplies the parts
only it can know -- how to build its dataset, what its network is, which
diagnostics matter -- and the framework supplies everything else. That is
the whole thesis of ``rade_qnet``, and this file is the test of it.

Running it
----------
::

    python examples/rade_qnet/phase3_train_hybrid_single_member.py

It reads the golden fixture captured in Phase 0, so it needs no external
data and completes in seconds. Artifacts land under ``artifacts/rade_qnet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rade_qnet.core.runtime.components import get_model
from rade_qnet.core.runtime.context import RunContext
from rade_qnet.core.runtime.hashing import abbreviate_digest, digest_spec
from rade_qnet.core.spec.run import parse_run_spec
from rade_qnet.models.hybrid_gnn_rnn.pipelines.train import HybridTrainPipeline

#: The Phase 0 fixture. The same inputs the original was captured on, so a
#: number produced here is directly comparable to one in the fixture.
FIXTURE = Path(__file__).resolve().parents[2] / (
    "tests/fixtures/rade_qnet/golden/hybrid_gnn_rnn/input"
)

#: Short enough to finish in seconds. The point of the example is that the
#: wiring works, not that the model is good.
N_EPOCHS = 5


def specification() -> dict[str, object]:
    """
    Build the run specification.

    Written as a plain dictionary rather than loaded from YAML so the
    script is self-contained and every setting is visible at the point it
    matters. A production run would load the identical structure from a
    file.

    Returns
    -------
    dict
        An unvalidated specification, ready for ``parse_run_spec``.
    """
    return {
        "task": "supervised",
        "name": "hybrid-single-member",
        "seed": 0,
        "model": {
            "name": "hybrid_gnn_rnn",
            "params": {
                "units": 16,
                "gnn_layers": 2,
                "rnn_layers": 2,
                "dropout": 0.1,
            },
        },
        "source": {
            "kind": "model",
            # Settings the framework owns. The model's data spec deliberately
            # has no copy of any of these -- a second place to set the
            # sequence length is a second place for it to be wrong.
            "transforms": {
                "sequence": {"length": 4},
                # Basis selection fitted on the full scaled history rather
                # than on training rows alone. This reproduces the original's
                # behaviour, which is defect 9 in the architecture notes; the
                # framework logs a lineage warning so the choice is recorded
                # in the bundle rather than only in this comment. Set it to
                # "train" for a run whose numbers you intend to defend.
                "reduction": {"method": "basis_selection", "fit_on": "all"},
            },
            "split": {
                "kind": "chronological",
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
            },
            "loader": {"batch_size": 16, "shuffle": True},
            # Settings only this model understands.
            "params": {
                "directory": str(FIXTURE),
                "variance_threshold": 0.99,
                "encoder": {},
                "graph": {"n_neighbours": 5},
            },
        },
        "training": {
            "engine": "torch",
            "epochs": N_EPOCHS,
            "loss": "mse",
        },
        "reports": {"enabled": ("summary", "curves")},
    }


def main() -> None:
    """Run the pipeline and print where everything landed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    spec = parse_run_spec(specification())
    digest = digest_spec(spec)
    short = abbreviate_digest(digest, length=8)
    context = RunContext(
        run_id=f"hybrid-{short}",
        # The digest, not a timestamp: two runs of the same specification
        # should be recognisable as the same run, and two runs that differ
        # anywhere should be recognisable as different.
        spec_digest=digest,
        output_directory=spec.output_root / f"hybrid-{short}",
        seed=spec.seed,
    )
    # Resolved through the registry by the name in the spec, rather than
    # imported directly, because that is the path a real run takes and so
    # it is the path worth exercising.
    definition = get_model(spec.model.name)()

    result = HybridTrainPipeline(context, spec, definition).run()

    print(f"\nrun directory: {context.output_directory}")
    print(f"epochs trained: {len(result.fit.history)}")
    for split, evaluation in result.evaluations.items():
        scores = ", ".join(f"{k}={v:.4f}" for k, v in sorted(evaluation.metrics.items()))
        print(f"  {split:<11} {scores}")
    print(f"\nreports: {context.reports_directory}")


if __name__ == "__main__":
    main()
