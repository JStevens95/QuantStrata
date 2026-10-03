"""
Re-score a saved model, predict with it, and search for a better one.

This is the Phase 5 deliverable in executable form, and the first script
here that does not train anything from scratch. It trains once to have
something to work with, and then spends the rest of its time doing what a
desk actually does with a model after it exists: ask whether it still
holds, ask it for numbers, and ask whether a different configuration would
be better.

What it demonstrates
--------------------
Three claims, each of which is checked rather than asserted:

1. **A saved model re-scores to the same numbers.** Not approximately. The
   evaluation pipeline reads the split indices and the fitted scalers out
   of the bundle rather than re-deriving them, so a re-score of unchanged
   data reproduces the training metrics exactly. The script compares them
   and says so.

2. **Predictions carry where they came from.** A bare array of numbers
   cannot be reconciled with anything three weeks later. Every prediction
   comes back with the bundle version, the spec digest, the fingerprint of
   the data it was trained on, and the fingerprint of the data it was run
   against -- so a mismatch is visible rather than inferred.

3. **A search spends its budget on trials that can run.** The flagship's
   tuning override discards configurations whose block width does not
   divide by its head count, which the framework cannot know about and
   which a naive grid would discover one wasted trial at a time.

Running it
----------
::

    python examples/rade_xl/phase5_evaluate_and_tune.py

It reads the golden fixture captured in Phase 0, so it needs no external
data. Artifacts land under ``artifacts/rade_xl``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rade_xl.api import evaluate, infer, train, tune

# Importing the model package is what registers it, so a specification can
# name it as a string. The framework deliberately does not import every
# model on its own behalf: a host that never uses the flagship should not
# pay for Torch to be loaded. The earlier examples get this for free by
# importing the flagship's pipeline directly; a script that goes through
# `api` has to ask for it, which is defect 12 in the architecture notes.
import rade_xl.models.hybrid_gnn_rnn  # noqa: F401  isort:skip

#: The Phase 0 fixture, as every example in this folder uses.
FIXTURE = Path(__file__).resolve().parents[2] / (
    "tests/fixtures/rade_xl/golden/hybrid_gnn_rnn/input"
)

#: Where everything this script writes ends up.
OUTPUT_ROOT = Path("artifacts/rade_xl/phase5")

#: Short enough to finish in seconds. The point is the wiring, not the fit.
N_EPOCHS = 3

#: How many configurations the search tries. Small enough to run in a
#: lunch break, large enough that the feasibility filter has something to
#: filter.
N_TRIALS = 6


def base_specification() -> dict[str, object]:
    """
    Build the run specification everything else starts from.

    Returns
    -------
    dict
        An unvalidated specification.
    """
    return {
        "task": "supervised",
        "name": "phase5-base",
        "seed": 0,
        "model": {"name": "hybrid_gnn_rnn", "params": {"units": 16}},
        "source": {
            "kind": "model",
            "transforms": {
                "sequence": {"length": 4},
                "reduction": {"method": "basis_selection", "fit_on": "train"},
            },
            "split": {
                "kind": "chronological",
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
            },
            "loader": {"batch_size": 16, "shuffle": True},
            "params": {
                "directory": str(FIXTURE),
                "variance_threshold": 0.99,
                "encoder": {},
                "graph": {"n_neighbours": 5},
            },
        },
        "training": {"engine": "torch", "epochs": N_EPOCHS, "loss": "mse"},
        "reports": {"enabled": ()},
        "output_root": str(OUTPUT_ROOT),
    }


def search_specification() -> dict[str, object]:
    """
    Build the search specification.

    The space crosses the shared width against the fusion head count,
    which is deliberately a combination with infeasible cells: a width of
    24 cannot be split into 4 heads of equal size, and the flagship's
    tuning override drops those before a trial is spent on them.

    Returns
    -------
    dict
        An unvalidated search specification.
    """
    base = base_specification()
    base["name"] = "phase5-trial"
    return {
        "model": "hybrid_gnn_rnn",
        "base": base,
        "space": {
            "model.params.units": [16, 24, 32],
            "model.params.fusion_heads": [1, 4],
            "training.learning_rate": {"low": 1e-4, "high": 1e-2, "log": True},
        },
        "trials": N_TRIALS,
        "sampler": "random",
        "objective": "mae",
        "direction": "minimise",
        "seed": 7,
        "name": "phase5-search",
    }


def show_evaluation(bundle: Path) -> None:
    """
    Re-score the bundle and report whether it reproduced.

    Parameters
    ----------
    bundle
        The saved model directory.
    """
    result = evaluate(bundle)

    print(f"\nre-scored {result.model_name} v{result.bundle_version}")
    print(f"  trained on {result.trained_on_fingerprint}")
    print(f"  scored against {result.source_fingerprint}")
    print(f"  source changed: {result.source_changed}")
    for split, evaluation in sorted(result.evaluations.items()):
        scores = ", ".join(f"{k}={v:.6f}" for k, v in sorted(evaluation.metrics.items()))
        print(f"  {split:<11} {scores}")
    # The per-target breakdown is the flagship's eval override, not the
    # framework's. An aggregate error hides the one target that did not
    # replicate, which is usually the one worth looking at.
    for key, note in sorted(result.notes.items()):
        print(f"  {key}: {note}")


def show_predictions(bundle: Path) -> None:
    """
    Predict with the bundle and report the provenance that came back.

    Parameters
    ----------
    bundle
        The saved model directory.
    """
    predictions = infer(bundle, split="test")

    print(f"\npredicted {predictions.n_predictions} value(s)")
    print(f"  in original units: {predictions.in_original_units}")
    for key, value in sorted(predictions.provenance.items()):
        print(f"  {key}: {value}")


def show_search() -> None:
    """Run the search and report what it found and what it skipped."""
    result = tune(search_specification(), output_root=OUTPUT_ROOT / "searches")

    print(f"\n{result.describe()}")
    for record in result.trials:
        if record.succeeded:
            print(f"  trial {record.trial}: {record.objective:.6f}  {record.overrides}")
        else:
            print(f"  trial {record.trial}: {record.failure_kind} -- {record.failure_message}")

    # The winning bundle is a bundle like any other, which is the point of
    # giving every trial its own directory: acting on a search does not
    # mean retraining its winner.
    if result.best is not None and result.best.bundle_directory is not None:
        print("\nthe winner, re-scored from disk:")
        show_evaluation(Path(result.best.bundle_directory))


def main() -> None:
    """Train once, then evaluate, predict and search."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    trained = train(base_specification(), output_root=OUTPUT_ROOT)
    bundle = Path(trained.bundle_directory)
    print(f"\ntrained and saved to {bundle}")

    show_evaluation(bundle)
    show_predictions(bundle)
    show_search()


if __name__ == "__main__":
    main()
